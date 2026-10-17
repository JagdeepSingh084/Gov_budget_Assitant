#!/usr/bin/env python3
"""
Embeddings generation service for the RAG pipeline.

Supports Gemini Embedding 2 as primary and SentenceTransformer as fallback.
Stores embeddings in a FAISS index for efficient similarity search.

Usage:
    python -m src.services.embeddings [--chunks-file CHUNKS_FILE] [--output-file OUTPUT_FILE] [--provider gemini|local]
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

try:
    import src.config as config
except ImportError:
    import config

MAX_RETRY_DELAY_SECONDS = 3600  # 1 hour limit
GEMINI_CHECKPOINT_FILE = "data/gemini_embeddings_checkpoint.npy"

# Cached singletons to avoid reloading models on every query
_GEMINI_CLIENT: Optional[genai.Client] = None
_LOCAL_MODEL: Optional[SentenceTransformer] = None


def get_gemini_client() -> genai.Client:
    """Returns cached Google GenAI client instance."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=config.GEMINI_API_KEY)
    return _GEMINI_CLIENT


def get_local_model() -> SentenceTransformer:
    """Returns cached SentenceTransformer model instance."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = SentenceTransformer(config.FALLBACK_QUERY_EMBEDDING_MODEL)
    return _LOCAL_MODEL


def is_gemini_model(model_name: Optional[str] = None) -> bool:
    """Checks whether a given model name refers to Gemini."""
    target = model_name or config.PRIMARY_GEMINI_EMBEDDING_MODEL
    return "gemini" in str(target).lower()


# ---------------------------------------------------------------------------
# Core Embedding Helpers
# ---------------------------------------------------------------------------

def _gemini_embed(
    texts: List[str],
    max_retries: int = 5
) -> np.ndarray:
    """
    Embeds a list of texts using Gemini API with automatic 429 rate-limit backoff.
    Parses Google's suggested retryDelay and pauses before retrying.
    If wait time escalates to > 1 hour, raises TimeoutError to trigger immediate fallback.
    """
    client = get_gemini_client()
    contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]

    for attempt in range(max_retries):
        try:
            resp = client.models.embed_content(
                model=config.PRIMARY_GEMINI_EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            vecs = np.array([emb.values for emb in resp.embeddings], dtype=np.float32)
            faiss.normalize_L2(vecs)
            return vecs
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if is_rate_limit and attempt < max_retries - 1:
                # Extract Google's suggested retry delay if present
                delay = None
                match_retry = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                if match_retry:
                    delay = float(match_retry.group(1))
                else:
                    match_delay = re.search(r"['\"]retryDelay['\"]:\s*['\"](\d+)s['\"]", err_str)
                    if match_delay:
                        delay = float(match_delay.group(1))

                wait_sec = (delay + 2.0) if delay is not None else min(60.0, (2 ** attempt) * 10.0 + 5.0)

                # If wait time escalates to more than 1 hour (e.g. daily quota reached), cancel immediately
                if wait_sec > MAX_RETRY_DELAY_SECONDS:
                    raise TimeoutError(
                        f"Wait time ({wait_sec:.0f}s / {wait_sec / 3600:.1f}h) exceeds 1 hour limit. "
                        "Gemini API quota exhausted."
                    )

                print(f"  [Rate limit 429] Quota exceeded. Waiting {wait_sec:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_sec)
            else:
                raise


def _local_embed(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """Embeds a list of texts using local SentenceTransformer."""
    model = get_local_model()
    vecs = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public Embedding API
# ---------------------------------------------------------------------------

def embed_query(query: str, target_dim: Optional[int] = None) -> np.ndarray:
    """
    Embeds a search query according to config settings:
    - Primary: Gemini Embedding 2
    - Fallback: Local SentenceTransformer (used on API failure or dimension mismatch)
    """
    if config.PRIMARY_QUEERY_EMBEDDING_PROVIDER.lower() == "gemini" and config.GEMINI_API_KEY:
        try:
            vec = _gemini_embed([query])
            if target_dim is None or vec.shape[1] == target_dim:
                return vec
            print(f"[Notice] FAISS index is {target_dim}-dim, but Gemini is {vec.shape[1]}-dim. Using fallback model.")
        except Exception as e:
            print(f"[Notice] Gemini query embedding failed ({e}); falling back to local model.")

    return _local_embed([query])


def generate_embeddings(chunks: List[Dict[str, Any]], provider: Optional[str] = None) -> np.ndarray:
    """
    Generates embeddings for chunks using the specified or configured provider.
    Includes rate-limit pacing and automatic disk checkpointing to safely resume on interruption.
    If retry wait time exceeds 1 hour or Gemini fails, previous partial embeddings are deleted
    and the fallback model is used.
    """
    texts = [c['text'] for c in chunks]
    selected_provider = (provider or config.PRIMARY_QUEERY_EMBEDDING_PROVIDER).lower()

    if selected_provider == "gemini" and config.GEMINI_API_KEY:
        try:
            print(f"Embedding {len(texts)} chunks with Gemini ({config.PRIMARY_GEMINI_EMBEDDING_MODEL})...")
            all_vecs = []
            start_idx = 0

            # Resume from checkpoint if present
            if os.path.exists(GEMINI_CHECKPOINT_FILE):
                try:
                    cached = np.load(GEMINI_CHECKPOINT_FILE)
                    if 0 < len(cached) <= len(texts):
                        all_vecs = [cached]
                        start_idx = len(cached)
                        print(f"  Resuming from checkpoint: {start_idx}/{len(texts)} chunks already completed.")
                except Exception:
                    all_vecs = []
                    start_idx = 0

            for i in range(start_idx, len(texts), config.EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + config.EMBEDDING_BATCH_SIZE]
                vec = _gemini_embed(batch)
                all_vecs.append(vec)

                # Save intermediate checkpoint every 3 batches or on completion
                if len(all_vecs) % 3 == 0 or (i + config.EMBEDDING_BATCH_SIZE >= len(texts)):
                    np.save(GEMINI_CHECKPOINT_FILE, np.vstack(all_vecs))

                processed_count = min(i + config.EMBEDDING_BATCH_SIZE, len(texts))
                print(f"  Processed {processed_count}/{len(texts)} chunks...")

                # Inter-batch pacing to avoid hitting burst rate limits
                time.sleep(2.0)

            final_embeddings = np.vstack(all_vecs)
            if os.path.exists(GEMINI_CHECKPOINT_FILE):
                try:
                    os.remove(GEMINI_CHECKPOINT_FILE)
                except OSError:
                    pass

            return final_embeddings
        except Exception as e:
            print(f"[Notice] Gemini chunk embedding cancelled ({e}).")
            print(f"Cleaning up previous/partial embeddings and falling back to {config.FALLBACK_QUERY_EMBEDDING_MODEL}...")

            # Clean up partial checkpoint
            if os.path.exists(GEMINI_CHECKPOINT_FILE):
                try:
                    os.remove(GEMINI_CHECKPOINT_FILE)
                    print(f"  [Clean up] Deleted checkpoint file: {GEMINI_CHECKPOINT_FILE}")
                except OSError as err:
                    print(f"  [Clean up] Error deleting {GEMINI_CHECKPOINT_FILE}: {err}")

            # Clean up previous partial embedding artifacts
            for ext in [".faiss", ".meta.pkl"]:
                artifact = f"data/embeddings{ext}"
                if os.path.exists(artifact):
                    try:
                        os.remove(artifact)
                        print(f"  [Clean up] Deleted previous embedding file: {artifact}")
                    except OSError as err:
                        print(f"  [Clean up] Error deleting {artifact}: {err}")

    print(f"Embedding {len(texts)} chunks with local model ({config.FALLBACK_QUERY_EMBEDDING_MODEL})...")
    return _local_embed(texts, show_progress=config.EMBEDDING_SHOW_PROGRESS_BAR)


# ---------------------------------------------------------------------------
# Storage & CLI
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: np.ndarray, chunks: List[Dict[str, Any]], output_file_base: str) -> None:
    """Saves embeddings in FAISS index and chunk metadata in pickle file."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, f"{output_file_base}.faiss")
    with open(f"{output_file_base}.meta.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {embeddings.shape[0]} embeddings ({dim} dims) to {output_file_base}.*")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks and store in FAISS.")
    parser.add_argument("--chunks-file", default="data/chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--output-file", default="data/embeddings", help="Base path for .faiss and .meta.pkl")
    parser.add_argument(
        "--provider",
        default=config.PRIMARY_QUEERY_EMBEDDING_PROVIDER,
        choices=["gemini", "local"],
        help="Embedding provider ('gemini' or 'local')"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.chunks_file):
        print(f"Error: Chunks file not found: {args.chunks_file}")
        sys.exit(1)

    with open(args.chunks_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    embeddings = generate_embeddings(chunks, provider=args.provider)
    save_embeddings(embeddings, chunks, args.output_file)


if __name__ == "__main__":
    main()
