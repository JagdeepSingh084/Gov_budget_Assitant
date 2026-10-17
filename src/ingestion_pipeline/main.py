#!/usr/bin/env python3
"""
Ingestion pipeline for the RAG system.
Executes chunking and embedding generation in sequence.

Usage:
    python -m src.ingestion_pipeline.main

This script:
1. Processes all files in the corpus/ directory to create data/chunks.jsonl
2. Generates embeddings from the chunks and saves to data/embeddings.faiss and data/embeddings.meta.pkl
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.injection.chunking import process_corpus_to_jsonl
    from src.services.embeddings import main as embeddings_main
except ImportError:
    from injection.chunking import process_corpus_to_jsonl
    from services.embeddings import main as embeddings_main


def run_ingestion_pipeline():
    """Run the complete ingestion pipeline: chunking -> embeddings."""
    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Generate chunks from corpus
    print("\nStep 1: Generating chunks from corpus...")
    corpus_dir = "corpus"
    chunks_file = "data/chunks.jsonl"

    # Ensure data directory exists
    os.makedirs(os.path.dirname(chunks_file), exist_ok=True)

    try:
        chunk_count = process_corpus_to_jsonl(corpus_dir=corpus_dir, output_jsonl=chunks_file)
        if chunk_count == 0:
            print("Error: No chunks were generated. Check the corpus directory.")
            return 1
    except Exception as e:
        print(f"Error during chunk generation: {e}")
        return 1

    # Step 2: Generate embeddings from chunks
    print("\nStep 2: Generating embeddings from chunks...")
    embeddings_file_base = "data/embeddings"

    try:
        # We'll call the embeddings main function with our specific arguments
        # Temporarily modify sys.argv to pass our arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            "embeddings.py",
            "--chunks-file", chunks_file,
            "--output-file", embeddings_file_base
        ]

        embeddings_main()

        # Restore original argv
        sys.argv = original_argv
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Ingestion pipeline completed successfully!")
    print(f"Chunks saved to: {chunks_file}")
    print(f"Embeddings index saved to: {embeddings_file_base}.faiss")
    print(f"Embeddings metadata saved to: {embeddings_file_base}.meta.pkl")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(run_ingestion_pipeline())