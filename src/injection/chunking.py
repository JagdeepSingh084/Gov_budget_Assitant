"""
Document- and structure-aware recursive chunking module for the RAG pipeline.

Handles:
1. Tabular chunks (CSV, XLS): passed through directly with `page=None` and verified metadata.
2. PDF text: page-aware AND structure-aware recursive chunking:
   - Preserves strict page boundaries via [Page X] tags (no cross-page bleeding).
   - Recursively splits within each page using structural separators in priority order:
     Paragraphs (\\n\\n) -> Lines/Table rows (\\n) -> Sentences -> Words.
   - Never slices a table row or sentence in half mid-thought.
   - Prepends a contextual attribution banner to every chunk: `[{source}, Page {page_num}]`.
   - Stores exact integer `page` number in chunk metadata for accurate citation.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

try:
    import config
except ImportError:
    from src import config


def recursive_split_text(
    text: str,
    max_words: int = 120,
    overlap_words: int = 25
) -> List[str]:
    """
    Recursively splits text by structural separators in priority order:
    1. Paragraphs (\\n\\n)
    2. Lines / Table rows (\\n)
    3. Sentences (ending in . / ! / ?)
    4. Word-level fallback (sliding window)
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    # 1. Try splitting by paragraph (\\n\\n)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = []
        current_len = 0
        for p in paragraphs:
            p_len = len(p.split())
            if current_len + p_len <= max_words:
                current_chunk.append(p)
                current_len += p_len
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                if p_len > max_words:
                    chunks.extend(recursive_split_text(p, max_words, overlap_words))
                    current_chunk, current_len = [], 0
                else:
                    current_chunk = [p]
                    current_len = p_len
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    # 2. Try splitting by line / table row (\\n)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        chunks = []
        current_chunk = []
        current_len = 0
        for l in lines:
            l_len = len(l.split())
            if current_len + l_len <= max_words:
                current_chunk.append(l)
                current_len += l_len
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                if l_len > max_words:
                    chunks.extend(recursive_split_text(l, max_words, overlap_words))
                    current_chunk, current_len = [], 0
                else:
                    current_chunk = [l]
                    current_len = l_len
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    # 3. Try splitting by sentence boundary
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) > 1:
        chunks = []
        current_chunk = []
        current_len = 0
        for s in sentences:
            s_len = len(s.split())
            if current_len + s_len <= max_words:
                current_chunk.append(s)
                current_len += s_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_len = s_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    # 4. Final fallback: word-level sliding window
    step = max(1, max_words - overlap_words)
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]


def create_chunks_from_text(
    text: Optional[str] = None,
    source: str = "unknown",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    chunks: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Creates contextual chunks from plain text (PDF) or pre-parsed tabular chunks (CSV/XLS).

    Args:
        text (str): Extracted PDF text with [Page X] markers.
        source (str): Source filename.
        chunk_size (int): Max words per chunk (default from config).
        overlap (int): Overlap in words between chunks.
        chunks (list): Pre-parsed tabular chunks (CSV/XLS).

    Returns:
        List of chunk dicts: {"text": ..., "source": ..., "page": ..., "chunk_index": ...}
    """
    if chunk_size is None:
        chunk_size = getattr(config, "CHUNK_SIZE", 120)
    if overlap is None:
        overlap = getattr(config, "CHUNK_OVERLAP", 25)

    # Case 1: Tabular data (CSV, XLS, XLSX)
    if chunks is not None:
        out = []
        for i, ch in enumerate(chunks):
            new_ch = ch.copy()
            new_ch.setdefault("chunk_index", i)
            new_ch.setdefault("source", source)
            new_ch.setdefault("page", None)
            out.append(new_ch)
        return out

    # Case 2: PDF text with [Page X] markers
    if not text or not text.strip():
        return []

    # Split text into pages using [Page X] markers
    page_splits = re.split(r'\[Page\s+(\d+)\]', text)
    if len(page_splits) <= 1:
        pages = [(1, text.strip())]
    else:
        pages = []
        for i in range(1, len(page_splits), 2):
            p_num = int(page_splits[i])
            p_content = page_splits[i + 1].strip() if i + 1 < len(page_splits) else ""
            if p_content:
                pages.append((p_num, p_content))

    chunks_out: List[Dict[str, Any]] = []
    chunk_index = 0

    for page_num, page_content in pages:
        # Recursively split the page by structure (paragraphs -> lines -> sentences)
        structural_segments = recursive_split_text(
            text=page_content,
            max_words=chunk_size,
            overlap_words=overlap
        )

        for segment in structural_segments:
            if not segment.strip():
                continue
            chunk_text = f"[{source}, Page {page_num}]\n{segment}"
            chunks_out.append({
                "text": chunk_text,
                "source": source,
                "page": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1

    return chunks_out


def process_corpus_to_jsonl(corpus_dir: str = "corpus", output_jsonl: str = "data/chunks.jsonl") -> int:
    """
    Process all files in the corpus directory and write contextual chunks to a JSONL file.
    """
    try:
        from injection.parser import parse_file
    except ImportError:
        from src.injection.parser import parse_file

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    def iter_corpus():
        for name in sorted(os.listdir(corpus_dir)):
            if not name.lower().endswith(('.pdf', '.csv', '.xls', '.xlsx')):
                continue
            yield os.path.join(corpus_dir, name), name

    total_chunks = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for fpath, src_name in iter_corpus():
            print(f"Parsing {src_name} ...")
            parsed_result = parse_file(fpath)

            if isinstance(parsed_result, list):
                file_chunks = create_chunks_from_text(chunks=parsed_result, source=src_name)
            else:
                file_chunks = create_chunks_from_text(text=parsed_result, source=src_name)

            for ch in file_chunks:
                fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                total_chunks += 1

        print(f"[OK] Wrote {total_chunks} chunks to {output_jsonl}")

    return total_chunks