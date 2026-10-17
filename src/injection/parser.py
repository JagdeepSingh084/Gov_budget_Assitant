"""
Simplified, document-aware parser module for 2-hour RAG workflow.

Handles:
1. Tabular files (CSV, XLS, XLSX):
   - Dynamic header detection (finds the first row with rich text column names)
   - Preserves introductory preamble notes as separate chunks
   - Merges parent super-headers (e.g. '2012-13 Budget - Total')
   - Serializes each row as: '[filename] Column: Value | Column: Value'

2. PDF files:
   - Line-by-line layout reconstruction using word Y-coordinates (preserves table lines)
   - Skips identical duplicate files (MD5 hash check)
   - Adds clean [Page X] markers for attribution
"""

import os
import re
import hashlib
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    import config
except ImportError:
    from src import config

# Simple in-memory set to skip indexing byte-identical duplicate files
_SEEN_HASHES = set()


def _file_hash(path: str) -> str:
    """Quick MD5 to detect duplicate files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()


def parse_tabular(file_path: str):
    """
    Parse CSV or Excel files into self-describing row chunks.
    Automatically finds the table header and captures preamble notes.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    # 1. Load data without assuming fixed headers
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, header=None)
    else:
        try:
            df = pd.read_csv(file_path, encoding='cp1252', header=None)
        except Exception:
            df = pd.read_csv(file_path, header=None)

    df = df.dropna(how='all').reset_index(drop=True)
    if df.empty:
        return []

    # 2. Find header row: the row with the most text columns in the first 15 rows
    header_idx = 0
    max_text_cols = 0
    for r in range(min(15, len(df))):
        vals = [str(v).strip() for v in df.iloc[r].values if pd.notna(v) and str(v).strip() not in ['', 'nan']]
        # Text labels are non-numeric strings of reasonable length
        text_cols = [v for v in vals if not re.match(r'^-?\d+(?:\.\d+)?%?$', v) and len(v) < 60]
        if len(text_cols) > max_text_cols:
            max_text_cols = len(text_cols)
            header_idx = r

    # 3. Extract preamble notes (any cell before the table with length > 60 chars)
    chunks = []
    chunk_idx = 0
    for r in range(header_idx):
        for val in df.iloc[r].values:
            val_str = str(val).strip()
            if pd.notna(val) and len(val_str) > 60:
                chunks.append({
                    "text": f"[{filename} Note] {val_str}",
                    "source": filename,
                    "row_index": r,
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1

    # 4. Resolve column names
    headers = [
        str(v).strip() if pd.notna(v) and str(v).strip() != 'nan' else f"col_{i}" 
        for i, v in enumerate(df.iloc[header_idx].values)
    ]
    
    # If the row immediately above has group titles (like years), merge them
    if header_idx > 0:
        prev_vals = [str(v).strip() if pd.notna(v) and str(v).strip() != 'nan' else '' 
                     for v in df.iloc[header_idx - 1].values]
        cur_group = ""
        for i in range(len(headers)):
            if i < len(prev_vals) and prev_vals[i]:
                cur_group = prev_vals[i]
            if cur_group and headers[i] in ['Plan', 'Non-Plan', 'Total']:
                headers[i] = f"{cur_group} - {headers[i]}"

    # 5. Serialize data rows into self-describing statements
    data_df = df.iloc[header_idx + 1:].reset_index(drop=True)
    for r_idx, row in data_df.iterrows():
        row_dict = {}
        row_parts = []
        for col_name, val in zip(headers, row.values):
            val_str = str(val).strip()
            if pd.notna(val) and val_str not in ['', 'nan']:
                row_dict[col_name] = val_str
                row_parts.append(f"{col_name}: {val_str}")

        if not row_parts:
            continue

        serialized = f"[{filename}] " + " | ".join(row_parts)
        chunks.append({
            "text": serialized,
            "source": filename,
            "row_index": header_idx + 1 + r_idx,
            "chunk_index": chunk_idx,
            "data": row_dict,
        })
        chunk_idx += 1

    return chunks


def parse_pdf(file_path: str) -> str:
    """
    Parse PDF into layout-preserving text with [Page X] markers.
    Reconstructs lines by word position so table rows stay intact.
    """
    filename = os.path.basename(file_path)

    # Skip byte-identical duplicates (e.g. duplicate years)
    fhash = _file_hash(file_path)
    if fhash in _SEEN_HASHES:
        return ""
    _SEEN_HASHES.add(fhash)

    try:
        doc = fitz.open(file_path)
    except Exception:
        return ""

    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = page.get_text("words")
        if not words:
            continue

        # Sort words primarily by vertical coordinate Y, secondarily by horizontal X
        words.sort(key=lambda w: (w[1], w[0]))

        # Cluster words into visual lines (tolerance of 3 pixels vertically)
        lines = []
        cur_y = None
        cur_line = []
        for w in words:
            if cur_y is None or abs(w[1] - cur_y) <= 3.0:
                cur_line.append(w[4])
                if cur_y is None:
                    cur_y = w[1]
            else:
                lines.append(" ".join(cur_line))
                cur_line = [w[4]]
                cur_y = w[1]
        if cur_line:
            lines.append(" ".join(cur_line))

        # Skip pages with negligible text
        if sum(len(l) for l in lines) < config.PDF_MIN_TEXT_PER_PAGE:
            continue

        # Simple filter for garbled legacy font ASCII lines (keep line if it has numbers)
        clean_lines = []
        for l in lines:
            has_numbers = bool(re.search(r'\d', l))
            is_garbled = sum(1 for c in l if c in ";'[]{}~^`") > 2
            if has_numbers or not is_garbled:
                clean_lines.append(l)

        if clean_lines:
            pages_text.append(f"[Page {page_num + 1}]\n" + "\n".join(clean_lines))

    doc.close()
    return "\n\n".join(pages_text)


def parse_csv(file_path: str):
    """Parse CSV using the tabular parser."""
    return parse_tabular(file_path)


def parse_xls(file_path: str):
    """Parse Excel using the tabular parser."""
    return parse_tabular(file_path)


def parse_file(file_path: str):
    """
    Unified entry point:
    - Tabular (CSV, XLS, XLSX) -> returns list of row chunk dicts
    - PDF -> returns layout-preserving text with [Page X] tags
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xls', '.xlsx', '.csv']:
        return parse_tabular(file_path)
    elif ext == '.pdf':
        return parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")