import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import json
from src.injection.parser import parse_file
from src.injection.chunking import create_chunks_from_text

CORPUS_DIR = "corpus"
OUTPUT_JSONL = "data/chunks.jsonl"

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

def iter_corpus():
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.lower().endswith(('.pdf', '.csv', '.xls', '.xlsx')):
            continue
        yield os.path.join(CORPUS_DIR, name), name

with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for fpath, src_name in iter_corpus():
        print(f"Parsing {src_name} ...")
        parsed_result = parse_file(fpath)

        # Handle different return types from parse_file
        if isinstance(parsed_result, list):
            # CSV/XLS files return a list of pre-structured chunks
            chunks = create_chunks_from_text(chunks=parsed_result, source=src_name)
        else:
            # PDF files return text with [Page X] markers
            chunks = create_chunks_from_text(text=parsed_result, source=src_name)

        for ch in chunks:
            fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {OUTPUT_JSONL}")