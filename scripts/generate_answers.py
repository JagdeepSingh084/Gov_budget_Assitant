import os
import sys
import json
import re

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import time
from src.retrieval.main import answer_question


def extract_eval_questions(eval_file=None):
    if eval_file is None:
        eval_file = os.path.join(ROOT_DIR, "reference", "questions_eval.jsonl")

    with open(eval_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Find blocks starting with {"question_id": "E...
    blocks = re.findall(r'\{\s*"question_id"\s*:\s*"E\d+".*?(?=\n\s*\{\s*"question_id"|\Z)', text, re.DOTALL)
    
    questions = []
    seen = set()
    for block in blocks:
        qid_m = re.search(r'"question_id"\s*:\s*"(E\d+)"', block)
        q_m = re.search(r'"question"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', block)
        if qid_m and q_m:
            qid = qid_m.group(1)
            raw_q = q_m.group(1)
            # Safely decode string escapes without corrupting native UTF-8 Devanagari characters
            try:
                q_text = json.loads(f'"{raw_q}"')
            except Exception:
                q_text = raw_q
            if qid not in seen:
                seen.add(qid)
                questions.append((qid, q_text))
    return sorted(questions, key=lambda x: int(x[0][1:]))


def main():
    questions = extract_eval_questions()
    print(f"Loaded {len(questions)} evaluation questions:")
    for qid, q in questions:
        print(f"  {qid}: {q[:70]}...")

    output_file = os.path.join(ROOT_DIR, "answers.jsonl")
    results = []

    print("\nRunning RAG system over evaluation questions...")
    for qid, q in questions:
        print(f"Processing {qid}...")
        try:
            entry = answer_question(q, as_assignment=True, question_id=qid)
            results.append(entry)
            ans_snippet = entry['answer'][:60] if entry['answered'] else "(abstained per corpus absence)"
            print(f"  -> Answered: {entry['answered']} | Sources: {len(entry['sources'])} | {ans_snippet}")
        except Exception as e:
            print(f"  -> Error on {qid}: {e}")
            results.append({
                "question_id": qid,
                "answered": False,
                "answer": "",
                "sources": []
            })
        time.sleep(2.5)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[OK] Wrote {len(results)} answers to {output_file}")



if __name__ == "__main__":
    main()
