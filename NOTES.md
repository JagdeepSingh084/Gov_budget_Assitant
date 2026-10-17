# Technical Notes & Future Scope: Budget RAG Assistant

## 1. How the System Works

The system operates across three tightly integrated stages:

1. **Ingestion & Lineage Enrichment:** Heterogeneous budget documents (PDF, CSV, XLS) are normalized into 200-word sliding chunks with 20-word overlap. Each chunk is prepended with explicit document and page metadata (`[Source: <file>, Page <N>]`) so evidence provenance is preserved end-to-end.
2. **Hybrid Retrieval with Weighted RRF:** BM25Okapi and FAISS dense vector search run in parallel. A Weighted Reciprocal Rank Fusion ($k=60$, BM25 weight $1.5\times$, Dense weight $1.0\times$) merges candidate pools. Token expansion maps financial years (`2024-25` $\rightarrow$ `2024`, `25`, `2025`, `2024-2025`) to prevent numeric misses on fiscal spans.
3. **Multilingual Routing & Fact-Grounded Synthesis:** Queries containing Devanagari script are detected and translated into English using Groq LLM, retaining proper scheme names (*Kishori*, *Ladli*, *Kanya Utthan*). The retrieved context and question are dispatched to Groq (`openai/gpt-oss-20b`) in strict JSON object mode. If the requested information is absent from the corpus, negative indicator analysis triggers explicit abstention (`"answered": false, "answer": ""`), preventing hallucinations.

---

## 2. AI Tool Misleading Instance & Debugging

During the evaluation pipeline setup, an AI coding tool suggested handling escaped characters in `questions_eval.jsonl` using `.encode('latin1').decode('unicode_escape')`. 

While this unescaped ASCII slashes, it corrupted native multi-byte UTF-8 Devanagari text into Latin-1 mojibake (e.g., turning `"दिल्ली के 2022-23..."` into `"à¤¦à¤¿à¤²à¥..."`). This corrupted string failed the regex language detector `[\u0900-\u097F]`, skipped query translation entirely, passed unsearchable gibberish to BM25, and caused questions E09 and E10 to fail retrieval.

**How it was caught:** Inspecting the console logs revealed garbled characters during question loading. We replaced the double-encode hack with native JSON string parsing (`json.loads(f'"{raw_q}"')`), which cleanly preserved UTF-8 Devanagari characters, triggered proper translation, and immediately restored retrieval to the correct gold documents.

---

## 3. What To Do Next With More Time

* **Visual Table Parsing:** Integrate `pdfplumber` or `Camelot` to extract structural grid tables with column/row coordinates, eliminating text-wrapping ambiguities in complex budget matrices.
* **Cross-Encoder Re-Ranking:** Deploy a cross-encoder (e.g., `bge-reranker-large`) over top-30 candidates to re-score nuanced passages prior to LLM synthesis.
* **Deterministic Arithmetic Validator:** Implement an automated consistency checker verifying that component figures (e.g., Part A + Part B) equal published budget totals.
