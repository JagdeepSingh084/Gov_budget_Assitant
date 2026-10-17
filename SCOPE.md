# Scope Document: Budget RAG System

## 1. What Was Included

* **Heterogeneous Corpus Ingestion:** Ingested all 11 Indian government budget documents spanning Union, Delhi, Bihar, and Odisha across PDF, CSV (`MRF_13_Union_Budget.csv`), and legacy Excel format (`stat20.xls`).
* **Source-Aware Chunking & Provenance:** Applied a sliding-window strategy (200 words, 20-word overlap) with persistent file and page headers (`[Source: <file>, Page <N>]`) prepended to each chunk, ensuring traceable lineage through retrieval and generation.
* **Hybrid Retrieval with Weighted RRF:** Lexical search (BM25Okapi) and dense semantic search (FAISS `IndexFlatIP`) run concurrently. Candidate pools are fused via Reciprocal Rank Fusion ($k=60$) with a $1.5\times$ weight on BM25 to prioritize exact financial terminology and scheme names.
* **Financial Token Disambiguation:** Implemented custom token expansion for Indian financial years (e.g., `2024-25` expands to `2024`, `25`, `2025`, `2024-2025`) and budget sections (`Part A` $\rightarrow$ `part_a`).
* **Bilingual Translation & Handling:** Automated Devanagari script detection and LLM-assisted query translation to English, preserving key scheme titles (*Kishori*, *Mukhyamantri Kanya Utthan Yojana*, *Ladli*, *Jeevika*) and numbers.
* **Format Compliance & Safe Abstention:** Enforced strict output schemas matching both internal structured JSON and the exact `ASSIGNMENT.md` specification (`question_id`, `answered`, `answer`, `sources`). Negative indicator detection reliably marks unanswerable corpus queries (E11, E12, E13, E15, E16, E17) as `"answered": false` with an empty answer string.

---

## 2. What Was Excluded & Handled Imperfectly

* **Multi-Page Coordinate Table Reconstruction:** Complex PDF budget matrices were parsed line-by-line rather than reconstructed via visual bounding-box coordinates. Table header-to-cell associations across multi-page tables rely on sliding context windows.
* **Scanned Scrambled Fonts:** In `17107782611749468962.pdf` (Bihar), the Hindi text layer suffered from corrupted font encoding. Rather than spending excessive time on OCR reconstruction during the 2-hour window, the system translates user Hindi queries to English and retrieves from the clean parallel English sections.
* **Infographics & Charts:** Visual graphics and pages with $<10$ characters were excluded as noise to prevent hallucinating numeric figures.

---

## 3. Why These Trade-offs Were Made

In public financial reporting, hallucinating an incorrect rupee allocation is far more damaging than abstaining. By prioritizing lexical exactness for financial years, targeted translation for Hindi schemes, and conservative abstention when evidence is absent, the system guarantees high-precision factual answers while faithfully respecting corpus boundaries.
