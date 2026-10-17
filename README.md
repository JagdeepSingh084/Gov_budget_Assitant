# 🇮🇳 Indian Gender & Union Budget RAG Assistant

A robust, bilingual (English & Hindi) Retrieval-Augmented Generation (RAG) system built over Indian government budget documents (Union, Delhi, Bihar, and Odisha).

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd RAG_Project
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
groq_api_key=your_groq_api_key_here
gemini_api_key=your_gemini_api_key_here  # Optional: for Gemini embeddings
```

### 3. Run Ingestion Pipeline (Chunking & Indexing)

```bash
python -m src.ingestion_pipeline.main
```
This parses all corpus files (`corpus/`), creates chunks (`data/chunks.jsonl`), generates dense embeddings (`data/embeddings.faiss`), and initializes the hybrid retriever.

### 4. Querying the System

#### A. Interactive Streamlit UI
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` to chat interactively with source badges and verification expanders.

#### B. Command Line Interface
```bash
# General Research Output (Strict JSON with unit and tests)
python -m src.retrieval.main "Which country was the first to undertake gender-responsive budgeting, and in what year?"

# Assignment Submission Format Output
python -m src.retrieval.main --assignment "What was the allocation for Menstrual Hygiene in Girls (KISHORI) in Delhi's 2023-24 gender budget?"

# Hindi Query (Auto-translated preserving Indian scheme names)
python -m src.retrieval.main "बिहार जेंडर बजट 2024-25 के अनुसार जीविका के अंतर्गत अब तक कितने स्वयं सहायता समूह गठित किए गए हैं?"
```

### 5. Generate Submission Answers (`answers.jsonl`)

Run the automated evaluation pipeline to answer all 18 evaluation questions (E01–E18) matching the exact `ASSIGNMENT.md` schema:

```bash
python scripts/generate_answers.py
```
This writes the final answers into [answers.jsonl](file:///c:/RAG_Project/answers.jsonl).

---

## 🏗️ Architecture

```
Corpus Files (PDF, CSV, XLS)
       │
       ▼
 [Parser & Sliding Chunking] ──► data/chunks.jsonl (with Document & Page provenance)
       │
       ├──────────────────────────────┐
       ▼                              ▼
 [BM25Okapi Lexical Index]     [FAISS IndexFlatIP Dense Index]
       │                              │
       └──────────────┬───────────────┘
                      ▼
        [Weighted RRF (k=60)]
                      │
                      ▼
     [Top Evidence Chunks + Groq LLM]
                      │
                      ▼
   [Structured JSON Output / ASSIGNMENT Schema]
```

---

## 📋 Evaluation & Documentation

* **[`answers.jsonl`](file:///c:/RAG_Project/answers.jsonl)**: Output for all 18 evaluation questions (`question_id`, `answered`, `answer`, `sources`).
* **[`SCOPE.md`](file:///c:/RAG_Project/SCOPE.md)**: Report on inclusions, exclusions, imperfectly handled elements, and trade-off rationale (~300 words).
* **[`NOTES.md`](file:///c:/RAG_Project/NOTES.md)**: Operational breakdown, future roadmap, and reflection on AI debugging (~300 words).
* **[`ASSIGNMENT.md`](file:///c:/RAG_Project/ASSIGNMENT.md)**: Original task guidelines and specifications.
