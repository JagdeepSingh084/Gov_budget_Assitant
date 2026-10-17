# Budget RAG Chatbot UI

Interactive Streamlit interface for querying Indian Government Budget reports (Union, Delhi, Bihar, and Odisha) in English and Hindi.

## How to Run

From the root project directory:

```powershell
streamlit run ui/app.py
```

Or using the Python virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py
```

The application will start and open automatically in your browser at:
`http://localhost:8501`

## Features

- **Interactive Bilingual Chat:** Accepts queries in both English and Hindi.
- **Accurate Grounding:** Uses Hybrid BM25 + FAISS Dense Retrieval with Reciprocal Rank Fusion (RRF).
- **Source Attribution:** Shows exact document name (`17107782611749468962.pdf`, `MRF_13_Union_Budget.csv`, etc.) and page number.
- **Verification Details:** Collapsible section displaying verification rationale and retrieved chunk snippets.
- **Example Prompts:** One-click sample questions in the sidebar for quick testing.
