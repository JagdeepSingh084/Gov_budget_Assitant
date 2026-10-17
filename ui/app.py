import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import src.config as config
from src.retrieval.main import answer_question, hybrid_search

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Budget AI Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, readable output
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        background-color: #f0f2f6;
        color: #31333f;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .stExpander {
        border-radius: 8px !important;
        background-color: rgba(240, 242, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 AI Powered Gender Responsive Budget Assistant")
    st.caption("Advance AI Based Retrieval Augmented Generation using Gemini and Groq API")
    
    st.markdown("---")
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "Which country was the first to undertake gender-responsive budgeting, and in what year?",
        "बिहार जेंडर बजट 2024-25 के अनुसार जीविका के अंतर्गत अब तक कितने स्वयं सहायता समूह गठित किए गए हैं?",
        "What is the total allocation for Part A in Union Budget 2024-25?",
        "What are the major objectives of Mission Vatsalya in gender budgeting?",
        "What was the total expenditure in Part B for 2022-23?"
    ]
    
    for ex in example_questions:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["pending_prompt"] = ex

    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.markdown(f"- **LLM Provider:** Groq (`{config.GROQ_MODEL}`)")
    st.markdown(f"- **Embedding Model:** `{config.FALLBACK_QUERY_EMBEDDING_MODEL}`")
    st.markdown("- **Index:** FAISS + BM25Okapi (RRF)")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state["pending_prompt"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🇮🇳 Indian Gender & Union Budget RAG Chatbot")
st.markdown(
    "Ask factual questions about Union, Delhi, Bihar, and Odisha Budget documents in **English** or **Hindi**."
)

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Render Existing Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display structured metadata badges if available
        meta = msg.get("metadata")
        if meta:
            badge_html = ""
            if meta.get("lang"):
                badge_html += f'<span class="metric-badge">🌐 Lang: {meta["lang"].upper()}</span>'
            if meta.get("unit"):
                badge_html += f'<span class="metric-badge">📏 Unit: {meta["unit"]}</span>'
            if meta.get("source_location") and meta["source_location"] != "unknown":
                badge_html += f'<span class="metric-badge">📍 {meta["source_location"]}</span>'
            if meta.get("source_files"):
                sources_str = ", ".join(meta["source_files"])
                badge_html += f'<span class="metric-badge">📄 {sources_str}</span>'
            
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)
            
            # Collapsible verification & sources
            if meta.get("tests") or meta.get("retrieved_chunks"):
                with st.expander("🔍 Verification & Retrieved Chunks", expanded=False):
                    if meta.get("tests"):
                        st.markdown(f"**Verification Rationale:**\n_{meta['tests']}_")
                    
                    chunks = meta.get("retrieved_chunks", [])
                    if chunks:
                        st.markdown("**Top Retrieved Evidence:**")
                        for idx, chunk in enumerate(chunks[:3], 1):
                            src = chunk.get("source", "Unknown")
                            pg = chunk.get("page", "?")
                            st.markdown(f"**{idx}. {src} (Page {pg})**")
                            st.code(chunk.get("text", "")[:350] + "...", language="text")


# ---------------------------------------------------------------------------
# Chat Input & Response Generation
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about the budget documents...")

# Check if a sample query was clicked in sidebar
if st.session_state.get("pending_prompt"):
    user_query = st.session_state.pop("pending_prompt")

if user_query:
    # Display and record user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching budget documents and verifying answer..."):
            try:
                # 1. Retrieve candidate chunks
                retrieved_chunks = hybrid_search(user_query, top_k=config.ANSWER_TOP_K)
                
                # 2. Synthesize strict JSON answer
                response_dict = answer_question(retrieved_chunks, query_or_none=user_query, as_dict=True)
                
                answer_text = response_dict.get("answer", "No answer provided.")
                st.markdown(answer_text)
                
                # Metadata badges
                badge_html = ""
                lang = response_dict.get("lang")
                unit = response_dict.get("unit")
                loc = response_dict.get("source_location")
                sources = response_dict.get("source_files", [])
                
                if lang:
                    badge_html += f'<span class="metric-badge">🌐 Lang: {lang.upper()}</span>'
                if unit:
                    badge_html += f'<span class="metric-badge">📏 Unit: {unit}</span>'
                if loc and loc != "unknown":
                    badge_html += f'<span class="metric-badge">📍 {loc}</span>'
                if sources:
                    badge_html += f'<span class="metric-badge">📄 {", ".join(sources)}</span>'
                
                if badge_html:
                    st.markdown(badge_html, unsafe_allow_html=True)
                
                # Expandable details
                tests_rationale = response_dict.get("tests", "")
                if tests_rationale or retrieved_chunks:
                    with st.expander("🔍 Verification & Retrieved Chunks", expanded=False):
                        if tests_rationale:
                            st.markdown(f"**Verification Rationale:**\n_{tests_rationale}_")
                        if retrieved_chunks:
                            st.markdown("**Top Retrieved Evidence:**")
                            for idx, chunk in enumerate(retrieved_chunks[:3], 1):
                                src = chunk.get("source", "Unknown")
                                pg = chunk.get("page", "?")
                                st.markdown(f"**{idx}. {src} (Page {pg})**")
                                st.code(chunk.get("text", "")[:350] + "...", language="text")
                
                # Save to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "metadata": {
                        "lang": lang,
                        "unit": unit,
                        "source_location": loc,
                        "source_files": sources,
                        "tests": tests_rationale,
                        "retrieved_chunks": retrieved_chunks[:3]
                    }
                })
            except Exception as e:
                err_msg = f"⚠️ An error occurred while processing your query: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
