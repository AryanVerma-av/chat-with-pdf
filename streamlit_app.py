import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings
from backend.app.rag.pipeline import RAGPipeline, DocumentNotLoadedError
from backend.app.services.pdf_loader import InvalidPDFError, EmptyPDFError
from backend.app.services.embeddings import MissingAPIKeyError, EmbeddingServiceError
from backend.app.services.llm import LLMServiceError

# Page Configuration (Zero emojis)
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Minimalist CSS
st.markdown(
    """
    <style>
    /* Minimalist styling overrides */
    .block-container {
        max-width: 680px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }
    .stButton button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 4px;
        border: 1px solid #0f172a;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #1e293b;
        color: #ffffff;
    }
    .source-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .source-header {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .source-snippet {
        color: #475569;
        font-size: 12px;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_effective_api_key() -> str:
    """Retrieves API key from Streamlit secrets, environment, or settings."""
    # 1. Check Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip():
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass

    # 2. Check settings / .env
    key = settings.gemini_api_key
    if key and key.strip():
        return key.strip()

    # 3. Check environment variable
    return os.getenv("GEMINI_API_KEY", "").strip()


# Resolve API Key
active_api_key = get_effective_api_key()

# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "processed_file_signature" not in st.session_state:
    st.session_state.processed_file_signature = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

pipeline: RAGPipeline = st.session_state.pipeline

# Application Header
st.title("PDF RAG ASSISTANT")
st.caption("Grounded question answering from your PDF document")
st.write("---")

# Sidebar for Configuration (Only if API key is missing)
if not active_api_key:
    with st.sidebar:
        st.subheader("Configuration")
        user_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste Gemini API key here...",
            help="Get an API key from https://aistudio.google.com/",
        )
        if user_key:
            active_api_key = user_key.strip()
            # Set in environment for current process
            os.environ["GEMINI_API_KEY"] = active_api_key
            st.success("API key configured.")

# Verify API Key availability
if not active_api_key:
    st.warning("Please configure your GEMINI_API_KEY in .streamlit/secrets.toml or backend/.env.")
    st.stop()

# 1. Document Upload Section
st.subheader("Upload your PDF")

uploaded_file = st.file_uploader(
    "Select or drop a PDF document",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

    # Only process if this file has not already been indexed in session
    if st.session_state.processed_file_signature != file_signature:
        with st.spinner("Processing PDF and generating embeddings..."):
            try:
                file_bytes = uploaded_file.read()
                result = pipeline.ingest_pdf(file_bytes=file_bytes, filename=uploaded_file.name)
                st.session_state.processed_file_signature = file_signature
                st.session_state.last_answer = None
                st.session_state.last_sources = []
                st.success(
                    f"Uploaded: {result['filename']} | {result['total_pages']} page(s) | {result['total_chunks']} chunk(s) indexed"
                )
            except (InvalidPDFError, EmptyPDFError) as exc:
                st.error(str(exc))
                st.session_state.processed_file_signature = None
            except MissingAPIKeyError as exc:
                st.error(str(exc))
                st.session_state.processed_file_signature = None
            except EmbeddingServiceError as exc:
                st.error(f"Embedding service error: {str(exc)}")
                st.session_state.processed_file_signature = None
            except Exception as exc:
                st.error(f"Failed to process PDF: {str(exc)}")
                st.session_state.processed_file_signature = None
    else:
        status_info = pipeline.get_status()
        st.info(
            f"Uploaded: {status_info['active_document']} | {status_info['total_pages']} page(s) | {status_info['total_chunks']} chunk(s) indexed"
        )

# 2. Question Answering Section
st.write("")
st.subheader("Ask a question")

with st.form("rag_question_form", clear_on_submit=False):
    user_question = st.text_input(
        "Question",
        placeholder="Ask a question about the document...",
        label_visibility="collapsed",
        disabled=not pipeline.is_document_loaded,
    )
    submit_button = st.form_submit_button("Ask", disabled=not pipeline.is_document_loaded)

if submit_button and user_question:
    with st.spinner("Searching document context and generating answer..."):
        try:
            response = pipeline.answer_question(user_question)
            st.session_state.last_answer = response.answer
            st.session_state.last_sources = response.sources
        except DocumentNotLoadedError as exc:
            st.error(str(exc))
        except (LLMServiceError, EmbeddingServiceError) as exc:
            st.error(f"AI Service error: {str(exc)}")
        except Exception as exc:
            st.error(f"Failed to answer question: {str(exc)}")

# 3. Answer Display Section
if st.session_state.last_answer:
    st.write("---")
    st.subheader("Answer")
    st.write(st.session_state.last_answer)

    if st.session_state.last_sources:
        st.write("")
        st.markdown("**Sources**")
        # Deduplicate pages for clear provenance display
        page_map = {}
        for src in st.session_state.last_sources:
            if src.page not in page_map:
                page_map[src.page] = src.snippet

        for page_num, snippet in page_map.items():
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-header">Page {page_num}</div>
                    <div class="source-snippet">"{snippet}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
