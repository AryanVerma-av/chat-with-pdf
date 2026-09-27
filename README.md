# PDF RAG Assistant

A minimal, professional Retrieval-Augmented Generation (RAG) web application for grounded question answering from uploaded PDF documents.

The system extracts page-by-page text from PDFs, generates dense vector embeddings, stores them in an in-memory ChromaDB vector store, retrieves relevant passages based on cosine similarity, and synthesizes answers using Google Gemini while strictly preserving source page citations.

---

## Architecture and Pipeline

```
PDF Document
  |
  v
PDF Document Loader (pypdf with page preservation)
  |
  v
Text Chunks (Recursive splitting with configurable size and overlap)
  |
  v
Embeddings (Google Gemini text-embedding-004)
  |
  v
Vector Store (ChromaDB with cosine distance metric)
  |
  v
Retriever (Top-K semantic similarity search)
  |
  v
Relevant Text (Page-tagged excerpts)
  |
  v
LLM (Google Gemini gemini-2.0-flash with strict grounding instructions)
  |
  v
Answer + Source Page Citations
```

---

## Project Structure

```
project/
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py                # FastAPI entry point and static mounting
|   |   |-- config.py              # Configuration via Pydantic BaseSettings
|   |   |-- models/
|   |   |   |-- __init__.py
|   |   |   |-- schemas.py         # Request and response Pydantic schemas
|   |   |-- routes/
|   |   |   |-- __init__.py
|   |   |   |-- rag.py             # Route handlers: /upload, /ask, /health, /clear
|   |   |-- services/
|   |   |   |-- __init__.py
|   |   |   |-- pdf_loader.py      # PDF text extractor with page metadata
|   |   |   |-- text_splitter.py   # Recursive text chunker
|   |   |   |-- embeddings.py      # Gemini embedding service wrapper
|   |   |   |-- vector_store.py    # ChromaDB collection manager
|   |   |   |-- llm.py             # Gemini LLM completion service
|   |   |-- rag/
|   |       |-- __init__.py
|   |       |-- pipeline.py        # End-to-end RAG orchestrator
|   |       |-- prompts.py         # Grounding system prompt and templates
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- test_api_routes.py     # Endpoint integration tests
|   |   |-- test_pdf_loader.py     # PDF parsing unit tests
|   |   |-- test_text_splitter.py  # Text chunking unit tests
|   |   |-- test_rag_pipeline.py   # RAG pipeline and vector search tests
|   |-- requirements.txt
|   |-- .env.example
|   |-- .env
|
|-- frontend/
|   |-- index.html                 # Minimal, accessible semantic HTML
|   |-- style.css                  # Clean, modern SaaS styling (no emojis, no clutter)
|   |-- app.js                     # State management, drag-and-drop, API integration
|
|-- sample_document.pdf            # Sample 3-page test PDF
|-- README.md
|-- .gitignore
```

---

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

---

## Local Setup and Installation

### 1. Clone or Open the Repository

```bash
cd "project #3"
```

### 2. Configure Environment Variables

Create or edit `backend/.env`:

```bash
# Windows PowerShell
copy backend\.env.example backend\.env
```

Open `backend/.env` and insert your Gemini API Key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004

CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVER_TOP_K=4

HOST=127.0.0.1
PORT=8000
```

### 3. Install Dependencies

Using PowerShell:

```powershell
# Navigate to backend
cd backend

# Create virtual environment (if not already present)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Unified Server (Recommended)

The backend server is configured to serve both the REST API and the minimal frontend interface on a single port:

```powershell
# From the project root with backend venv activated:
.\backend\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Option B: Separate Frontend Server

If you prefer to run the frontend independently:

1. Start the backend:
```powershell
.\backend\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

2. In another terminal, serve the frontend:
```powershell
cd frontend
python -m http.server 3000
```

Navigate to: `http://localhost:3000`

### Option C: Streamlit Web App (Local or Cloud Deployment)

Run the full application as a Streamlit app:

```powershell
.\backend\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Navigate to:
```
http://localhost:8501
```

#### Deploying to Streamlit Community Cloud (share.streamlit.io):
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Select your repository and set the **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings...** and add your secret in the **Secrets** section:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy**. Streamlit Cloud uses the root `requirements.txt` to install dependencies and run the app.

---

## Example Usage

1. Open `http://127.0.0.1:8000` in your web browser.
2. Click **Select PDF** or drag and drop `sample_document.pdf` into the upload zone.
3. Observe the loading state: `"Processing PDF and generating embeddings..."`.
4. Once processed, the document badge confirms: `"Uploaded: sample_document.pdf | 3 page(s) | 3 chunk(s) indexed"`.
5. Enter a question:
   - Question: `What was the total revenue in fiscal year 2025?`
   - Answer: `Total operating revenue reached 12.8 million dollars, representing a 22 percent increase.`
   - Source: `Page 2: "Total operating revenue reached 12.8 million dollars..."`
6. Test hallucination resistance by asking an out-of-scope question:
   - Question: `What is the capital of Australia?`
   - Answer: `I cannot find the answer to this question in the uploaded document.`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service health and active document metadata. |
| `POST` | `/upload` | Uploads, chunks, embeds, and indexes a PDF document. |
| `POST` | `/ask` | Queries the document and generates a grounded response with citations. |
| `POST` | `/clear` | Clears the active document and resets the vector store. |

### API Request Examples

#### Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Upload PDF
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@sample_document.pdf"
```

#### Ask Question
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the primary objective of the initiative?"}'
```

---

## Running the Automated Test Suite

The test suite validates PDF page extraction, recursive chunking, ChromaDB vector indexing, prompt formatting, and end-to-end API workflows:

```powershell
.\backend\venv\Scripts\python.exe -m pytest backend/tests -v
```

All 14 unit and integration tests pass without external dependencies.

---

## Troubleshooting

### 1. `GEMINI_API_KEY is not configured`
- **Cause**: The `GEMINI_API_KEY` in `backend/.env` is blank or missing.
- **Solution**: Open `backend/.env`, set `GEMINI_API_KEY=your_key_here`, and restart the backend server.

### 2. `Invalid file format`
- **Cause**: Uploaded file does not have a `.pdf` extension or lacks `%PDF` header bytes.
- **Solution**: Ensure the uploaded file is an actual PDF document.

### 3. `The PDF contains no extractable text`
- **Cause**: Scanned document containing images of text rather than extractable text layers.
- **Solution**: Upload a native PDF with selectable text. Scanned image documents requiring OCR are not supported in this minimal pipeline.

### 4. `No PDF document has been uploaded yet`
- **Cause**: A question was submitted before uploading a PDF or after clearing the index.
- **Solution**: Upload a valid PDF before asking questions.

---

## Verification Checklist

- [x] PDF Document Loader: Page-by-page text extraction with 1-indexed page metadata (`pypdf`).
- [x] Text Splitter: Recursive character text splitting adhering to chunk size and overlap boundaries.
- [x] Embedding Generation: Google Gemini `text-embedding-004` integration with batching.
- [x] Vector Store: ChromaDB with cosine distance metric and in-memory isolation.
- [x] Vector Retriever: Semantic similarity search returning top-K relevant chunks.
- [x] LLM Integration: Google Gemini `gemini-2.0-flash` with temperature 0.0 for factual precision.
- [x] Grounded Prompt: Strict system instructions requiring explicit fallback when context lacks information.
- [x] Source Citations: Page numbers and excerpt snippets traced directly to source PDF pages.
- [x] API Layer: Fully implemented `POST /upload`, `POST /ask`, `GET /health`, and `POST /clear`.
- [x] Frontend UI: Minimalist, professional SaaS design with clear loading, ready, answer, and error states.
- [x] Zero Emojis: Enforced across UI, code comments, documentation, and model responses.
- [x] Test Coverage: 14 automated unit and integration tests passing.
