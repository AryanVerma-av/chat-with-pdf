import io
import pytest
from reportlab.pdfgen import canvas
from backend.app.services.text_splitter import TextChunk
from backend.app.services.vector_store import ChromaVectorStore
from backend.app.rag.prompts import build_rag_prompt
from backend.app.rag.pipeline import RAGPipeline, DocumentNotLoadedError


class MockEmbeddingService:
    """Deterministic embedding service for automated testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Map word presence to synthetic 3D vector for deterministic similarity
        embeddings = []
        for text in texts:
            t = text.lower()
            embeddings.append([
                1.0 if "revenue" in t else 0.1,
                1.0 if "growth" in t else 0.1,
                1.0 if "executive" in t else 0.1,
            ])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "revenue" in t else 0.1,
            1.0 if "growth" in t else 0.1,
            1.0 if "executive" in t else 0.1,
        ]


class MockLLMService:
    """Mock LLM service that echoes context and simulates grounded answers."""

    def generate(self, prompt: str) -> str:
        if "revenue" in prompt.lower():
            return "According to page 2, the 2023 revenue reached four million dollars."
        return "I cannot find the answer to this question in the uploaded document."


def create_test_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "Introduction to the company foundation on page one.")
    c.showPage()
    c.drawString(100, 700, "Financial section: In 2023 total revenue reached four million dollars.")
    c.showPage()
    c.drawString(100, 700, "Executive concluding remarks on page three.")
    c.showPage()
    c.save()
    return buf.getvalue()


def test_chroma_vector_store():
    """Verifies adding chunks, querying nearest vector, and resetting collection."""
    store = ChromaVectorStore()
    chunks = [
        TextChunk(chunk_id="c1", text="Company overview", page_number=1, source_filename="test.pdf"),
        TextChunk(chunk_id="c2", text="Financial revenue of $4M", page_number=2, source_filename="test.pdf"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    store.add_chunks(chunks, embeddings)

    assert store.count() == 2
    assert not store.is_empty()

    # Query closest to chunk 2 ([0.0, 1.0, 0.0])
    results = store.query(query_embedding=[0.0, 0.9, 0.1], top_k=1)
    assert len(results) == 1
    assert results[0].page_number == 2
    assert "Financial revenue" in results[0].text

    # Test reset
    store.reset()
    assert store.count() == 0
    assert store.is_empty()


def test_rag_prompt_builder():
    """Verifies prompt generation and page tag inclusion."""
    context_items = [
        {"page": 2, "text": "Total revenue reached four million dollars."},
    ]
    prompt = build_rag_prompt(question="What was the revenue?", context_items=context_items)

    assert "[Source Page 2]:" in prompt
    assert "Total revenue reached four million dollars." in prompt
    assert "What was the revenue?" in prompt
    assert "CRITICAL INSTRUCTIONS:" in prompt


def test_rag_pipeline_end_to_end():
    """Tests full pipeline flow: ingest PDF -> index -> retrieve -> answer + citation."""
    pipeline = RAGPipeline(
        embedding_service=MockEmbeddingService(),
        llm_service=MockLLMService(),
        vector_store=ChromaVectorStore(),
    )

    # Pre-upload query must fail
    with pytest.raises(DocumentNotLoadedError):
        pipeline.answer_question("What is the revenue?")

    # Ingest test PDF
    pdf_bytes = create_test_pdf()
    status = pipeline.ingest_pdf(file_bytes=pdf_bytes, filename="annual_report.pdf")

    assert status["filename"] == "annual_report.pdf"
    assert status["total_pages"] == 3
    assert status["total_chunks"] >= 3
    assert pipeline.is_document_loaded

    # Query relevant question
    response = pipeline.answer_question("What was the total revenue in 2023?")

    assert "four million dollars" in response.answer
    assert len(response.sources) > 0
    assert any(s.page == 2 for s in response.sources)

    # Clear pipeline
    pipeline.clear()
    assert not pipeline.is_document_loaded
