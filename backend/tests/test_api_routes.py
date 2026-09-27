import io
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from backend.app.main import app
from backend.app.routes.rag import rag_pipeline
from backend.tests.test_rag_pipeline import MockEmbeddingService, MockLLMService


def create_sample_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "Financial data on page one: revenue is four million.")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_mock_pipeline():
    """Inject mock services into the global pipeline instance for API testing."""
    rag_pipeline.embedding_service = MockEmbeddingService()
    rag_pipeline.llm_service = MockLLMService()
    rag_pipeline.clear()
    yield
    rag_pipeline.clear()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pdf_loaded"] is False
    assert data["total_pages"] == 0


def test_ask_before_upload_fails():
    response = client.post("/ask", json={"question": "What is the revenue?"})
    assert response.status_code == 400
    assert "No PDF document has been uploaded yet" in response.json()["detail"]


def test_upload_invalid_filetype():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_upload_and_ask_success():
    pdf_bytes = create_sample_pdf_bytes()
    upload_res = client.post(
        "/upload",
        files={"file": ("annual_report.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "success"
    assert upload_data["total_pages"] == 1
    assert upload_data["total_chunks"] >= 1

    # Now health check should report active document
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["pdf_loaded"] is True
    assert health_res.json()["active_document"] == "annual_report.pdf"

    # Now ask question
    ask_res = client.post("/ask", json={"question": "What is the revenue?"})
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert "four million" in ask_data["answer"]
    assert len(ask_data["sources"]) > 0
    assert ask_data["sources"][0]["page"] == 1

    # Clear document
    clear_res = client.post("/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "cleared"

    # Health check should now be empty
    health_after_clear = client.get("/health")
    assert health_after_clear.json()["pdf_loaded"] is False
