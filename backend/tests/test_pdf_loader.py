import io
import pytest
from reportlab.pdfgen import canvas
from backend.app.services.pdf_loader import PDFLoader, InvalidPDFError, EmptyPDFError


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with known text on each page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages_text:
        c.drawString(100, 700, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_pdf_loader_success():
    """Verifies that pages and text are extracted accurately with 1-indexed page metadata."""
    sample_texts = [
        "Page one introduces the company overview and foundation date.",
        "Page two details the 2023 financial revenue which reached four million dollars.",
        "Page three contains concluding executive remarks.",
    ]
    pdf_bytes = create_sample_pdf(sample_texts)

    pages = PDFLoader.load(file_bytes=pdf_bytes, filename="sample.pdf")

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "Page one introduces" in pages[0].text
    assert pages[1].page_number == 2
    assert "four million dollars" in pages[1].text
    assert pages[2].page_number == 3
    assert "concluding executive remarks" in pages[2].text


def test_pdf_loader_empty_bytes():
    """Verifies that empty byte input raises InvalidPDFError."""
    with pytest.raises(InvalidPDFError, match="Uploaded file is empty"):
        PDFLoader.load(file_bytes=b"", filename="empty.pdf")


def test_pdf_loader_invalid_magic_bytes():
    """Verifies that non-PDF content raises InvalidPDFError."""
    with pytest.raises(InvalidPDFError, match="does not appear to be a valid PDF document"):
        PDFLoader.load(file_bytes=b"PLAIN TEXT NOT A PDF", filename="fake.pdf")


def test_pdf_loader_empty_text_pdf():
    """Verifies that a PDF with blank pages raises EmptyPDFError."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()  # Empty page
    c.save()

    with pytest.raises(EmptyPDFError, match="contains no extractable text"):
        PDFLoader.load(file_bytes=buf.getvalue(), filename="blank.pdf")
