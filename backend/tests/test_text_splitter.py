import pytest
from backend.app.services.pdf_loader import PageContent
from backend.app.services.text_splitter import RecursiveTextSplitter


def test_text_splitter_page_metadata_preserved():
    """Asserts that page numbers are preserved across all split chunks."""
    pages = [
        PageContent(page_number=1, text="This is short content on page one."),
        PageContent(page_number=2, text="This is short content on page two."),
    ]
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_pages(pages=pages, source_filename="doc.pdf")

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "doc.pdf"
    assert chunks[1].page_number == 2
    assert chunks[1].source_filename == "doc.pdf"


def test_text_splitter_long_text_chunks():
    """Asserts that long text is split into chunks smaller than chunk_size."""
    paragraph = (
        "Retrieval-Augmented Generation enhances large language models by integrating external knowledge. "
        "Instead of relying purely on parameter weights, the system queries a vector database for relevant documents. "
        "The retrieved passages are then formatted into a context window alongside the user prompt. "
        "This drastically reduces hallucination and provides verifiable source citations."
    )
    pages = [PageContent(page_number=1, text=paragraph)]
    splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=30)
    chunks = splitter.split_pages(pages=pages, source_filename="rag_intro.pdf")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_number == 1
        assert len(chunk.text) <= 150  # adheres to chunk bounds within margin


def test_text_splitter_overlap_validation():
    """Asserts that setting chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be strictly smaller than chunk_size"):
        RecursiveTextSplitter(chunk_size=200, chunk_overlap=200)
