import logging
from backend.app.config import settings
from backend.app.models.schemas import QueryResponse, SourceCitation
from backend.app.services.pdf_loader import PDFLoader
from backend.app.services.text_splitter import RecursiveTextSplitter
from backend.app.services.embeddings import GeminiEmbeddingService
from backend.app.services.vector_store import ChromaVectorStore
from backend.app.services.llm import GeminiLLMService
from backend.app.rag.prompts import build_rag_prompt

logger = logging.getLogger(__name__)


class DocumentNotLoadedError(Exception):
    """Raised when an action requires an active document but none is loaded."""
    pass


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for PDF question answering."""

    def __init__(
        self,
        embedding_service: GeminiEmbeddingService | None = None,
        llm_service: GeminiLLMService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.embedding_service = embedding_service or GeminiEmbeddingService()
        self.llm_service = llm_service or GeminiLLMService()
        self.vector_store = vector_store or ChromaVectorStore()
        self.text_splitter = RecursiveTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        self.active_filename: str | None = None
        self.total_pages: int = 0
        self.total_chunks: int = 0

    @property
    def is_document_loaded(self) -> bool:
        return self.active_filename is not None and not self.vector_store.is_empty()

    def get_status(self) -> dict:
        return {
            "pdf_loaded": self.is_document_loaded,
            "active_document": self.active_filename,
            "total_pages": self.total_pages,
            "total_chunks": self.vector_store.count(),
        }

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> dict:
        """Executes full PDF ingestion: loading, chunking, embedding, and vector indexing."""
        # 1. Extract text and page metadata
        pages = PDFLoader.load(file_bytes=file_bytes, filename=filename)

        # 2. Chunk text with page provenance
        chunks = self.text_splitter.split_pages(pages=pages, source_filename=filename)
        if not chunks:
            raise ValueError(f"No valid text chunks could be produced from '{filename}'.")

        # 3. Generate embeddings for all chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_documents(chunk_texts)

        # 4. Reset vector store to ensure single active PDF scope
        self.vector_store.reset()

        # 5. Index chunks and embeddings
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        # 6. Update pipeline state
        self.active_filename = filename
        self.total_pages = len(pages)
        self.total_chunks = len(chunks)

        logger.info(
            f"Successfully indexed '{filename}': {self.total_pages} pages, {self.total_chunks} chunks."
        )

        return {
            "filename": filename,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
        }

    def answer_question(self, question: str) -> QueryResponse:
        """Retrieves relevant chunks and generates a grounded response with page citations."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if not self.is_document_loaded:
            raise DocumentNotLoadedError(
                "No PDF document has been uploaded yet. Please upload a PDF before asking questions."
            )

        # 1. Embed query vector
        query_vector = self.embedding_service.embed_query(cleaned_question)

        # 2. Retrieve top-K relevant chunks
        retrieved_chunks = self.vector_store.query(
            query_embedding=query_vector,
            top_k=settings.RETRIEVER_TOP_K,
        )

        if not retrieved_chunks:
            return QueryResponse(
                question=cleaned_question,
                answer="I cannot find the answer to this question in the uploaded document.",
                sources=[],
            )

        # 3. Assemble context for LLM prompt
        context_items = [
            {"page": c.page_number, "text": c.text}
            for c in retrieved_chunks
        ]
        prompt = build_rag_prompt(question=cleaned_question, context_items=context_items)

        # 4. Generate answer using grounded LLM
        answer = self.llm_service.generate(prompt=prompt)

        # 5. Extract unique page sources and snippets
        seen_pages = set()
        sources: list[SourceCitation] = []
        for c in retrieved_chunks:
            # Create a concise snippet for verification
            snippet_text = c.text.strip().replace("\n", " ")
            if len(snippet_text) > 180:
                snippet_text = snippet_text[:177] + "..."

            # Record citation (avoid duplicate snippet spam for same page if text is identical)
            sources.append(
                SourceCitation(
                    page=c.page_number,
                    snippet=snippet_text,
                )
            )

        return QueryResponse(
            question=cleaned_question,
            answer=answer,
            sources=sources,
        )

    def clear(self) -> None:
        """Resets the pipeline and purges vector store."""
        self.vector_store.reset()
        self.active_filename = None
        self.total_pages = 0
        self.total_chunks = 0
