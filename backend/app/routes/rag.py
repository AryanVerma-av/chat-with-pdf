from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.app.models.schemas import (
    HealthResponse,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    ClearResponse,
)
from backend.app.rag.pipeline import RAGPipeline, DocumentNotLoadedError
from backend.app.services.pdf_loader import InvalidPDFError, EmptyPDFError, PDFProcessingError
from backend.app.services.embeddings import MissingAPIKeyError, EmbeddingServiceError
from backend.app.services.llm import LLMServiceError
from backend.app.services.vector_store import VectorStoreError

router = APIRouter()

# Global pipeline instance
rag_pipeline = RAGPipeline()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning system status and active document stats."""
    status_info = rag_pipeline.get_status()
    return HealthResponse(
        status="healthy",
        pdf_loaded=status_info["pdf_loaded"],
        active_document=status_info["active_document"],
        total_pages=status_info["total_pages"],
        total_chunks=status_info["total_chunks"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads, extracts, chunks, vectorizes, and indexes a PDF document."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a file with a .pdf extension.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    try:
        result = rag_pipeline.ingest_pdf(file_bytes=file_bytes, filename=file.filename)
        return UploadResponse(
            status="success",
            filename=result["filename"],
            total_pages=result["total_pages"],
            total_chunks=result["total_chunks"],
            message=f"Document '{result['filename']}' processed successfully with {result['total_pages']} pages and {result['total_chunks']} chunks indexed.",
        )
    except (InvalidPDFError, EmptyPDFError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except (VectorStoreError, PDFProcessingError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during PDF ingestion: {str(exc)}",
        )


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Retrieves relevant document excerpts and generates a grounded response."""
    try:
        return rag_pipeline.answer_question(question=request.question)
    except DocumentNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except (EmbeddingServiceError, LLMServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing question: {str(exc)}",
        )


@router.post("/clear", response_model=ClearResponse)
async def clear_document():
    """Resets the active document index and cleans vector storage."""
    rag_pipeline.clear()
    return ClearResponse(
        status="cleared",
        message="Active document and vector store cleared successfully.",
    )
