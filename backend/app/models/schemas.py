from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status indicator")
    pdf_loaded: bool = Field(..., description="Whether a PDF document is currently indexed")
    active_document: str | None = Field(default=None, description="Filename of the active PDF")
    total_pages: int = Field(default=0, description="Total pages in the active document")
    total_chunks: int = Field(default=0, description="Total chunks indexed in the vector store")


class UploadResponse(BaseModel):
    status: str = Field(..., description="Status of the upload operation")
    filename: str = Field(..., description="Name of the processed PDF file")
    total_pages: int = Field(..., description="Number of extracted pages")
    total_chunks: int = Field(..., description="Number of chunks generated and stored")
    message: str = Field(..., description="Human-readable status summary")


class SourceCitation(BaseModel):
    page: int = Field(..., description="1-indexed PDF page number")
    snippet: str = Field(..., description="Relevant text excerpt from this page")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question to answer")


class QueryResponse(BaseModel):
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer based on the PDF context")
    sources: list[SourceCitation] = Field(default_factory=list, description="List of source citations with page numbers")


class ClearResponse(BaseModel):
    status: str = Field(..., description="Status of the clear operation")
    message: str = Field(..., description="Confirmation message")
