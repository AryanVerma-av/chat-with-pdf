import io
from dataclasses import dataclass
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(Exception):
    """Base exception for PDF processing failures."""
    pass


class InvalidPDFError(PDFProcessingError):
    """Raised when the uploaded file is not a valid PDF."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when the PDF has no pages or contains no extractable text."""
    pass


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str


class PDFLoader:
    """Extracts text content and page metadata from PDF documents."""

    @staticmethod
    def load(file_bytes: bytes, filename: str = "document.pdf") -> list[PageContent]:
        if not file_bytes:
            raise InvalidPDFError("Uploaded file is empty (0 bytes).")

        # Validate PDF magic bytes (%PDF)
        if not file_bytes.startswith(b"%PDF"):
            raise InvalidPDFError(
                f"File '{filename}' does not appear to be a valid PDF document (missing PDF magic header)."
            )

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except PdfReadError as exc:
            raise InvalidPDFError(f"Could not read PDF '{filename}'. File may be corrupt or encrypted: {str(exc)}")
        except Exception as exc:
            raise PDFProcessingError(f"Unexpected error while opening PDF '{filename}': {str(exc)}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise EmptyPDFError(f"PDF '{filename}' contains no pages.")

        pages_extracted: list[PageContent] = []
        total_text_chars = 0

        for idx, page in enumerate(reader.pages):
            page_number = idx + 1
            try:
                extracted = page.extract_text() or ""
                cleaned = extracted.strip()
                total_text_chars += len(cleaned)
                if cleaned:
                    pages_extracted.append(PageContent(page_number=page_number, text=cleaned))
            except Exception as exc:
                # Log or handle individual page extraction failures gracefully
                continue

        if total_text_chars == 0 or len(pages_extracted) == 0:
            raise EmptyPDFError(
                f"PDF '{filename}' contains no extractable text across {total_pages} page(s). "
                "Scanned documents requiring OCR are not supported."
            )

        return pages_extracted
