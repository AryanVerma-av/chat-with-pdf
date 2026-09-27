from dataclasses import dataclass
from backend.app.services.pdf_loader import PageContent


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page_number: int
    source_filename: str


class RecursiveTextSplitter:
    """Splits text into chunks recursively based on natural separators while preserving metadata."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively breaks text down using the given separators."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split using the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks adhering to chunk_size and chunk_overlap
        good_splits: list[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Combines short splits up to chunk_size with chunk_overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_doc else 0)
            if total + piece_len > self.chunk_size and current_doc:
                doc_text = separator.join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)

                # Keep overlap pieces from the end
                while total > self.chunk_overlap and current_doc:
                    removed = current_doc.pop(0)
                    total -= len(removed) + (len(separator) if current_doc else 0)

            current_doc.append(piece)
            total += piece_len

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs

    def split_pages(self, pages: list[PageContent], source_filename: str) -> list[TextChunk]:
        """Splits page contents into chunks, propagating page numbers and source metadata."""
        all_chunks: list[TextChunk] = []

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # If page is already small enough, keep as single chunk
            if len(page_text) <= self.chunk_size:
                chunk_id = f"{source_filename}_p{page.page_number}_c0"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=page_text,
                        page_number=page.page_number,
                        source_filename=source_filename,
                    )
                )
            else:
                chunks = self._split_text(page_text, self.separators)
                for c_idx, chunk_text in enumerate(chunks):
                    cleaned = chunk_text.strip()
                    if cleaned:
                        chunk_id = f"{source_filename}_p{page.page_number}_c{c_idx}"
                        all_chunks.append(
                            TextChunk(
                                chunk_id=chunk_id,
                                text=cleaned,
                                page_number=page.page_number,
                                source_filename=source_filename,
                            )
                        )

        return all_chunks
