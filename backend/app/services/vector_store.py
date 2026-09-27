from dataclasses import dataclass
import chromadb
from backend.app.services.text_splitter import TextChunk


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    distance: float


class ChromaVectorStore:
    """Manages an in-memory vector collection for document chunks using ChromaDB."""

    COLLECTION_NAME = "active_pdf_collection"

    def __init__(self):
        # Using EphemeralClient keeps the index fast and cleanly scoped in-memory
        self._client = chromadb.EphemeralClient()
        self._collection = None
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initializes or resets the active collection."""
        try:
            # Delete if exists
            try:
                self._client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass

            self._collection = self._client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize vector store collection: {str(exc)}") from exc

    def reset(self) -> None:
        """Purges all stored chunks and re-initializes an empty collection."""
        self._initialize_collection()

    def count(self) -> int:
        """Returns the number of indexed chunks."""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def is_empty(self) -> bool:
        """Checks if the vector store has no indexed chunks."""
        return self.count() == 0

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        """Adds text chunks with embeddings and metadata to the vector store."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "page_number": int(chunk.page_number),
                "source_filename": str(chunk.source_filename),
                "chunk_id": str(chunk.chunk_id),
            }
            for chunk in chunks
        ]

        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to insert chunks into vector store: {str(exc)}") from exc

    def query(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        """Performs cosine similarity search against indexed chunks."""
        if self.is_empty():
            return []

        actual_k = min(top_k, self.count())
        if actual_k <= 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )

            retrieved: list[RetrievedChunk] = []
            if not results or not results["documents"] or not results["documents"][0]:
                return retrieved

            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results["distances"][0] if results.get("distances") else []
            ids = results["ids"][0] if results.get("ids") else []

            for i in range(len(docs)):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else 0.0
                chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"

                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=docs[i],
                        page_number=int(meta.get("page_number", 1)),
                        source_filename=str(meta.get("source_filename", "")),
                        distance=float(dist),
                    )
                )

            return retrieved

        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {str(exc)}") from exc
