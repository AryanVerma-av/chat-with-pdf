import time
from google import genai
from google.genai.errors import APIError
from backend.app.config import settings


class EmbeddingServiceError(Exception):
    """Base exception for embedding failures."""
    pass


class MissingAPIKeyError(EmbeddingServiceError):
    """Raised when an API key is required but not provided."""
    pass


class GeminiEmbeddingService:
    """Handles text embedding generation via Google GenAI API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key
        self.model = model or settings.GEMINI_EMBEDDING_MODEL
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        key = self.api_key or settings.gemini_api_key
        if not key or not key.strip():
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file."
            )
        if self._client is None:
            self._client = genai.Client(api_key=key.strip())
        return self._client

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generates embedding vectors for a batch of text chunks."""
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Process in batches to adhere to payload limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                if not response or not response.embeddings:
                    raise EmbeddingServiceError("Embedding service returned an empty response.")

                for emb in response.embeddings:
                    if emb.values is None:
                        raise EmbeddingServiceError("Embedding response contained null values.")
                    all_embeddings.append(list(emb.values))

            except MissingAPIKeyError:
                raise
            except APIError as exc:
                raise EmbeddingServiceError(f"Gemini Embedding API error: {exc.message}") from exc
            except Exception as exc:
                raise EmbeddingServiceError(f"Failed to generate embeddings: {str(exc)}") from exc

            # Slight delay between batches if processing large numbers of chunks
            if i + batch_size < len(texts):
                time.sleep(0.05)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generates an embedding vector for a single query string."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        client = self._get_client()
        try:
            response = client.models.embed_content(
                model=self.model,
                contents=text.strip(),
            )
            if not response or not response.embeddings or not response.embeddings[0].values:
                raise EmbeddingServiceError("Embedding service returned an empty vector for the query.")

            return list(response.embeddings[0].values)

        except MissingAPIKeyError:
            raise
        except APIError as exc:
            raise EmbeddingServiceError(f"Gemini Embedding API error: {exc.message}") from exc
        except Exception as exc:
            raise EmbeddingServiceError(f"Failed to embed query: {str(exc)}") from exc
