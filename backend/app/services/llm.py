import time
import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError
from backend.app.config import settings
from backend.app.services.embeddings import MissingAPIKeyError

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Base exception for LLM generation failures."""
    pass


class GeminiLLMService:
    """Handles text generation using Google Gemini models with grounding constraints and automatic fallback."""

    FALLBACK_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ]

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key
        self.model = model or settings.GEMINI_MODEL
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

    def generate(self, prompt: str) -> str:
        """Generates grounded answer text based on the provided prompt with automatic high-demand fallback."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        client = self._get_client()

        # Build candidate list starting with configured model, followed by fallbacks
        candidates = [self.model]
        for fb in self.FALLBACK_MODELS:
            if fb not in candidates:
                candidates.append(fb)

        config = types.GenerateContentConfig(
            temperature=0.0,  # Strict deterministic grounding
            max_output_tokens=1000,
        )

        last_error: Exception | None = None

        for candidate_model in candidates:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt.strip(),
                    config=config,
                )

                if not response or not response.text:
                    return "I cannot find the answer to this question in the uploaded document."

                return response.text.strip()

            except MissingAPIKeyError:
                raise
            except APIError as exc:
                last_error = exc
                err_msg = (exc.message or str(exc)).lower()
                # Check for transient high demand / quota / unavailable conditions
                if any(x in err_msg for x in ["high demand", "resource_exhausted", "unavailable", "503", "429"]):
                    logger.warning(
                        f"Model {candidate_model} experienced high demand/rate limit. Falling back to next available model."
                    )
                    time.sleep(0.5)
                    continue
                # For non-transient API errors, if we still have fallback candidates, try them
                continue
            except Exception as exc:
                last_error = exc
                continue

        # If all candidates failed, raise descriptive error
        if last_error:
            if isinstance(last_error, APIError):
                raise LLMServiceError(f"Gemini LLM API error: {last_error.message}") from last_error
            raise LLMServiceError(f"Failed to generate answer from LLM: {str(last_error)}") from last_error

        return "I cannot find the answer to this question in the uploaded document."
