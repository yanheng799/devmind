"""DashScope embedding model for text-embedding-v4."""

import logging
from typing import Any

import requests

from config import get_settings


logger = logging.getLogger(__name__)


class DashScopeEmbeddingModel:
    """DashScope text embedding model using text-embedding-v4.

    Uses Alibaba DashScope API for generating embeddings.
    """

    DEFAULT_DIM = 1024

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize the DashScope embedding model.

        Args:
            api_key: DashScope API key
            api_base: API base URL
            model: Model name
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        config = settings.get_embedding_config()

        self.api_key = api_key or config.get("api_key", "")
        self.api_base = (api_base or config.get("api_base") or
                        "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or config.get("model_name", "text-embedding-v4")
        self.timeout = timeout or config.get("timeout", 60)
        self.dim = config.get("dim", self.DEFAULT_DIM)

        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. "
                "Set DEVMIND_EMBEDDING_API_KEY in your environment or .env file."
            )

        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session.

        Returns:
            Session object
        """
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate embeddings for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding vector or list of vectors
        """
        session = self._get_session()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        try:
            response = session.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            embeddings = [item["embedding"] for item in result["data"]]

            if single_input:
                return embeddings[0]
            return embeddings

        except requests.RequestException as e:
            logger.error(f"DashScope API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}")
            raise

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed(text)

    def close(self) -> None:
        """Close HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None


class MockDashScopeEmbeddingModel(DashScopeEmbeddingModel):
    """Mock DashScope embedding model for testing.

    Returns deterministic embeddings without API calls.
    """

    def __init__(self) -> None:
        """Initialize mock embedding model."""
        self.api_key = "mock_key"
        self.api_base = "http://mock"
        self.model = "text-embedding-v4"
        self.timeout = 60
        self.dim = DashScopeEmbeddingModel.DEFAULT_DIM
        self._session = None

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate mock embeddings."""
        import hashlib

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings: list[list[float]] = []
        for text in texts:
            hash_bytes = hashlib.md5(text.encode()).digest()

            vector: list[float] = []
            for i in range(self.dim):
                byte_idx = i % len(hash_bytes)
                val = (hash_bytes[byte_idx] + i) / 255.0
                vector.append(val)

            norm = sum(x * x for x in vector) ** 0.5
            vector = [x / (norm + 1e-8) for x in vector]
            embeddings.append(vector)

        if single_input:
            return embeddings[0]
        return embeddings

    def close(self) -> None:
        """No-op for mock."""
        pass


__all__ = [
    "DashScopeEmbeddingModel",
    "MockDashScopeEmbeddingModel",
]
