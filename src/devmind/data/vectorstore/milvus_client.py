"""Milvus vector store for event similarity search."""

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    connections,
    DataType,
    FieldSchema,
    utility,
)

from devmind.config import get_settings


logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Vector store using Milvus for event similarity search.

    Stores and retrieves historical events based on semantic similarity
    using embedding vectors.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name of the Milvus collection
            host: Milvus host
            port: Milvus port
            embedding_dim: Embedding dimension (from settings if None)
        """
        settings = get_settings()
        config = settings.get_embedding_config()

        self.EMBEDDING_DIM = embedding_dim or config.get("dim", 1024)

        self.collection_name = collection_name or settings.milvus_collection_name
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port

        # Connect to Milvus
        self._connect()
        self._collection: Collection | None = None

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name of the Milvus collection
            host: Milvus host
            port: Milvus port
        """
        settings = get_settings()

        self.collection_name = collection_name or settings.milvus_collection_name
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port

        # Connect to Milvus
        self._connect()
        self._collection: Collection | None = None

    def _connect(self) -> None:
        """Connect to Milvus server."""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )

    def _get_collection(self) -> Collection:
        """Get or create collection.

        Returns:
            Collection object
        """
        if self._collection is None:
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
            else:
                self._create_collection()
        return self._collection

    def _create_collection(self) -> None:
        """Create a new collection."""
        fields = [
            FieldSchema(
                name="vector_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="event_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.EMBEDDING_DIM,
            ),
            FieldSchema(
                name="event_type",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=1000,
            ),
            FieldSchema(
                name="stock_code",
                dtype=DataType.VARCHAR,
                max_length=16,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Historical events for similarity search",
        )

        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        # Create index on embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

    def insert_event(
        self,
        vector_id: str,
        event_id: str,
        embedding: list[float],
        event_type: str,
        description: str,
        stock_code: str,
    ) -> None:
        """Insert an event into the vector store.

        Args:
            vector_id: Unique vector ID
            event_id: Event ID
            embedding: Embedding vector
            event_type: Event type
            description: Event description
            stock_code: Related stock code
        """
        collection = self._get_collection()

        data = [{
            "vector_id": vector_id,
            "event_id": event_id,
            "embedding": embedding,
            "event_type": event_type,
            "description": description,
            "stock_code": stock_code,
        }]

        collection.insert(data)

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        event_type: str | None = None,
        stock_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar events.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            event_type: Filter by event type
            stock_code: Filter by stock code

        Returns:
            List of similar event dicts with scores
        """
        collection = self._get_collection()

        # Load collection into memory
        collection.load()

        # Build filter expression
        filter_expr = None
        if event_type and stock_code:
            filter_expr = f'event_type == "{event_type}" && stock_code == "{stock_code}"'
        elif event_type:
            filter_expr = f'event_type == "{event_type}"'
        elif stock_code:
            filter_expr = f'stock_code == "{stock_code}"'

        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["event_id", "event_type", "description", "stock_code"],
        )

        # Parse results
        events: list[dict[str, Any]] = []
        for hit in results[0]:
            events.append({
                "event_id": hit.entity.get("event_id"),
                "event_type": hit.entity.get("event_type"),
                "description": hit.entity.get("description"),
                "stock_code": hit.entity.get("stock_code"),
                "score": float(hit.score),
            })

        return events

    def delete_event(self, vector_id: str) -> None:
        """Delete an event from the vector store.

        Args:
            vector_id: Vector ID to delete
        """
        collection = self._get_collection()
        collection.delete(f"vector_id == '{vector_id}'")

    def close(self) -> None:
        """Close connection to Milvus."""
        connections.disconnect("default")


class MockVectorStore(MilvusVectorStore):
    """Mock vector store for testing.

    Uses simple in-memory storage instead of Milvus.
    """

    def __init__(self, embedding_dim: int | None = None) -> None:
        """Initialize mock vector store.

        Args:
            embedding_dim: Embedding dimension
        """
        settings = get_settings()
        config = settings.get_embedding_config()

        self.EMBEDDING_DIM = embedding_dim or config.get("dim", 1024)
        self._events: dict[str, dict[str, Any]] = {}

    def insert_event(
        self,
        vector_id: str,
        event_id: str,
        embedding: list[float],
        event_type: str,
        description: str,
        stock_code: str,
    ) -> None:
        """Insert an event into mock storage."""
        self._events[vector_id] = {
            "event_id": event_id,
            "embedding": embedding,
            "event_type": event_type,
            "description": description,
            "stock_code": stock_code,
        }

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        event_type: str | None = None,
        stock_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar events using simple cosine similarity."""
        import numpy as np

        results: list[tuple[str, float]] = []

        query_vec = np.array(query_embedding)

        for vector_id, event in self._events.items():
            # Apply filters
            if event_type and event["event_type"] != event_type:
                continue
            if stock_code and event["stock_code"] != stock_code:
                continue

            # Calculate cosine similarity
            event_vec = np.array(event["embedding"])
            similarity = float(np.dot(query_vec, event_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(event_vec) + 1e-8
            ))
            results.append((vector_id, similarity))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)

        events: list[dict[str, Any]] = []
        for vector_id, score in results[:top_k]:
            event = self._events[vector_id].copy()
            event["score"] = score
            events.append(event)

        return events

    def delete_event(self, vector_id: str) -> None:
        """Delete an event from mock storage."""
        if vector_id in self._events:
            del self._events[vector_id]

    def close(self) -> None:
        """No-op for mock store."""
        pass


class EmbeddingModel:
    """Text embedding model using sentence-transformers.

    Uses text2vec-base-chinese for Chinese text embeddings.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedding model.

        Args:
            model_name: Model name (uses settings if None)
        """
        settings = get_settings()
        embedding_config = settings.get_embedding_config()

        self.model_name = model_name or embedding_config["model_name"]
        self.device = embedding_config.get("device", "cpu")
        self.batch_size = embedding_config.get("batch_size", 32)

        # Lazy load model
        self._model: Any = None

    def _get_model(self) -> Any:
        """Get or load the model.

        Returns:
            SentenceTransformer model
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate embeddings for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding vector or list of vectors
        """
        model = self._get_model()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if single_input:
            return embeddings[0].tolist()
        return [e.tolist() for e in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed(text)


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing.

    Returns deterministic random embeddings.
    """

    def __init__(self) -> None:
        """Initialize mock embedding model."""
        import hashlib

        import numpy as np

        settings = get_settings()
        config = settings.get_embedding_config()

        self._hash_func = hashlib.md5
        self._dim = config.get("dim", 1024)

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate mock embeddings."""
        import numpy as np

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings: list[list[float]] = []
        for text in texts:
            # Use hash of text for deterministic embeddings
            hash_bytes = self._hash_func(text.encode()).digest()
            # Convert to array of floats
            arr = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            # Pad or truncate to desired dimension
            if len(arr) < self._dim:
                arr = np.pad(arr, (0, self._dim - len(arr)))
            else:
                arr = arr[:self._dim]
            # Normalize
            arr = arr / (np.linalg.norm(arr) + 1e-8)
            embeddings.append(arr.tolist())

        if single_input:
            return embeddings[0]
        return embeddings


class APIDashScopeEmbeddingModel(EmbeddingModel):
    """API-based embedding model using DashScope text-embedding-v4.

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
        self.api_base = (
            api_base
            or config.get("api_base")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or config.get("model_name", "text-embedding-v4")
        self.timeout = timeout or config.get("timeout", 60)
        self.dim = config.get("dim", self.DEFAULT_DIM)

        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. "
                "Set DEVMIND_EMBEDDING_API_KEY in your environment or .env file."
            )

        self._session: Any = None

    def _get_session(self) -> Any:
        """Get or create HTTP session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate embeddings for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding vector or list of vectors
        """
        import requests

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


class MockAPIDashScopeEmbeddingModel(APIDashScopeEmbeddingModel):
    """Mock DashScope embedding model for testing.

    Returns deterministic embeddings without API calls.
    """

    def __init__(self) -> None:
        """Initialize mock embedding model."""
        self.api_key = "mock_key"
        self.api_base = "http://mock"
        self.model = "text-embedding-v4"
        self.timeout = 60
        self.dim = APIDashScopeEmbeddingModel.DEFAULT_DIM
        self._session = None

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Generate mock embeddings."""
        import hashlib

        import numpy as np

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings: list[list[float]] = []
        for text in texts:
            hash_bytes = hashlib.md5(text.encode()).digest()
            arr = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

            if len(arr) < self.dim:
                arr = np.pad(arr, (0, self.dim - len(arr)))
            else:
                arr = arr[:self.dim]

            arr = arr / (np.linalg.norm(arr) + 1e-8)
            embeddings.append(arr.tolist())

        if single_input:
            return embeddings[0]
        return embeddings

    def close(self) -> None:
        """No-op for mock."""
        pass


def get_embedding_model(use_mock: bool = False, use_api: bool = True) -> EmbeddingModel | APIDashScopeEmbeddingModel:
    """Get appropriate embedding model based on settings.

    Args:
        use_mock: Use mock model for testing
        use_api: Use API-based model (DashScope) or local model

    Returns:
        Embedding model instance
    """
    settings = get_settings()
    config = settings.get_embedding_config()

    if use_mock:
        return MockAPIDashScopeEmbeddingModel()

    if use_api or config.get("provider") == "dashscope":
        return APIDashScopeEmbeddingModel()
    else:
        return EmbeddingModel()
