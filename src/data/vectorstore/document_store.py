"""Document vector store for RAG knowledge base."""

import logging
import uuid
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    connections,
    DataType,
    FieldSchema,
    utility,
)

from config import get_settings


logger = logging.getLogger(__name__)


class DocumentVectorStore:
    """Vector store for RAG document chunks.

    Stores and retrieves document chunks based on semantic similarity
    using embedding vectors.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Initialize the document vector store.

        Args:
            collection_name: Name of the Milvus collection
            host: Milvus host
            port: Milvus port
            embedding_dim: Embedding dimension (from settings if None)
        """
        settings = get_settings()
        config = settings.get_embedding_config()
        doc_config = settings.get_document_config()

        self.EMBEDDING_DIM = embedding_dim or config.get("dim", 1024)

        self.collection_name = collection_name or doc_config["collection_name"]
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port

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
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.EMBEDDING_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=5000,
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document chunks for RAG knowledge base",
        )

        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        logger.info(f"Created collection: {self.collection_name}")

    def insertChunk(
        self,
        chunk_id: str,
        document_id: str,
        embedding: list[float],
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert a document chunk.

        Args:
            chunk_id: Unique chunk identifier
            document_id: Parent document ID
            embedding: Embedding vector
            content: Chunk content
            doc_type: Document type
            metadata: Additional metadata

        Returns:
            Vector ID
        """
        vector_id = metadata.get("vector_id", str(uuid.uuid4())) if metadata else str(uuid.uuid4())

        collection = self._get_collection()

        data = [{
            "vector_id": vector_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "embedding": embedding,
            "content": content[:5000],
            "doc_type": doc_type,
            "metadata": metadata or {},
        }]

        collection.insert(data)
        logger.debug(f"Inserted chunk: {chunk_id}")

        return vector_id

    def searchChunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_type: str | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            doc_type: Filter by document type
            document_id: Filter by document ID

        Returns:
            List of similar chunk dicts with scores
        """
        collection = self._get_collection()
        collection.load()

        filter_expr = None
        if doc_type and document_id:
            filter_expr = f'doc_type == "{doc_type}" && document_id == "{document_id}"'
        elif doc_type:
            filter_expr = f'doc_type == "{doc_type}"'
        elif document_id:
            filter_expr = f'document_id == "{document_id}"'

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
            output_fields=["chunk_id", "document_id", "content", "doc_type", "metadata"],
        )

        chunks: list[dict[str, Any]] = []
        for hit in results[0]:
            chunks.append({
                "vector_id": hit.id,
                "chunk_id": hit.entity.get("chunk_id"),
                "document_id": hit.entity.get("document_id"),
                "content": hit.entity.get("content"),
                "doc_type": hit.entity.get("doc_type"),
                "metadata": hit.entity.get("metadata", {}),
                "score": float(hit.score),
            })

        return chunks

    def deleteChunks(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        collection = self._get_collection()
        expr = f'document_id == "{document_id}"'
        collection.delete(expr)
        logger.info(f"Deleted chunks for document: {document_id}")
        return 0

    def deleteChunk(self, vector_id: str) -> None:
        """Delete a single chunk.

        Args:
            vector_id: Vector ID to delete
        """
        collection = self._get_collection()
        collection.delete(f'vector_id == "{vector_id}"')
        logger.debug(f"Deleted chunk: {vector_id}")

    def close(self) -> None:
        """Close connection to Milvus."""
        connections.disconnect("default")


class MockDocumentVectorStore(DocumentVectorStore):
    """Mock vector store for testing.

    Uses simple in-memory storage instead of Milvus.
    """

    def __init__(self, embedding_dim: int | None = None) -> None:
        """Initialize mock document vector store.

        Args:
            embedding_dim: Embedding dimension
        """
        settings = get_settings()
        config = settings.get_embedding_config()

        self.EMBEDDING_DIM = embedding_dim or config.get("dim", 1024)
        self._chunks: dict[str, dict[str, Any]] = {}

    def insertChunk(
        self,
        chunk_id: str,
        document_id: str,
        embedding: list[float],
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert a chunk into mock storage."""
        import hashlib

        vector_id = hashlib.md5(chunk_id.encode()).hexdigest()
        self._chunks[vector_id] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "embedding": embedding,
            "content": content,
            "doc_type": doc_type,
            "metadata": metadata or {},
        }
        return vector_id

    def searchChunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_type: str | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using simple cosine similarity."""
        import numpy as np

        results: list[tuple[str, float]] = []
        query_vec = np.array(query_embedding)

        for vector_id, chunk in self._chunks.items():
            if doc_type and chunk["doc_type"] != doc_type:
                continue
            if document_id and chunk["document_id"] != document_id:
                continue

            chunk_vec = np.array(chunk["embedding"])
            similarity = float(np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
            ))
            results.append((vector_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)

        chunks: list[dict[str, Any]] = []
        for vector_id, score in results[:top_k]:
            chunk = self._chunks[vector_id].copy()
            chunk["score"] = score
            chunk["vector_id"] = vector_id
            chunks.append(chunk)

        return chunks

    def deleteChunks(self, document_id: str) -> int:
        """Delete chunks by document ID."""
        to_delete = [
            vector_id for vector_id, chunk in self._chunks.items()
            if chunk["document_id"] == document_id
        ]
        for vector_id in to_delete:
            del self._chunks[vector_id]
        return len(to_delete)

    def deleteChunk(self, vector_id: str) -> None:
        """Delete a single chunk."""
        if vector_id in self._chunks:
            del self._chunks[vector_id]

    def close(self) -> None:
        """No-op for mock store."""
        pass


__all__ = [
    "DocumentVectorStore",
    "MockDocumentVectorStore",
]
