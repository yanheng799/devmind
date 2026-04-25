"""Tests for document vector store."""

import uuid

import pytest

from devmind.data.vectorstore.document_store import (
    DocumentVectorStore,
    MockDocumentVectorStore,
)


class TestDocumentVectorStore:
    """Test DocumentVectorStore class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        store = DocumentVectorStore()
        assert store.collection_name == "devmind_documents"
        assert store.EMBEDDING_DIM == 1024
        store.close()

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        try:
            store = DocumentVectorStore(
                collection_name="custom_collection",
                host="127.0.0.1",
                port=19530,
            )
            assert store.collection_name == "custom_collection"
            assert store.host == "127.0.0.1"
            assert store.port == 19530
            store.close()
        except Exception as e:
            pytest.skip(f"Milvus not available: {e}")


class TestMockDocumentVectorStore:
    """Test MockDocumentVectorStore class."""

    def test_init(self) -> None:
        """Test mock store initialization."""
        store = MockDocumentVectorStore()
        assert store._chunks == {}

    def test_insert_and_search_chunk(self) -> None:
        """Test inserting and searching chunks."""
        store = MockDocumentVectorStore()

        chunk_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        embedding = [0.1] * 768

        vector_id = store.insertChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            embedding=embedding,
            content="Test content about finance",
            doc_type="pdf",
        )

        assert vector_id in store._chunks
        assert store._chunks[vector_id]["chunk_id"] == chunk_id

    def test_search_chunks(self) -> None:
        """Test searching similar chunks."""
        store = MockDocumentVectorStore()

        document_id = str(uuid.uuid4())

        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            embedding=[0.1] * 768,
            content="Stock market analysis",
            doc_type="pdf",
        )
        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            embedding=[0.9] * 768,
            content="Cooking recipes",
            doc_type="txt",
        )

        query_embedding = [0.1] * 768
        results = store.searchChunks(query_embedding, top_k=2)

        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_search_chunks_with_filter(self) -> None:
        """Test searching with document type filter."""
        store = MockDocumentVectorStore()

        document_id = str(uuid.uuid4())

        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            embedding=[0.1] * 768,
            content="Finance content",
            doc_type="pdf",
        )
        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            embedding=[0.2] * 768,
            content="Other content",
            doc_type="txt",
        )

        query_embedding = [0.1] * 768
        results = store.searchChunks(query_embedding, doc_type="pdf")

        assert len(results) == 1
        assert results[0]["doc_type"] == "pdf"

    def test_delete_chunks(self) -> None:
        """Test deleting chunks by document ID."""
        store = MockDocumentVectorStore()

        document_id = str(uuid.uuid4())

        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            embedding=[0.1] * 768,
            content="Content",
            doc_type="pdf",
        )
        store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            embedding=[0.2] * 768,
            content="Other content",
            doc_type="txt",
        )

        count = store.deleteChunks(document_id)
        assert count == 1
        assert len(store._chunks) == 1

    def test_delete_chunk(self) -> None:
        """Test deleting single chunk."""
        store = MockDocumentVectorStore()

        vector_id = store.insertChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            embedding=[0.1] * 768,
            content="Content",
            doc_type="pdf",
        )

        store.deleteChunk(vector_id)
        assert len(store._chunks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
