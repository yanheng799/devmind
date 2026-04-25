"""FAQ document ingestion into vector store."""

import logging
import uuid
from pathlib import Path

from devmind.config import get_settings
from devmind.data.processors.document_processor import DocumentProcessor
from devmind.data.vectorstore.dashscope_embedding import DashScopeEmbeddingModel
from devmind.data.vectorstore.document_store import DocumentVectorStore


logger = logging.getLogger(__name__)


class FaqIngestor:
    """Ingest FAQ markdown files into the document vector store.

    Skips files whose checksum already exists in the store,
    ensuring data is only vectorized once.
    """

    def __init__(
        self,
        source_dir: Path | str | None = None,
        collection_name: str | None = None,
        force: bool = False,
    ) -> None:
        """Initialize the FAQ ingestor.

        Args:
            source_dir: Directory of FAQ .md files (from settings if None)
            collection_name: Milvus collection name (from settings if None)
            force: If True, re-ingest even if data already exists
        """
        settings = get_settings()
        rag_config = settings.get_rag_config()

        self.source_dir = Path(source_dir) if source_dir else Path(rag_config["faq_source_dir"])
        self.collection_name = collection_name or rag_config["faq_collection_name"]
        self.force = force

        self._processor = DocumentProcessor()
        self._embedding_model = DashScopeEmbeddingModel(
            api_key="sk-901f743555d246408a9492dc96d57caa",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4"
        )
        self._store = DocumentVectorStore(collection_name=self.collection_name)

    def ingestAll(self) -> int:
        """Ingest all FAQ markdown files from source directory.

        Returns:
            Total number of chunks inserted.
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"FAQ source directory not found: {self.source_dir}")

        md_files = sorted(self.source_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"No .md files found in {self.source_dir}")
            return 0

        logger.info(f"Found {len(md_files)} FAQ files in {self.source_dir}")

        total_inserted = 0
        for file_path in md_files:
            inserted = self.ingestFile(file_path)
            total_inserted += inserted

        logger.info(f"Ingestion complete: {total_inserted} chunks total")
        return total_inserted

    def ingestFile(self, file_path: Path) -> int:
        """Parse, chunk, embed, and store a single FAQ file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Number of chunks inserted.
        """
        file_path = Path(file_path)
        document_id = self._computeDocumentId(file_path)

        if not self.force and self._isFileIngested(document_id):
            logger.info(f"Skipping {file_path.name}: already ingested")
            return 0

        if self.force:
            self._store.deleteChunks(document_id)
            logger.info(f"Force re-ingesting {file_path.name}")

        parsed = self._processor.parseFile(file_path)
        chunks = self._processor.chunkText(parsed["text"])

        if not chunks:
            logger.warning(f"No chunks generated from {file_path.name}")
            return 0

        embeddings = self._embedChunks(chunks)

        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}:{idx}"))
            self._store.insertChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                embedding=embedding,
                content=chunk_text,
                doc_type="faq",
                metadata={"filename": file_path.name, "checksum": document_id},
            )

        logger.info(f"Ingested {file_path.name}: {len(chunks)} chunks")
        return len(chunks)

    def _computeDocumentId(self, file_path: Path) -> str:
        """Compute document_id from file checksum + filename.

        Args:
            file_path: Path to the file.

        Returns:
            Document ID string.
        """
        checksum = self._processor.calculateChecksum(file_path)
        return f"{file_path.stem}_{checksum[:16]}"

    def _isFileIngested(self, document_id: str) -> bool:
        """Check if a document_id already has chunks in the store.

        Args:
            document_id: Document ID to check.

        Returns:
            True if document already exists in store.
        """
        collection = self._store._get_collection()
        collection.load()
        try:
            results = collection.query(
                expr=f'document_id == "{document_id}"',
                output_fields=["vector_id"],
                limit=1,
            )
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Failed to check ingestion status: {e}")
            return False

    def _embedChunks(self, chunks: list[str], batch_size: int = 10) -> list[list[float]]:
        """Batch-embed text chunks with size limit per request.

        Args:
            chunks: List of text chunks.
            batch_size: Max number of texts per API call.

        Returns:
            List of embedding vectors.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            result = self._embedding_model.embed(batch)
            if isinstance(result, list) and result and isinstance(result[0], list):
                all_embeddings.extend(result)
            else:
                all_embeddings.append(result)  # type: ignore[arg-type]
        return all_embeddings

    def close(self) -> None:
        """Clean up resources."""
        self._embedding_model.close()
        self._store.close()
