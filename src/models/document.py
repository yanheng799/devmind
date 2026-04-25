"""Document data models for RAG knowledge base."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """A chunk of a document for vectorization.

    Attributes:
        chunk_id: Unique chunk identifier
        document_id: Parent document ID
        chunk_index: Order in the document
        content: Chunk text content
        embedding_id: Vector store embedding ID
        page_number: Page number (for PDF)
        metadata: Additional metadata
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., ge=0, description="Order in the document")
    content: str = Field(..., min_length=1, description="Chunk text content")
    embedding_id: str | None = Field(default=None, description="Vector store embedding ID")
    page_number: int | None = Field(default=None, ge=1, description="Page number for PDF")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Validate content is not just whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be empty or whitespace only")
        return value


class Document(BaseModel):
    """Document metadata and status.

    Attributes:
        document_id: Unique document identifier
        filename: Original filename
        file_path: Full file path
        file_type: File extension/type
        file_size: File size in bytes
        title: Extracted or assigned title
        author: Extracted author (if available)
        checksum: SHA256 checksum for deduplication
        upload_time: When document was uploaded
        last_modified: File last modified timestamp
        chunk_count: Number of chunks created
        status: Processing status
        metadata: Additional metadata
    """

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    file_path: str = Field(..., min_length=1, description="Full file path")
    file_type: str = Field(..., min_length=1, max_length=32, description="File extension/type")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    title: str | None = Field(default=None, max_length=500, description="Document title")
    author: str | None = Field(default=None, max_length=255, description="Document author")
    checksum: str = Field(..., min_length=64, max_length=64, description="SHA256 checksum")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    last_modified: datetime = Field(..., description="File last modified timestamp")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks created")
    status: str = Field(
        default="processing",
        description="Processing status: processing, completed, failed",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, value: str) -> str:
        """Validate file type is supported."""
        supported_types = {"txt", "md", "pdf", "docx"}
        value_lower = value.lower().lstrip(".")
        if value_lower not in supported_types:
            raise ValueError(
                f"Unsupported file type: {value}. Supported: {supported_types}"
            )
        return value_lower

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        """Validate status value."""
        valid_statuses = {"processing", "completed", "failed"}
        value_lower = value.lower()
        if value_lower not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}, got: {value}")
        return value_lower

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        """Validate checksum format (SHA256 = 64 hex chars)."""
        if len(value) != 64:
            raise ValueError(f"Checksum must be 64 characters (SHA256), got {len(value)}")
        try:
            int(value, 16)
        except ValueError:
            raise ValueError("Checksum must be hexadecimal string")
        return value


class DocumentSearchResult(BaseModel):
    """Result from document search.

    Attributes:
        chunk_id: Chunk identifier
        document_id: Parent document ID
        filename: Source filename
        content: Chunk content
        score: Similarity score
        doc_type: Document type
        metadata: Additional metadata
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Source filename")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    doc_type: str = Field(..., description="Document type")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


__all__ = [
    "DocumentChunk",
    "Document",
    "DocumentSearchResult",
]
