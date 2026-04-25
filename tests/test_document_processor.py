"""Tests for document processor."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from devmind.data.processors.document_processor import (
    DocumentProcessor,
    MockDocumentProcessor,
)


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_init(self) -> None:
        """Test processor initialization."""
        processor = DocumentProcessor()
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert isinstance(processor.supported_formats, set)

    def test_init_custom_params(self) -> None:
        """Test processor with custom parameters."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 20

    def test_validate_file_not_found(self) -> None:
        """Test validation with non-existent file."""
        processor = DocumentProcessor()
        is_valid, error = processor.validateFile("/nonexistent/file.pdf")
        assert not is_valid
        assert "not found" in error.lower()

    def test_validate_file_unsupported_type(self) -> None:
        """Test validation with unsupported file type."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            is_valid, error = processor.validateFile(temp_path)
            assert not is_valid
            assert "unsupported" in error.lower()
        finally:
            Path(temp_path).unlink()

    def test_calculate_checksum(self) -> None:
        """Test checksum calculation."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for checksum")
            temp_path = f.name

        try:
            checksum = processor.calculateChecksum(temp_path)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

            expected = hashlib.sha256(b"test content for checksum").hexdigest()
            assert checksum == expected
        finally:
            Path(temp_path).unlink()

    def test_parse_text_file(self) -> None:
        """Test parsing text file."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("This is a test document.\n\nWith multiple paragraphs.")
            temp_path = f.name

        try:
            result = processor.parseFile(temp_path)
            assert "text" in result
            assert "title" in result
            assert "metadata" in result
            assert "test document" in result["text"].lower()
        finally:
            Path(temp_path).unlink()

    def test_parse_markdown_file(self) -> None:
        """Test parsing markdown file with title extraction."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("# Document Title\n\nThis is content.")
            temp_path = f.name

        try:
            result = processor.parseFile(temp_path)
            assert result["title"] == "Document Title"
            assert "This is content" in result["text"]
        finally:
            Path(temp_path).unlink()

    def test_chunk_text(self) -> None:
        """Test text chunking."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = "word " * 100
        chunks = processor.chunkText(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks)

    def test_chunk_text_empty(self) -> None:
        """Test chunking empty text."""
        processor = DocumentProcessor()
        chunks = processor.chunkText("")
        assert chunks == []

    def test_chunk_text_short(self) -> None:
        """Test chunking short text."""
        processor = DocumentProcessor(chunk_size=500)
        chunks = processor.chunkText("Short text.")
        assert len(chunks) == 1
        assert "Short text" in chunks[0]


class TestMockDocumentProcessor:
    """Test MockDocumentProcessor class."""

    def test_validate_file_supported(self) -> None:
        """Test mock validation with supported type."""
        processor = MockDocumentProcessor()
        is_valid, error = processor.validateFile("test.pdf")
        assert is_valid
        assert error is None

    def test_validate_file_unsupported(self) -> None:
        """Test mock validation with unsupported type."""
        processor = MockDocumentProcessor()
        is_valid, error = processor.validateFile("test.xyz")
        assert not is_valid
        assert error is not None

    def test_calculate_checksum(self) -> None:
        """Test mock checksum is deterministic."""
        processor = MockDocumentProcessor()
        checksum1 = processor.calculateChecksum("/path/to/file.pdf")
        checksum2 = processor.calculateChecksum("/path/to/file.pdf")
        assert checksum1 == checksum2
        assert len(checksum1) == 64

    def test_parse_file(self) -> None:
        """Test mock parsing."""
        processor = MockDocumentProcessor()
        result = processor.parseFile("/path/to/file.pdf")

        assert "text" in result
        assert "title" in result
        assert "metadata" in result
        assert result["title"] == "file"
        assert result["metadata"].get("mock") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
