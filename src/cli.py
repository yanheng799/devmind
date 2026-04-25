"""Command-line interface for DEVMIND."""

import argparse
import json
import logging
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings
from data.database.database import PredictionDatabase
from data.processors.document_processor import DocumentProcessor
from data.vectorstore.document_store import DocumentVectorStore
from data.vectorstore.milvus_client import get_embedding_model
from data.watchers.directory_watcher import DirectoryWatcher


def json_serialize(obj: object) -> object:
    """Custom JSON serializer for datetime and other types.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_upload_doc(args: argparse.Namespace) -> int:
    """Handle upload-doc command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        settings = get_settings()

        processor = DocumentProcessor()
        embedding_model = get_embedding_model(use_mock=args.mock)
        vector_store = DocumentVectorStore()

        with PredictionDatabase() as db:
            file_path = Path(args.file)

            is_valid, error = processor.validateFile(file_path)
            if not is_valid:
                logger.error(f"File validation failed: {error}")
                return 1

            checksum = processor.calculateChecksum(file_path)

            existing_doc = db.get_document_by_checksum(checksum)
            if existing_doc and not args.force:
                logger.info(f"Document already exists: {existing_doc['document_id']}")
                print(f"Document already uploaded: {existing_doc['document_id']}")
                return 0

            parsed = processor.parseFile(file_path)
            chunks = processor.chunkText(parsed["text"])

            document_id = str(uuid.uuid4())
            file_stat = file_path.stat()

            document = {
                "document_id": document_id,
                "filename": file_path.name,
                "file_path": str(file_path.absolute()),
                "file_type": file_path.suffix.lstrip(".").lower(),
                "file_size": file_stat.st_size,
                "title": parsed.get("title"),
                "author": parsed.get("author"),
                "checksum": checksum,
                "upload_time": datetime.now(),
                "last_modified": datetime.fromtimestamp(file_stat.st_mtime),
                "chunk_count": len(chunks),
                "status": "processing",
                "metadata": parsed.get("metadata", {}),
            }

            db.insert_document(document)

            for idx, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                embedding = embedding_model.embed_single(chunk_text)

                vector_id = vector_store.insertChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    embedding=embedding,
                    content=chunk_text,
                    doc_type=document["file_type"],
                    metadata={"index": idx},
                )

                db.insert_chunk({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": idx,
                    "content": chunk_text,
                    "embedding_id": vector_id,
                })

            db.update_document_status(document_id, "completed", len(chunks))

            print(f"Successfully uploaded document: {document_id}")
            print(f"Filename: {document['filename']}")
            print(f"Chunks: {len(chunks)}")

        vector_store.close()
        return 0

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        return 1


def cmd_list_docs(args: argparse.Namespace) -> int:
    """Handle list-docs command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with PredictionDatabase() as db:
            documents = db.list_documents(
                file_type=args.type,
                status=args.status,
                limit=args.limit,
            )

            if not documents:
                print("No documents found.")
                return 0

            print(f"Found {len(documents)} document(s):\n")

            for doc in documents:
                print(f"ID: {doc['document_id']}")
                print(f"Filename: {doc['filename']}")
                print(f"Type: {doc['file_type']}")
                print(f"Size: {doc['file_size']:,} bytes")
                print(f"Status: {doc['status']}")
                print(f"Chunks: {doc['chunk_count']}")
                print(f"Uploaded: {doc['upload_time']}")
                if doc.get("title"):
                    print(f"Title: {doc['title']}")
                print("-" * 60)

        return 0

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return 1


def cmd_delete_doc(args: argparse.Namespace) -> int:
    """Handle delete-doc command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        settings = get_settings()

        with PredictionDatabase() as db:
            doc = db.get_document(args.doc_id)
            if not doc:
                logger.error(f"Document not found: {args.doc_id}")
                return 1

            vector_store = DocumentVectorStore()
            vector_store.deleteChunks(args.doc_id)
            vector_store.close()

            db.delete_document(args.doc_id)

            print(f"Deleted document: {args.doc_id}")

        return 0

    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        return 1


def cmd_search_docs(args: argparse.Namespace) -> int:
    """Handle search-docs command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        settings = get_settings()

        embedding_model = get_embedding_model(use_mock=args.mock)
        vector_store = DocumentVectorStore()

        query_embedding = embedding_model.embed_single(args.query)

        results = vector_store.searchChunks(
            query_embedding=query_embedding,
            top_k=args.top_k,
            doc_type=args.type,
        )

        vector_store.close()

        if not results:
            print("No matching chunks found.")
            return 0

        print(f"Found {len(results)} result(s):\n")

        for idx, result in enumerate(results, start=1):
            print(f"--- Result {idx} (Score: {result['score']:.3f}) ---")
            print(f"Document ID: {result['document_id']}")
            print(f"Type: {result['doc_type']}")
            content = result['content']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"Content: {content}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return 1


def cmd_watch_dir(args: argparse.Namespace) -> int:
    """Handle watch-dir command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        settings = get_settings()
        watch_dir = Path(args.dir) if args.dir else settings.doc_upload_dir

        if not watch_dir.exists():
            watch_dir.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor()
        embedding_model = get_embedding_model(use_mock=args.mock)
        vector_store = DocumentVectorStore()

        running = True

        def stop_handler(signum: int, frame: Any) -> None:
            nonlocal running
            print("\nStopping directory watcher...")
            running = False

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        def process_file(file_path: Path) -> None:
            nonlocal processor, embedding_model, vector_store

            try:
                ext = file_path.suffix.lstrip(".").lower()
                if ext not in {"txt", "md", "pdf", "docx"}:
                    return

                print(f"Processing file: {file_path.name}")

                with PredictionDatabase() as db:
                    is_valid, error = processor.validateFile(file_path)
                    if not is_valid:
                        logger.error(f"File validation failed: {error}")
                        return

                    checksum = processor.calculateChecksum(file_path)
                    existing_doc = db.get_document_by_checksum(checksum)
                    if existing_doc:
                        print(f"  Skipping (already exists): {existing_doc['document_id']}")
                        return

                    parsed = processor.parseFile(file_path)
                    chunks = processor.chunkText(parsed["text"])

                    document_id = str(uuid.uuid4())
                    file_stat = file_path.stat()

                    document = {
                        "document_id": document_id,
                        "filename": file_path.name,
                        "file_path": str(file_path.absolute()),
                        "file_type": ext,
                        "file_size": file_stat.st_size,
                        "title": parsed.get("title"),
                        "author": parsed.get("author"),
                        "checksum": checksum,
                        "upload_time": datetime.now(),
                        "last_modified": datetime.fromtimestamp(file_stat.st_mtime),
                        "chunk_count": len(chunks),
                        "status": "processing",
                        "metadata": parsed.get("metadata", {}),
                    }

                    db.insert_document(document)

                    for idx, chunk_text in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        embedding = embedding_model.embed_single(chunk_text)

                        vector_id = vector_store.insertChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            embedding=embedding,
                            content=chunk_text,
                            doc_type=ext,
                            metadata={"index": idx},
                        )

                        db.insert_chunk({
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "chunk_index": idx,
                            "content": chunk_text,
                            "embedding_id": vector_id,
                        })

                    db.update_document_status(document_id, "completed", len(chunks))

                    print(f"  Uploaded: {document_id} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        watcher = DirectoryWatcher(
            watch_dir=watch_dir,
            callback=process_file,
        )

        print(f"Watching directory: {watch_dir}")
        print(f"Supported formats: txt, md, pdf, docx")
        print("Press Ctrl+C to stop\n")

        watcher.start()

        try:
            while running:
                import time

                time.sleep(1)
        finally:
            watcher.stop()
            vector_store.close()

        return 0

    except Exception as e:
        logger.error(f"Directory watch failed: {e}")
        return 1


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="DEVMIND - Knowledge Management and Stock Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a document
  devmind upload-doc path/to/document.pdf

  # List uploaded documents
  devmind list-docs

  # Search document content
  devmind search-docs "降准政策影响"

  # Watch directory for new documents
  devmind watch-dir --dir data/documents
        """,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock collectors and models (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload document command
    upload_parser = subparsers.add_parser(
        "upload-doc",
        help="Upload a document to the RAG knowledge base",
    )
    upload_parser.add_argument(
        "file",
        help="Path to document file",
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-upload even if file exists",
    )

    # List documents command
    list_parser = subparsers.add_parser(
        "list-docs",
        help="List uploaded documents",
    )
    list_parser.add_argument(
        "--type",
        help="Filter by file type (pdf, docx, txt, md)",
    )
    list_parser.add_argument(
        "--status",
        help="Filter by status (processing, completed, failed)",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of documents (default: 50)",
    )

    # Delete document command
    delete_parser = subparsers.add_parser(
        "delete-doc",
        help="Delete a document from the knowledge base",
    )
    delete_parser.add_argument(
        "doc_id",
        help="Document ID to delete",
    )

    # Search documents command
    search_parser = subparsers.add_parser(
        "search-docs",
        help="Search document content",
    )
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--type",
        help="Filter by document type",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )

    # Watch directory command
    watch_parser = subparsers.add_parser(
        "watch-dir",
        help="Watch directory for new documents",
    )
    watch_parser.add_argument(
        "--dir",
        help="Directory to watch (default: data/documents)",
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0

    # Validate settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure DEVMIND_LLM_API_KEY is set")
        return 1

    # Dispatch command
    if args.command == "upload-doc":
        return cmd_upload_doc(args)
    elif args.command == "list-docs":
        return cmd_list_docs(args)
    elif args.command == "delete-doc":
        return cmd_delete_doc(args)
    elif args.command == "search-docs":
        return cmd_search_docs(args)
    elif args.command == "watch-dir":
        return cmd_watch_dir(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
