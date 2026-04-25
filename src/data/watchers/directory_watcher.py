"""Directory watcher for monitoring document uploads."""

import logging
import threading
import time
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)


class DirectoryWatcher:
    """Monitor directory for new documents and process them.

    Uses watchdog library for file system event monitoring.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        callback: Callable[[Path], None],
        pattern: str = "*",
        recursive: bool = True,
    ) -> None:
        """Initialize the directory watcher.

        Args:
            watch_dir: Directory to watch
            callback: Function to call for new files
            pattern: Glob pattern for matching files
            recursive: Watch subdirectories
        """
        self.watch_dir = Path(watch_dir)
        self.callback = callback
        self.pattern = pattern
        self.recursive = recursive
        self._observer: Any = None
        self._running = False
        self._lock = threading.Lock()

        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created watch directory: {self.watch_dir}")

    def start(self) -> None:
        """Start watching directory."""
        with self._lock:
            if self._running:
                logger.warning("Watcher already running")
                return

            try:
                from watchdog.observers import Observer
                from watchdog.events import PatternMatchingEventHandler
            except ImportError:
                raise ImportError(
                    "watchdog is required for directory watching. "
                    "Install with: pip install watchdog"
                )

            class Handler(PatternMatchingEventHandler):
                def __init__(self_outer: "Handler", callback_inner: Callable[[Path], None]) -> None:
                    super().__init__(patterns=["*"], ignore_directories=True)
                    self_outer.callback = callback_inner
                    self_outer._seen_files: set[str] = set()

                def on_created(self_outer, event) -> None:
                    path = Path(event.src_path)
                    if path.is_file():
                        self_outer._processFile(path)

                def on_moved(self_outer, event) -> None:
                    if event.dest_path:
                        path = Path(event.dest_path)
                        if path.is_file():
                            self_outer._processFile(path)

                def _processFile(self_outer, path: Path) -> None:
                    # Debounce: wait a bit for file to be fully written
                    time.sleep(0.5)
                    if path.exists() and path.stat().st_size > 0:
                        try:
                            self_outer.callback(path)
                        except Exception as e:
                            logger.error(f"Error processing {path}: {e}")

            handler = Handler(self.callback)
            self._observer = Observer()
            self._observer.schedule(
                handler,
                str(self.watch_dir),
                recursive=self.recursive,
            )
            self._observer.start()
            self._running = True

            logger.info(f"Started watching: {self.watch_dir}")

    def stop(self) -> None:
        """Stop watching directory."""
        with self._lock:
            if not self._running:
                return

            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None

            self._running = False
            logger.info(f"Stopped watching: {self.watch_dir}")

    def is_running(self) -> bool:
        """Check if watcher is running.

        Returns:
            True if running
        """
        with self._lock:
            return self._running

    def __enter__(self) -> "DirectoryWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit."""
        self.stop()


class MockDirectoryWatcher(DirectoryWatcher):
    """Mock directory watcher for testing.

    Simulates file detection without actual monitoring.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        callback: Callable[[Path], None],
        pattern: str = "*",
        recursive: bool = True,
    ) -> None:
        """Initialize mock watcher."""
        super().__init__(watch_dir, callback, pattern, recursive)
        self._mock_running = False
        self._mock_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start mock watcher (just sets flag)."""
        with self._lock:
            if self._mock_running:
                return
            self._mock_running = True
            logger.info(f"Mock watcher started for: {self.watch_dir}")

    def stop(self) -> None:
        """Stop mock watcher."""
        with self._lock:
            if not self._mock_running:
                return
            self._mock_running = False
            logger.info(f"Mock watcher stopped for: {self.watch_dir}")

    def is_running(self) -> bool:
        """Check if mock watcher is running."""
        with self._lock:
            return self._mock_running

    def simulateFile(self, file_path: str | Path) -> None:
        """Simulate a file being detected.

        Args:
            file_path: Path to simulate
        """
        if self._mock_running:
            path = Path(file_path)
            try:
                self.callback(path)
            except Exception as e:
                logger.error(f"Error in mock callback for {path}: {e}")

    def scanExisting(self) -> list[Path]:
        """Return list of existing files in watch directory.

        Returns:
            List of existing file paths
        """
        if not self.watch_dir.exists():
            return []

        files: list[Path] = []
        for path in self.watch_dir.rglob("*") if self.recursive else self.watch_dir.glob("*"):
            if path.is_file():
                files.append(path)
        return files


__all__ = [
    "DirectoryWatcher",
    "MockDirectoryWatcher",
]
