"""Tests for directory watcher."""

import tempfile
from pathlib import Path

import pytest

from devmind.data.watchers.directory_watcher import (
    DirectoryWatcher,
    MockDirectoryWatcher,
)


class TestDirectoryWatcher:
    """Test DirectoryWatcher class."""

    def test_init(self) -> None:
        """Test watcher initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = lambda p: None
            watcher = DirectoryWatcher(tmpdir, callback)
            assert watcher.watch_dir == Path(tmpdir)
            assert watcher.callback == callback
            assert not watcher.is_running()

    def test_init_creates_directory(self) -> None:
        """Test that initialization creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watch_path = Path(tmpdir) / "new_watch_dir"
            assert not watch_path.exists()

            callback = lambda p: None
            watcher = DirectoryWatcher(watch_path, callback)

            assert watch_path.exists()
            assert watch_path.is_dir()


class TestMockDirectoryWatcher:
    """Test MockDirectoryWatcher class."""

    def test_init(self) -> None:
        """Test mock watcher initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = lambda p: None
            watcher = MockDirectoryWatcher(tmpdir, callback)
            assert watcher.watch_dir == Path(tmpdir)
            assert not watcher.is_running()

    def test_start_stop(self) -> None:
        """Test starting and stopping mock watcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = lambda p: None
            watcher = MockDirectoryWatcher(tmpdir, callback)

            watcher.start()
            assert watcher.is_running()

            watcher.stop()
            assert not watcher.is_running()

    def test_simulate_file(self) -> None:
        """Test simulating file detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detected_files = []

            def callback(path: Path) -> None:
                detected_files.append(path)

            watcher = MockDirectoryWatcher(tmpdir, callback)
            watcher.start()

            test_file = Path(tmpdir) / "test.pdf"
            watcher.simulateFile(test_file)

            assert len(detected_files) == 1
            assert detected_files[0] == test_file

    def test_simulate_file_not_running(self) -> None:
        """Test simulating file when not running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detected_files = []

            def callback(path: Path) -> None:
                detected_files.append(path)

            watcher = MockDirectoryWatcher(tmpdir, callback)

            test_file = Path(tmpdir) / "test.pdf"
            watcher.simulateFile(test_file)

            assert len(detected_files) == 0

    def test_scan_existing(self) -> None:
        """Test scanning existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = MockDirectoryWatcher(tmpdir, lambda p: None)

            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.pdf").touch()
            Path(tmpdir, "subdir").mkdir()
            Path(tmpdir, "subdir", "file3.md").touch()

            files = watcher.scanExisting()

            assert len(files) == 3
            assert all(isinstance(f, Path) for f in files)

    def test_context_manager(self) -> None:
        """Test using watcher as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with MockDirectoryWatcher(tmpdir, lambda p: None) as watcher:
                assert watcher.is_running()

            assert not watcher.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
