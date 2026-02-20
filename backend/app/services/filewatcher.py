"""File watcher service for detecting new and modified documents."""

import fnmatch
import hashlib
import logging
import shutil
import threading
from pathlib import Path
from typing import Callable, List, Optional

from sqlalchemy.orm import Session
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.core.config import KnowledgeForgeConfig
from app.models.database import Document

logger = logging.getLogger(__name__)

BUFFER_SIZE = 65536


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def _matches_patterns(file_name: str, patterns: List[str]) -> bool:
    """Check if a file name matches any of the configured glob patterns.

    Args:
        file_name: The file name to check.
        patterns: List of glob patterns to match against.

    Returns:
        True if the file name matches at least one pattern.
    """
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in patterns)


def _get_file_type(file_name: str) -> str:
    """Extract the file type (extension without dot) from a file name.

    Args:
        file_name: The file name to extract the type from.

    Returns:
        Lowercase file extension without the leading dot.
    """
    return Path(file_name).suffix.lstrip(".").lower()


class FileWatcherHandler(FileSystemEventHandler):
    """Watchdog event handler that processes new and modified files."""

    def __init__(
        self,
        config: KnowledgeForgeConfig,
        session_factory: Callable[[], Session],
        on_new_document: Optional[Callable[[Document], None]] = None,
        workflow_name: Optional[str] = None,
        force_rerun: bool = False,
    ) -> None:
        """Initialize the file watcher handler.

        Args:
            config: KnowledgeForge configuration object.
            session_factory: Callable that returns a new SQLAlchemy session.
            on_new_document: Optional callback invoked when a new document
                record is created, receiving the Document instance.
            workflow_name: Optional workflow identifier to tag documents with.
            force_rerun: When True, skip the hash-unchanged check.
        """
        super().__init__()
        self.config = config
        self.session_factory = session_factory
        self.on_new_document = on_new_document
        self.workflow_name = workflow_name
        self.force_rerun = force_rerun

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The watchdog file system event.
        """
        if event.is_directory:
            return
        self._process_file(str(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The watchdog file system event.
        """
        if event.is_directory:
            return
        self._process_file(str(event.src_path))

    def _process_file(self, file_path: str) -> None:
        """Process a detected file: hash, version, copy, and record.

        Args:
            file_path: Absolute path to the detected file.
        """
        path = Path(file_path)
        file_name = path.name

        if not _matches_patterns(file_name, self.config.source.file_patterns):
            return

        if not path.exists() or not path.is_file():
            return

        try:
            file_hash = compute_file_hash(file_path)
        except OSError:
            logger.error("Failed to compute hash for %s", file_path)
            return

        session = self.session_factory()
        try:
            self._ingest_file(session, path, file_name, file_hash)
        except Exception:
            session.rollback()
            logger.exception("Failed to ingest file %s", file_path)
        finally:
            session.close()

    def _ingest_file(
        self,
        session: Session,
        path: Path,
        file_name: str,
        file_hash: str,
    ) -> None:
        """Ingest a single file into the system.

        Args:
            session: Active SQLAlchemy session.
            path: Path object for the source file.
            file_name: Name of the file.
            file_hash: SHA-256 hash of the file contents.
        """
        query = session.query(Document).filter_by(file_name=file_name)
        if self.workflow_name is not None:
            query = query.filter_by(workflow_id=self.workflow_name)
        else:
            query = query.filter(Document.workflow_id.is_(None))
        existing = query.order_by(Document.version.desc()).first()

        if existing is not None and existing.file_hash == file_hash and not self.force_rerun:
            logger.info("Skipping %s - hash unchanged (version %d)", file_name, existing.version)
            return

        new_version = (existing.version + 1) if existing is not None else 1

        # Copy to staging
        staging_dir = Path(self.config.source.staging_folder)
        staging_dir.mkdir(parents=True, exist_ok=True)

        stem = path.stem
        suffix = path.suffix
        if new_version > 1:
            staging_name = f"{stem}_v{new_version}{suffix}"
        else:
            staging_name = file_name

        staging_path = staging_dir / staging_name
        shutil.copy2(str(path), str(staging_path))

        doc = Document(
            file_name=file_name,
            file_path=str(staging_path),
            file_type=_get_file_type(file_name),
            version=new_version,
            file_hash=file_hash,
            workflow_id=self.workflow_name,
            status="pending",
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)

        logger.info(
            "Ingested %s (version %d, hash=%s)",
            file_name,
            new_version,
            file_hash[:12],
        )

        if self.on_new_document is not None:
            self.on_new_document(doc)


class FileWatcher:
    """Manages the watchdog observer for monitoring source folders."""

    def __init__(
        self,
        config: KnowledgeForgeConfig,
        session_factory: Callable[[], Session],
        on_new_document: Optional[Callable[[Document], None]] = None,
        workflow_name: Optional[str] = None,
        force_rerun: bool = False,
    ) -> None:
        """Initialize the file watcher.

        Args:
            config: KnowledgeForge configuration object.
            session_factory: Callable that returns a new SQLAlchemy session.
            on_new_document: Optional callback invoked for new documents.
            workflow_name: Optional workflow identifier to tag documents with.
            force_rerun: When True, skip the hash-unchanged check.
        """
        self.config = config
        self.session_factory = session_factory
        self.on_new_document = on_new_document
        self.workflow_name = workflow_name
        self.force_rerun = force_rerun
        self._handler = FileWatcherHandler(
            config, session_factory, on_new_document,
            workflow_name=workflow_name, force_rerun=force_rerun,
        )
        self._observer: Optional[Observer] = None
        self._thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self) -> None:
        """Start the file watcher in a background thread."""
        watch_folder = Path(self.config.source.watch_folder)
        watch_folder.mkdir(parents=True, exist_ok=True)

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(watch_folder),
            recursive=False,
        )
        self._observer.daemon = True
        self._observer.start()
        self.is_running = True
        logger.info("File watcher started on %s", watch_folder)

    def stop(self) -> None:
        """Stop the file watcher gracefully."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self.is_running = False
        logger.info("File watcher stopped")

    def scan_existing(self) -> List[Document]:
        """Scan the source folder for existing files and ingest them.

        Returns:
            List of newly created Document records.
        """
        watch_folder = Path(self.config.source.watch_folder)
        if not watch_folder.exists():
            logger.warning("Watch folder does not exist: %s", watch_folder)
            return []

        ingested: List[Document] = []

        def track_document(doc: Document) -> None:
            ingested.append(doc)
            if self.on_new_document is not None:
                self.on_new_document(doc)

        # Temporarily swap callback to track ingested docs
        original_callback = self._handler.on_new_document
        self._handler.on_new_document = track_document

        try:
            for file_path in sorted(watch_folder.iterdir()):
                if not file_path.is_file():
                    continue
                if _matches_patterns(file_path.name, self.config.source.file_patterns):
                    self._handler._process_file(str(file_path))
        finally:
            self._handler.on_new_document = original_callback

        logger.info("Scan complete: %d files ingested", len(ingested))
        return ingested
