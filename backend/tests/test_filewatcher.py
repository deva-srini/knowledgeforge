"""Unit tests for the file watcher service."""

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import KnowledgeForgeConfig, SourceConfig
from app.db.session import init_db
from app.models.database import Document
from app.services.filewatcher import FileWatcher, compute_file_hash


@pytest.fixture
def test_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create temporary source and staging directories."""
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    return {"source": source, "staging": staging}


@pytest.fixture
def config(test_dirs: dict[str, Path]) -> KnowledgeForgeConfig:
    """Create a test configuration pointing to temporary directories."""
    return KnowledgeForgeConfig(
        source=SourceConfig(
            watch_folder=str(test_dirs["source"]),
            staging_folder=str(test_dirs["staging"]),
            file_patterns=["*.pdf", "*.docx", "*.txt"],
        )
    )


@pytest.fixture
def db_session_factory(tmp_path: Path) -> sessionmaker[Session]:
    """Create an in-memory SQLite database and return a session factory."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine)
    return factory


@pytest.fixture
def watcher(
    config: KnowledgeForgeConfig,
    db_session_factory: sessionmaker[Session],
) -> FileWatcher:
    """Create a FileWatcher instance for testing."""
    return FileWatcher(config, db_session_factory)


class TestComputeFileHash:
    """Tests for the compute_file_hash function."""

    def test_hash_consistency(self, tmp_path: Path) -> None:
        """Test that hashing the same file twice returns the same hash."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")
        hash1 = compute_file_hash(str(file))
        hash2 = compute_file_hash(str(file))
        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test that different file contents produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content A")
        file2.write_text("content B")
        assert compute_file_hash(str(file1)) != compute_file_hash(str(file2))

    def test_hash_is_sha256(self, tmp_path: Path) -> None:
        """Test that the hash is a valid SHA-256 hex string."""
        file = tmp_path / "test.txt"
        file.write_text("test")
        file_hash = compute_file_hash(str(file))
        assert len(file_hash) == 64
        assert all(c in "0123456789abcdef" for c in file_hash)


class TestFileWatcherScanExisting:
    """Tests for the scan_existing method."""

    def test_scan_picks_up_matching_files(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
        db_session_factory: sessionmaker[Session],
    ) -> None:
        """Test that scan_existing finds all matching files."""
        (test_dirs["source"] / "doc1.pdf").write_bytes(b"pdf content 1")
        (test_dirs["source"] / "doc2.txt").write_bytes(b"text content")

        docs = watcher.scan_existing()
        assert len(docs) == 2

        session = db_session_factory()
        all_docs = session.query(Document).all()
        assert len(all_docs) == 2
        session.close()

    def test_scan_ignores_non_matching_files(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
        db_session_factory: sessionmaker[Session],
    ) -> None:
        """Test that scan_existing ignores files not matching patterns."""
        (test_dirs["source"] / "image.png").write_bytes(b"png data")
        (test_dirs["source"] / "data.csv").write_bytes(b"csv data")
        (test_dirs["source"] / "doc.pdf").write_bytes(b"pdf data")

        docs = watcher.scan_existing()
        assert len(docs) == 1
        assert docs[0].file_name == "doc.pdf"

    def test_scan_skips_duplicate_hash(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
        db_session_factory: sessionmaker[Session],
    ) -> None:
        """Test that scanning the same file twice skips the duplicate."""
        (test_dirs["source"] / "doc.pdf").write_bytes(b"same content")

        docs_first = watcher.scan_existing()
        assert len(docs_first) == 1

        docs_second = watcher.scan_existing()
        assert len(docs_second) == 0

        session = db_session_factory()
        assert session.query(Document).count() == 1
        session.close()

    def test_scan_creates_new_version_on_modified_file(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
        db_session_factory: sessionmaker[Session],
    ) -> None:
        """Test that a modified file creates a new version."""
        pdf_file = test_dirs["source"] / "report.pdf"
        pdf_file.write_bytes(b"version 1 content")

        docs_v1 = watcher.scan_existing()
        assert len(docs_v1) == 1
        assert docs_v1[0].version == 1

        # Modify the file
        pdf_file.write_bytes(b"version 2 content - updated")

        docs_v2 = watcher.scan_existing()
        assert len(docs_v2) == 1
        assert docs_v2[0].version == 2

        session = db_session_factory()
        all_docs = (
            session.query(Document)
            .filter_by(file_name="report.pdf")
            .order_by(Document.version)
            .all()
        )
        assert len(all_docs) == 2
        assert all_docs[0].version == 1
        assert all_docs[1].version == 2
        session.close()


class TestFileWatcherCopyToStaging:
    """Tests for staging folder file copy behavior."""

    def test_file_copied_to_staging(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
    ) -> None:
        """Test that ingested files are copied to the staging folder."""
        (test_dirs["source"] / "doc.pdf").write_bytes(b"pdf content")

        docs = watcher.scan_existing()
        assert len(docs) == 1

        staging_files = list(test_dirs["staging"].iterdir())
        assert len(staging_files) == 1
        assert staging_files[0].name == "doc.pdf"

    def test_versioned_file_name_in_staging(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
    ) -> None:
        """Test that version 2+ files get versioned names in staging."""
        pdf_file = test_dirs["source"] / "doc.pdf"
        pdf_file.write_bytes(b"v1")
        watcher.scan_existing()

        pdf_file.write_bytes(b"v2")
        watcher.scan_existing()

        staging_files = sorted(f.name for f in test_dirs["staging"].iterdir())
        assert "doc.pdf" in staging_files
        assert "doc_v2.pdf" in staging_files


class TestFileWatcherDocumentRecord:
    """Tests for document record creation."""

    def test_document_record_fields(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
        db_session_factory: sessionmaker[Session],
    ) -> None:
        """Test that the document record has all expected fields populated."""
        (test_dirs["source"] / "manual.docx").write_bytes(b"docx content here")

        docs = watcher.scan_existing()
        assert len(docs) == 1

        session = db_session_factory()
        doc = session.query(Document).first()
        assert doc is not None
        assert doc.file_name == "manual.docx"
        assert doc.file_type == "docx"
        assert doc.version == 1
        assert doc.status == "pending"
        assert doc.file_hash is not None
        assert len(doc.file_hash) == 64
        assert doc.file_path.endswith("manual.docx")
        assert doc.created_at is not None
        session.close()


class TestFileWatcherCallback:
    """Tests for the on_new_document callback."""

    def test_callback_invoked_on_new_document(
        self,
        config: KnowledgeForgeConfig,
        db_session_factory: sessionmaker[Session],
        test_dirs: dict[str, Path],
    ) -> None:
        """Test that the callback fires for each new document."""
        callback_docs: list[Document] = []

        def track(doc: Document) -> None:
            callback_docs.append(doc)

        watcher = FileWatcher(config, db_session_factory, on_new_document=track)
        (test_dirs["source"] / "a.pdf").write_bytes(b"aaa")
        (test_dirs["source"] / "b.pdf").write_bytes(b"bbb")

        watcher.scan_existing()
        assert len(callback_docs) == 2


class TestFileWatcherStartStop:
    """Tests for watcher lifecycle (start/stop)."""

    def test_start_and_stop(
        self,
        watcher: FileWatcher,
        test_dirs: dict[str, Path],
    ) -> None:
        """Test that the watcher can start and stop without errors."""
        watcher.start()
        assert watcher.is_running is True

        watcher.stop()
        assert watcher.is_running is False

    def test_watcher_detects_new_file(
        self,
        config: KnowledgeForgeConfig,
        test_dirs: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Test that a running watcher detects a newly dropped file."""
        # Use file-based SQLite so the watchdog thread shares the same DB
        db_path = tmp_path / "watcher_test.db"
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        init_db(engine)
        factory = sessionmaker(bind=engine)

        file_watcher = FileWatcher(config, factory)
        file_watcher.start()
        try:
            time.sleep(0.5)
            (test_dirs["source"] / "new_doc.pdf").write_bytes(b"new pdf content")
            time.sleep(2)

            session = factory()
            docs = session.query(Document).all()
            assert len(docs) >= 1
            assert any(d.file_name == "new_doc.pdf" for d in docs)
            session.close()
        finally:
            file_watcher.stop()
            engine.dispose()
