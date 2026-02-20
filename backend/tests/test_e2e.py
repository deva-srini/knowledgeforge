"""End-to-end integration tests for the KnowledgeForge pipeline.

Uses the real travel.pdf reference document and real services (parser,
extractor, transformer, chunker, embedder, indexer) with isolated temp
directories and in-memory SQLite. These tests are marked with @pytest.mark.e2e
and can be skipped via ``pytest -m "not e2e"``.
"""

import os
import shutil
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import KnowledgeForgeConfig
from app.metrics.collector import MetricsCollector
from app.models.database import Base, Document, WorkflowRun, WorkflowStage
from app.services.chunking import StructureAwareChunker
from app.services.embedding import Embedder
from app.services.extraction import ContentExtractor
from app.services.indexing import ChromaIndexer
from app.services.parsing import DocumentParser
from app.services.transformation import ContentTransformer
from app.services.workflow import STAGE_NAMES, WorkflowOrchestrator

TRAVEL_PDF = os.path.join(
    os.path.dirname(__file__), "..", "..", "reference", "travel.pdf"
)


@pytest.fixture(scope="module")
def tmp_dirs():
    """Create temp source, staging, and chromadb directories."""
    base = tempfile.mkdtemp(prefix="kf_e2e_")
    source_dir = os.path.join(base, "source")
    staging_dir = os.path.join(base, "staging")
    chromadb_dir = os.path.join(base, "chromadb")
    os.makedirs(source_dir)
    os.makedirs(staging_dir)
    os.makedirs(chromadb_dir)
    yield {
        "base": base,
        "source": source_dir,
        "staging": staging_dir,
        "chromadb": chromadb_dir,
    }
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def e2e_config(tmp_dirs):
    """Create a KnowledgeForgeConfig pointing to temp dirs and in-memory SQLite."""
    return KnowledgeForgeConfig(
        source={
            "watch_folder": tmp_dirs["source"],
            "staging_folder": tmp_dirs["staging"],
        },
        indexing={
            "chromadb_path": tmp_dirs["chromadb"],
            "default_collection": "e2e_test",
        },
        database={"url": "sqlite:///:memory:"},
    )


@pytest.fixture(scope="module")
def db_session_factory(e2e_config):
    """Create an in-memory SQLite database with all tables."""
    engine = create_engine(
        e2e_config.database.url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return factory


@pytest.fixture(scope="module")
def orchestrator(e2e_config, db_session_factory):
    """Create a real WorkflowOrchestrator with all real services."""
    parser = DocumentParser(e2e_config)
    extractor = ContentExtractor(e2e_config)
    transformer = ContentTransformer(e2e_config)
    chunker = StructureAwareChunker(e2e_config)
    embedder = Embedder(e2e_config)
    indexer = ChromaIndexer(e2e_config)
    return WorkflowOrchestrator(
        e2e_config,
        db_session_factory,
        parser=parser,
        extractor=extractor,
        transformer=transformer,
        chunker=chunker,
        embedder=embedder,
        indexer=indexer,
    )


def _create_document(session: Session, file_path: str, **kwargs) -> Document:
    """Create and persist a Document record.

    Args:
        session: Active SQLAlchemy session.
        file_path: Path to the source file.
        **kwargs: Optional overrides for Document fields.

    Returns:
        The persisted Document instance.
    """
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    defaults = {
        "file_name": file_name,
        "file_path": file_path,
        "file_type": ext.lstrip(".").lower(),
        "file_hash": "e2e_test_hash",
        "status": "pending",
    }
    defaults.update(kwargs)
    doc = Document(**defaults)
    session.add(doc)
    session.commit()
    session.refresh(doc)
    session.expunge(doc)
    return doc


@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end integration tests using the real pipeline."""

    def test_full_pipeline_pdf(self, orchestrator, db_session_factory):
        """Process travel.pdf end-to-end and verify success status."""
        session = db_session_factory()
        doc = _create_document(session, os.path.abspath(TRAVEL_PDF))
        session.close()

        run = orchestrator.process_document(doc)

        assert run.status == "success"
        assert run.total_chunks > 0
        assert run.total_tokens > 0

        # Verify document status in DB
        session = db_session_factory()
        db_doc = session.query(Document).filter_by(id=doc.id).one()
        assert db_doc.status == "indexed"

        # Verify all 6 stages succeeded
        stages = (
            session.query(WorkflowStage).filter_by(run_id=run.id).all()
        )
        assert len(stages) == 6
        for stage in stages:
            assert stage.status == "success", (
                f"Stage '{stage.stage_name}' has status '{stage.status}'"
            )
        session.close()

    def test_chromadb_has_chunks(self, orchestrator, db_session_factory, e2e_config):
        """After indexing, ChromaDB collection contains chunks with metadata."""
        import chromadb

        client = chromadb.PersistentClient(
            path=e2e_config.indexing.chromadb_path
        )
        collection = client.get_collection(name="e2e_test")
        results = collection.get(include=["metadatas", "documents"])

        assert len(results["ids"]) > 0
        assert results["metadatas"] is not None
        assert len(results["metadatas"]) > 0
        # Check metadata fields exist on first chunk
        first_meta = results["metadatas"][0]
        assert "document_id" in first_meta
        assert "file_name" in first_meta
        assert "chunk_index" in first_meta

    def test_metrics_after_processing(self, db_session_factory, e2e_config):
        """MetricsCollector returns correct counts after pipeline run."""
        collector = MetricsCollector(
            session_factory=db_session_factory,
            config=e2e_config,
        )
        metrics = collector.collect()

        assert metrics.total_documents >= 1
        assert metrics.successful_runs >= 1
        assert metrics.total_chunks > 0
        assert metrics.total_tokens > 0

    def test_reprocess_creates_new_run(self, orchestrator, db_session_factory):
        """Reprocessing the same document creates a second WorkflowRun."""
        session = db_session_factory()
        # Get the first document (from test_full_pipeline_pdf)
        doc = session.query(Document).first()
        assert doc is not None
        session.expunge(doc)
        session.close()

        run2 = orchestrator.process_document(doc)

        assert run2.status == "success"

        # Verify there are now 2 runs for this document
        session = db_session_factory()
        runs = (
            session.query(WorkflowRun)
            .filter_by(document_id=doc.id)
            .all()
        )
        assert len(runs) >= 2
        session.close()

    def test_versioning_on_modified_file(self, orchestrator, db_session_factory):
        """A document with a different version is indexed separately."""
        session = db_session_factory()
        doc = _create_document(
            session,
            os.path.abspath(TRAVEL_PDF),
            version=2,
            file_hash="version2_hash",
        )
        session.close()

        run = orchestrator.process_document(doc)

        assert run.status == "success"

        # Verify the new doc is indexed
        session = db_session_factory()
        db_doc = session.query(Document).filter_by(id=doc.id).one()
        assert db_doc.status == "indexed"
        assert db_doc.version == 2
        session.close()

    def test_failed_document_recovery(self, orchestrator, db_session_factory):
        """Processing a nonexistent file fails, then a valid file succeeds."""
        session = db_session_factory()
        bad_doc = _create_document(session, "/nonexistent/path/fake.pdf")
        session.close()

        bad_run = orchestrator.process_document(bad_doc)
        assert bad_run.status == "failed"

        # Verify bad doc is marked failed
        session = db_session_factory()
        db_bad = session.query(Document).filter_by(id=bad_doc.id).one()
        assert db_bad.status == "failed"
        session.close()

        # Process a valid file — should succeed independently
        session = db_session_factory()
        good_doc = _create_document(
            session,
            os.path.abspath(TRAVEL_PDF),
            file_hash="recovery_hash",
        )
        session.close()

        good_run = orchestrator.process_document(good_doc)
        assert good_run.status == "success"

        session = db_session_factory()
        db_good = session.query(Document).filter_by(id=good_doc.id).one()
        assert db_good.status == "indexed"
        session.close()
