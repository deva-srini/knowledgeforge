"""Unit tests for the MetricsCollector service.

All tests use an in-memory SQLite database with pre-seeded records and a
mocked ChromaDB client to avoid filesystem side-effects.
"""

from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import KnowledgeForgeConfig
from app.metrics.collector import MetricsCollector, PipelineMetrics, StageMetrics
from app.models.database import Base, Document, WorkflowRun, WorkflowStage


@pytest.fixture()
def db_session_factory():
    """Create an in-memory SQLite database and return a session factory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return factory


@pytest.fixture()
def config():
    """Return a default KnowledgeForgeConfig."""
    return KnowledgeForgeConfig()


def _seed_document(session, *, status="indexed", days_ago=0):
    """Insert a Document record and return it."""
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    doc = Document(
        file_name=f"doc_{status}_{days_ago}.pdf",
        file_path=f"/tmp/doc_{status}_{days_ago}.pdf",
        file_type="pdf",
        version=1,
        file_hash=f"hash_{status}_{days_ago}",
        status=status,
        created_at=created,
    )
    session.add(doc)
    session.flush()
    return doc


def _seed_workflow_run(
    session,
    document_id,
    *,
    status="success",
    total_chunks=10,
    total_tokens=100,
    duration_seconds=5.0,
):
    """Insert a WorkflowRun record and return it."""
    started = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
    completed = datetime.now(timezone.utc)
    run = WorkflowRun(
        document_id=document_id,
        status=status,
        started_at=started,
        completed_at=completed if status in ("success", "failed") else None,
        total_chunks=total_chunks,
        total_tokens=total_tokens,
    )
    session.add(run)
    session.flush()
    return run


def _seed_stage(
    session,
    run_id,
    stage_name,
    *,
    status="success",
    duration_seconds=1.0,
):
    """Insert a WorkflowStage record and return it."""
    started = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
    completed = datetime.now(timezone.utc)
    stage = WorkflowStage(
        run_id=run_id,
        stage_name=stage_name,
        status=status,
        started_at=started,
        completed_at=completed if status != "pending" else None,
    )
    session.add(stage)
    session.flush()
    return stage


class TestMetricsCollector:
    """Tests for the MetricsCollector."""

    def test_empty_database_returns_zeros(self, db_session_factory, config):
        """Empty database yields zero-valued metrics."""
        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        assert metrics.total_documents == 0
        assert metrics.documents_by_status == {}
        assert metrics.total_chunks == 0
        assert metrics.total_tokens == 0
        assert metrics.avg_chunks_per_document == 0.0
        assert metrics.total_workflow_runs == 0
        assert metrics.successful_runs == 0
        assert metrics.failed_runs == 0
        assert metrics.avg_processing_time_seconds == 0.0
        assert metrics.stage_metrics == []
        assert metrics.documents_last_7_days == 0

    def test_documents_by_status(self, db_session_factory, config):
        """Documents are correctly grouped by status."""
        session = db_session_factory()
        _seed_document(session, status="indexed")
        _seed_document(session, status="indexed")
        _seed_document(session, status="failed")
        _seed_document(session, status="pending")
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        assert metrics.total_documents == 4
        assert metrics.documents_by_status["indexed"] == 2
        assert metrics.documents_by_status["failed"] == 1
        assert metrics.documents_by_status["pending"] == 1

    def test_total_chunks_and_tokens(self, db_session_factory, config):
        """Total chunks/tokens summed from successful runs only."""
        session = db_session_factory()
        doc1 = _seed_document(session, status="indexed")
        doc2 = _seed_document(session, status="failed")
        _seed_workflow_run(session, doc1.id, status="success", total_chunks=10, total_tokens=100)
        _seed_workflow_run(session, doc2.id, status="failed", total_chunks=5, total_tokens=50)
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        # Only successful run's chunks/tokens counted
        assert metrics.total_chunks == 10
        assert metrics.total_tokens == 100

    def test_avg_chunks_per_document(self, db_session_factory, config):
        """Average chunks per document = total_chunks / indexed count."""
        session = db_session_factory()
        doc1 = _seed_document(session, status="indexed")
        doc2 = _seed_document(session, status="indexed")
        _seed_workflow_run(session, doc1.id, status="success", total_chunks=10, total_tokens=100)
        _seed_workflow_run(session, doc2.id, status="success", total_chunks=20, total_tokens=200)
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        # 30 total chunks / 2 indexed docs = 15.0
        assert metrics.avg_chunks_per_document == 15.0

    def test_workflow_run_counts(self, db_session_factory, config):
        """Successful and failed run counts are correct."""
        session = db_session_factory()
        doc = _seed_document(session, status="indexed")
        _seed_workflow_run(session, doc.id, status="success")
        _seed_workflow_run(session, doc.id, status="success")
        _seed_workflow_run(session, doc.id, status="failed")
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        assert metrics.total_workflow_runs == 3
        assert metrics.successful_runs == 2
        assert metrics.failed_runs == 1

    def test_avg_processing_time(self, db_session_factory, config):
        """Average processing time computed from completed runs."""
        session = db_session_factory()
        doc = _seed_document(session, status="indexed")
        _seed_workflow_run(session, doc.id, status="success", duration_seconds=4.0)
        _seed_workflow_run(session, doc.id, status="success", duration_seconds=6.0)
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        # Average of ~4 and ~6 seconds
        assert 4.0 <= metrics.avg_processing_time_seconds <= 6.5

    def test_stage_metrics_aggregation(self, db_session_factory, config):
        """Per-stage metrics aggregate correctly across multiple runs."""
        session = db_session_factory()
        doc = _seed_document(session, status="indexed")
        run1 = _seed_workflow_run(session, doc.id, status="success")
        run2 = _seed_workflow_run(session, doc.id, status="success")

        _seed_stage(session, run1.id, "parse", status="success", duration_seconds=2.0)
        _seed_stage(session, run2.id, "parse", status="success", duration_seconds=4.0)
        _seed_stage(session, run1.id, "embed", status="success", duration_seconds=1.0)
        _seed_stage(session, run2.id, "embed", status="failed", duration_seconds=1.0)
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        stage_map = {s.stage_name: s for s in metrics.stage_metrics}

        assert "parse" in stage_map
        assert stage_map["parse"].total_runs == 2
        assert stage_map["parse"].successful == 2
        assert stage_map["parse"].failed == 0

        assert "embed" in stage_map
        assert stage_map["embed"].total_runs == 2
        assert stage_map["embed"].successful == 1
        assert stage_map["embed"].failed == 1

    def test_documents_last_7_days(self, db_session_factory, config):
        """Only documents created in the last 7 days are counted."""
        session = db_session_factory()
        _seed_document(session, status="indexed", days_ago=1)
        _seed_document(session, status="indexed", days_ago=3)
        _seed_document(session, status="indexed", days_ago=10)
        session.commit()
        session.close()

        with patch("app.metrics.collector.MetricsCollector._count_chromadb_collections", return_value=0):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        assert metrics.documents_last_7_days == 2

    def test_print_metrics_no_error(self, db_session_factory, config, capsys):
        """print_metrics runs without errors and produces output."""
        metrics = PipelineMetrics(
            total_documents=5,
            documents_by_status={"indexed": 3, "failed": 2},
            total_chunks=100,
            total_tokens=1000,
            avg_chunks_per_document=33.3,
            total_workflow_runs=5,
            successful_runs=3,
            failed_runs=2,
            avg_processing_time_seconds=4.5,
            stage_metrics=[
                StageMetrics(
                    stage_name="parse",
                    total_runs=5,
                    successful=4,
                    failed=1,
                    avg_duration_seconds=1.2,
                ),
            ],
            documents_last_7_days=3,
            chromadb_collections=2,
        )

        collector = MetricsCollector(db_session_factory, config)
        collector.print_metrics(metrics)

        output = capsys.readouterr().out
        assert "KnowledgeForge Metrics" in output
        assert "Documents: 5" in output
        assert "Successful: 3" in output
        assert "parse" in output

    def test_chromadb_collections_count(self, db_session_factory, config):
        """ChromaDB collection count is included in metrics."""
        mock_client = MagicMock()
        mock_client.list_collections.return_value = ["coll1", "coll2", "coll3"]

        with patch("chromadb.PersistentClient", return_value=mock_client):
            collector = MetricsCollector(db_session_factory, config)
            metrics = collector.collect()

        assert metrics.chromadb_collections == 3
