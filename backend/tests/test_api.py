"""Unit tests for KnowledgeForge API endpoints."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import documents, health
from app.models.database import Base, Document, WorkflowRun, WorkflowStage


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with routers but no lifespan."""
    test_app = FastAPI(title="KnowledgeForge", version="0.1.0")
    test_app.include_router(health.router, prefix="/api/v1", tags=["health"])
    test_app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
    return test_app


@pytest.fixture()
def db_session_factory():
    """Create an in-memory SQLite database with tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return factory


@pytest.fixture()
def mock_orchestrator():
    """Create a mock workflow orchestrator."""
    return MagicMock()


@pytest.fixture()
def mock_watcher():
    """Create a mock file watcher with is_running=True."""
    return SimpleNamespace(is_running=True)


@pytest.fixture()
def client(db_session_factory, mock_orchestrator, mock_watcher):
    """Create a FastAPI TestClient with mocked app state."""
    test_app = _create_test_app()
    test_app.state.session_factory = db_session_factory
    test_app.state.orchestrator = mock_orchestrator
    test_app.state.watcher = mock_watcher
    return TestClient(test_app, raise_server_exceptions=True)


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns status ok."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_includes_version_and_watcher(self, client: TestClient) -> None:
        """Health endpoint includes version string and watcher_active flag."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "0.1.0"
        assert data["watcher_active"] is True


class TestListDocuments:
    """Tests for the GET /api/v1/documents endpoint."""

    def test_list_documents_empty(self, client: TestClient) -> None:
        """Returns empty list when no documents exist."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    def test_list_documents_with_data(
        self, client: TestClient, db_session_factory: sessionmaker
    ) -> None:
        """Returns documents when they exist in the database."""
        session = db_session_factory()
        doc = Document(
            id="doc-1",
            file_name="test.pdf",
            file_path="/tmp/test.pdf",
            file_type="pdf",
            file_hash="abc123",
            status="indexed",
        )
        session.add(doc)
        session.commit()
        session.close()

        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["file_name"] == "test.pdf"


class TestDocumentStatus:
    """Tests for the GET /api/v1/documents/{doc_id}/status endpoint."""

    def test_document_status_found(
        self, client: TestClient, db_session_factory: sessionmaker
    ) -> None:
        """Returns document detail when document exists."""
        session = db_session_factory()
        doc = Document(
            id="doc-status-1",
            file_name="report.pdf",
            file_path="/tmp/report.pdf",
            file_type="pdf",
            file_hash="def456",
            status="indexed",
        )
        session.add(doc)
        session.commit()
        session.close()

        response = client.get("/api/v1/documents/doc-status-1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "doc-status-1"
        assert data["file_name"] == "report.pdf"

    def test_document_status_not_found(self, client: TestClient) -> None:
        """Returns 404 for non-existent document."""
        response = client.get("/api/v1/documents/nonexistent/status")
        assert response.status_code == 404

    def test_document_status_includes_stages(
        self, client: TestClient, db_session_factory: sessionmaker
    ) -> None:
        """Returns workflow runs with nested stage data."""
        session = db_session_factory()
        now = datetime.now(timezone.utc)

        doc = Document(
            id="doc-stages-1",
            file_name="data.pdf",
            file_path="/tmp/data.pdf",
            file_type="pdf",
            file_hash="ghi789",
            status="indexed",
        )
        session.add(doc)
        session.flush()

        run = WorkflowRun(
            id="run-1",
            document_id="doc-stages-1",
            status="success",
            started_at=now,
            completed_at=now,
            total_chunks=10,
            total_tokens=500,
        )
        session.add(run)
        session.flush()

        stage = WorkflowStage(
            id="stage-1",
            run_id="run-1",
            stage_name="parse",
            status="success",
            started_at=now,
            completed_at=now,
        )
        session.add(stage)
        session.commit()
        session.close()

        response = client.get("/api/v1/documents/doc-stages-1/status")
        assert response.status_code == 200
        data = response.json()
        assert len(data["workflow_runs"]) == 1
        assert data["workflow_runs"][0]["total_chunks"] == 10
        assert len(data["workflow_runs"][0]["stages"]) == 1
        assert data["workflow_runs"][0]["stages"][0]["stage_name"] == "parse"


class TestProcessDocuments:
    """Tests for the POST /api/v1/documents/process endpoint."""

    def test_process_documents_pending(
        self,
        client: TestClient,
        db_session_factory: sessionmaker,
        mock_orchestrator: MagicMock,
    ) -> None:
        """Processes all pending documents when no file_path provided."""
        session = db_session_factory()
        for i in range(3):
            doc = Document(
                id=f"pending-{i}",
                file_name=f"file{i}.pdf",
                file_path=f"/tmp/file{i}.pdf",
                file_type="pdf",
                file_hash=f"hash{i}",
                status="pending",
            )
            session.add(doc)
        session.commit()
        session.close()

        response = client.post("/api/v1/documents/process")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Processing started"
        assert data["document_count"] == 3

    def test_process_specific_file(
        self, client: TestClient, mock_orchestrator: MagicMock
    ) -> None:
        """Creates document record and schedules processing for a specific file."""
        response = client.post(
            "/api/v1/documents/process",
            json={"file_path": "/tmp/new_doc.pdf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Processing started"
        assert data["document_count"] == 1


class TestReprocessDocument:
    """Tests for the POST /api/v1/documents/{doc_id}/reprocess endpoint."""

    def test_reprocess_document_found(
        self,
        client: TestClient,
        db_session_factory: sessionmaker,
        mock_orchestrator: MagicMock,
    ) -> None:
        """Schedules reprocessing for an existing document."""
        session = db_session_factory()
        doc = Document(
            id="reprocess-1",
            file_name="old.pdf",
            file_path="/tmp/old.pdf",
            file_type="pdf",
            file_hash="oldhash",
            status="indexed",
        )
        session.add(doc)
        session.commit()
        session.close()

        response = client.post("/api/v1/documents/reprocess-1/reprocess")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Reprocessing started"
        assert data["document_id"] == "reprocess-1"

    def test_reprocess_document_not_found(self, client: TestClient) -> None:
        """Returns 404 for non-existent document."""
        response = client.post("/api/v1/documents/nonexistent/reprocess")
        assert response.status_code == 404
