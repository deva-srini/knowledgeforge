"""Unit tests for SQLAlchemy ORM models and session management."""

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import Base, Document, WorkflowRun, WorkflowStage
from app.db.session import init_db


@pytest.fixture
def db_session() -> Session:
    """Create an in-memory SQLite database and return a session."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_document(db_session: Session) -> Document:
    """Create and return a sample document in the database."""
    doc = Document(
        file_name="test.pdf",
        file_path="/staging/test.pdf",
        file_type="pdf",
        version=1,
        file_hash="abc123hash",
        status="pending",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def sample_workflow_run(
    db_session: Session, sample_document: Document
) -> WorkflowRun:
    """Create and return a sample workflow run for the sample document."""
    run = WorkflowRun(
        document_id=sample_document.id,
        status="pending",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run


class TestInitDb:
    """Tests for database initialization."""

    def test_creates_all_tables(self) -> None:
        """Test that init_db creates all three expected tables."""
        engine = create_engine("sqlite:///:memory:")
        init_db(engine)

        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        assert "documents" in table_names
        assert "workflow_runs" in table_names
        assert "workflow_stages" in table_names
        engine.dispose()

    def test_idempotent_init(self) -> None:
        """Test that calling init_db twice does not raise errors."""
        engine = create_engine("sqlite:///:memory:")
        init_db(engine)
        init_db(engine)

        inspector = inspect(engine)
        assert len(inspector.get_table_names()) == 3
        engine.dispose()


class TestDocumentModel:
    """Tests for the Document ORM model."""

    def test_create_document(self, db_session: Session) -> None:
        """Test creating a document with required fields."""
        doc = Document(
            file_name="report.pdf",
            file_path="/staging/report.pdf",
            file_type="pdf",
            file_hash="sha256hash",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.id is not None
        assert doc.version == 1
        assert doc.status == "pending"
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_query_document(self, db_session: Session, sample_document: Document) -> None:
        """Test querying a document by ID."""
        result = db_session.query(Document).filter_by(id=sample_document.id).first()

        assert result is not None
        assert result.file_name == "test.pdf"
        assert result.file_type == "pdf"
        assert result.file_hash == "abc123hash"

    def test_update_document_status(
        self, db_session: Session, sample_document: Document
    ) -> None:
        """Test updating a document's status."""
        sample_document.status = "indexed"
        db_session.commit()
        db_session.refresh(sample_document)

        assert sample_document.status == "indexed"

    def test_document_versioning(self, db_session: Session) -> None:
        """Test creating multiple versions of the same document."""
        doc_v1 = Document(
            file_name="guide.docx",
            file_path="/staging/guide_v1.docx",
            file_type="docx",
            version=1,
            file_hash="hash_v1",
        )
        doc_v2 = Document(
            file_name="guide.docx",
            file_path="/staging/guide_v2.docx",
            file_type="docx",
            version=2,
            file_hash="hash_v2",
        )
        db_session.add_all([doc_v1, doc_v2])
        db_session.commit()

        results = (
            db_session.query(Document)
            .filter_by(file_name="guide.docx")
            .order_by(Document.version)
            .all()
        )
        assert len(results) == 2
        assert results[0].version == 1
        assert results[1].version == 2

    def test_document_repr(self, sample_document: Document) -> None:
        """Test the string representation of a document."""
        repr_str = repr(sample_document)
        assert "Document" in repr_str
        assert "test.pdf" in repr_str


class TestWorkflowRunModel:
    """Tests for the WorkflowRun ORM model."""

    def test_create_workflow_run(
        self, db_session: Session, sample_document: Document
    ) -> None:
        """Test creating a workflow run linked to a document."""
        run = WorkflowRun(
            document_id=sample_document.id,
            status="in_progress",
        )
        db_session.add(run)
        db_session.commit()

        assert run.id is not None
        assert run.document_id == sample_document.id
        assert run.total_chunks == 0
        assert run.total_tokens == 0
        assert run.started_at is not None

    def test_workflow_run_completion(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test updating a workflow run upon completion."""
        sample_workflow_run.status = "success"
        sample_workflow_run.completed_at = datetime.now(timezone.utc)
        sample_workflow_run.total_chunks = 15
        sample_workflow_run.total_tokens = 5000
        db_session.commit()
        db_session.refresh(sample_workflow_run)

        assert sample_workflow_run.status == "success"
        assert sample_workflow_run.completed_at is not None
        assert sample_workflow_run.total_chunks == 15
        assert sample_workflow_run.total_tokens == 5000

    def test_workflow_run_failure(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test recording a failed workflow run."""
        sample_workflow_run.status = "failed"
        sample_workflow_run.error_message = "Parsing failed: corrupt PDF"
        sample_workflow_run.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        assert sample_workflow_run.status == "failed"
        assert "corrupt PDF" in sample_workflow_run.error_message

    def test_workflow_run_document_relationship(
        self, db_session: Session, sample_document: Document, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test the relationship between workflow run and document."""
        assert sample_workflow_run.document.id == sample_document.id
        assert sample_document.workflow_runs[0].id == sample_workflow_run.id


class TestWorkflowStageModel:
    """Tests for the WorkflowStage ORM model."""

    def test_create_all_stages(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test creating all pipeline stages for a run."""
        stage_names = ["parse", "extract", "organise", "chunk", "embed", "index"]
        for name in stage_names:
            stage = WorkflowStage(
                run_id=sample_workflow_run.id,
                stage_name=name,
                status="pending",
            )
            db_session.add(stage)
        db_session.commit()

        stages = (
            db_session.query(WorkflowStage)
            .filter_by(run_id=sample_workflow_run.id)
            .all()
        )
        assert len(stages) == 6
        assert {s.stage_name for s in stages} == set(stage_names)

    def test_stage_lifecycle(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test a stage going through its full lifecycle."""
        stage = WorkflowStage(
            run_id=sample_workflow_run.id,
            stage_name="parse",
            status="pending",
        )
        db_session.add(stage)
        db_session.commit()

        # Start stage
        stage.status = "in_progress"
        stage.started_at = datetime.now(timezone.utc)
        db_session.commit()
        assert stage.status == "in_progress"

        # Complete stage
        stage.status = "success"
        stage.completed_at = datetime.now(timezone.utc)
        stage.metadata_json = json.dumps({"pages": 10, "tokens": 3500})
        db_session.commit()

        assert stage.status == "success"
        assert stage.completed_at is not None
        metadata = json.loads(stage.metadata_json)
        assert metadata["pages"] == 10

    def test_stage_skipped(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test marking a stage as skipped."""
        stage = WorkflowStage(
            run_id=sample_workflow_run.id,
            stage_name="organise",
            status="skipped",
        )
        db_session.add(stage)
        db_session.commit()

        assert stage.status == "skipped"
        assert stage.started_at is None
        assert stage.completed_at is None

    def test_stage_run_relationship(
        self, db_session: Session, sample_workflow_run: WorkflowRun
    ) -> None:
        """Test the relationship between stage and workflow run."""
        stage = WorkflowStage(
            run_id=sample_workflow_run.id,
            stage_name="embed",
            status="pending",
        )
        db_session.add(stage)
        db_session.commit()
        db_session.refresh(sample_workflow_run)

        assert stage.workflow_run.id == sample_workflow_run.id
        assert any(s.id == stage.id for s in sample_workflow_run.stages)


class TestForeignKeyConstraints:
    """Tests for foreign key relationship enforcement."""

    def test_cascade_delete_document(
        self, db_session: Session, sample_document: Document
    ) -> None:
        """Test that deleting a document cascades to runs and stages."""
        run = WorkflowRun(document_id=sample_document.id, status="success")
        db_session.add(run)
        db_session.commit()

        stage = WorkflowStage(
            run_id=run.id, stage_name="parse", status="success"
        )
        db_session.add(stage)
        db_session.commit()

        run_id = run.id

        db_session.delete(sample_document)
        db_session.commit()

        assert db_session.query(Document).count() == 0
        assert db_session.query(WorkflowRun).filter_by(id=run_id).first() is None
        assert db_session.query(WorkflowStage).count() == 0
