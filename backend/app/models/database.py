"""SQLAlchemy ORM models for KnowledgeForge metadata database."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _generate_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Document(Base):
    """Represents an ingested document tracked in the system."""

    __tablename__ = "documents"

    id: str = Column(String, primary_key=True, default=_generate_uuid)
    file_name: str = Column(String, nullable=False)
    file_path: str = Column(String, nullable=False)
    file_type: str = Column(String, nullable=False)
    version: int = Column(Integer, nullable=False, default=1)
    file_hash: str = Column(String, nullable=False)
    workflow_id: str = Column(String, nullable=True, default=None)
    status: str = Column(String, nullable=False, default="pending")
    created_at: datetime = Column(DateTime, nullable=False, default=_utcnow)
    updated_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    workflow_runs = relationship(
        "WorkflowRun", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """Return string representation of the document."""
        return (
            f"<Document(id={self.id!r}, file_name={self.file_name!r}, "
            f"version={self.version}, status={self.status!r})>"
        )


class WorkflowRun(Base):
    """Represents a single processing run for a document."""

    __tablename__ = "workflow_runs"

    id: str = Column(String, primary_key=True, default=_generate_uuid)
    document_id: str = Column(
        String, ForeignKey("documents.id"), nullable=False
    )
    workflow_id: str = Column(String, nullable=True, default=None)
    status: str = Column(String, nullable=False, default="pending")
    started_at: datetime = Column(DateTime, nullable=False, default=_utcnow)
    completed_at: datetime = Column(DateTime, nullable=True)
    error_message: str = Column(Text, nullable=True)
    total_chunks: int = Column(Integer, nullable=False, default=0)
    total_tokens: int = Column(Integer, nullable=False, default=0)

    document = relationship("Document", back_populates="workflow_runs")
    stages = relationship(
        "WorkflowStage", back_populates="workflow_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """Return string representation of the workflow run."""
        return (
            f"<WorkflowRun(id={self.id!r}, document_id={self.document_id!r}, "
            f"status={self.status!r})>"
        )


class WorkflowStage(Base):
    """Represents a single stage within a workflow run."""

    __tablename__ = "workflow_stages"

    id: str = Column(String, primary_key=True, default=_generate_uuid)
    run_id: str = Column(
        String, ForeignKey("workflow_runs.id"), nullable=False
    )
    stage_name: str = Column(String, nullable=False)
    status: str = Column(String, nullable=False, default="pending")
    started_at: datetime = Column(DateTime, nullable=True)
    completed_at: datetime = Column(DateTime, nullable=True)
    metadata_json: str = Column(Text, nullable=True)
    error_message: str = Column(Text, nullable=True)

    workflow_run = relationship("WorkflowRun", back_populates="stages")

    def __repr__(self) -> str:
        """Return string representation of the workflow stage."""
        return (
            f"<WorkflowStage(id={self.id!r}, stage_name={self.stage_name!r}, "
            f"status={self.status!r})>"
        )
