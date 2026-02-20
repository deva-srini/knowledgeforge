"""Pydantic request/response schemas for KnowledgeForge API."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Response schema for a document."""

    id: str
    file_name: str
    file_path: str
    file_type: str
    version: int
    file_hash: str
    workflow_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkflowStageResponse(BaseModel):
    """Response schema for a workflow stage."""

    id: str
    run_id: str
    stage_name: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata_json: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class WorkflowRunResponse(BaseModel):
    """Response schema for a workflow run."""

    id: str
    document_id: str
    workflow_id: Optional[str] = None
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_chunks: int = 0
    total_tokens: int = 0
    stages: List[WorkflowStageResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class DocumentDetailResponse(BaseModel):
    """Response schema for a document with workflow details."""

    id: str
    file_name: str
    file_path: str
    file_type: str
    version: int
    file_hash: str
    workflow_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    workflow_runs: List[WorkflowRunResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""

    documents: List[DocumentResponse]
    total: int


class ProcessRequest(BaseModel):
    """Request schema for triggering document processing."""

    file_path: Optional[str] = Field(
        default=None,
        description="Specific file path to process. If None, process all pending.",
    )
    force: bool = Field(
        default=False,
        description="Force re-processing even if already indexed.",
    )


class StageMetricsResponse(BaseModel):
    """Response schema for per-stage metrics."""

    stage_name: str
    total_runs: int
    successful: int
    failed: int
    avg_duration_seconds: float


class MetricsResponse(BaseModel):
    """Response schema for pipeline-wide metrics."""

    total_documents: int
    documents_by_status: Dict[str, int]
    total_chunks: int
    total_tokens: int
    avg_chunks_per_document: float
    total_workflow_runs: int
    successful_runs: int
    failed_runs: int
    avg_processing_time_seconds: float
    stage_metrics: List[StageMetricsResponse]
    documents_last_7_days: int
    chromadb_collections: int


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    status: str
    version: str
    watcher_active: bool
