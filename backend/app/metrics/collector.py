"""Metrics collection and aggregation for KnowledgeForge pipeline.

Queries the metadata database to compute document, workflow, and stage-level
statistics. Also inspects ChromaDB for collection counts.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List

from sqlalchemy.orm import Session

from app.core.config import KnowledgeForgeConfig
from app.models.database import Document, WorkflowRun, WorkflowStage

logger = logging.getLogger(__name__)


def _ensure_naive(dt: datetime) -> datetime:
    """Strip timezone info for safe comparison with SQLite-stored datetimes.

    Args:
        dt: A datetime that may or may not be timezone-aware.

    Returns:
        A timezone-naive datetime (UTC assumed).
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


@dataclass
class StageMetrics:
    """Aggregated metrics for a single pipeline stage.

    Attributes:
        stage_name: Name of the pipeline stage.
        total_runs: Total number of stage executions.
        successful: Count of successful executions.
        failed: Count of failed executions.
        avg_duration_seconds: Average duration across completed executions.
    """

    stage_name: str
    total_runs: int
    successful: int
    failed: int
    avg_duration_seconds: float


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire processing pipeline.

    Attributes:
        total_documents: Total documents in the system.
        documents_by_status: Document counts grouped by status.
        total_chunks: Sum of chunks across successful runs.
        total_tokens: Sum of tokens across successful runs.
        avg_chunks_per_document: Average chunks per indexed document.
        total_workflow_runs: Total workflow run count.
        successful_runs: Count of successful runs.
        failed_runs: Count of failed runs.
        avg_processing_time_seconds: Average duration of completed runs.
        stage_metrics: Per-stage aggregated metrics.
        documents_last_7_days: Documents created in the last 7 days.
        chromadb_collections: Number of ChromaDB collections.
    """

    total_documents: int
    documents_by_status: Dict[str, int]
    total_chunks: int
    total_tokens: int
    avg_chunks_per_document: float
    total_workflow_runs: int
    successful_runs: int
    failed_runs: int
    avg_processing_time_seconds: float
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    documents_last_7_days: int = 0
    chromadb_collections: int = 0


class MetricsCollector:
    """Collects and aggregates pipeline metrics from the database and ChromaDB.

    Args:
        session_factory: Callable that returns a new SQLAlchemy session.
        config: KnowledgeForge configuration object.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        config: KnowledgeForgeConfig,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session.
            config: KnowledgeForge configuration object.
        """
        self.session_factory = session_factory
        self.config = config

    def collect(self) -> PipelineMetrics:
        """Collect and aggregate all pipeline metrics.

        Opens a database session, queries documents, workflow runs, and
        stages, then aggregates the results into a PipelineMetrics object.

        Returns:
            A PipelineMetrics dataclass with all aggregated metrics.
        """
        session = self.session_factory()
        try:
            return self._collect_from_session(session)
        finally:
            session.close()

    def _collect_from_session(self, session: Session) -> PipelineMetrics:
        """Aggregate metrics within an open session.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            A PipelineMetrics dataclass with all aggregated metrics.
        """
        # Document metrics
        docs = session.query(Document).all()
        total_documents = len(docs)

        documents_by_status: Dict[str, int] = {}
        for doc in docs:
            documents_by_status[doc.status] = documents_by_status.get(doc.status, 0) + 1

        indexed_count = documents_by_status.get("indexed", 0)

        # Documents in last 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        cutoff_naive = cutoff.replace(tzinfo=None)
        documents_last_7_days = sum(
            1 for doc in docs
            if doc.created_at is not None and _ensure_naive(doc.created_at) >= cutoff_naive
        )

        # Workflow run metrics
        runs = session.query(WorkflowRun).all()
        total_workflow_runs = len(runs)
        successful_runs_list = [r for r in runs if r.status == "success"]
        failed_runs_list = [r for r in runs if r.status == "failed"]

        successful_runs = len(successful_runs_list)
        failed_runs = len(failed_runs_list)

        total_chunks = sum(r.total_chunks for r in successful_runs_list)
        total_tokens = sum(r.total_tokens for r in successful_runs_list)

        avg_chunks_per_document = (
            total_chunks / indexed_count if indexed_count > 0 else 0.0
        )

        # Average processing time for completed runs
        durations: List[float] = []
        for r in runs:
            if r.completed_at is not None and r.started_at is not None:
                duration = (r.completed_at - r.started_at).total_seconds()
                durations.append(duration)

        avg_processing_time_seconds = (
            sum(durations) / len(durations) if durations else 0.0
        )

        # Stage metrics
        stages = session.query(WorkflowStage).all()
        stage_metrics = self._aggregate_stage_metrics(stages)

        # ChromaDB collection count
        chromadb_collections = self._count_chromadb_collections()

        return PipelineMetrics(
            total_documents=total_documents,
            documents_by_status=documents_by_status,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            avg_chunks_per_document=avg_chunks_per_document,
            total_workflow_runs=total_workflow_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            avg_processing_time_seconds=avg_processing_time_seconds,
            stage_metrics=stage_metrics,
            documents_last_7_days=documents_last_7_days,
            chromadb_collections=chromadb_collections,
        )

    def _aggregate_stage_metrics(
        self, stages: List[WorkflowStage]
    ) -> List[StageMetrics]:
        """Group workflow stages by name and compute aggregated metrics.

        Args:
            stages: All WorkflowStage records from the database.

        Returns:
            A list of StageMetrics, one per unique stage name.
        """
        stage_data: Dict[str, Dict[str, List[float] | int]] = {}

        for stage in stages:
            name = stage.stage_name
            if name not in stage_data:
                stage_data[name] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "durations": [],
                }

            entry = stage_data[name]
            entry["total"] = int(entry["total"]) + 1

            if stage.status == "success":
                entry["successful"] = int(entry["successful"]) + 1
            elif stage.status == "failed":
                entry["failed"] = int(entry["failed"]) + 1

            if stage.completed_at is not None and stage.started_at is not None:
                duration = (stage.completed_at - stage.started_at).total_seconds()
                durations_list: List[float] = entry["durations"]  # type: ignore[assignment]
                durations_list.append(duration)

        result: List[StageMetrics] = []
        for name, data in stage_data.items():
            durations_list = data["durations"]  # type: ignore[assignment]
            avg_dur = (
                sum(durations_list) / len(durations_list)
                if durations_list
                else 0.0
            )
            result.append(
                StageMetrics(
                    stage_name=name,
                    total_runs=int(data["total"]),
                    successful=int(data["successful"]),
                    failed=int(data["failed"]),
                    avg_duration_seconds=avg_dur,
                )
            )

        return result

    def _count_chromadb_collections(self) -> int:
        """Count the number of collections in ChromaDB.

        Returns:
            Number of collections, or 0 if ChromaDB is unavailable.
        """
        try:
            import chromadb

            client = chromadb.PersistentClient(
                path=self.config.indexing.chromadb_path
            )
            return len(client.list_collections())
        except Exception:
            logger.warning("Failed to count ChromaDB collections", exc_info=True)
            return 0

    def print_metrics(self, metrics: PipelineMetrics) -> None:
        """Print formatted metrics to stdout.

        Args:
            metrics: The PipelineMetrics to display.
        """
        print("\n=== KnowledgeForge Metrics ===")

        print(f"\nDocuments: {metrics.total_documents}")
        for status, count in sorted(metrics.documents_by_status.items()):
            print(f"  {status.capitalize():<12} {count}")
        print(f"  Last 7 days: {metrics.documents_last_7_days}")

        print(f"\nWorkflow Runs: {metrics.total_workflow_runs}")
        print(f"  Successful: {metrics.successful_runs}")
        print(f"  Failed:     {metrics.failed_runs}")
        print(f"  Total Chunks: {metrics.total_chunks}")
        print(f"  Total Tokens: {metrics.total_tokens}")
        print(f"  Avg Chunks/Doc: {metrics.avg_chunks_per_document:.1f}")
        print(f"  Avg Processing Time: {metrics.avg_processing_time_seconds:.1f}s")

        if metrics.stage_metrics:
            print("\nStage Metrics:")
            for sm in metrics.stage_metrics:
                print(
                    f"  {sm.stage_name:<12} "
                    f"runs={sm.total_runs} "
                    f"ok={sm.successful} "
                    f"fail={sm.failed} "
                    f"avg={sm.avg_duration_seconds:.2f}s"
                )

        print(f"\nChromaDB Collections: {metrics.chromadb_collections}")
