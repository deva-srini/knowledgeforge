"""Metrics API endpoint for KnowledgeForge."""

from dataclasses import asdict

from fastapi import APIRouter, Request

from app.metrics.collector import MetricsCollector
from app.models.schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """Return aggregated pipeline metrics.

    Collects document, workflow, stage, and ChromaDB metrics via
    the MetricsCollector and returns them as a structured response.
    """
    collector = MetricsCollector(
        session_factory=request.app.state.session_factory,
        config=request.app.state.config,
    )
    metrics = collector.collect()
    return MetricsResponse(**asdict(metrics))
