"""Health check endpoint for KnowledgeForge."""

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status including version and watcher state."""
    watcher_active = False
    watcher = getattr(request.app.state, "watcher", None)
    if watcher is not None:
        watcher_active = getattr(watcher, "is_running", False)

    return HealthResponse(
        status="ok",
        version=request.app.version,
        watcher_active=watcher_active,
    )
