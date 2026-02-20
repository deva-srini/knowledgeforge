"""FastAPI application with lifespan managing the full pipeline."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from knowledgeforge root (one level above backend/)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

from app.api.v1 import documents, health, metrics, workflows
from app.core.config import load_config
from app.observability.tracing import setup_tracing
from app.db.session import get_engine, get_session_factory, init_db
from app.models.database import Document
from app.services.filewatcher import FileWatcher
from app.services.workflow import WorkflowOrchestrator
from app.services.workflow_registry import WorkflowRegistryManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup: loads config, initialises the database, creates the
    workflow orchestrator, and starts the file watcher. On shutdown:
    stops the file watcher.
    """
    logger.info("Starting KnowledgeForge...")

    # Load config and init DB
    config = load_config()
    setup_tracing(config)
    engine = get_engine(config.database.url)
    init_db(engine)
    session_factory = get_session_factory(engine)

    # Create default orchestrator (for non-workflow usage)
    orchestrator = WorkflowOrchestrator(config, session_factory)

    # Wire file watcher to orchestrator (default/fallback watcher)
    def on_new_document(doc: Document) -> None:
        """Process newly detected documents through the pipeline."""
        orchestrator.process_document(doc)

    watcher = FileWatcher(config, session_factory, on_new_document=on_new_document)
    watcher.start()
    watcher.scan_existing()

    # Try to start workflow registry manager
    registry_manager: Optional[WorkflowRegistryManager] = None
    try:
        from pathlib import Path

        workflows_dir = (
            Path(__file__).resolve().parent.parent.parent / "workflows"
        )
        if workflows_dir.exists():
            registry_manager = WorkflowRegistryManager(
                config, session_factory, str(workflows_dir)
            )
            registry_manager.load_and_start()
            logger.info("Workflow registry manager started")
    except Exception:
        logger.exception("Failed to start workflow registry manager")

    # Store on app.state for route access
    app.state.config = config
    app.state.watcher = watcher
    app.state.orchestrator = orchestrator
    app.state.session_factory = session_factory
    app.state.registry_manager = registry_manager

    yield

    # Shutdown
    if registry_manager is not None:
        registry_manager.stop_all()
    watcher.stop()
    logger.info("Shutting down KnowledgeForge...")


app = FastAPI(
    title="KnowledgeForge",
    description="Knowledge ingestion, processing, and indexing module",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(workflows.router, prefix="/api/v1", tags=["workflows"])
