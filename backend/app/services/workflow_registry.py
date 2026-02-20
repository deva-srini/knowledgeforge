"""Runtime workflow registry manager for KnowledgeForge.

Manages the lifecycle of all active workflows, each with its own
WorkflowOrchestrator and FileWatcher. Supports hot-reload via periodic
sync of the registry.yaml file.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import KnowledgeForgeConfig
from app.core.workflow_config import (
    ResolvedWorkflowConfig,
    load_all_active_workflows,
    load_registry,
)
from app.models.database import Document
from app.services.filewatcher import FileWatcher
from app.services.workflow import WorkflowOrchestrator

logger = logging.getLogger(__name__)

DEFAULT_SYNC_INTERVAL_SECONDS = 30.0


class _WorkflowInstance:
    """Internal state for a single running workflow."""

    def __init__(
        self,
        config: ResolvedWorkflowConfig,
        orchestrator: WorkflowOrchestrator,
        watcher: FileWatcher,
    ) -> None:
        """Initialize a workflow instance.

        Args:
            config: The resolved workflow configuration.
            orchestrator: The workflow orchestrator.
            watcher: The file watcher for this workflow.
        """
        self.config = config
        self.orchestrator = orchestrator
        self.watcher = watcher


class WorkflowRegistryManager:
    """Manages lifecycle of all active workflows from the registry.

    Creates one WorkflowOrchestrator and one FileWatcher per active workflow.
    Supports hot-reload by periodically re-reading registry.yaml.
    """

    def __init__(
        self,
        base_config: KnowledgeForgeConfig,
        session_factory: Callable[[], Session],
        workflows_dir: Optional[str] = None,
        sync_interval: float = DEFAULT_SYNC_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the registry manager.

        Args:
            base_config: The global KnowledgeForge configuration.
            session_factory: Callable that returns a new SQLAlchemy session.
            workflows_dir: Path to the workflows directory.
            sync_interval: Seconds between automatic registry syncs.
        """
        self.base_config = base_config
        self.session_factory = session_factory
        self.workflows_dir = workflows_dir
        self.sync_interval = sync_interval
        self._instances: Dict[str, _WorkflowInstance] = {}
        self._sync_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def load_and_start(self) -> None:
        """Load all active workflows from the registry and start their watchers."""
        workflows = load_all_active_workflows(
            self.base_config, self.workflows_dir
        )
        for wf_config in workflows:
            self._start_workflow(wf_config)

        # Start background sync timer
        self._schedule_sync()

        logger.info(
            "Registry manager started with %d workflow(s)", len(self._instances)
        )

    def stop_all(self) -> None:
        """Stop all watchers and the sync timer gracefully."""
        # Cancel sync timer
        if self._sync_timer is not None:
            self._sync_timer.cancel()
            self._sync_timer = None

        with self._lock:
            for name, instance in self._instances.items():
                try:
                    instance.watcher.stop()
                    logger.info("Stopped watcher for workflow '%s'", name)
                except Exception:
                    logger.exception("Error stopping watcher for '%s'", name)
            self._instances.clear()

        logger.info("Registry manager stopped")

    def sync(self) -> None:
        """Re-read registry.yaml and diff against current state.

        - Newly activated workflows: start them.
        - Newly deactivated workflows: stop them.
        - Already active and unchanged: no-op.
        """
        try:
            registry = load_registry(self.workflows_dir)
        except FileNotFoundError:
            logger.warning("Registry file not found during sync")
            return

        active_names = {
            e.name for e in registry.workflows if e.active
        }
        current_names = set(self._instances.keys())

        # Stop deactivated workflows
        to_stop = current_names - active_names
        for name in to_stop:
            self._stop_workflow(name)

        # Start newly activated workflows
        to_start = active_names - current_names
        if to_start:
            all_workflows = load_all_active_workflows(
                self.base_config, self.workflows_dir
            )
            for wf_config in all_workflows:
                if wf_config.name in to_start:
                    self._start_workflow(wf_config)

        if to_stop or to_start:
            logger.info(
                "Sync complete: started=%d, stopped=%d, total=%d",
                len(to_start),
                len(to_stop),
                len(self._instances),
            )

    def process_file(
        self, workflow_name: str, file_path: str, force: bool = False
    ) -> Optional[Document]:
        """Manually trigger processing for a specific workflow.

        Args:
            workflow_name: Name of the workflow to use.
            file_path: Path to the file to process.
            force: Whether to force re-processing.

        Returns:
            The Document record, or None if the workflow is not found.
        """
        with self._lock:
            instance = self._instances.get(workflow_name)

        if instance is None:
            logger.error("Workflow '%s' not found or not active", workflow_name)
            return None

        # Use the watcher handler to process the file
        instance.watcher._handler._process_file(file_path)
        return None

    def list_workflows(self) -> List[Dict[str, object]]:
        """Return status of all registered workflows.

        Returns:
            List of dicts with workflow name, active, watch_folder, collection.
        """
        with self._lock:
            result: List[Dict[str, object]] = []
            for name, instance in self._instances.items():
                result.append({
                    "name": name,
                    "active": True,
                    "watch_folder": instance.config.source.watch_folder,
                    "collection": instance.config.indexing.default_collection,
                    "force_rerun": instance.config.force_rerun,
                })
        return result

    def _start_workflow(self, wf_config: ResolvedWorkflowConfig) -> None:
        """Create orchestrator and watcher for a workflow, then start watching.

        Args:
            wf_config: The resolved workflow configuration.
        """
        with self._lock:
            if wf_config.name in self._instances:
                logger.warning(
                    "Workflow '%s' already running, skipping", wf_config.name
                )
                return

            # Build a full KnowledgeForgeConfig for the services
            svc_config = KnowledgeForgeConfig(
                source=wf_config.source,
                processing=wf_config.processing,
                indexing=wf_config.indexing,
                database=self.base_config.database,
                observability=self.base_config.observability,
            )

            orchestrator = WorkflowOrchestrator(
                svc_config,
                self.session_factory,
                workflow_config=wf_config,
            )

            def on_new_document(doc: Document) -> None:
                """Process newly detected documents through the pipeline."""
                orchestrator.process_document(doc)

            watcher = FileWatcher(
                svc_config,
                self.session_factory,
                on_new_document=on_new_document,
                workflow_name=wf_config.name,
                force_rerun=wf_config.force_rerun,
            )

            watcher.start()
            watcher.scan_existing()

            self._instances[wf_config.name] = _WorkflowInstance(
                config=wf_config,
                orchestrator=orchestrator,
                watcher=watcher,
            )

        logger.info(
            "Started workflow '%s' watching '%s'",
            wf_config.name,
            wf_config.source.watch_folder,
        )

    def _stop_workflow(self, name: str) -> None:
        """Stop and remove a workflow by name.

        Args:
            name: The workflow name to stop.
        """
        with self._lock:
            instance = self._instances.pop(name, None)

        if instance is not None:
            instance.watcher.stop()
            logger.info("Stopped workflow '%s'", name)

    def _schedule_sync(self) -> None:
        """Schedule the next background sync."""
        if self.sync_interval <= 0:
            return

        def _sync_and_reschedule() -> None:
            try:
                self.sync()
            except Exception:
                logger.exception("Error during registry sync")
            finally:
                self._schedule_sync()

        self._sync_timer = threading.Timer(
            self.sync_interval, _sync_and_reschedule
        )
        self._sync_timer.daemon = True
        self._sync_timer.start()
