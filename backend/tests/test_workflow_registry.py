"""Unit tests for the WorkflowRegistryManager service."""

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import KnowledgeForgeConfig, SourceConfig
from app.db.session import init_db
from app.models.database import Document
from app.services.workflow_registry import WorkflowRegistryManager


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dict as YAML to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> Dict[str, Path]:
    """Create temporary directories for workflows, source, staging."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    return {"workflows": wf_dir, "source": source, "staging": staging}


@pytest.fixture
def base_config(tmp_dirs: Dict[str, Path]) -> KnowledgeForgeConfig:
    """Create a base config pointing to temp directories."""
    return KnowledgeForgeConfig(
        source=SourceConfig(
            watch_folder=str(tmp_dirs["source"]),
            staging_folder=str(tmp_dirs["staging"]),
            file_patterns=["*.pdf"],
        )
    )


@pytest.fixture
def db_session_factory(tmp_path: Path) -> sessionmaker[Session]:
    """Create an in-memory SQLite database and return a session factory."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return sessionmaker(bind=engine)


def _setup_registry(
    tmp_dirs: Dict[str, Path],
    entries: list[Dict[str, Any]],
    workflow_yamls: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Create registry.yaml and workflow YAML files."""
    _write_yaml(
        tmp_dirs["workflows"] / "registry.yaml",
        {"workflows": entries},
    )
    if workflow_yamls:
        for filename, data in workflow_yamls.items():
            _write_yaml(tmp_dirs["workflows"] / filename, data)


class TestWorkflowRegistryManager:
    """Tests for the WorkflowRegistryManager."""

    def test_load_and_start_single_workflow(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """Loading a single active workflow starts its watcher."""
        source_dir = tmp_dirs["source"] / "factsheets"
        source_dir.mkdir()

        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "test_wf",
                "config": "test_wf.yaml",
                "active": True,
            }],
            workflow_yamls={
                "test_wf.yaml": {
                    "source": {
                        "watch_folder": str(source_dir),
                    },
                    "indexing": {"default_collection": "test_coll"},
                },
            },
        )

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,  # Disable auto-sync for tests
        )
        manager.load_and_start()

        try:
            workflows = manager.list_workflows()
            assert len(workflows) == 1
            assert workflows[0]["name"] == "test_wf"
            assert workflows[0]["active"] is True
            assert workflows[0]["collection"] == "test_coll"
        finally:
            manager.stop_all()

    def test_load_and_start_no_workflows(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """Empty registry starts no workflows."""
        _setup_registry(tmp_dirs, entries=[])

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()

        try:
            assert manager.list_workflows() == []
        finally:
            manager.stop_all()

    def test_stop_all(self, tmp_dirs, base_config, db_session_factory):
        """stop_all clears all instances."""
        source_dir = tmp_dirs["source"] / "wf1"
        source_dir.mkdir()

        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "wf1",
                "config": "wf1.yaml",
                "active": True,
            }],
            workflow_yamls={
                "wf1.yaml": {
                    "source": {"watch_folder": str(source_dir)},
                },
            },
        )

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()
        assert len(manager.list_workflows()) == 1

        manager.stop_all()
        assert manager.list_workflows() == []

    def test_sync_activates_new_workflow(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """sync() detects a newly activated workflow and starts it."""
        source_dir = tmp_dirs["source"] / "new_wf"
        source_dir.mkdir()

        # Start with empty registry
        _setup_registry(tmp_dirs, entries=[])

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()
        assert len(manager.list_workflows()) == 0

        # Update registry to add a workflow
        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "new_wf",
                "config": "new_wf.yaml",
                "active": True,
            }],
            workflow_yamls={
                "new_wf.yaml": {
                    "source": {"watch_folder": str(source_dir)},
                },
            },
        )

        manager.sync()
        try:
            assert len(manager.list_workflows()) == 1
            assert manager.list_workflows()[0]["name"] == "new_wf"
        finally:
            manager.stop_all()

    def test_sync_deactivates_workflow(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """sync() detects a deactivated workflow and stops it."""
        source_dir = tmp_dirs["source"] / "wf_to_deactivate"
        source_dir.mkdir()

        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "wf_to_deactivate",
                "config": "wf.yaml",
                "active": True,
            }],
            workflow_yamls={
                "wf.yaml": {
                    "source": {"watch_folder": str(source_dir)},
                },
            },
        )

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()
        assert len(manager.list_workflows()) == 1

        # Deactivate the workflow
        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "wf_to_deactivate",
                "config": "wf.yaml",
                "active": False,
            }],
        )

        manager.sync()
        try:
            assert len(manager.list_workflows()) == 0
        finally:
            manager.stop_all()

    def test_list_workflows_includes_metadata(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """list_workflows returns watch_folder, collection, force_rerun."""
        source_dir = tmp_dirs["source"] / "meta_wf"
        source_dir.mkdir()

        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "meta_wf",
                "config": "meta_wf.yaml",
                "active": True,
            }],
            workflow_yamls={
                "meta_wf.yaml": {
                    "source": {"watch_folder": str(source_dir)},
                    "indexing": {"default_collection": "my_collection"},
                    "force_rerun": True,
                },
            },
        )

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()

        try:
            wfs = manager.list_workflows()
            assert len(wfs) == 1
            wf = wfs[0]
            assert wf["name"] == "meta_wf"
            assert wf["collection"] == "my_collection"
            assert wf["force_rerun"] is True
            assert str(source_dir) in wf["watch_folder"]
        finally:
            manager.stop_all()

    def test_inactive_workflows_not_started(
        self, tmp_dirs, base_config, db_session_factory
    ):
        """Inactive workflows in the registry are not started."""
        _setup_registry(
            tmp_dirs,
            entries=[{
                "name": "inactive",
                "config": "inactive.yaml",
                "active": False,
            }],
            workflow_yamls={"inactive.yaml": {}},
        )

        manager = WorkflowRegistryManager(
            base_config,
            db_session_factory,
            workflows_dir=str(tmp_dirs["workflows"]),
            sync_interval=0,
        )
        manager.load_and_start()

        try:
            assert manager.list_workflows() == []
        finally:
            manager.stop_all()
