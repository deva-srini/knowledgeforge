"""Unit tests for workflow configuration models, registry, and overlay loader."""

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from app.core.config import KnowledgeForgeConfig
from app.core.workflow_config import (
    ResolvedWorkflowConfig,
    RegistryEntry,
    StageToggle,
    StagesConfig,
    WorkflowDefinition,
    WorkflowRegistry,
    _deep_merge,
    load_all_active_workflows,
    load_registry,
    load_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dict as YAML to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def _make_registry(
    tmp_path: Path,
    entries: list[Dict[str, Any]] | None = None,
) -> Path:
    """Create a workflows directory with a registry.yaml and return its path."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(exist_ok=True)
    if entries is None:
        entries = [
            {
                "name": "test_wf",
                "config": "test_wf.yaml",
                "active": True,
                "description": "Test workflow",
            }
        ]
    _write_yaml(wf_dir / "registry.yaml", {"workflows": entries})
    return wf_dir


def _make_workflow_yaml(wf_dir: Path, filename: str, data: Dict[str, Any]) -> None:
    """Write a workflow YAML into the given workflows directory."""
    _write_yaml(wf_dir / filename, data)


# ---------------------------------------------------------------------------
# StageToggle / StagesConfig
# ---------------------------------------------------------------------------

class TestStageToggle:
    """Tests for the StageToggle model."""

    def test_default_enabled(self) -> None:
        """Default StageToggle is enabled."""
        toggle = StageToggle()
        assert toggle.enabled is True

    def test_explicit_disabled(self) -> None:
        """StageToggle can be set to disabled."""
        toggle = StageToggle(enabled=False)
        assert toggle.enabled is False


class TestStagesConfig:
    """Tests for the StagesConfig model."""

    def test_all_enabled_by_default(self) -> None:
        """All stages default to enabled."""
        stages = StagesConfig()
        for name in ("parse", "extract", "transform", "chunk", "embed", "index"):
            assert stages.is_enabled(name) is True

    def test_disable_specific_stage(self) -> None:
        """A single stage can be disabled."""
        stages = StagesConfig(transform=StageToggle(enabled=False))
        assert stages.is_enabled("parse") is True
        assert stages.is_enabled("transform") is False

    def test_is_enabled_unknown_stage(self) -> None:
        """is_enabled raises ValueError for unknown stage names."""
        stages = StagesConfig()
        with pytest.raises(ValueError, match="Unknown stage name"):
            stages.is_enabled("nonexistent")

    def test_from_dict(self) -> None:
        """StagesConfig can be built from a dict (as from YAML)."""
        raw = {
            "parse": {"enabled": True},
            "embed": {"enabled": False},
            "index": {"enabled": False},
        }
        stages = StagesConfig(**raw)
        assert stages.is_enabled("parse") is True
        assert stages.is_enabled("embed") is False
        assert stages.is_enabled("index") is False
        # Unspecified stages default to enabled
        assert stages.is_enabled("chunk") is True


# ---------------------------------------------------------------------------
# WorkflowDefinition
# ---------------------------------------------------------------------------

class TestWorkflowDefinition:
    """Tests for the WorkflowDefinition model."""

    def test_empty_definition(self) -> None:
        """An empty definition has all None overrides."""
        defn = WorkflowDefinition()
        assert defn.source is None
        assert defn.processing is None
        assert defn.indexing is None
        assert defn.stages is None
        assert defn.force_rerun is False

    def test_with_overrides(self) -> None:
        """WorkflowDefinition accepts override dicts."""
        defn = WorkflowDefinition(
            source={"watch_folder": "/custom"},
            processing={"chunking": {"chunk_size_tokens": 256}},
            force_rerun=True,
        )
        assert defn.source == {"watch_folder": "/custom"}
        assert defn.force_rerun is True


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    """Tests for the _deep_merge helper."""

    def test_simple_override(self) -> None:
        """Override replaces a top-level key."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self) -> None:
        """Nested dicts are merged recursively."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_adds_new_keys(self) -> None:
        """Override can add keys not in the base."""
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_base_not_mutated(self) -> None:
        """The base dict is not mutated."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}

    def test_non_dict_override_replaces(self) -> None:
        """A non-dict value in override replaces a dict in base."""
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        assert _deep_merge(base, override) == {"a": "replaced"}


# ---------------------------------------------------------------------------
# RegistryEntry / WorkflowRegistry
# ---------------------------------------------------------------------------

class TestRegistryModels:
    """Tests for registry Pydantic models."""

    def test_registry_entry_defaults(self) -> None:
        """RegistryEntry has sensible defaults."""
        entry = RegistryEntry(name="wf1", config="wf1.yaml")
        assert entry.active is True
        assert entry.description == ""

    def test_workflow_registry_empty(self) -> None:
        """Empty registry is valid."""
        registry = WorkflowRegistry()
        assert registry.workflows == []

    def test_workflow_registry_with_entries(self) -> None:
        """Registry can hold multiple entries."""
        registry = WorkflowRegistry(
            workflows=[
                RegistryEntry(name="a", config="a.yaml"),
                RegistryEntry(name="b", config="b.yaml", active=False),
            ]
        )
        assert len(registry.workflows) == 2
        assert registry.workflows[1].active is False


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------

class TestLoadRegistry:
    """Tests for the load_registry function."""

    def test_load_valid_registry(self, tmp_path: Path) -> None:
        """Load a valid registry.yaml."""
        wf_dir = _make_registry(tmp_path)
        registry = load_registry(str(wf_dir))
        assert len(registry.workflows) == 1
        assert registry.workflows[0].name == "test_wf"

    def test_load_registry_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError when registry.yaml is missing."""
        with pytest.raises(FileNotFoundError, match="Registry file not found"):
            load_registry(str(tmp_path / "nonexistent"))

    def test_load_empty_registry(self, tmp_path: Path) -> None:
        """Empty YAML produces an empty registry."""
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        (wf_dir / "registry.yaml").write_text("")
        registry = load_registry(str(wf_dir))
        assert registry.workflows == []


# ---------------------------------------------------------------------------
# load_workflow
# ---------------------------------------------------------------------------

class TestLoadWorkflow:
    """Tests for the load_workflow function."""

    def test_load_workflow_basic(self, tmp_path: Path) -> None:
        """Load a workflow and verify overrides are applied."""
        wf_dir = _make_registry(tmp_path)
        _make_workflow_yaml(wf_dir, "test_wf.yaml", {
            "source": {"watch_folder": "/custom/source"},
            "processing": {"chunking": {"chunk_size_tokens": 256}},
            "indexing": {"default_collection": "custom_coll"},
            "stages": {"embed": {"enabled": False}},
            "force_rerun": True,
        })
        base = KnowledgeForgeConfig()

        resolved = load_workflow("test_wf", base, str(wf_dir))

        assert resolved.name == "test_wf"
        assert resolved.source.watch_folder == "/custom/source"
        # Base default preserved for staging_folder
        assert resolved.source.staging_folder == "./data/staging"
        assert resolved.processing.chunking.chunk_size_tokens == 256
        # Base default preserved for embedding
        assert resolved.processing.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert resolved.indexing.default_collection == "custom_coll"
        assert resolved.stages.is_enabled("embed") is False
        assert resolved.stages.is_enabled("parse") is True
        assert resolved.force_rerun is True
        assert resolved.base_config is base

    def test_load_workflow_not_in_registry(self, tmp_path: Path) -> None:
        """ValueError when workflow name not in registry."""
        wf_dir = _make_registry(tmp_path)
        base = KnowledgeForgeConfig()

        with pytest.raises(ValueError, match="not found in registry"):
            load_workflow("missing_wf", base, str(wf_dir))

    def test_load_workflow_config_file_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError when workflow YAML is missing."""
        wf_dir = _make_registry(tmp_path)
        # Don't create test_wf.yaml
        base = KnowledgeForgeConfig()

        with pytest.raises(FileNotFoundError, match="Workflow config not found"):
            load_workflow("test_wf", base, str(wf_dir))

    def test_load_workflow_empty_yaml(self, tmp_path: Path) -> None:
        """Empty workflow YAML inherits all base config values."""
        wf_dir = _make_registry(tmp_path)
        (wf_dir / "test_wf.yaml").write_text("")
        base = KnowledgeForgeConfig()

        resolved = load_workflow("test_wf", base, str(wf_dir))

        assert resolved.source.watch_folder == base.source.watch_folder
        assert resolved.processing.chunking.chunk_size_tokens == base.processing.chunking.chunk_size_tokens
        assert resolved.indexing.default_collection == base.indexing.default_collection
        assert resolved.force_rerun is False

    def test_load_workflow_partial_nested_override(self, tmp_path: Path) -> None:
        """Partial nested override preserves sibling fields."""
        wf_dir = _make_registry(tmp_path)
        _make_workflow_yaml(wf_dir, "test_wf.yaml", {
            "processing": {
                "chunking": {"chunk_size_tokens": 128, "chunk_overlap_tokens": 10},
            },
        })
        base = KnowledgeForgeConfig()

        resolved = load_workflow("test_wf", base, str(wf_dir))

        assert resolved.processing.chunking.chunk_size_tokens == 128
        assert resolved.processing.chunking.chunk_overlap_tokens == 10
        # Sibling field preserved
        assert resolved.processing.chunking.strategy == "structure_aware"
        # Other processing sections preserved
        assert resolved.processing.parsing.library == "docling"


# ---------------------------------------------------------------------------
# load_all_active_workflows
# ---------------------------------------------------------------------------

class TestLoadAllActiveWorkflows:
    """Tests for the load_all_active_workflows function."""

    def test_loads_only_active(self, tmp_path: Path) -> None:
        """Only active workflows are loaded."""
        wf_dir = _make_registry(tmp_path, entries=[
            {"name": "active_wf", "config": "active.yaml", "active": True},
            {"name": "inactive_wf", "config": "inactive.yaml", "active": False},
        ])
        _make_workflow_yaml(wf_dir, "active.yaml", {"force_rerun": True})
        _make_workflow_yaml(wf_dir, "inactive.yaml", {})
        base = KnowledgeForgeConfig()

        workflows = load_all_active_workflows(base, str(wf_dir))

        assert len(workflows) == 1
        assert workflows[0].name == "active_wf"
        assert workflows[0].force_rerun is True

    def test_skips_missing_config(self, tmp_path: Path) -> None:
        """Workflows with missing YAML are skipped (not errored)."""
        wf_dir = _make_registry(tmp_path, entries=[
            {"name": "good", "config": "good.yaml", "active": True},
            {"name": "missing", "config": "missing.yaml", "active": True},
        ])
        _make_workflow_yaml(wf_dir, "good.yaml", {})
        base = KnowledgeForgeConfig()

        workflows = load_all_active_workflows(base, str(wf_dir))

        assert len(workflows) == 1
        assert workflows[0].name == "good"

    def test_empty_registry(self, tmp_path: Path) -> None:
        """Empty registry returns empty list."""
        wf_dir = _make_registry(tmp_path, entries=[])
        base = KnowledgeForgeConfig()

        workflows = load_all_active_workflows(base, str(wf_dir))

        assert workflows == []


# ---------------------------------------------------------------------------
# ResolvedWorkflowConfig
# ---------------------------------------------------------------------------

class TestResolvedWorkflowConfig:
    """Tests for the ResolvedWorkflowConfig model."""

    def test_default_construction(self) -> None:
        """Can be constructed with just a name."""
        resolved = ResolvedWorkflowConfig(name="test")
        assert resolved.name == "test"
        assert resolved.force_rerun is False
        assert resolved.stages.is_enabled("parse") is True

    def test_serialization_roundtrip(self) -> None:
        """Can serialize and deserialize."""
        resolved = ResolvedWorkflowConfig(
            name="test",
            force_rerun=True,
            stages=StagesConfig(embed=StageToggle(enabled=False)),
        )
        data = resolved.model_dump(exclude={"base_config"})
        restored = ResolvedWorkflowConfig(**data)
        assert restored.name == "test"
        assert restored.force_rerun is True
        assert restored.stages.is_enabled("embed") is False
