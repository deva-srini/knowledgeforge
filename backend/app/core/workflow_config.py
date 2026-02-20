"""Workflow configuration models, registry, and overlay loader for KnowledgeForge.

Supports per-workflow configurations that override the global kf_config.yaml.
Each workflow YAML defines only the fields it wants to change; everything else
is inherited from the base KnowledgeForgeConfig.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from app.core.config import (
    KnowledgeForgeConfig,
    IndexingConfig,
    ProcessingConfig,
    SourceConfig,
)

logger = logging.getLogger(__name__)

# Default path to the workflows directory (sibling of kf_config.yaml)
_WORKFLOWS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "workflows"
)


class StageToggle(BaseModel):
    """Toggle for an individual pipeline stage."""

    enabled: bool = Field(default=True, description="Whether this stage is enabled")


class StagesConfig(BaseModel):
    """Configuration for enabling/disabling individual pipeline stages."""

    parse: StageToggle = Field(default_factory=StageToggle)
    extract: StageToggle = Field(default_factory=StageToggle)
    transform: StageToggle = Field(default_factory=StageToggle)
    chunk: StageToggle = Field(default_factory=StageToggle)
    embed: StageToggle = Field(default_factory=StageToggle)
    index: StageToggle = Field(default_factory=StageToggle)

    def is_enabled(self, stage_name: str) -> bool:
        """Check whether a given stage is enabled.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            True if the stage is enabled, False otherwise.

        Raises:
            ValueError: If the stage name is not recognised.
        """
        toggle = getattr(self, stage_name, None)
        if toggle is None:
            raise ValueError(f"Unknown stage name: {stage_name!r}")
        return toggle.enabled


class WorkflowDefinition(BaseModel):
    """Raw workflow YAML content with optional override sections."""

    source: Optional[Dict[str, Any]] = Field(
        default=None, description="Source config overrides"
    )
    processing: Optional[Dict[str, Any]] = Field(
        default=None, description="Processing config overrides"
    )
    indexing: Optional[Dict[str, Any]] = Field(
        default=None, description="Indexing config overrides"
    )
    stages: Optional[Dict[str, Any]] = Field(
        default=None, description="Per-stage enable/disable toggles"
    )
    force_rerun: bool = Field(
        default=False, description="Whether to force re-processing on every run"
    )


class ResolvedWorkflowConfig(BaseModel):
    """Fully merged workflow configuration inheriting from the base config."""

    name: str = Field(description="Workflow name")
    source: SourceConfig = Field(default_factory=SourceConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    force_rerun: bool = Field(default=False)

    # Carry the full base config for services that need database/observability
    base_config: Optional[KnowledgeForgeConfig] = Field(
        default=None, description="Original base config for database/observability access"
    )

    model_config = {"arbitrary_types_allowed": True}


class RegistryEntry(BaseModel):
    """A single workflow entry in the registry."""

    name: str = Field(description="Unique workflow name")
    config: str = Field(description="Filename of the workflow YAML")
    active: bool = Field(default=True, description="Whether this workflow is active")
    description: str = Field(default="", description="Human-readable description")


class WorkflowRegistry(BaseModel):
    """Top-level registry listing all workflows."""

    workflows: List[RegistryEntry] = Field(default_factory=list)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary (not mutated).
        override: Override dictionary whose values take precedence.

    Returns:
        A new merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_registry(
    workflows_dir: Optional[str] = None,
) -> WorkflowRegistry:
    """Load the workflow registry from registry.yaml.

    Args:
        workflows_dir: Path to the workflows directory.
            Defaults to ``knowledgeforge/workflows/``.

    Returns:
        A validated WorkflowRegistry.

    Raises:
        FileNotFoundError: If the registry file does not exist.
    """
    wf_dir = Path(workflows_dir) if workflows_dir else _WORKFLOWS_DIR
    registry_path = wf_dir / "registry.yaml"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with open(registry_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return WorkflowRegistry(**raw)


def load_workflow(
    name: str,
    base_config: KnowledgeForgeConfig,
    workflows_dir: Optional[str] = None,
) -> ResolvedWorkflowConfig:
    """Load a single workflow definition and merge with the base config.

    Args:
        name: Workflow name (must match a registry entry).
        base_config: The global KnowledgeForgeConfig to use as the base.
        workflows_dir: Path to the workflows directory.

    Returns:
        A ResolvedWorkflowConfig with overrides applied.

    Raises:
        FileNotFoundError: If the workflow YAML does not exist.
        ValueError: If the workflow name is not found in the registry.
    """
    wf_dir = Path(workflows_dir) if workflows_dir else _WORKFLOWS_DIR
    registry = load_registry(str(wf_dir))

    # Find the entry
    entry: Optional[RegistryEntry] = None
    for e in registry.workflows:
        if e.name == name:
            entry = e
            break

    if entry is None:
        raise ValueError(
            f"Workflow {name!r} not found in registry. "
            f"Available: {[e.name for e in registry.workflows]}"
        )

    workflow_path = wf_dir / entry.config
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {workflow_path}")

    with open(workflow_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    definition = WorkflowDefinition(**raw)

    return _resolve_workflow(name, definition, base_config)


def _resolve_workflow(
    name: str,
    definition: WorkflowDefinition,
    base_config: KnowledgeForgeConfig,
) -> ResolvedWorkflowConfig:
    """Merge a workflow definition with the base config.

    Args:
        name: Workflow name.
        definition: The parsed workflow definition.
        base_config: The global base config.

    Returns:
        A fully resolved workflow configuration.
    """
    # Deep-merge source
    base_source = base_config.source.model_dump()
    merged_source = (
        _deep_merge(base_source, definition.source)
        if definition.source
        else base_source
    )

    # Deep-merge processing
    base_processing = base_config.processing.model_dump()
    merged_processing = (
        _deep_merge(base_processing, definition.processing)
        if definition.processing
        else base_processing
    )

    # Deep-merge indexing
    base_indexing = base_config.indexing.model_dump()
    merged_indexing = (
        _deep_merge(base_indexing, definition.indexing)
        if definition.indexing
        else base_indexing
    )

    # Build stages config
    stages = StagesConfig(**(definition.stages or {}))

    return ResolvedWorkflowConfig(
        name=name,
        source=SourceConfig(**merged_source),
        processing=ProcessingConfig(**merged_processing),
        indexing=IndexingConfig(**merged_indexing),
        stages=stages,
        force_rerun=definition.force_rerun,
        base_config=base_config,
    )


def load_all_active_workflows(
    base_config: KnowledgeForgeConfig,
    workflows_dir: Optional[str] = None,
) -> List[ResolvedWorkflowConfig]:
    """Load all active workflows from the registry.

    Args:
        base_config: The global KnowledgeForgeConfig.
        workflows_dir: Path to the workflows directory.

    Returns:
        List of resolved configurations for all active workflows.
    """
    wf_dir = Path(workflows_dir) if workflows_dir else _WORKFLOWS_DIR
    registry = load_registry(str(wf_dir))

    workflows: List[ResolvedWorkflowConfig] = []
    for entry in registry.workflows:
        if not entry.active:
            logger.info("Skipping inactive workflow: %s", entry.name)
            continue

        workflow_path = wf_dir / entry.config
        if not workflow_path.exists():
            logger.warning(
                "Workflow config not found for '%s': %s", entry.name, workflow_path
            )
            continue

        with open(workflow_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        definition = WorkflowDefinition(**raw)
        resolved = _resolve_workflow(entry.name, definition, base_config)
        workflows.append(resolved)

    logger.info("Loaded %d active workflow(s)", len(workflows))
    return workflows
