"""Configuration loader and Pydantic validation for KnowledgeForge."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class SourceConfig(BaseModel):
    """Configuration for document source and staging folders."""

    watch_folder: str = Field(
        default="./data/source",
        description="Folder to watch for new documents",
    )
    staging_folder: str = Field(
        default="./data/staging",
        description="Internal staging folder for processing",
    )
    file_patterns: List[str] = Field(
        default=["*.pdf", "*.docx", "*.xlsx", "*.html", "*.pptx"],
        description="File glob patterns to watch for",
    )


class ParsingConfig(BaseModel):
    """Configuration for document parsing."""

    library: str = Field(
        default="docling",
        description="Parsing library to use",
    )
    pipeline: Literal["standard", "vlm"] = Field(
        default="standard",
        description="'standard' uses TableFormer PDF pipeline; 'vlm' uses a VLM model",
    )
    vlm_model: str = Field(
        default="granite_docling",
        description="VLM preset when pipeline='vlm'. Options: granite_docling, smoldocling, deepseek_ocr",
    )
    generate_page_images: bool = Field(
        default=False,
        description="Render full-page images during PDF parsing (required for page-level image export)",
    )
    generate_picture_images: bool = Field(
        default=False,
        description="Crop and store individual picture images during PDF parsing (enables get_image() on PictureItem)",
    )

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v: str) -> str:
        """Validate pipeline is a known value."""
        valid = {"standard", "vlm"}
        if v not in valid:
            raise ValueError(f"pipeline must be one of {valid}, got '{v}'")
        return v


class ExtractionConfig(BaseModel):
    """Configuration for content extraction strategy."""

    strategy: str = Field(
        default="auto",
        description="Extraction strategy: auto, ocr, direct, or agentic",
    )
    save_picture_images: bool = Field(
        default=False,
        description=(
            "Save extracted picture images to disk. Requires "
            "processing.parsing.generate_picture_images=true."
        ),
    )
    picture_images_dir: str = Field(
        default="./data/images",
        description="Directory where extracted picture PNG files are saved.",
    )
    describe_pictures: bool = Field(
        default=False,
        description=(
            "Call a Claude Vision model to generate rich text descriptions "
            "for each picture at index time. Requires save_picture_images=true "
            "and a valid Anthropic API key in vision_api_key_env."
        ),
    )
    vision_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model ID used for picture description (Option B).",
    )
    vision_api_key_env: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Environment variable name holding the Anthropic API key.",
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate extraction strategy is a known value."""
        valid_strategies = {"auto", "ocr", "direct", "agentic"}
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid extraction strategy '{v}'. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )
        return v


class OrganisationConfig(BaseModel):
    """Configuration for content organisation/transformation."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable content transformation",
    )
    table_format: str = Field(
        default="markdown",
        description="Format to convert tables into",
    )
    markdown_generation: bool = Field(
        default=True,
        description="Whether to generate structured markdown from the raw document",
    )


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    strategy: str = Field(
        default="structure_aware",
        description="Chunking strategy to use",
    )
    chunk_size_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum chunk size in tokens",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive chunks",
    )
    skip_threshold_tokens: int = Field(
        default=1000,
        gt=0,
        description="Skip chunking if document is below this token count",
    )

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        """Validate that overlap does not exceed chunk size."""
        chunk_size = info.data.get("chunk_size_tokens", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be less than "
                f"chunk_size_tokens ({chunk_size})"
            )
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: str = Field(
        default="sentence_transformers",
        description="Embedding provider: sentence_transformers, openai, or cohere",
    )
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Number of chunks to embed per batch",
    )
    api_key_env: str = Field(
        default="",
        description="Environment variable name for API key (e.g. COHERE_API_KEY)",
    )


class ProcessingConfig(BaseModel):
    """Configuration for the full processing pipeline."""

    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    organisation: OrganisationConfig = Field(default_factory=OrganisationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class IndexingConfig(BaseModel):
    """Configuration for vector store indexing."""

    vector_store: str = Field(
        default="chromadb",
        description="Vector store backend to use",
    )
    chromadb_path: str = Field(
        default="./data/chromadb",
        description="Path for persistent ChromaDB storage",
    )
    default_collection: str = Field(
        default="default",
        description="Default ChromaDB collection name",
    )
    collection_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of file path patterns to collection names",
    )


class DatabaseConfig(BaseModel):
    """Configuration for the metadata database."""

    url: str = Field(
        default="sqlite:///./data/knowledgeforge.db",
        description="SQLAlchemy database connection URL",
    )


class ObservabilityConfig(BaseModel):
    """Configuration for observability and tracing."""

    langsmith_enabled: bool = Field(
        default=False,
        description="Whether to enable LangSmith tracing",
    )
    langsmith_project: str = Field(
        default="knowledgeforge",
        description="LangSmith project name",
    )


class KnowledgeForgeConfig(BaseModel):
    """Root configuration for KnowledgeForge."""

    source: SourceConfig = Field(default_factory=SourceConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def load_config(config_path: Optional[str] = None) -> KnowledgeForgeConfig:
    """Load and validate KnowledgeForge configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, searches for
            kf_config.yaml in the project root (two levels up from this file).

    Returns:
        A validated KnowledgeForgeConfig object with all settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file contains invalid values.
    """
    if config_path is None:
        # Default: knowledgeforge/kf_config.yaml (project root relative to backend/)
        default_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "kf_config.yaml"
        )
        config_path = str(default_path)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    return KnowledgeForgeConfig(**raw_config)
