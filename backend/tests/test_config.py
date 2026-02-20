"""Unit tests for KnowledgeForge configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from app.core.config import (
    KnowledgeForgeConfig,
    load_config,
)


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Create a temporary valid config file."""
    config_data = {
        "source": {
            "watch_folder": "./test/source",
            "staging_folder": "./test/staging",
            "file_patterns": ["*.pdf", "*.docx"],
        },
        "processing": {
            "parsing": {"library": "docling"},
            "extraction": {"strategy": "direct"},
            "organisation": {"enabled": False, "table_format": "markdown"},
            "chunking": {
                "strategy": "structure_aware",
                "chunk_size_tokens": 256,
                "chunk_overlap_tokens": 25,
                "skip_threshold_tokens": 500,
            },
            "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "indexing": {
            "vector_store": "chromadb",
            "chromadb_path": "./test/chromadb",
            "default_collection": "test_collection",
            "collection_mapping": {"docs/*.pdf": "docs_index"},
        },
        "database": {"url": "sqlite:///./test/test.db"},
        "observability": {
            "langsmith_enabled": True,
            "langsmith_project": "test_project",
        },
    }
    config_file = tmp_path / "kf_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@pytest.fixture
def minimal_config_file(tmp_path: Path) -> Path:
    """Create a config file with minimal/no overrides to test defaults."""
    config_file = tmp_path / "kf_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump({}, f)
    return config_file


@pytest.fixture
def empty_config_file(tmp_path: Path) -> Path:
    """Create a completely empty YAML config file."""
    config_file = tmp_path / "kf_config.yaml"
    config_file.write_text("")
    return config_file


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, valid_config_file: Path) -> None:
        """Test loading a fully specified config file."""
        config = load_config(str(valid_config_file))

        assert config.source.watch_folder == "./test/source"
        assert config.source.staging_folder == "./test/staging"
        assert config.source.file_patterns == ["*.pdf", "*.docx"]
        assert config.processing.extraction.strategy == "direct"
        assert config.processing.organisation.enabled is False
        assert config.processing.chunking.chunk_size_tokens == 256
        assert config.processing.chunking.chunk_overlap_tokens == 25
        assert config.indexing.default_collection == "test_collection"
        assert config.indexing.collection_mapping == {"docs/*.pdf": "docs_index"}
        assert config.database.url == "sqlite:///./test/test.db"
        assert config.observability.langsmith_enabled is True
        assert config.observability.langsmith_project == "test_project"

    def test_load_minimal_config_uses_defaults(
        self, minimal_config_file: Path
    ) -> None:
        """Test that missing fields fall back to defaults."""
        config = load_config(str(minimal_config_file))

        assert config.source.watch_folder == "./data/source"
        assert config.source.staging_folder == "./data/staging"
        assert len(config.source.file_patterns) == 5
        assert config.processing.parsing.library == "docling"
        assert config.processing.extraction.strategy == "auto"
        assert config.processing.organisation.enabled is True
        assert config.processing.organisation.markdown_generation is True
        assert config.processing.chunking.chunk_size_tokens == 512
        assert config.processing.chunking.chunk_overlap_tokens == 50
        assert config.processing.chunking.skip_threshold_tokens == 1000
        assert config.processing.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.indexing.vector_store == "chromadb"
        assert config.indexing.default_collection == "default"
        assert config.indexing.collection_mapping == {}
        assert config.database.url == "sqlite:///./data/knowledgeforge.db"
        assert config.observability.langsmith_enabled is False

    def test_load_empty_config_uses_defaults(self, empty_config_file: Path) -> None:
        """Test that an empty YAML file produces all defaults."""
        config = load_config(str(empty_config_file))
        assert isinstance(config, KnowledgeForgeConfig)
        assert config.source.watch_folder == "./data/source"

    def test_load_config_file_not_found(self) -> None:
        """Test that a missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_custom_path(self, valid_config_file: Path) -> None:
        """Test loading config from a custom path."""
        config = load_config(str(valid_config_file))
        assert config.source.watch_folder == "./test/source"

    def test_partial_config_merges_with_defaults(self, tmp_path: Path) -> None:
        """Test that a partial config merges correctly with defaults."""
        partial_config = {
            "source": {"watch_folder": "/custom/path"},
            "processing": {
                "chunking": {"chunk_size_tokens": 1024},
            },
        }
        config_file = tmp_path / "partial.yaml"
        with open(config_file, "w") as f:
            yaml.dump(partial_config, f)

        config = load_config(str(config_file))

        # Overridden values
        assert config.source.watch_folder == "/custom/path"
        assert config.processing.chunking.chunk_size_tokens == 1024

        # Defaults preserved
        assert config.source.staging_folder == "./data/staging"
        assert config.processing.extraction.strategy == "auto"
        assert config.indexing.default_collection == "default"


class TestConfigValidation:
    """Tests for config validation rules."""

    def test_invalid_extraction_strategy(self, tmp_path: Path) -> None:
        """Test that an invalid extraction strategy raises ValueError."""
        bad_config = {
            "processing": {"extraction": {"strategy": "invalid_strategy"}}
        }
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="Invalid extraction strategy"):
            load_config(str(config_file))

    def test_negative_chunk_size_rejected(self, tmp_path: Path) -> None:
        """Test that a negative chunk_size_tokens is rejected."""
        bad_config = {
            "processing": {"chunking": {"chunk_size_tokens": -1}}
        }
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError):
            load_config(str(config_file))

    def test_overlap_exceeding_chunk_size_rejected(self, tmp_path: Path) -> None:
        """Test that overlap >= chunk_size is rejected."""
        bad_config = {
            "processing": {
                "chunking": {
                    "chunk_size_tokens": 100,
                    "chunk_overlap_tokens": 100,
                }
            }
        }
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="chunk_overlap_tokens"):
            load_config(str(config_file))

    def test_markdown_generation_override(self, tmp_path: Path) -> None:
        """Test that markdown_generation can be set to False via YAML."""
        config_data = {
            "processing": {
                "organisation": {"markdown_generation": False}
            }
        }
        config_file = tmp_path / "md_off.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.processing.organisation.markdown_generation is False
        # Other defaults still hold
        assert config.processing.organisation.enabled is True
        assert config.processing.organisation.table_format == "markdown"

    def test_valid_extraction_strategies(self, tmp_path: Path) -> None:
        """Test that all valid extraction strategies are accepted."""
        for strategy in ("auto", "ocr", "direct", "agentic"):
            config_data = {
                "processing": {"extraction": {"strategy": strategy}}
            }
            config_file = tmp_path / f"{strategy}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            config = load_config(str(config_file))
            assert config.processing.extraction.strategy == strategy


class TestKnowledgeForgeConfig:
    """Tests for the KnowledgeForgeConfig model directly."""

    def test_default_construction(self) -> None:
        """Test creating config with all defaults."""
        config = KnowledgeForgeConfig()
        assert config.source.watch_folder == "./data/source"
        assert config.processing.chunking.chunk_size_tokens == 512
        assert config.indexing.vector_store == "chromadb"

    def test_model_serialization(self) -> None:
        """Test that config can be serialized to dict and back."""
        config = KnowledgeForgeConfig()
        config_dict = config.model_dump()
        restored = KnowledgeForgeConfig(**config_dict)
        assert config == restored

    def test_extraction_picture_defaults(self) -> None:
        """Test that picture-related extraction fields default to safe off values."""
        config = KnowledgeForgeConfig()
        ext = config.processing.extraction
        assert ext.save_picture_images is False
        assert ext.picture_images_dir == "./data/images"
        assert ext.describe_pictures is False
        assert ext.vision_model == "claude-haiku-4-5-20251001"
        assert ext.vision_api_key_env == "ANTHROPIC_API_KEY"

    def test_extraction_picture_flags_can_be_enabled(self, tmp_path: Path) -> None:
        """Test that save_picture_images and describe_pictures can be enabled."""
        config_data = {
            "processing": {
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": "/tmp/test_images",
                    "describe_pictures": True,
                    "vision_model": "claude-opus-4-6",
                    "vision_api_key_env": "MY_API_KEY",
                }
            }
        }
        config_file = tmp_path / "pic_flags.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        ext = config.processing.extraction
        assert ext.save_picture_images is True
        assert ext.picture_images_dir == "/tmp/test_images"
        assert ext.describe_pictures is True
        assert ext.vision_model == "claude-opus-4-6"
        assert ext.vision_api_key_env == "MY_API_KEY"

    def test_extraction_defaults_preserved_with_partial_picture_config(
        self, tmp_path: Path
    ) -> None:
        """Test that unspecified picture fields keep their defaults."""
        config_data = {
            "processing": {
                "extraction": {
                    "save_picture_images": True,
                }
            }
        }
        config_file = tmp_path / "partial_pic.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        ext = config.processing.extraction
        assert ext.save_picture_images is True
        # Other fields keep defaults
        assert ext.picture_images_dir == "./data/images"
        assert ext.describe_pictures is False
        assert ext.vision_model == "claude-haiku-4-5-20251001"
