"""Tests for the embedding generator service."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.chunking import Chunk
from app.services.embedding import (
    EmbeddedChunk,
    Embedder,
    EmbedResult,
    SentenceTransformerClient,
    _PROVIDER_REGISTRY,
)
from app.services.extraction import ContentType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "test content",
    chunk_index: int = 0,
    header_path: str = "Section 1",
    page_number: int = 1,
) -> Chunk:
    """Create a Chunk for testing."""
    return Chunk(
        content=content,
        content_type=ContentType.TEXT,
        chunk_index=chunk_index,
        header_path=header_path,
        page_number=page_number,
        token_count=len(content.split()),
        metadata={"source": "test"},
    )


def _fake_embedding(dim: int = 384) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.1] * dim


def _make_mock_client() -> MagicMock:
    """Create a mock EmbeddingClient with default behaviour."""
    mock_client = MagicMock()
    mock_client.embed_batch.return_value = [_fake_embedding()]
    mock_client.model_name.return_value = "test-model"
    mock_client.dimension.return_value = 384
    return mock_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Default config for embedding tests."""
    return KnowledgeForgeConfig()


@pytest.fixture
def small_batch_config() -> KnowledgeForgeConfig:
    """Config with batch_size=2 for testing batching."""
    return KnowledgeForgeConfig(
        processing={"embedding": {"batch_size": 2}}
    )


@pytest.fixture
def mock_client() -> MagicMock:
    """A mock EmbeddingClient."""
    return _make_mock_client()


@pytest.fixture
def embedder_with_mock(
    config: KnowledgeForgeConfig, mock_client: MagicMock
) -> Embedder:
    """Embedder with _create_client patched to return a mock."""
    with patch.object(Embedder, "_create_client", return_value=mock_client):
        return Embedder(config)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestEmbeddedChunkDataclass:
    """Tests for EmbeddedChunk construction and defaults."""

    def test_embedded_chunk_dataclass(self) -> None:
        """EmbeddedChunk stores chunk, embedding, and metadata."""
        chunk = _make_chunk()
        embedding = _fake_embedding()
        ec = EmbeddedChunk(chunk=chunk, embedding=embedding)

        assert ec.chunk is chunk
        assert ec.embedding == embedding
        assert ec.metadata == {}

    def test_embedded_chunk_with_metadata(self) -> None:
        """EmbeddedChunk accepts custom metadata."""
        chunk = _make_chunk()
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=_fake_embedding(),
            metadata={"model": "test-model"},
        )
        assert ec.metadata == {"model": "test-model"}


class TestEmbedResultDataclass:
    """Tests for EmbedResult construction and stats."""

    def test_embed_result_dataclass(self) -> None:
        """EmbedResult tracks embedded chunks, skipped count, and total."""
        chunk = _make_chunk()
        ec = EmbeddedChunk(chunk=chunk, embedding=_fake_embedding())
        result = EmbedResult(
            embedded_chunks=[ec],
            skipped_count=1,
            total_chunks=2,
        )

        assert len(result.embedded_chunks) == 1
        assert result.skipped_count == 1
        assert result.total_chunks == 2


# ---------------------------------------------------------------------------
# SentenceTransformerClient tests
# ---------------------------------------------------------------------------


class TestSentenceTransformerClient:
    """Tests for SentenceTransformerClient."""

    def test_lazy_model_loading(self) -> None:
        """Model is not loaded at init, only on first embed call."""
        client = SentenceTransformerClient(
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Model should not be loaded yet.
        assert client._model is None

    @patch("app.services.embedding.SentenceTransformerClient._load_model")
    def test_embed_triggers_load(self, mock_load: MagicMock) -> None:
        """First embed_batch call triggers model loading."""
        client = SentenceTransformerClient(
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Set up mock model after _load_model is called.
        def side_effect() -> None:
            mock_model = MagicMock()
            mock_model.encode.return_value = [
                MagicMock(tolist=lambda: _fake_embedding())
            ]
            client._model = mock_model
            client._dimension = 384

        mock_load.side_effect = side_effect
        client.embed_batch(["hello"])
        mock_load.assert_called_once()

    def test_model_name(self) -> None:
        """model_name returns the configured model identifier."""
        client = SentenceTransformerClient(model_id="my-model")
        assert client.model_name() == "my-model"


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


class TestEmbedder:
    """Tests for the Embedder orchestrator."""

    def test_embedder_creates_sentence_transformer_client(
        self, config: KnowledgeForgeConfig
    ) -> None:
        """Factory creates SentenceTransformerClient for default config."""
        mock_cls = MagicMock()
        mock_cls.return_value = _make_mock_client()
        with patch.dict(
            _PROVIDER_REGISTRY, {"sentence_transformers": mock_cls}
        ):
            embedder = Embedder(config)
            mock_cls.assert_called_once_with(
                model_id="sentence-transformers/all-MiniLM-L6-v2"
            )
            assert embedder._client is mock_cls.return_value

    def test_embed_single_chunk(
        self, embedder_with_mock: Embedder, mock_client: MagicMock
    ) -> None:
        """Single chunk is embedded with correct output shape."""
        mock_client.embed_batch.return_value = [_fake_embedding()]

        chunk = _make_chunk(content="hello world")
        result = embedder_with_mock.embed([chunk])

        assert result.total_chunks == 1
        assert result.skipped_count == 0
        assert len(result.embedded_chunks) == 1
        assert result.embedded_chunks[0].chunk is chunk
        assert len(result.embedded_chunks[0].embedding) == 384

    def test_embed_multiple_batches(
        self, small_batch_config: KnowledgeForgeConfig
    ) -> None:
        """Chunks are split into batches of configured size."""
        mock_client = _make_mock_client()
        mock_client.embed_batch.side_effect = [
            [_fake_embedding(), _fake_embedding()],  # Batch 1
            [_fake_embedding()],                      # Batch 2
        ]
        with patch.object(
            Embedder, "_create_client", return_value=mock_client
        ):
            embedder = Embedder(small_batch_config)

        chunks = [_make_chunk(content=f"chunk {i}") for i in range(3)]
        result = embedder.embed(chunks)

        assert result.total_chunks == 3
        assert len(result.embedded_chunks) == 3
        assert result.skipped_count == 0
        assert mock_client.embed_batch.call_count == 2

    def test_batch_failure_falls_back_to_individual(
        self, embedder_with_mock: Embedder, mock_client: MagicMock
    ) -> None:
        """Batch error triggers individual chunk retry."""
        mock_client.embed_batch.side_effect = [
            RuntimeError("batch failed"),
            [_fake_embedding()],
            [_fake_embedding()],
        ]

        chunks = [_make_chunk(content=f"chunk {i}") for i in range(2)]
        result = embedder_with_mock.embed(chunks)

        assert result.total_chunks == 2
        assert len(result.embedded_chunks) == 2
        assert result.skipped_count == 0

    def test_individual_failure_skipped(
        self, embedder_with_mock: Embedder, mock_client: MagicMock
    ) -> None:
        """Failed individual chunk is logged and skipped."""
        mock_client.embed_batch.side_effect = [
            RuntimeError("batch failed"),
            [_fake_embedding()],
            RuntimeError("chunk failed"),
        ]

        chunks = [_make_chunk(content=f"chunk {i}") for i in range(2)]
        result = embedder_with_mock.embed(chunks)

        assert result.total_chunks == 2
        assert len(result.embedded_chunks) == 1
        assert result.skipped_count == 1

    def test_empty_input_returns_empty(
        self, embedder_with_mock: Embedder
    ) -> None:
        """Empty chunk list returns empty EmbedResult."""
        result = embedder_with_mock.embed([])

        assert result.total_chunks == 0
        assert len(result.embedded_chunks) == 0
        assert result.skipped_count == 0

    def test_chunk_metadata_preserved(
        self, embedder_with_mock: Embedder, mock_client: MagicMock
    ) -> None:
        """Original chunk data is intact in EmbeddedChunk."""
        mock_client.embed_batch.return_value = [_fake_embedding()]

        chunk = _make_chunk(
            content="important text",
            chunk_index=5,
            header_path="Chapter 1 > Section 2",
            page_number=3,
        )
        result = embedder_with_mock.embed([chunk])

        ec = result.embedded_chunks[0]
        assert ec.chunk.content == "important text"
        assert ec.chunk.chunk_index == 5
        assert ec.chunk.header_path == "Chapter 1 > Section 2"
        assert ec.chunk.page_number == 3
        assert ec.chunk.metadata == {"source": "test"}
        assert ec.metadata["model"] == "test-model"
        assert ec.metadata["dimension"] == 384
