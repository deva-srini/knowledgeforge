"""Tests for the ChromaDB indexer service."""

from unittest.mock import patch

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.chunking import Chunk
from app.services.embedding import EmbeddedChunk
from app.services.extraction import ContentType
from app.services.indexing import ChromaIndexer, IndexResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "test content",
    chunk_index: int = 0,
    header_path: str = "Section 1",
    page_number: int = 1,
    content_type: ContentType = ContentType.TEXT,
    token_count: int = 10,
) -> Chunk:
    """Create a Chunk for testing."""
    return Chunk(
        content=content,
        content_type=content_type,
        chunk_index=chunk_index,
        header_path=header_path,
        page_number=page_number,
        token_count=token_count,
        metadata={},
    )


def _make_embedded_chunk(
    content: str = "test content",
    chunk_index: int = 0,
    header_path: str = "Section 1",
    page_number: int = 1,
    content_type: ContentType = ContentType.TEXT,
    token_count: int = 10,
    dim: int = 8,
) -> EmbeddedChunk:
    """Create an EmbeddedChunk with a fake embedding vector."""
    chunk = _make_chunk(
        content=content,
        chunk_index=chunk_index,
        header_path=header_path,
        page_number=page_number,
        content_type=content_type,
        token_count=token_count,
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=[0.1 * (i + 1) for i in range(dim)],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Default config for indexing tests."""
    return KnowledgeForgeConfig()


@pytest.fixture
def mapped_config() -> KnowledgeForgeConfig:
    """Config with collection_mapping for pattern tests."""
    return KnowledgeForgeConfig(
        indexing={
            "default_collection": "general",
            "collection_mapping": {
                "*.pdf": "pdfs",
                "*.docx": "docs",
                "reports/*": "reports",
            },
        }
    )


@pytest.fixture
def indexer(
    config: KnowledgeForgeConfig, tmp_path: "pytest.TempPathFactory"
) -> ChromaIndexer:
    """ChromaIndexer using PersistentClient in a temp directory."""
    config = KnowledgeForgeConfig(
        indexing={"chromadb_path": str(tmp_path / "chromadb")}
    )
    return ChromaIndexer(config)


@pytest.fixture
def mapped_indexer(
    mapped_config: KnowledgeForgeConfig, tmp_path: "pytest.TempPathFactory"
) -> ChromaIndexer:
    """ChromaIndexer with collection mapping in a temp directory."""
    mapped_config = KnowledgeForgeConfig(
        indexing={
            "chromadb_path": str(tmp_path / "chromadb"),
            "default_collection": "general",
            "collection_mapping": {
                "*.pdf": "pdfs",
                "*.docx": "docs",
                "reports/*": "reports",
            },
        }
    )
    return ChromaIndexer(mapped_config)


# ---------------------------------------------------------------------------
# IndexResult dataclass tests
# ---------------------------------------------------------------------------


class TestIndexResultDataclass:
    """Tests for IndexResult construction and defaults."""

    def test_index_result_dataclass(self) -> None:
        """IndexResult stores collection name, IDs, and count."""
        result = IndexResult(
            collection_name="test_collection",
            indexed_ids=["doc1_0", "doc1_1"],
            total_indexed=2,
        )
        assert result.collection_name == "test_collection"
        assert result.indexed_ids == ["doc1_0", "doc1_1"]
        assert result.total_indexed == 2

    def test_index_result_defaults(self) -> None:
        """IndexResult defaults to empty list and zero count."""
        result = IndexResult(collection_name="default")
        assert result.indexed_ids == []
        assert result.total_indexed == 0


# ---------------------------------------------------------------------------
# Collection resolution tests
# ---------------------------------------------------------------------------


class TestResolveCollection:
    """Tests for collection name resolution via file path patterns."""

    def test_resolve_collection_default(self, indexer: ChromaIndexer) -> None:
        """No mapping configured returns default collection."""
        assert indexer._resolve_collection("anything.pdf") == "default"

    def test_resolve_collection_pattern_match(
        self, mapped_indexer: ChromaIndexer
    ) -> None:
        """Pattern match returns the mapped collection name."""
        assert mapped_indexer._resolve_collection("report.pdf") == "pdfs"
        assert mapped_indexer._resolve_collection("notes.docx") == "docs"

    def test_resolve_collection_first_match_wins(
        self, mapped_indexer: ChromaIndexer
    ) -> None:
        """When multiple patterns could match, first match wins."""
        # "reports/q1.pdf" matches both "*.pdf" and "reports/*".
        # Dict ordering: "*.pdf" comes first → "pdfs" wins.
        result = mapped_indexer._resolve_collection("reports/q1.pdf")
        assert result == "pdfs"

    def test_resolve_collection_fallback(
        self, mapped_indexer: ChromaIndexer
    ) -> None:
        """Unmatched file path falls back to default collection."""
        assert mapped_indexer._resolve_collection("data.csv") == "general"


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------


class TestIndex:
    """Tests for the index method."""

    def test_index_single_chunk(self, indexer: ChromaIndexer) -> None:
        """Single chunk is indexed with correct ID returned."""
        ec = _make_embedded_chunk(content="hello world", chunk_index=0)
        result = indexer.index(
            embedded_chunks=[ec],
            document_id="doc1",
            file_name="test.pdf",
            version=1,
            file_path="test.pdf",
        )

        assert result.total_indexed == 1
        assert result.indexed_ids == ["doc1_0"]
        assert result.collection_name == "default"

    def test_index_multiple_chunks(self, indexer: ChromaIndexer) -> None:
        """Multiple chunks are indexed and all IDs returned."""
        chunks = [
            _make_embedded_chunk(content=f"chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        result = indexer.index(
            embedded_chunks=chunks,
            document_id="doc2",
            file_name="multi.pdf",
            version=1,
            file_path="multi.pdf",
        )

        assert result.total_indexed == 3
        assert result.indexed_ids == ["doc2_0", "doc2_1", "doc2_2"]

    def test_upsert_replaces_existing(self, indexer: ChromaIndexer) -> None:
        """Re-indexing the same document replaces data, count unchanged."""
        ec_v1 = _make_embedded_chunk(
            content="version one", chunk_index=0, token_count=5
        )
        indexer.index(
            embedded_chunks=[ec_v1],
            document_id="doc3",
            file_name="test.pdf",
            version=1,
            file_path="test.pdf",
        )

        # Re-index with updated content.
        ec_v2 = _make_embedded_chunk(
            content="version two", chunk_index=0, token_count=5
        )
        result = indexer.index(
            embedded_chunks=[ec_v2],
            document_id="doc3",
            file_name="test.pdf",
            version=2,
            file_path="test.pdf",
        )

        assert result.total_indexed == 1
        # Verify the collection only has 1 entry (not 2).
        collection = indexer._client.get_collection("default")
        assert collection.count() == 1
        # Verify content was updated.
        got = collection.get(ids=["doc3_0"])
        assert got["documents"][0] == "version two"  # type: ignore[index]

    def test_deterministic_ids(self, indexer: ChromaIndexer) -> None:
        """ID format is '{document_id}_{chunk_index}'."""
        chunks = [
            _make_embedded_chunk(chunk_index=0),
            _make_embedded_chunk(chunk_index=5),
            _make_embedded_chunk(chunk_index=10),
        ]
        result = indexer.index(
            embedded_chunks=chunks,
            document_id="abc123",
            file_name="test.pdf",
            version=1,
            file_path="test.pdf",
        )

        assert result.indexed_ids == ["abc123_0", "abc123_5", "abc123_10"]

    def test_metadata_stored_correctly(self, indexer: ChromaIndexer) -> None:
        """All metadata fields are present and correct in ChromaDB."""
        ec = _make_embedded_chunk(
            content="metadata test",
            chunk_index=3,
            header_path="Chapter 1 > Intro",
            page_number=2,
            content_type=ContentType.TABLE,
            token_count=42,
        )
        indexer.index(
            embedded_chunks=[ec],
            document_id="meta_doc",
            file_name="report.pdf",
            version=7,
            file_path="report.pdf",
        )

        collection = indexer._client.get_collection("default")
        got = collection.get(ids=["meta_doc_3"], include=["metadatas"])
        meta = got["metadatas"][0]  # type: ignore[index]

        assert meta["document_id"] == "meta_doc"
        assert meta["file_name"] == "report.pdf"
        assert meta["version"] == 7
        assert meta["page_number"] == 2
        assert meta["chunk_index"] == 3
        assert meta["content_type"] == "table"
        assert meta["header_path"] == "Chapter 1 > Intro"
        assert meta["token_count"] == 42

    def test_empty_input(self, indexer: ChromaIndexer) -> None:
        """Empty list produces an empty IndexResult."""
        result = indexer.index(
            embedded_chunks=[],
            document_id="empty_doc",
            file_name="empty.pdf",
            version=1,
            file_path="empty.pdf",
        )

        assert result.total_indexed == 0
        assert result.indexed_ids == []
        assert result.collection_name == "default"

    def test_collection_created_on_demand(
        self, mapped_indexer: ChromaIndexer
    ) -> None:
        """Collection is auto-created if it does not exist yet."""
        ec = _make_embedded_chunk(content="new collection", chunk_index=0)
        result = mapped_indexer.index(
            embedded_chunks=[ec],
            document_id="doc_new",
            file_name="data.pdf",
            version=1,
            file_path="data.pdf",
        )

        assert result.collection_name == "pdfs"
        # Verify the collection exists and has data.
        collection = mapped_indexer._client.get_collection("pdfs")
        assert collection.count() == 1


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    """Tests for the delete_document method."""

    def test_delete_document(self, indexer: ChromaIndexer) -> None:
        """Deletes all chunks for a document_id."""
        chunks = [
            _make_embedded_chunk(content=f"chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        indexer.index(
            embedded_chunks=chunks,
            document_id="del_doc",
            file_name="delete_me.pdf",
            version=1,
            file_path="delete_me.pdf",
        )

        deleted = indexer.delete_document("del_doc")

        assert deleted == 3
        collection = indexer._client.get_collection("default")
        assert collection.count() == 0

    def test_delete_nonexistent_document(
        self, indexer: ChromaIndexer
    ) -> None:
        """Deleting a document that doesn't exist returns 0."""
        # First create the collection so it exists.
        ec = _make_embedded_chunk(chunk_index=0)
        indexer.index(
            embedded_chunks=[ec],
            document_id="other",
            file_name="other.pdf",
            version=1,
            file_path="other.pdf",
        )

        deleted = indexer.delete_document("nonexistent")
        assert deleted == 0

    def test_delete_from_missing_collection(
        self, indexer: ChromaIndexer
    ) -> None:
        """Deleting from a non-existent collection returns 0."""
        deleted = indexer.delete_document(
            "doc1", collection_name="no_such_collection"
        )
        assert deleted == 0
