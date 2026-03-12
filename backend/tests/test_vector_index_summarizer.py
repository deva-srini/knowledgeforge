"""Tests for the vector index summarizer service."""

import tomllib

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.vector_index_summarizer import VectorIndexSummarizer


@pytest.fixture
def config(tmp_path):
    """Create a KnowledgeForgeConfig pointing at a temporary ChromaDB dir."""
    return KnowledgeForgeConfig(
        indexing={
            "chromadb_path": str(tmp_path / "chromadb"),
            "summary_path": str(tmp_path / "summary.toml"),
        },
    )


@pytest.fixture
def summarizer(config):
    """Create a VectorIndexSummarizer from the test config."""
    return VectorIndexSummarizer(config)


def _seed_collection(summarizer, name, chunks):
    """Seed a ChromaDB collection with test chunk data.

    Args:
        summarizer: VectorIndexSummarizer instance (used to access _client).
        name: Collection name.
        chunks: List of dicts with keys: id, embedding, document, metadata.
    """
    collection = summarizer._client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


class TestVectorIndexSummarizer:
    """Tests for VectorIndexSummarizer."""

    def test_empty_database(self, summarizer):
        """No collections returns empty summary."""
        summary = summarizer.generate()
        assert summary.total_collections == 0
        assert summary.total_chunks == 0
        assert summary.collections == []

    def test_single_collection_stats(self, summarizer):
        """One collection with chunks produces correct aggregate stats."""
        _seed_collection(
            summarizer,
            "test_col",
            [
                {
                    "id": "doc1_0",
                    "embedding": [0.1, 0.2, 0.3],
                    "document": "First chunk text",
                    "metadata": {
                        "document_id": "doc1",
                        "file_name": "report.pdf",
                        "version": 1,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "text",
                        "header_path": "Introduction",
                        "token_count": 50,
                    },
                },
                {
                    "id": "doc1_1",
                    "embedding": [0.4, 0.5, 0.6],
                    "document": "Second chunk text",
                    "metadata": {
                        "document_id": "doc1",
                        "file_name": "report.pdf",
                        "version": 1,
                        "page_number": 2,
                        "chunk_index": 1,
                        "content_type": "table",
                        "header_path": "Financials",
                        "token_count": 100,
                    },
                },
            ],
        )

        summary = summarizer.generate()

        assert summary.total_collections == 1
        assert summary.total_chunks == 2
        assert len(summary.collections) == 1

        col = summary.collections[0]
        assert col.name == "test_col"
        assert col.total_chunks == 2
        assert "text" in col.content_types
        assert "table" in col.content_types

        assert len(col.documents) == 1
        doc = col.documents[0]
        assert doc.file_name == "report.pdf"
        assert doc.chunks == 2
        assert doc.tokens == 150
        assert doc.version == 1
        assert "Introduction" in doc.sections
        assert "Financials" in doc.sections

    def test_document_descriptions_merged(self, summarizer):
        """Descriptions from config are matched to documents by file_name."""
        _seed_collection(
            summarizer,
            "test_col",
            [
                {
                    "id": "doc1_0",
                    "embedding": [0.1, 0.2, 0.3],
                    "document": "Some content",
                    "metadata": {
                        "document_id": "doc1",
                        "file_name": "report.pdf",
                        "version": 1,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "text",
                        "header_path": "",
                        "token_count": 25,
                    },
                },
            ],
        )

        descriptions = {"report.pdf": "Q1 fund performance report"}
        summary = summarizer.generate(document_descriptions=descriptions)

        doc = summary.collections[0].documents[0]
        assert doc.description == "Q1 fund performance report"

    def test_multiple_collections(self, summarizer):
        """Multiple collections each summarized independently."""
        _seed_collection(
            summarizer,
            "alpha",
            [
                {
                    "id": "a_0",
                    "embedding": [1.0, 0.0, 0.0],
                    "document": "Alpha content",
                    "metadata": {
                        "document_id": "a",
                        "file_name": "alpha.pdf",
                        "version": 1,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "text",
                        "header_path": "Intro",
                        "token_count": 30,
                    },
                },
            ],
        )
        _seed_collection(
            summarizer,
            "beta",
            [
                {
                    "id": "b_0",
                    "embedding": [0.0, 1.0, 0.0],
                    "document": "Beta content",
                    "metadata": {
                        "document_id": "b",
                        "file_name": "beta.pdf",
                        "version": 2,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "image",
                        "header_path": "Charts",
                        "token_count": 10,
                    },
                },
            ],
        )

        summary = summarizer.generate()

        assert summary.total_collections == 2
        assert summary.total_chunks == 2

        col_names = [c.name for c in summary.collections]
        assert "alpha" in col_names
        assert "beta" in col_names

        beta = next(c for c in summary.collections if c.name == "beta")
        assert beta.documents[0].version == 2

    def test_chunk_structure_detected(self, summarizer):
        """Metadata fields, has_embeddings, has_document_content detected."""
        _seed_collection(
            summarizer,
            "test_col",
            [
                {
                    "id": "doc1_0",
                    "embedding": [0.1, 0.2, 0.3],
                    "document": "Content here",
                    "metadata": {
                        "document_id": "doc1",
                        "file_name": "file.pdf",
                        "version": 1,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "text",
                        "header_path": "Section 1",
                        "token_count": 40,
                    },
                },
            ],
        )

        summary = summarizer.generate()
        col = summary.collections[0]

        assert col.has_embeddings is True
        assert col.has_document_content is True
        assert "document_id" in col.metadata_fields
        assert "file_name" in col.metadata_fields
        assert "token_count" in col.metadata_fields

    def test_write_toml(self, summarizer, config, tmp_path):
        """TOML file written and is valid TOML."""
        _seed_collection(
            summarizer,
            "test_col",
            [
                {
                    "id": "doc1_0",
                    "embedding": [0.1, 0.2, 0.3],
                    "document": "Test content",
                    "metadata": {
                        "document_id": "doc1",
                        "file_name": "test.pdf",
                        "version": 1,
                        "page_number": 1,
                        "chunk_index": 0,
                        "content_type": "text",
                        "header_path": "Intro",
                        "token_count": 20,
                    },
                },
            ],
        )

        summary = summarizer.generate()
        output_path = str(tmp_path / "output.toml")
        summarizer.write_toml(summary, output_path=output_path)

        # Verify file exists and is valid TOML
        with open(output_path, "rb") as f:
            parsed = tomllib.load(f)

        assert "meta" in parsed
        assert parsed["meta"]["total_collections"] == 1
        assert parsed["meta"]["total_chunks"] == 1
        assert parsed["meta"]["embedding_dimension"] == 3
        assert len(parsed["collections"]) == 1
        assert parsed["collections"][0]["name"] == "test_col"
        assert len(parsed["collections"][0]["documents"]) == 1
        assert parsed["collections"][0]["documents"][0]["file_name"] == "test.pdf"
