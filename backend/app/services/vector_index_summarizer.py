"""Vector index summarizer for generating TOML summaries of ChromaDB contents.

Scans all ChromaDB collections and produces a structured summary including
collection stats, per-document breakdowns, content types, and sections.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import tomli_w

from app.core.config import KnowledgeForgeConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentSummary:
    """Summary of a single document's chunks within a collection."""

    file_name: str
    description: str
    chunks: int
    tokens: int
    content_types: List[str]
    sections: List[str]
    version: int


@dataclass
class CollectionSummary:
    """Summary of a single ChromaDB collection."""

    name: str
    total_chunks: int
    content_types: List[str]
    metadata_fields: List[str]
    has_embeddings: bool
    has_document_content: bool
    documents: List[DocumentSummary] = field(default_factory=list)


@dataclass
class VectorIndexSummary:
    """Complete summary of all ChromaDB vector index contents."""

    generated_at: str
    chromadb_path: str
    embedding_model: str
    embedding_dimension: int
    total_collections: int
    total_chunks: int
    collections: List[CollectionSummary] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the summary to a nested dict suitable for TOML serialization."""
        return {
            "meta": {
                "generated_at": self.generated_at,
                "chromadb_path": self.chromadb_path,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "total_collections": self.total_collections,
                "total_chunks": self.total_chunks,
            },
            "collections": [
                {
                    "name": c.name,
                    "total_chunks": c.total_chunks,
                    "content_types": c.content_types,
                    "metadata_fields": c.metadata_fields,
                    "has_embeddings": c.has_embeddings,
                    "has_document_content": c.has_document_content,
                    "documents": [
                        {
                            "file_name": d.file_name,
                            "description": d.description,
                            "chunks": d.chunks,
                            "tokens": d.tokens,
                            "content_types": d.content_types,
                            "sections": d.sections,
                            "version": d.version,
                        }
                        for d in c.documents
                    ],
                }
                for c in self.collections
            ],
        }


class VectorIndexSummarizer:
    """Scans ChromaDB collections and generates a structured TOML summary."""

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        self._config = config
        self._client = chromadb.PersistentClient(
            path=config.indexing.chromadb_path
        )

    def generate(
        self, document_descriptions: Optional[Dict[str, str]] = None
    ) -> VectorIndexSummary:
        """Scan all ChromaDB collections and build a complete summary.

        Args:
            document_descriptions: Optional mapping of file_name to
                human-readable description strings.

        Returns:
            A VectorIndexSummary covering all collections and documents.
        """
        descriptions = document_descriptions or {}
        collection_names = [c.name for c in self._client.list_collections()]

        collection_summaries: List[CollectionSummary] = []
        total_chunks = 0

        for col_name in sorted(collection_names):
            collection = self._client.get_collection(name=col_name)
            col_summary = self._summarize_collection(
                collection, descriptions
            )
            collection_summaries.append(col_summary)
            total_chunks += col_summary.total_chunks

        embedding_model = self._config.processing.embedding.model
        # Peek at embedding dimension from first non-empty collection
        embedding_dimension = self._detect_embedding_dimension(
            collection_names
        )

        return VectorIndexSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            chromadb_path=self._config.indexing.chromadb_path,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            total_collections=len(collection_names),
            total_chunks=total_chunks,
            collections=collection_summaries,
        )

    def write_toml(
        self,
        summary: VectorIndexSummary,
        output_path: Optional[str] = None,
    ) -> None:
        """Write summary to a TOML file.

        Args:
            summary: The VectorIndexSummary to serialize.
            output_path: File path to write. Falls back to
                config.indexing.summary_path.
        """
        path = Path(output_path or self._config.indexing.summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = summary.to_dict()
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

        logger.info("Vector index summary written to %s", path)

    def _summarize_collection(
        self,
        collection: chromadb.Collection,
        descriptions: Dict[str, str],
    ) -> CollectionSummary:
        """Build a CollectionSummary from a ChromaDB collection.

        Args:
            collection: The ChromaDB collection object.
            descriptions: file_name -> description mapping.

        Returns:
            A CollectionSummary with per-document breakdowns.
        """
        result = collection.get(include=["metadatas", "documents"])
        ids = result["ids"]
        metadatas = result["metadatas"] or []
        documents = result["documents"] or []

        chunk_count = len(ids)

        # Detect has_embeddings by peeking at one entry
        has_embeddings = False
        if chunk_count > 0:
            peek = collection.get(limit=1, include=["embeddings"])
            peek_embeddings = peek.get("embeddings")
            if peek_embeddings is not None and len(peek_embeddings) > 0:
                has_embeddings = peek_embeddings[0] is not None

        has_document_content = any(
            d is not None and len(d) > 0 for d in documents
        )

        # Collect metadata field names from first chunk
        metadata_fields: List[str] = []
        if metadatas and len(metadatas) > 0:
            metadata_fields = sorted(metadatas[0].keys())

        # Group chunks by file_name
        doc_chunks: Dict[str, list] = defaultdict(list)
        for i, meta in enumerate(metadatas):
            fname = meta.get("file_name", "unknown")
            doc_chunks[fname].append(meta)

        # Build per-document summaries
        all_content_types: set = set()
        doc_summaries: List[DocumentSummary] = []

        for fname in sorted(doc_chunks.keys()):
            chunks = doc_chunks[fname]
            token_sum = sum(c.get("token_count", 0) for c in chunks)
            ctypes = sorted(
                set(c.get("content_type", "") for c in chunks if c.get("content_type"))
            )
            sections = sorted(
                set(c.get("header_path", "") for c in chunks if c.get("header_path"))
            )
            max_version = max(c.get("version", 1) for c in chunks)

            all_content_types.update(ctypes)

            doc_summaries.append(
                DocumentSummary(
                    file_name=fname,
                    description=descriptions.get(fname, ""),
                    chunks=len(chunks),
                    tokens=token_sum,
                    content_types=ctypes,
                    sections=sections,
                    version=max_version,
                )
            )

        return CollectionSummary(
            name=collection.name,
            total_chunks=chunk_count,
            content_types=sorted(all_content_types),
            metadata_fields=metadata_fields,
            has_embeddings=has_embeddings,
            has_document_content=has_document_content,
            documents=doc_summaries,
        )

    def _detect_embedding_dimension(
        self, collection_names: List[str]
    ) -> int:
        """Detect embedding dimension by peeking at the first non-empty collection.

        Args:
            collection_names: List of collection names to check.

        Returns:
            Embedding dimension, or 0 if no embeddings found.
        """
        for col_name in collection_names:
            collection = self._client.get_collection(name=col_name)
            if collection.count() == 0:
                continue
            peek = collection.get(limit=1, include=["embeddings"])
            embeddings = peek.get("embeddings")
            if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
                return len(embeddings[0])
        return 0
