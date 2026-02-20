"""ChromaDB indexer service for persisting embedded chunks into a vector store.

Takes a list of EmbeddedChunk objects from the embedding step and upserts
them into a ChromaDB collection. Supports configurable collection mapping
via file-path glob patterns and deterministic document IDs for safe
re-indexing.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

from app.core.config import KnowledgeForgeConfig
from app.services.embedding import EmbeddedChunk

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of an indexing operation.

    Attributes:
        collection_name: Name of the ChromaDB collection used.
        indexed_ids: List of ChromaDB document IDs that were upserted.
        total_indexed: Count of chunks indexed.
    """

    collection_name: str
    indexed_ids: List[str] = field(default_factory=list)
    total_indexed: int = 0


class ChromaIndexer:
    """Indexes embedded chunks into ChromaDB collections.

    Uses a PersistentClient for durable storage and supports file-path
    pattern matching to route documents into different collections.
    Pre-computed embeddings are passed directly — no embedding function
    is set on the collection.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the indexer from configuration.

        Args:
            config: KnowledgeForge configuration object.
        """
        self._default_collection = config.indexing.default_collection
        self._collection_mapping = config.indexing.collection_mapping
        self._client = chromadb.PersistentClient(
            path=config.indexing.chromadb_path
        )

    def index(
        self,
        embedded_chunks: List[EmbeddedChunk],
        document_id: str,
        file_name: str,
        version: int,
        file_path: str,
    ) -> IndexResult:
        """Index a list of embedded chunks into ChromaDB.

        Resolves the target collection from file_path patterns, builds
        deterministic IDs, and performs a single upsert call.

        Args:
            embedded_chunks: List of EmbeddedChunk objects to index.
            document_id: Unique identifier for the source document.
            file_name: Original file name of the document.
            version: Document version number.
            file_path: Full file path, used for collection routing.

        Returns:
            IndexResult with collection name and indexed IDs.
        """
        if not embedded_chunks:
            collection_name = self._resolve_collection(file_path)
            return IndexResult(collection_name=collection_name)

        collection_name = self._resolve_collection(file_path)
        collection = self._client.get_or_create_collection(
            name=collection_name
        )

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[dict] = []  # type: ignore[type-arg]

        for ec in embedded_chunks:
            chunk = ec.chunk
            chunk_id = f"{document_id}_{chunk.chunk_index}"
            ids.append(chunk_id)
            embeddings.append(ec.embedding)
            documents.append(chunk.content)
            metadatas.append({
                "document_id": document_id,
                "file_name": file_name,
                "version": version,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content_type": chunk.content_type.value,
                "header_path": chunk.header_path,
                "token_count": chunk.token_count,
            })

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d chunks into collection '%s' for document '%s'",
            len(ids),
            collection_name,
            document_id,
        )

        return IndexResult(
            collection_name=collection_name,
            indexed_ids=ids,
            total_indexed=len(ids),
        )

    def _resolve_collection(self, file_path: str) -> str:
        """Resolve which collection to use based on file path patterns.

        Iterates through collection_mapping patterns (fnmatch) and returns
        the first matching collection name. Falls back to default_collection.

        Args:
            file_path: The file path to match against patterns.

        Returns:
            Collection name string.
        """
        for pattern, collection_name in self._collection_mapping.items():
            if fnmatch.fnmatch(file_path, pattern):
                return collection_name
        return self._default_collection

    def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Delete all chunks for a document from a collection.

        Args:
            document_id: The document ID whose chunks should be removed.
            collection_name: Target collection. Uses default if not given.

        Returns:
            Number of chunks deleted.
        """
        target = collection_name or self._default_collection
        try:
            collection = self._client.get_collection(name=target)
        except Exception:
            logger.warning(
                "Collection '%s' not found for delete_document('%s')",
                target,
                document_id,
            )
            return 0

        # Query for all chunks belonging to this document.
        results = collection.get(
            where={"document_id": document_id},
        )

        matched_ids = results["ids"]
        if not matched_ids:
            return 0

        collection.delete(ids=matched_ids)

        logger.info(
            "Deleted %d chunks for document '%s' from collection '%s'",
            len(matched_ids),
            document_id,
            target,
        )

        return len(matched_ids)
