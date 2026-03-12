"""Semantic overlap detection for ChromaDB collections.

Finds semantically similar chunks within a single collection or across
two collections. Runs post-indexing as a standalone analysis tool,
fully decoupled from the ingestion pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class OverlapMatch:
    """A pair of semantically similar chunks.

    Attributes:
        chunk_id: ChromaDB ID of chunk A.
        chunk_content_preview: First 100 chars of chunk A content.
        similar_chunk_id: ChromaDB ID of chunk B.
        similar_content_preview: First 100 chars of chunk B content.
        similarity: Cosine similarity score (0-1).
        chunk_document: file_name metadata of chunk A.
        similar_document: file_name metadata of chunk B.
    """

    chunk_id: str
    chunk_content_preview: str
    similar_chunk_id: str
    similar_content_preview: str
    similarity: float
    chunk_document: str
    similar_document: str


@dataclass
class OverlapReport:
    """Summary of overlap detection results.

    Attributes:
        mode: "within_collection" or "cross_collection".
        collection_a: Source collection name.
        collection_b: Target collection name (None for within-collection).
        total_chunks_analyzed: Number of chunks queried.
        overlaps_found: Number of overlap matches found.
        threshold_used: Similarity threshold applied.
        matches: List of OverlapMatch results.
    """

    mode: str
    collection_a: str
    collection_b: Optional[str]
    total_chunks_analyzed: int
    overlaps_found: int
    threshold_used: float
    matches: List[OverlapMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to a dictionary."""
        return {
            "mode": self.mode,
            "collection_a": self.collection_a,
            "collection_b": self.collection_b,
            "total_chunks_analyzed": self.total_chunks_analyzed,
            "overlaps_found": self.overlaps_found,
            "threshold_used": self.threshold_used,
            "matches": [
                {
                    "chunk_id": m.chunk_id,
                    "chunk_content_preview": m.chunk_content_preview,
                    "similar_chunk_id": m.similar_chunk_id,
                    "similar_content_preview": m.similar_content_preview,
                    "similarity": m.similarity,
                    "chunk_document": m.chunk_document,
                    "similar_document": m.similar_document,
                }
                for m in self.matches
            ],
        }

    def print_report(self) -> None:
        """Print a formatted CLI report."""
        if self.mode == "within_collection":
            header = f'Overlap Report: within-collection "{self.collection_a}"'
        else:
            header = (
                f'Overlap Report: cross-collection '
                f'"{self.collection_a}" -> "{self.collection_b}"'
            )

        print(f"\n{header}")
        print(f"  Chunks analyzed: {self.total_chunks_analyzed}")
        print(f"  Threshold: {self.threshold_used}")
        print(f"  Overlaps found: {self.overlaps_found}")

        if self.matches:
            print("\n  Matches:")
            for i, m in enumerate(self.matches, 1):
                print(
                    f"    {i}. [{m.similarity:.2f}] "
                    f"{m.chunk_id} ({m.chunk_document}) ~ "
                    f"{m.similar_chunk_id} ({m.similar_document})"
                )
                print(
                    f'       "{m.chunk_content_preview}"  ~  '
                    f'"{m.similar_content_preview}"'
                )
        print()


def _preview(text: str, length: int = 100) -> str:
    """Truncate text to a preview length."""
    if len(text) <= length:
        return text
    return text[:length - 3] + "..."


def _get_file_name(metadata: Optional[Dict]) -> str:
    """Extract file_name from metadata, with fallback."""
    if metadata and "file_name" in metadata:
        return metadata["file_name"]
    return "unknown"


class OverlapDetector:
    """Detects semantically similar chunks in ChromaDB collections.

    Uses cosine distance from ChromaDB (collections must be created with
    hnsw:space=cosine) to find overlapping content. Similarity is computed
    as 1.0 - distance.
    """

    def __init__(self, chromadb_path: str) -> None:
        self._client = chromadb.PersistentClient(path=chromadb_path)

    def detect_within_collection(
        self,
        collection_name: str,
        similarity_threshold: float = 0.85,
        max_results: int = 5,
    ) -> OverlapReport:
        """Find overlapping chunks within a single collection.

        Fetches all chunks, queries each embedding against the same
        collection, and filters out self-matches and symmetric duplicates.

        Args:
            collection_name: Name of the ChromaDB collection to scan.
            similarity_threshold: Minimum cosine similarity to report (0-1).
            max_results: Maximum similar chunks to return per query chunk.

        Returns:
            OverlapReport with detected matches.
        """
        collection = self._client.get_collection(name=collection_name)
        all_data = collection.get(
            include=["embeddings", "metadatas", "documents"],
        )

        ids = all_data["ids"]
        embeddings = all_data["embeddings"]
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        if len(ids) == 0 or embeddings is None or len(embeddings) == 0:
            return OverlapReport(
                mode="within_collection",
                collection_a=collection_name,
                collection_b=None,
                total_chunks_analyzed=0,
                overlaps_found=0,
                threshold_used=similarity_threshold,
            )

        # Query all embeddings in a single batch
        # Request extra results to account for self-match filtering
        query_results = collection.query(
            query_embeddings=embeddings,
            n_results=min(max_results + 1, len(ids)),
            include=["distances", "metadatas", "documents"],
        )

        seen_pairs: Set[Tuple[str, str]] = set()
        matches: List[OverlapMatch] = []

        for i, chunk_id in enumerate(ids):
            result_ids = query_results["ids"][i]
            result_distances = query_results["distances"][i]
            result_metadatas = query_results["metadatas"][i]
            result_documents = query_results["documents"][i]

            for j, similar_id in enumerate(result_ids):
                # Skip self-match
                if similar_id == chunk_id:
                    continue

                distance = result_distances[j]
                similarity = 1.0 - distance

                if similarity < similarity_threshold:
                    continue

                # Deduplicate symmetric pairs (A~B == B~A)
                pair_key = tuple(sorted((chunk_id, similar_id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                matches.append(
                    OverlapMatch(
                        chunk_id=chunk_id,
                        chunk_content_preview=_preview(documents[i] or ""),
                        similar_chunk_id=similar_id,
                        similar_content_preview=_preview(
                            result_documents[j] or ""
                        ),
                        similarity=similarity,
                        chunk_document=_get_file_name(metadatas[i]),
                        similar_document=_get_file_name(result_metadatas[j]),
                    )
                )

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)

        return OverlapReport(
            mode="within_collection",
            collection_a=collection_name,
            collection_b=None,
            total_chunks_analyzed=len(ids),
            overlaps_found=len(matches),
            threshold_used=similarity_threshold,
            matches=matches,
        )

    def detect_cross_collection(
        self,
        source_collection: str,
        target_collection: str,
        similarity_threshold: float = 0.85,
        max_results: int = 5,
    ) -> OverlapReport:
        """Find overlapping chunks between two collections.

        Fetches all chunks from the source collection and queries their
        embeddings against the target collection.

        Args:
            source_collection: Name of the source ChromaDB collection.
            target_collection: Name of the target ChromaDB collection.
            similarity_threshold: Minimum cosine similarity to report (0-1).
            max_results: Maximum similar chunks to return per query chunk.

        Returns:
            OverlapReport with detected matches.
        """
        source = self._client.get_collection(name=source_collection)
        target = self._client.get_collection(name=target_collection)

        source_data = source.get(
            include=["embeddings", "metadatas", "documents"],
        )

        source_ids = source_data["ids"]
        source_embeddings = source_data["embeddings"]
        source_documents = source_data["documents"]
        source_metadatas = source_data["metadatas"]

        if len(source_ids) == 0 or source_embeddings is None or len(source_embeddings) == 0:
            return OverlapReport(
                mode="cross_collection",
                collection_a=source_collection,
                collection_b=target_collection,
                total_chunks_analyzed=0,
                overlaps_found=0,
                threshold_used=similarity_threshold,
            )

        target_count = target.count()
        if target_count == 0:
            return OverlapReport(
                mode="cross_collection",
                collection_a=source_collection,
                collection_b=target_collection,
                total_chunks_analyzed=len(source_ids),
                overlaps_found=0,
                threshold_used=similarity_threshold,
            )

        query_results = target.query(
            query_embeddings=source_embeddings,
            n_results=min(max_results, target_count),
            include=["distances", "metadatas", "documents"],
        )

        matches: List[OverlapMatch] = []

        for i, chunk_id in enumerate(source_ids):
            result_ids = query_results["ids"][i]
            result_distances = query_results["distances"][i]
            result_metadatas = query_results["metadatas"][i]
            result_documents = query_results["documents"][i]

            for j, similar_id in enumerate(result_ids):
                distance = result_distances[j]
                similarity = 1.0 - distance

                if similarity < similarity_threshold:
                    continue

                matches.append(
                    OverlapMatch(
                        chunk_id=chunk_id,
                        chunk_content_preview=_preview(
                            source_documents[i] or ""
                        ),
                        similar_chunk_id=similar_id,
                        similar_content_preview=_preview(
                            result_documents[j] or ""
                        ),
                        similarity=similarity,
                        chunk_document=_get_file_name(source_metadatas[i]),
                        similar_document=_get_file_name(result_metadatas[j]),
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)

        return OverlapReport(
            mode="cross_collection",
            collection_a=source_collection,
            collection_b=target_collection,
            total_chunks_analyzed=len(source_ids),
            overlaps_found=len(matches),
            threshold_used=similarity_threshold,
            matches=matches,
        )
