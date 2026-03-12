"""Tests for the semantic overlap detection service."""

import pytest

from app.services.overlap_detection import OverlapDetector, OverlapReport


@pytest.fixture
def detector(tmp_path):
    """Create an OverlapDetector backed by a temporary ChromaDB directory."""
    return OverlapDetector(chromadb_path=str(tmp_path / "chromadb"))


def _seed_collection(detector, name, ids, embeddings, documents, metadatas=None):
    """Seed a ChromaDB collection with test data."""
    collection = detector._client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    if metadatas is None:
        metadatas = [{"file_name": f"doc_{i}.pdf"} for i in range(len(ids))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


class TestWithinCollection:
    """Tests for detect_within_collection."""

    def test_no_overlaps(self, detector):
        """Dissimilar chunks produce zero overlaps."""
        _seed_collection(
            detector,
            "test_col",
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["about cats", "about dogs"],
        )
        report = detector.detect_within_collection("test_col", similarity_threshold=0.85)
        assert report.overlaps_found == 0
        assert report.total_chunks_analyzed == 2
        assert report.mode == "within_collection"
        assert report.matches == []

    def test_detects_similar(self, detector):
        """Near-identical chunks are flagged."""
        _seed_collection(
            detector,
            "test_col",
            ids=["a", "b", "c"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],  # very similar to a
                [0.0, 0.0, 1.0],   # dissimilar
            ],
            documents=["revenue grew 12%", "revenue increased 12%", "unrelated topic"],
        )
        report = detector.detect_within_collection("test_col", similarity_threshold=0.8)
        assert report.overlaps_found >= 1
        # The similar pair should be a~b
        similar_ids = {(m.chunk_id, m.similar_chunk_id) for m in report.matches}
        assert ("a", "b") in similar_ids or ("b", "a") in similar_ids

    def test_respects_threshold(self, detector):
        """Chunks below threshold are not flagged."""
        _seed_collection(
            detector,
            "test_col",
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            documents=["text one", "text two"],
        )
        # Very high threshold should exclude moderate similarity
        report = detector.detect_within_collection("test_col", similarity_threshold=0.99)
        assert report.overlaps_found == 0

    def test_deduplicates_symmetric_pairs(self, detector):
        """A~B and B~A should appear only once."""
        _seed_collection(
            detector,
            "test_col",
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]],
            documents=["text a", "text b"],
        )
        report = detector.detect_within_collection("test_col", similarity_threshold=0.5)
        # Should have exactly 1 match, not 2
        pair_count = sum(
            1 for m in report.matches
            if {m.chunk_id, m.similar_chunk_id} == {"a", "b"}
        )
        assert pair_count == 1

    def test_empty_collection(self, detector):
        """Empty collection returns empty report."""
        detector._client.get_or_create_collection(
            name="empty_col",
            metadata={"hnsw:space": "cosine"},
        )
        report = detector.detect_within_collection("empty_col")
        assert report.total_chunks_analyzed == 0
        assert report.overlaps_found == 0
        assert report.matches == []


class TestCrossCollection:
    """Tests for detect_cross_collection."""

    def test_detects_similar(self, detector):
        """Matching chunks across two collections are found."""
        _seed_collection(
            detector,
            "source",
            ids=["s1"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["revenue grew"],
            metadatas=[{"file_name": "source.pdf"}],
        )
        _seed_collection(
            detector,
            "target",
            ids=["t1", "t2"],
            embeddings=[[0.99, 0.1, 0.0], [0.0, 1.0, 0.0]],
            documents=["revenue increased", "unrelated"],
            metadatas=[{"file_name": "target1.pdf"}, {"file_name": "target2.pdf"}],
        )
        report = detector.detect_cross_collection(
            "source", "target", similarity_threshold=0.8
        )
        assert report.mode == "cross_collection"
        assert report.overlaps_found >= 1
        assert report.matches[0].chunk_id == "s1"
        assert report.matches[0].similar_chunk_id == "t1"

    def test_no_overlaps(self, detector):
        """Dissimilar chunks across collections produce zero overlaps."""
        _seed_collection(
            detector,
            "source",
            ids=["s1"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["about cats"],
        )
        _seed_collection(
            detector,
            "target",
            ids=["t1"],
            embeddings=[[0.0, 1.0, 0.0]],
            documents=["about dogs"],
        )
        report = detector.detect_cross_collection(
            "source", "target", similarity_threshold=0.85
        )
        assert report.overlaps_found == 0

    def test_empty_target(self, detector):
        """Empty target collection returns zero overlaps."""
        _seed_collection(
            detector,
            "source",
            ids=["s1"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["some content"],
        )
        detector._client.get_or_create_collection(
            name="empty_target",
            metadata={"hnsw:space": "cosine"},
        )
        report = detector.detect_cross_collection(
            "source", "empty_target", similarity_threshold=0.85
        )
        assert report.total_chunks_analyzed == 1
        assert report.overlaps_found == 0


class TestOverlapReport:
    """Tests for OverlapReport methods."""

    def test_to_dict(self):
        """to_dict returns a serializable dictionary."""
        report = OverlapReport(
            mode="within_collection",
            collection_a="test",
            collection_b=None,
            total_chunks_analyzed=10,
            overlaps_found=0,
            threshold_used=0.85,
        )
        d = report.to_dict()
        assert d["mode"] == "within_collection"
        assert d["collection_a"] == "test"
        assert d["collection_b"] is None
        assert d["matches"] == []

    def test_print_report(self, capsys):
        """print_report outputs formatted text."""
        report = OverlapReport(
            mode="within_collection",
            collection_a="my_col",
            collection_b=None,
            total_chunks_analyzed=5,
            overlaps_found=0,
            threshold_used=0.85,
        )
        report.print_report()
        captured = capsys.readouterr()
        assert "my_col" in captured.out
        assert "Chunks analyzed: 5" in captured.out
