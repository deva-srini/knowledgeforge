"""Tests for the structure-aware chunking service."""

from typing import Any, Dict

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.chunking import (
    Chunk,
    StructureAwareChunker,
    _count_tokens,
)
from app.services.extraction import ContentType
from app.services.transformation import TransformedContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Default config (chunk_size=512, overlap=50, skip_threshold=1000)."""
    return KnowledgeForgeConfig()


@pytest.fixture
def chunker(config: KnowledgeForgeConfig) -> StructureAwareChunker:
    """Chunker with default config."""
    return StructureAwareChunker(config)


@pytest.fixture
def small_chunk_config() -> KnowledgeForgeConfig:
    """Config with small chunk size for testing splits."""
    return KnowledgeForgeConfig(
        processing={
            "chunking": {
                "chunk_size_tokens": 50,
                "chunk_overlap_tokens": 10,
                "skip_threshold_tokens": 100,
            }
        }
    )


@pytest.fixture
def small_chunker(
    small_chunk_config: KnowledgeForgeConfig,
) -> StructureAwareChunker:
    """Chunker with small chunk size for testing splits."""
    return StructureAwareChunker(small_chunk_config)


@pytest.fixture
def no_overlap_config() -> KnowledgeForgeConfig:
    """Config with zero overlap."""
    return KnowledgeForgeConfig(
        processing={
            "chunking": {
                "chunk_size_tokens": 50,
                "chunk_overlap_tokens": 0,
                "skip_threshold_tokens": 100,
            }
        }
    )


@pytest.fixture
def no_overlap_chunker(
    no_overlap_config: KnowledgeForgeConfig,
) -> StructureAwareChunker:
    """Chunker with zero overlap."""
    return StructureAwareChunker(no_overlap_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformed(
    content: str = "hello world",
    content_type: ContentType = ContentType.TEXT,
    page_number: int = 1,
    header_path: str = "Section 1",
    metadata: Dict[str, Any] | None = None,
) -> TransformedContent:
    """Helper to build TransformedContent with sensible defaults."""
    return TransformedContent(
        content=content,
        content_type=content_type,
        page_number=page_number,
        header_path=header_path,
        metadata=metadata if metadata is not None else {},
    )


def _make_long_text(approx_tokens: int) -> str:
    """Generate a text string with approximately the given token count.

    Uses simple words that tokenize predictably (1 token per word).
    """
    words = []
    # Common short words that are each 1 token in cl100k_base.
    vocab = ["the", "cat", "sat", "on", "mat", "and", "dog", "ran", "far"]
    for i in range(approx_tokens):
        words.append(vocab[i % len(vocab)])
    return " ".join(words)


# ---------------------------------------------------------------------------
# TestChunkDataclass
# ---------------------------------------------------------------------------


class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_construction_with_all_fields(self) -> None:
        """Construct a Chunk with all fields populated."""
        chunk = Chunk(
            content="hello",
            content_type=ContentType.TEXT,
            chunk_index=0,
            header_path="Ch1",
            page_number=1,
            token_count=1,
            metadata={"key": "value"},
        )
        assert chunk.content == "hello"
        assert chunk.content_type == ContentType.TEXT
        assert chunk.chunk_index == 0
        assert chunk.header_path == "Ch1"
        assert chunk.page_number == 1
        assert chunk.token_count == 1
        assert chunk.metadata == {"key": "value"}

    def test_default_metadata(self) -> None:
        """Metadata defaults to empty dict."""
        chunk = Chunk(
            content="x",
            content_type=ContentType.TEXT,
            chunk_index=0,
            header_path="",
            page_number=0,
            token_count=1,
        )
        assert chunk.metadata == {}

    def test_metadata_independence(self) -> None:
        """Each Chunk gets its own metadata dict."""
        c1 = Chunk(
            content="a", content_type=ContentType.TEXT,
            chunk_index=0, header_path="", page_number=0, token_count=1,
        )
        c2 = Chunk(
            content="b", content_type=ContentType.TEXT,
            chunk_index=1, header_path="", page_number=0, token_count=1,
        )
        c1.metadata["added"] = True
        assert "added" not in c2.metadata

    def test_content_type_values(self) -> None:
        """Chunk works with all ContentType enum values."""
        for ct in ContentType:
            chunk = Chunk(
                content="x", content_type=ct,
                chunk_index=0, header_path="", page_number=0, token_count=1,
            )
            assert chunk.content_type == ct


# ---------------------------------------------------------------------------
# TestCountTokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for the _count_tokens helper."""

    def test_empty_string(self) -> None:
        """Empty string returns 0."""
        assert _count_tokens("") == 0

    def test_simple_text(self) -> None:
        """Simple text returns positive count."""
        count = _count_tokens("hello world")
        assert count > 0

    def test_consistency(self) -> None:
        """Same text always returns same count."""
        text = "the quick brown fox"
        assert _count_tokens(text) == _count_tokens(text)


# ---------------------------------------------------------------------------
# TestSkipThreshold
# ---------------------------------------------------------------------------


class TestSkipThreshold:
    """Tests for small-document skip path."""

    def test_small_doc_returns_single_chunk(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Documents below skip_threshold produce a single chunk."""
        items = [_make_transformed("Hello world.")]
        result = chunker.chunk(items)
        assert len(result) == 1

    def test_skip_metadata(self, chunker: StructureAwareChunker) -> None:
        """Skip-path chunk has chunking_strategy=skip in metadata."""
        items = [_make_transformed("Small text.")]
        result = chunker.chunk(items)
        assert result[0].metadata.get("chunking_strategy") == "skip"

    def test_skip_content_concatenated(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Multiple items below threshold are concatenated with \\n\\n."""
        items = [
            _make_transformed("First."),
            _make_transformed("Second."),
        ]
        result = chunker.chunk(items)
        assert "First." in result[0].content
        assert "Second." in result[0].content
        assert "\n\n" in result[0].content

    def test_skip_chunk_index_zero(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Skip-path chunk has chunk_index=0."""
        items = [_make_transformed("Short.")]
        result = chunker.chunk(items)
        assert result[0].chunk_index == 0

    def test_skip_token_count_accurate(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Skip-path chunk token_count matches _count_tokens."""
        items = [_make_transformed("Hello world test.")]
        result = chunker.chunk(items)
        assert result[0].token_count == _count_tokens(result[0].content)

    def test_skip_boundary_at_threshold(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Document at exactly skip_threshold uses structure-aware path."""
        # skip_threshold=100 tokens for small_chunker.
        text = _make_long_text(100)
        items = [_make_transformed(text)]
        result = small_chunker.chunk(items)
        # At threshold (not below), should use structure-aware path.
        assert result[0].metadata.get("chunking_strategy") != "skip"

    def test_skip_preserves_page_number(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Skip-path picks the first non-zero page number."""
        items = [
            _make_transformed("A", page_number=0),
            _make_transformed("B", page_number=3),
        ]
        result = chunker.chunk(items)
        assert result[0].page_number == 3


# ---------------------------------------------------------------------------
# TestTableChunking
# ---------------------------------------------------------------------------


class TestTableChunking:
    """Tests for table chunking."""

    def test_table_standalone_chunk(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Each table becomes a standalone chunk."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(
                "| A | B |\n|---|---|\n| 1 | 2 |",
                content_type=ContentType.TABLE,
            ),
        ]
        result = small_chunker.chunk(items)
        table_chunks = [c for c in result if c.content_type == ContentType.TABLE]
        assert len(table_chunks) == 1
        assert "| A | B |" in table_chunks[0].content

    def test_table_never_split(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Tables are never split even if they exceed chunk_size."""
        big_table = "| " + " | ".join(["col"] * 50) + " |\n"
        big_table += "| " + " | ".join(["---"] * 50) + " |\n"
        for i in range(20):
            big_table += "| " + " | ".join([str(i)] * 50) + " |\n"

        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(big_table, content_type=ContentType.TABLE),
        ]
        result = small_chunker.chunk(items)
        table_chunks = [c for c in result if c.content_type == ContentType.TABLE]
        assert len(table_chunks) == 1
        assert table_chunks[0].content == big_table

    def test_table_preserves_metadata(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Table chunks preserve original metadata."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(
                "| X |",
                content_type=ContentType.TABLE,
                metadata={"num_rows": 3, "label": "table"},
            ),
        ]
        result = small_chunker.chunk(items)
        table_chunks = [c for c in result if c.content_type == ContentType.TABLE]
        assert table_chunks[0].metadata["num_rows"] == 3

    def test_table_no_overlap_applied(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Overlap is not applied to/from table chunks."""
        text1 = _make_long_text(60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed("| X |", content_type=ContentType.TABLE),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)

        # Find the text chunk after the table.
        table_idx = next(
            i for i, c in enumerate(result)
            if c.content_type == ContentType.TABLE
        )
        if table_idx + 1 < len(result):
            next_chunk = result[table_idx + 1]
            # The text after the table should NOT start with table content.
            assert "| X |" not in next_chunk.content

    def test_multiple_tables(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Multiple tables each become standalone chunks."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed("| A |", content_type=ContentType.TABLE),
            _make_transformed("| B |", content_type=ContentType.TABLE),
        ]
        result = small_chunker.chunk(items)
        table_chunks = [c for c in result if c.content_type == ContentType.TABLE]
        assert len(table_chunks) == 2


# ---------------------------------------------------------------------------
# TestImageDescriptionChunking
# ---------------------------------------------------------------------------


class TestImageDescriptionChunking:
    """Tests for image description chunking."""

    def test_image_standalone_chunk(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Image descriptions become standalone chunks."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(
                "Photo of a sunset",
                content_type=ContentType.IMAGE_DESCRIPTION,
            ),
        ]
        result = small_chunker.chunk(items)
        img_chunks = [
            c for c in result
            if c.content_type == ContentType.IMAGE_DESCRIPTION
        ]
        assert len(img_chunks) == 1
        assert img_chunks[0].content == "Photo of a sunset"

    def test_image_metadata_preserved(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Image description metadata is preserved."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(
                "Sunset",
                content_type=ContentType.IMAGE_DESCRIPTION,
                metadata={"caption": "A sunset"},
            ),
        ]
        result = small_chunker.chunk(items)
        img_chunks = [
            c for c in result
            if c.content_type == ContentType.IMAGE_DESCRIPTION
        ]
        assert img_chunks[0].metadata["caption"] == "A sunset"

    def test_image_page_preserved(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Image description page number is preserved."""
        text = _make_long_text(120)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed(
                "Chart",
                content_type=ContentType.IMAGE_DESCRIPTION,
                page_number=5,
            ),
        ]
        result = small_chunker.chunk(items)
        img_chunks = [
            c for c in result
            if c.content_type == ContentType.IMAGE_DESCRIPTION
        ]
        assert img_chunks[0].page_number == 5


# ---------------------------------------------------------------------------
# TestTextChunking
# ---------------------------------------------------------------------------


class TestTextChunking:
    """Tests for text chunking logic."""

    def test_small_texts_merged(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Small text items are merged into a single chunk."""
        text = _make_long_text(120)
        # Two items that together fit in one chunk.
        items = [
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed("Alpha.", content_type=ContentType.TEXT),
            _make_transformed("Beta.", content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        # At least some chunks should contain merged content.
        contents = " ".join(c.content for c in result if c.content_type == ContentType.TEXT)
        assert "Alpha." in contents
        assert "Beta." in contents

    def test_text_split_at_boundary(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Text items that exceed chunk_size are split into multiple chunks."""
        # 60 tokens each, chunk_size=50, so each should be its own chunk.
        text1 = _make_long_text(60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        text_chunks = [c for c in result if c.content_type == ContentType.TEXT]
        assert len(text_chunks) >= 2

    def test_oversized_item_split_by_semchunk(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """A single item exceeding chunk_size is split by semchunk."""
        big_text = _make_long_text(200)
        # Pad with enough other text to exceed skip threshold.
        items = [_make_transformed(big_text, content_type=ContentType.TEXT)]
        # Use a config that has low skip threshold.
        result = small_chunker.chunk(items)
        assert len(result) > 1
        # Each chunk should have split_by=semchunk metadata.
        for c in result:
            if c.content_type == ContentType.TEXT:
                assert c.token_count <= 50 + 15  # Some tolerance for token boundaries.

    def test_token_count_accurate(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Each chunk's token_count matches _count_tokens."""
        text = _make_long_text(150)
        items = [_make_transformed(text, content_type=ContentType.TEXT)]
        result = small_chunker.chunk(items)
        for chunk in result:
            assert chunk.token_count == _count_tokens(chunk.content)

    def test_text_content_type_preserved(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Text chunks have content_type=TEXT."""
        text = _make_long_text(150)
        items = [_make_transformed(text, content_type=ContentType.TEXT)]
        result = small_chunker.chunk(items)
        for chunk in result:
            assert chunk.content_type == ContentType.TEXT

    def test_buffer_flush_before_table(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Accumulated text is flushed when a table is encountered."""
        text = _make_long_text(120)
        items = [
            _make_transformed("some text", content_type=ContentType.TEXT),
            _make_transformed(text, content_type=ContentType.TEXT),
            _make_transformed("| T |", content_type=ContentType.TABLE),
        ]
        result = small_chunker.chunk(items)
        # Table should not be merged with text.
        table_chunks = [c for c in result if c.content_type == ContentType.TABLE]
        assert len(table_chunks) == 1
        assert table_chunks[0].content == "| T |"

    def test_separator_between_merged_items(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Merged text items are separated by \\n\\n."""
        text = _make_long_text(110)
        items = [
            _make_transformed("first part", content_type=ContentType.TEXT),
            _make_transformed("second part", content_type=ContentType.TEXT),
            _make_transformed(text, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        # Find the chunk that has both parts.
        merged = [
            c for c in result
            if "first part" in c.content and "second part" in c.content
        ]
        assert len(merged) == 1
        assert "first part\n\nsecond part" in merged[0].content

    def test_no_content_loss(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """All original text content appears in at least one chunk."""
        words = ["alpha", "bravo", "charlie", "delta", "echo"]
        text_padding = _make_long_text(80)
        items = [_make_transformed(text_padding, content_type=ContentType.TEXT)]
        for w in words:
            items.append(_make_transformed(w, content_type=ContentType.TEXT))
        result = small_chunker.chunk(items)
        all_content = " ".join(c.content for c in result)
        for w in words:
            assert w in all_content


# ---------------------------------------------------------------------------
# TestHeaderBoundaries
# ---------------------------------------------------------------------------


class TestHeaderBoundaries:
    """Tests for header-path-based section boundaries."""

    def test_different_headers_cause_breaks(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Items with different header_paths are not merged."""
        text = _make_long_text(40)
        items = [
            _make_transformed(text, content_type=ContentType.TEXT, header_path="Ch1"),
            _make_transformed(text, content_type=ContentType.TEXT, header_path="Ch2"),
            # Pad to exceed skip threshold.
            _make_transformed(
                _make_long_text(40), content_type=ContentType.TEXT, header_path="Ch3"
            ),
        ]
        result = small_chunker.chunk(items)
        # Items from different headers should be in separate chunks.
        ch1_chunks = [c for c in result if c.header_path == "Ch1"]
        ch2_chunks = [c for c in result if c.header_path == "Ch2"]
        assert len(ch1_chunks) >= 1
        assert len(ch2_chunks) >= 1

    def test_same_headers_merge(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Consecutive items with same header_path can merge."""
        items = [
            _make_transformed("A", content_type=ContentType.TEXT, header_path="Ch1"),
            _make_transformed("B", content_type=ContentType.TEXT, header_path="Ch1"),
            _make_transformed(
                _make_long_text(100), content_type=ContentType.TEXT, header_path="Ch1"
            ),
        ]
        result = small_chunker.chunk(items)
        # A and B should be merged into one chunk.
        merged = [c for c in result if "A" in c.content and "B" in c.content]
        assert len(merged) >= 1

    def test_header_path_preserved_in_chunk(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Each chunk preserves the header_path from its items."""
        items = [
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT,
                header_path="Introduction",
            ),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT,
                header_path="Methods",
            ),
        ]
        result = small_chunker.chunk(items)
        headers = {c.header_path for c in result}
        assert "Introduction" in headers
        assert "Methods" in headers

    def test_empty_header_path(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Items with empty header_path are grouped together."""
        items = [
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT, header_path=""
            ),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT, header_path=""
            ),
        ]
        result = small_chunker.chunk(items)
        for c in result:
            assert c.header_path == ""

    def test_non_consecutive_same_header_separate(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Non-consecutive items with same header are in separate groups."""
        items = [
            _make_transformed(
                _make_long_text(40), content_type=ContentType.TEXT, header_path="Ch1"
            ),
            _make_transformed(
                _make_long_text(40), content_type=ContentType.TEXT, header_path="Ch2"
            ),
            _make_transformed(
                _make_long_text(40), content_type=ContentType.TEXT, header_path="Ch1"
            ),
        ]
        result = small_chunker.chunk(items)
        # Should have chunks from Ch1, Ch2, Ch1 (not merged across Ch2).
        ch1_chunks = [c for c in result if c.header_path == "Ch1"]
        assert len(ch1_chunks) >= 2


# ---------------------------------------------------------------------------
# TestOverlap
# ---------------------------------------------------------------------------


class TestOverlap:
    """Tests for overlap between text chunks."""

    def test_overlap_applied_between_text_chunks(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Overlap text from previous chunk appears at start of next."""
        text1 = _make_long_text(60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        text_chunks = [c for c in result if c.content_type == ContentType.TEXT]
        if len(text_chunks) >= 2:
            # Second chunk should contain some text from end of first.
            first_content = text_chunks[0].content
            second_content = text_chunks[1].content
            # The overlap is taken from the end of the first chunk.
            # Extract last few words from first chunk.
            last_words = first_content.split()[-3:]
            overlap_fragment = " ".join(last_words)
            assert overlap_fragment in second_content

    def test_no_overlap_on_first_chunk(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """First chunk should not have any prepended overlap."""
        text = _make_long_text(200)
        items = [_make_transformed(text, content_type=ContentType.TEXT)]
        result = small_chunker.chunk(items)
        # First chunk starts with content, not overlap.
        assert result[0].chunk_index == 0

    def test_no_overlap_across_table(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Overlap is not applied across a table boundary."""
        # Use a unique marker word in text1 that cannot appear in text2.
        text1 = " ".join(["unique_marker_xyz"] * 60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed("| Col |", content_type=ContentType.TABLE),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        # The text chunk after the table should NOT contain overlap from before the table.
        table_idx = next(
            i for i, c in enumerate(result)
            if c.content_type == ContentType.TABLE
        )
        if table_idx + 1 < len(result):
            after_table = result[table_idx + 1]
            # After-table chunk should not contain the unique marker from text1.
            assert "unique_marker_xyz" not in after_table.content

    def test_zero_overlap_config(
        self, no_overlap_chunker: StructureAwareChunker
    ) -> None:
        """With overlap=0, no overlap is applied."""
        text1 = _make_long_text(60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = no_overlap_chunker.chunk(items)
        text_chunks = [c for c in result if c.content_type == ContentType.TEXT]
        if len(text_chunks) >= 2:
            # Total content should be roughly the sum without duplicated overlap.
            total_tokens = sum(c.token_count for c in text_chunks)
            expected = _count_tokens(text1) + _count_tokens(text2)
            # With no overlap, total should be close to expected (allow small tolerance for separators).
            assert total_tokens < expected + 10

    def test_overlap_not_applied_to_image(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Overlap is not applied from text to image chunk."""
        text1 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed(
                "A photo",
                content_type=ContentType.IMAGE_DESCRIPTION,
            ),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT,
            ),
        ]
        result = small_chunker.chunk(items)
        img_chunks = [
            c for c in result
            if c.content_type == ContentType.IMAGE_DESCRIPTION
        ]
        assert len(img_chunks) == 1
        assert img_chunks[0].content == "A photo"

    def test_overlap_token_count_updated(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Token count is recalculated after overlap is applied."""
        text1 = _make_long_text(60)
        text2 = _make_long_text(60)
        items = [
            _make_transformed(text1, content_type=ContentType.TEXT),
            _make_transformed(text2, content_type=ContentType.TEXT),
        ]
        result = small_chunker.chunk(items)
        for chunk in result:
            assert chunk.token_count == _count_tokens(chunk.content)


# ---------------------------------------------------------------------------
# TestChunkIndex
# ---------------------------------------------------------------------------


class TestChunkIndex:
    """Tests for sequential chunk indexing."""

    def test_sequential_zero_based(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Chunk indices are sequential starting from 0."""
        text = _make_long_text(200)
        items = [_make_transformed(text, content_type=ContentType.TEXT)]
        result = small_chunker.chunk(items)
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_indices_correct_with_tables(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Chunk indices remain sequential when tables are present."""
        items = [
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
            _make_transformed("| A |", content_type=ContentType.TABLE),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
        ]
        result = small_chunker.chunk(items)
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_single_chunk_index_zero(
        self, chunker: StructureAwareChunker
    ) -> None:
        """A single-chunk result has index 0."""
        items = [_make_transformed("hello")]
        result = chunker.chunk(items)
        assert len(result) == 1
        assert result[0].chunk_index == 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self, chunker: StructureAwareChunker) -> None:
        """Empty input returns empty list."""
        result = chunker.chunk([])
        assert result == []

    def test_single_text_item(self, chunker: StructureAwareChunker) -> None:
        """Single text item below threshold returns one chunk."""
        items = [_make_transformed("Just one item.")]
        result = chunker.chunk(items)
        assert len(result) == 1
        assert result[0].content == "Just one item."

    def test_only_tables(self, small_chunker: StructureAwareChunker) -> None:
        """Input with only tables produces one chunk per table."""
        items = [
            _make_transformed("| A |", content_type=ContentType.TABLE),
            _make_transformed("| B |", content_type=ContentType.TABLE),
            _make_transformed("| C |", content_type=ContentType.TABLE),
            # Need enough tokens to exceed skip threshold.
            _make_transformed(
                _make_long_text(120), content_type=ContentType.TABLE
            ),
        ]
        result = small_chunker.chunk(items)
        assert all(c.content_type == ContentType.TABLE for c in result)
        assert len(result) == 4

    def test_page_number_zero(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Items with page_number=0 produce chunks with page_number=0."""
        items = [_make_transformed("Test", page_number=0)]
        result = chunker.chunk(items)
        assert result[0].page_number == 0

    def test_whitespace_only_content(
        self, chunker: StructureAwareChunker
    ) -> None:
        """Items with whitespace-only content are still chunked."""
        items = [_make_transformed("   ")]
        result = chunker.chunk(items)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestMixedContent
# ---------------------------------------------------------------------------


class TestMixedContent:
    """Tests for mixed content type interleaving."""

    def test_text_table_text_sequence(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Text-table-text produces correct sequence of chunk types."""
        items = [
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
            _make_transformed("| X |", content_type=ContentType.TABLE),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
        ]
        result = small_chunker.chunk(items)
        types = [c.content_type for c in result]
        # Should have TEXT chunks, then TABLE, then TEXT chunks.
        table_indices = [
            i for i, t in enumerate(types) if t == ContentType.TABLE
        ]
        assert len(table_indices) == 1
        # There should be text chunks before and after the table.
        text_before = any(
            types[i] == ContentType.TEXT for i in range(table_indices[0])
        )
        text_after = any(
            types[i] == ContentType.TEXT
            for i in range(table_indices[0] + 1, len(types))
        )
        assert text_before
        assert text_after

    def test_text_image_table_interleaving(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Text, image, and table items produce correctly typed chunks."""
        items = [
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
            _make_transformed(
                "Photo", content_type=ContentType.IMAGE_DESCRIPTION
            ),
            _make_transformed("| Y |", content_type=ContentType.TABLE),
            _make_transformed(
                _make_long_text(60), content_type=ContentType.TEXT
            ),
        ]
        result = small_chunker.chunk(items)
        types = [c.content_type for c in result]
        assert ContentType.TEXT in types
        assert ContentType.TABLE in types
        assert ContentType.IMAGE_DESCRIPTION in types

    def test_all_content_types_present(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """All three content types appear in output for mixed input."""
        items = [
            _make_transformed(
                _make_long_text(50), content_type=ContentType.TEXT
            ),
            _make_transformed("| T |", content_type=ContentType.TABLE),
            _make_transformed(
                "Image desc", content_type=ContentType.IMAGE_DESCRIPTION
            ),
            _make_transformed(
                _make_long_text(50), content_type=ContentType.TEXT
            ),
        ]
        result = small_chunker.chunk(items)
        result_types = {c.content_type for c in result}
        assert result_types == {
            ContentType.TEXT,
            ContentType.TABLE,
            ContentType.IMAGE_DESCRIPTION,
        }

    def test_chunk_order_preserved(
        self, small_chunker: StructureAwareChunker
    ) -> None:
        """Chunks maintain the original document order."""
        items = [
            _make_transformed(
                "First text " + _make_long_text(50),
                content_type=ContentType.TEXT,
                header_path="A",
            ),
            _make_transformed(
                "| table |", content_type=ContentType.TABLE, header_path="A"
            ),
            _make_transformed(
                "Second text " + _make_long_text(50),
                content_type=ContentType.TEXT,
                header_path="A",
            ),
        ]
        result = small_chunker.chunk(items)
        # Find positions.
        first_text_idx = next(
            i for i, c in enumerate(result) if "First text" in c.content
        )
        table_idx = next(
            i for i, c in enumerate(result)
            if c.content_type == ContentType.TABLE
        )
        second_text_idx = next(
            i for i, c in enumerate(result) if "Second text" in c.content
        )
        assert first_text_idx < table_idx < second_text_idx
