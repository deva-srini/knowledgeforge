"""Tests for the content transformation service."""

import unicodedata
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.extraction import ContentType, ExtractedContent
from app.services.transformation import (
    ContentTransformer,
    TransformedContent,
    TransformResult,
    _UNICODE_REPLACEMENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Default config with organisation enabled."""
    return KnowledgeForgeConfig()


@pytest.fixture
def disabled_config() -> KnowledgeForgeConfig:
    """Config with organisation disabled."""
    return KnowledgeForgeConfig(
        processing={"organisation": {"enabled": False}}
    )


@pytest.fixture
def transformer(config: KnowledgeForgeConfig) -> ContentTransformer:
    """ContentTransformer with default (enabled) config."""
    return ContentTransformer(config)


@pytest.fixture
def disabled_transformer(
    disabled_config: KnowledgeForgeConfig,
) -> ContentTransformer:
    """ContentTransformer with organisation disabled."""
    return ContentTransformer(disabled_config)


def _make_extracted(
    content: str = "hello",
    content_type: ContentType = ContentType.TEXT,
    page_number: int = 1,
    header_path: str = "Section 1",
    metadata: dict | None = None,
) -> ExtractedContent:
    """Helper to build ExtractedContent with sensible defaults."""
    return ExtractedContent(
        content=content,
        content_type=content_type,
        page_number=page_number,
        header_path=header_path,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# TestTransformedContentDataclass
# ---------------------------------------------------------------------------


class TestTransformedContentDataclass:
    """Tests for the TransformedContent dataclass."""

    def test_default_values(self) -> None:
        """Construct with minimal args, verify defaults."""
        tc = TransformedContent(content="text", content_type=ContentType.TEXT)
        assert tc.content == "text"
        assert tc.content_type == ContentType.TEXT
        assert tc.page_number == 0
        assert tc.header_path == ""
        assert tc.metadata == {}

    def test_full_construction(self) -> None:
        """Construct with all fields populated."""
        meta = {"key": "value"}
        tc = TransformedContent(
            content="hello",
            content_type=ContentType.TABLE,
            page_number=5,
            header_path="Ch1 > Sec2",
            metadata=meta,
        )
        assert tc.content == "hello"
        assert tc.content_type == ContentType.TABLE
        assert tc.page_number == 5
        assert tc.header_path == "Ch1 > Sec2"
        assert tc.metadata == {"key": "value"}

    def test_content_type_matches_extracted(self) -> None:
        """ContentType enum values are shared with extraction module."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.TABLE.value == "table"
        assert ContentType.IMAGE_DESCRIPTION.value == "image_description"

    def test_metadata_independence(self) -> None:
        """Two instances do not share metadata dicts."""
        a = TransformedContent(content="a", content_type=ContentType.TEXT)
        b = TransformedContent(content="b", content_type=ContentType.TEXT)
        a.metadata["x"] = 1
        assert "x" not in b.metadata


# ---------------------------------------------------------------------------
# TestPassThrough
# ---------------------------------------------------------------------------


class TestPassThrough:
    """Tests for disabled/pass-through behaviour."""

    def test_disabled_passes_through_text(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """Text content unchanged when disabled."""
        items = [_make_extracted(content="  hello  world  ")]
        result = disabled_transformer.transform(items)
        assert result.items[0].content == "  hello  world  "

    def test_disabled_passes_through_table(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """Table content unchanged when disabled."""
        raw = "|  A |B  |\n|---|---|\n| 1| 2|"
        items = [_make_extracted(content=raw, content_type=ContentType.TABLE)]
        result = disabled_transformer.transform(items)
        assert result.items[0].content == raw

    def test_disabled_passes_through_image(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """Image description unchanged when disabled."""
        items = [
            _make_extracted(
                content="A photo\u2019s caption",
                content_type=ContentType.IMAGE_DESCRIPTION,
            )
        ]
        result = disabled_transformer.transform(items)
        assert result.items[0].content == "A photo\u2019s caption"

    def test_disabled_preserves_all_fields(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """page_number, header_path, and original metadata are preserved."""
        items = [
            _make_extracted(
                page_number=7,
                header_path="Ch > Sec",
                metadata={"source": "test"},
            )
        ]
        result = disabled_transformer.transform(items)
        r = result.items[0]
        assert r.page_number == 7
        assert r.header_path == "Ch > Sec"
        assert r.metadata["source"] == "test"

    def test_disabled_metadata_has_transformed_false(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """metadata['transformed'] is False when disabled."""
        items = [_make_extracted()]
        result = disabled_transformer.transform(items)
        assert result.items[0].metadata["transformed"] is False


# ---------------------------------------------------------------------------
# TestWhitespaceCleaning
# ---------------------------------------------------------------------------


class TestWhitespaceCleaning:
    """Tests for _clean_whitespace."""

    def test_trailing_spaces_removed(
        self, transformer: ContentTransformer
    ) -> None:
        """Lines with trailing spaces are trimmed."""
        result = transformer._clean_whitespace("hello   \nworld  ")
        assert result == "hello\nworld"

    def test_multiple_blank_lines_collapsed(
        self, transformer: ContentTransformer
    ) -> None:
        """3+ blank lines become one blank line."""
        result = transformer._clean_whitespace("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_crlf_normalized_to_lf(
        self, transformer: ContentTransformer
    ) -> None:
        """CRLF and CR replaced with LF."""
        result = transformer._clean_whitespace("a\r\nb\rc")
        assert result == "a\nb\nc"

    def test_multiple_spaces_collapsed(
        self, transformer: ContentTransformer
    ) -> None:
        """Multiple spaces within a line become one."""
        result = transformer._clean_whitespace("hello    world")
        assert result == "hello world"

    def test_leading_trailing_stripped(
        self, transformer: ContentTransformer
    ) -> None:
        """Entire string leading/trailing whitespace removed."""
        result = transformer._clean_whitespace("\n\n  hello  \n\n")
        assert result == "hello"

    def test_tabs_normalized(
        self, transformer: ContentTransformer
    ) -> None:
        """Tab characters normalized to single space."""
        result = transformer._clean_whitespace("hello\t\tworld")
        assert result == "hello world"

    def test_already_clean_text_unchanged(
        self, transformer: ContentTransformer
    ) -> None:
        """Clean text passes through without modification."""
        result = transformer._clean_whitespace("hello world")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# TestEncodingNormalization
# ---------------------------------------------------------------------------


class TestEncodingNormalization:
    """Tests for _normalize_encoding."""

    def test_smart_quotes_replaced(
        self, transformer: ContentTransformer
    ) -> None:
        """Smart single and double quotes become ASCII."""
        text = "\u2018hello\u2019 \u201cworld\u201d"
        result = transformer._normalize_encoding(text)
        assert result == "'hello' \"world\""

    def test_em_dash_replaced(
        self, transformer: ContentTransformer
    ) -> None:
        """Em dash becomes hyphen."""
        result = transformer._normalize_encoding("word\u2014word")
        assert result == "word-word"

    def test_en_dash_replaced(
        self, transformer: ContentTransformer
    ) -> None:
        """En dash becomes hyphen."""
        result = transformer._normalize_encoding("1\u20132")
        assert result == "1-2"

    def test_ellipsis_replaced(
        self, transformer: ContentTransformer
    ) -> None:
        """Horizontal ellipsis becomes three dots."""
        result = transformer._normalize_encoding("wait\u2026")
        assert result == "wait..."

    def test_non_breaking_space_replaced(
        self, transformer: ContentTransformer
    ) -> None:
        """Non-breaking space becomes regular space."""
        result = transformer._normalize_encoding("hello\u00a0world")
        assert result == "hello world"

    def test_zero_width_chars_removed(
        self, transformer: ContentTransformer
    ) -> None:
        """Zero-width characters are removed."""
        result = transformer._normalize_encoding(
            "he\u200bll\u200co\u200d\ufeff"
        )
        assert result == "hello"

    def test_nfc_normalization(
        self, transformer: ContentTransformer
    ) -> None:
        """Decomposed Unicode normalized to NFC (composed form)."""
        # e + combining acute accent (NFD) -> e-acute (NFC)
        decomposed = "e\u0301"
        result = transformer._normalize_encoding(decomposed)
        expected = unicodedata.normalize("NFC", decomposed)
        assert result == expected
        assert len(result) == 1  # single composed character


# ---------------------------------------------------------------------------
# TestTableTransformation
# ---------------------------------------------------------------------------


class TestTableTransformation:
    """Tests for table-specific transformations."""

    def test_table_cell_padding_normalized(
        self, transformer: ContentTransformer
    ) -> None:
        """Inconsistent padding becomes consistent '| cell |'."""
        raw = "|  Name|Age  |\n|---|---|\n|  Alice|30  |"
        result = transformer._transform_table(raw)
        lines = result.split("\n")
        assert lines[0] == "| Name | Age |"
        assert lines[2] == "| Alice | 30 |"

    def test_table_empty_rows_removed(
        self, transformer: ContentTransformer
    ) -> None:
        """Blank lines within table are removed."""
        raw = "| A | B |\n\n|---|---|\n\n| 1 | 2 |"
        result = transformer._transform_table(raw)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "" not in lines

    def test_table_separator_row_preserved(
        self, transformer: ContentTransformer
    ) -> None:
        """Separator row is kept with consistent formatting."""
        raw = "| H1 | H2 |\n|---|---|\n| a | b |"
        result = transformer._transform_table(raw)
        lines = result.split("\n")
        assert "---" in lines[1]
        assert lines[1].startswith("|")
        assert lines[1].endswith("|")

    def test_table_content_cleaned(
        self, transformer: ContentTransformer
    ) -> None:
        """Cell text has whitespace trimmed."""
        raw = "|   hello   |   world   |\n|---|---|\n|  a  |  b  |"
        result = transformer._transform_table(raw)
        lines = result.split("\n")
        assert lines[0] == "| hello | world |"

    def test_table_encoding_cleaned(
        self, transformer: ContentTransformer
    ) -> None:
        """Unicode artifacts in table cells are normalized."""
        raw = "| Name | Quote |\n|---|---|\n| Alice | \u201chello\u201d |"
        result = transformer._transform_table(raw)
        assert '"hello"' in result

    def test_simple_markdown_table_roundtrip(
        self, transformer: ContentTransformer
    ) -> None:
        """A well-formed table passes through cleanly."""
        clean = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        result = transformer._transform_table(clean)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "| A | B |"
        assert lines[2] == "| 1 | 2 |"


# ---------------------------------------------------------------------------
# TestTextTransformation
# ---------------------------------------------------------------------------


class TestTextTransformation:
    """Tests for text-specific transformations."""

    def test_text_whitespace_and_encoding_combined(
        self, transformer: ContentTransformer
    ) -> None:
        """Both cleaning passes applied together."""
        raw = "  \u201chello\u201d    world  \n\n\n\n  end  "
        result = transformer._transform_text(raw)
        assert result == '"hello" world\n\nend'

    def test_empty_text_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """Empty string input produces empty string."""
        result = transformer._transform_text("")
        assert result == ""

    def test_whitespace_only_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """Whitespace-only input produces empty string."""
        result = transformer._transform_text("   \n\n\t  ")
        assert result == ""

    def test_multiline_text_cleaned(
        self, transformer: ContentTransformer
    ) -> None:
        """Multi-paragraph text cleaned properly."""
        raw = "First paragraph.  \n\nSecond paragraph.   \n\n\n\nThird."
        result = transformer._transform_text(raw)
        assert result == "First paragraph.\n\nSecond paragraph.\n\nThird."


# ---------------------------------------------------------------------------
# TestImageDescriptionTransformation
# ---------------------------------------------------------------------------


class TestImageDescriptionTransformation:
    """Tests for image description transformations."""

    def test_image_description_cleaned(
        self, transformer: ContentTransformer
    ) -> None:
        """Image descriptions get text cleaning."""
        items = [
            _make_extracted(
                content="  A photo   of a cat  ",
                content_type=ContentType.IMAGE_DESCRIPTION,
            )
        ]
        result = transformer.transform(items)
        assert result.items[0].content == "A photo of a cat"

    def test_image_encoding_normalized(
        self, transformer: ContentTransformer
    ) -> None:
        """Unicode in image descriptions is normalized."""
        items = [
            _make_extracted(
                content="Cat\u2019s \u201cphoto\u201d",
                content_type=ContentType.IMAGE_DESCRIPTION,
            )
        ]
        result = transformer.transform(items)
        assert result.items[0].content == "Cat's \"photo\""


# ---------------------------------------------------------------------------
# TestTransformBatchBehavior
# ---------------------------------------------------------------------------


class TestTransformBatchBehavior:
    """Tests for batch transform behaviour."""

    def test_empty_list_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """transform([]) returns empty items list."""
        result = transformer.transform([])
        assert result.items == []

    def test_multiple_items_all_transformed(
        self, transformer: ContentTransformer
    ) -> None:
        """List of mixed types all come back as TransformedContent."""
        items = [
            _make_extracted(content="text", content_type=ContentType.TEXT),
            _make_extracted(
                content="| A |\n|---|\n| 1 |",
                content_type=ContentType.TABLE,
            ),
            _make_extracted(
                content="image desc",
                content_type=ContentType.IMAGE_DESCRIPTION,
            ),
        ]
        result = transformer.transform(items)
        assert len(result.items) == 3
        assert all(isinstance(r, TransformedContent) for r in result.items)

    def test_output_length_matches_input(
        self, transformer: ContentTransformer
    ) -> None:
        """Output list has same length as input."""
        items = [_make_extracted() for _ in range(5)]
        result = transformer.transform(items)
        assert len(result.items) == len(items)

    def test_metadata_has_transformed_true(
        self, transformer: ContentTransformer
    ) -> None:
        """All items have metadata['transformed'] == True when enabled."""
        items = [_make_extracted(), _make_extracted()]
        result = transformer.transform(items)
        assert all(r.metadata["transformed"] is True for r in result.items)

    def test_original_items_not_mutated(
        self, transformer: ContentTransformer
    ) -> None:
        """Original ExtractedContent metadata dicts are not modified."""
        original_meta = {"source": "test"}
        items = [_make_extracted(metadata=dict(original_meta))]
        transformer.transform(items)
        # Original item should not have "transformed" key
        assert "transformed" not in items[0].metadata


# ---------------------------------------------------------------------------
# TestPerItemErrorHandling
# ---------------------------------------------------------------------------


class TestPerItemErrorHandling:
    """Tests for per-item error handling."""

    def test_error_in_one_item_does_not_block_others(
        self, transformer: ContentTransformer
    ) -> None:
        """If one item causes an exception, others still process."""
        good = _make_extracted(content="  hello  ")
        bad = _make_extracted(content="fine")

        with patch.object(
            transformer,
            "_transform_item",
            side_effect=[ValueError("boom"), TransformedContent(
                content="fine",
                content_type=ContentType.TEXT,
                page_number=1,
                header_path="Section 1",
                metadata={"transformed": True},
            )],
        ):
            result = transformer.transform([good, bad])

        assert len(result.items) == 2
        # First item fell back to pass-through
        assert result.items[0].metadata["transformed"] is False
        # Second item was transformed normally
        assert result.items[1].metadata["transformed"] is True

    def test_error_item_passes_through_unchanged(
        self, transformer: ContentTransformer
    ) -> None:
        """Failed item falls back to pass-through with original content."""
        item = _make_extracted(content="original text", page_number=3)

        with patch.object(
            transformer,
            "_transform_item",
            side_effect=RuntimeError("encoding crash"),
        ):
            result = transformer.transform([item])

        assert len(result.items) == 1
        assert result.items[0].content == "original text"
        assert result.items[0].page_number == 3
        assert result.items[0].metadata["transformed"] is False


# ---------------------------------------------------------------------------
# TestTransformResultDataclass
# ---------------------------------------------------------------------------


class TestTransformResultDataclass:
    """Tests for the TransformResult dataclass."""

    def test_default_document_markdown(self) -> None:
        """document_markdown defaults to empty string."""
        tr = TransformResult(items=[])
        assert tr.items == []
        assert tr.document_markdown == ""

    def test_full_construction(self) -> None:
        """Construct with items and markdown populated."""
        item = TransformedContent(
            content="hello", content_type=ContentType.TEXT
        )
        tr = TransformResult(
            items=[item], document_markdown="# Heading\n\nBody"
        )
        assert len(tr.items) == 1
        assert tr.items[0].content == "hello"
        assert tr.document_markdown == "# Heading\n\nBody"


# ---------------------------------------------------------------------------
# TestMarkdownGeneration
# ---------------------------------------------------------------------------


@pytest.fixture
def md_disabled_config() -> KnowledgeForgeConfig:
    """Config with organisation enabled but markdown_generation disabled."""
    return KnowledgeForgeConfig(
        processing={"organisation": {"enabled": True, "markdown_generation": False}}
    )


@pytest.fixture
def md_disabled_transformer(
    md_disabled_config: KnowledgeForgeConfig,
) -> ContentTransformer:
    """ContentTransformer with markdown_generation disabled."""
    return ContentTransformer(md_disabled_config)


class TestMarkdownGeneration:
    """Tests for structured markdown generation from raw documents."""

    def test_generates_markdown_from_raw_document(
        self, transformer: ContentTransformer
    ) -> None:
        """Calls export_to_markdown with correct parameters."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Title\n\nContent"

        result = transformer.transform([], raw_document=mock_doc)

        mock_doc.export_to_markdown.assert_called_once_with(
            image_placeholder="<!-- image -->",
            page_break_placeholder="<!-- page break -->",
            escape_underscores=False,
            escape_html=False,
        )
        assert result.document_markdown == "# Title\n\nContent"

    def test_empty_when_no_raw_document(
        self, transformer: ContentTransformer
    ) -> None:
        """document_markdown is empty when raw_document is None."""
        result = transformer.transform([])
        assert result.document_markdown == ""

    def test_empty_when_raw_document_omitted(
        self, transformer: ContentTransformer
    ) -> None:
        """document_markdown is empty when raw_document is not passed."""
        result = transformer.transform([_make_extracted()])
        assert result.document_markdown == ""

    def test_empty_when_organisation_disabled(
        self, disabled_transformer: ContentTransformer
    ) -> None:
        """document_markdown is empty when organisation is disabled."""
        mock_doc = MagicMock()
        result = disabled_transformer.transform([], raw_document=mock_doc)
        assert result.document_markdown == ""
        mock_doc.export_to_markdown.assert_not_called()

    def test_empty_when_markdown_generation_disabled(
        self, md_disabled_transformer: ContentTransformer
    ) -> None:
        """document_markdown is empty when markdown_generation is False."""
        mock_doc = MagicMock()
        result = md_disabled_transformer.transform([], raw_document=mock_doc)
        assert result.document_markdown == ""
        mock_doc.export_to_markdown.assert_not_called()

    def test_cleaning_applied_to_markdown(
        self, transformer: ContentTransformer
    ) -> None:
        """Encoding normalization and whitespace cleaning are applied."""
        mock_doc = MagicMock()
        # Smart quotes and extra whitespace
        mock_doc.export_to_markdown.return_value = (
            "# Title\n\n\n\n\u201cHello\u201d   world  \n\n\nEnd"
        )
        result = transformer.transform([], raw_document=mock_doc)
        assert result.document_markdown == '# Title\n\n"Hello" world\n\nEnd'

    def test_export_failure_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """Exception in export_to_markdown returns empty markdown."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.side_effect = RuntimeError("export failed")
        result = transformer.transform([_make_extracted()], raw_document=mock_doc)
        assert result.document_markdown == ""

    def test_export_failure_does_not_affect_items(
        self, transformer: ContentTransformer
    ) -> None:
        """Per-item results are unaffected when markdown generation fails."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.side_effect = RuntimeError("export failed")
        items = [_make_extracted(content="hello")]
        result = transformer.transform(items, raw_document=mock_doc)
        assert len(result.items) == 1
        assert result.items[0].metadata["transformed"] is True
        assert result.document_markdown == ""

    def test_empty_export_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """Empty string from export returns empty markdown."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = ""
        result = transformer.transform([], raw_document=mock_doc)
        assert result.document_markdown == ""

    def test_whitespace_only_export_returns_empty(
        self, transformer: ContentTransformer
    ) -> None:
        """Whitespace-only export returns empty markdown."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "   \n\n\t  "
        result = transformer.transform([], raw_document=mock_doc)
        assert result.document_markdown == ""

    def test_per_item_results_unaffected_by_markdown(
        self, transformer: ContentTransformer
    ) -> None:
        """Per-item transform results are the same with or without markdown."""
        items = [
            _make_extracted(content="  hello  ", content_type=ContentType.TEXT),
        ]
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Doc"

        result = transformer.transform(items, raw_document=mock_doc)
        assert result.items[0].content == "hello"
        assert result.document_markdown == "# Doc"

    def test_markdown_independent_of_per_item_errors(
        self, transformer: ContentTransformer
    ) -> None:
        """Markdown generation succeeds even if per-item transform fails."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Heading"
        item = _make_extracted(content="text")

        with patch.object(
            transformer,
            "_transform_item",
            side_effect=ValueError("item boom"),
        ):
            result = transformer.transform([item], raw_document=mock_doc)

        assert result.items[0].metadata["transformed"] is False
        assert result.document_markdown == "# Heading"

    def test_preserves_heading_hierarchy(
        self, transformer: ContentTransformer
    ) -> None:
        """Heading levels (#, ##, ###) are preserved in output."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = (
            "# H1\n\n## H2\n\n### H3\n\nParagraph"
        )
        result = transformer.transform([], raw_document=mock_doc)
        assert "# H1" in result.document_markdown
        assert "## H2" in result.document_markdown
        assert "### H3" in result.document_markdown

    def test_preserves_image_and_page_break_placeholders(
        self, transformer: ContentTransformer
    ) -> None:
        """Image and page break placeholders are kept in output."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = (
            "# Title\n\n<!-- image -->\n\n<!-- page break -->\n\nEnd"
        )
        result = transformer.transform([], raw_document=mock_doc)
        assert "<!-- image -->" in result.document_markdown
        assert "<!-- page break -->" in result.document_markdown
