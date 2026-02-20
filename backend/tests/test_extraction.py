"""Unit tests for the content extraction service."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import KnowledgeForgeConfig
from app.services.extraction import (
    ContentExtractor,
    ContentType,
    ExtractedContent,
)
from app.services.parsing import DocumentParser, ParseResult

# Paths to reference files
REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "reference"
INCOME_CSV = REFERENCE_DIR / "household_income_survey(2015).csv"
BGF_PDF = REFERENCE_DIR / "bgf_factsheet.pdf"


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Create a default test configuration."""
    return KnowledgeForgeConfig()


@pytest.fixture
def extractor(config: KnowledgeForgeConfig) -> ContentExtractor:
    """Create a ContentExtractor with default config."""
    return ContentExtractor(config)


@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    """Create a sample HTML file with headings, text, and a table."""
    html_file = tmp_path / "sample.html"
    html_file.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Test Document</title></head>
<body>
<h1>Chapter 1: Introduction</h1>
<p>This is the first paragraph of the document.</p>
<p>This is the second paragraph with more content.</p>
<h2>Section 1.1: Background</h2>
<p>Background information goes here.</p>
<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>
<h2>Section 1.2: Summary</h2>
<p>Final summary paragraph.</p>
</body>
</html>"""
    )
    return html_file


class TestFallbackExtraction:
    """Tests for raw text fallback extraction."""

    def test_fallback_with_raw_text(self, extractor: ContentExtractor) -> None:
        """Test extraction from a ParseResult without a Docling document."""
        parse_result = ParseResult(
            page_count=1,
            estimated_token_count=50,
            content_types={"text": 1},
            raw_document=None,
            raw_text="This is plain text content from a fallback parse.",
        )
        results = extractor.extract(parse_result)

        assert len(results) == 1
        assert results[0].content_type == ContentType.TEXT
        assert results[0].page_number == 1
        assert "plain text" in results[0].content
        assert results[0].metadata["source"] == "fallback"

    def test_fallback_with_empty_text(self, extractor: ContentExtractor) -> None:
        """Test extraction from an empty fallback ParseResult."""
        parse_result = ParseResult(
            page_count=0,
            estimated_token_count=0,
            raw_document=None,
            raw_text="",
        )
        results = extractor.extract(parse_result)
        assert results == []


class TestHTMLExtraction:
    """Tests for extraction from parsed HTML documents."""

    def test_extract_html_text_and_table(
        self,
        extractor: ContentExtractor,
        sample_html_file: Path,
        config: KnowledgeForgeConfig,
    ) -> None:
        """Test that HTML extraction produces both text and table items."""
        parser = DocumentParser(config)
        parse_result = parser.parse(str(sample_html_file))
        results = extractor.extract(parse_result)

        assert len(results) > 0
        types = {r.content_type for r in results}
        assert ContentType.TEXT in types

        table_items = [r for r in results if r.content_type == ContentType.TABLE]
        assert len(table_items) >= 1
        table_content = table_items[0].content
        assert "Alpha" in table_content or "Name" in table_content

    def test_extract_html_header_paths(
        self,
        extractor: ContentExtractor,
        sample_html_file: Path,
        config: KnowledgeForgeConfig,
    ) -> None:
        """Test that extracted items have heading hierarchy paths."""
        parser = DocumentParser(config)
        parse_result = parser.parse(str(sample_html_file))
        results = extractor.extract(parse_result)

        items_with_headers = [r for r in results if r.header_path]
        assert len(items_with_headers) > 0


@pytest.mark.skipif(not INCOME_CSV.exists(), reason="CSV file not found")
class TestCSVFallbackExtraction:
    """Tests for extraction from CSV via fallback parsing."""

    def test_extract_csv_fallback_returns_single_item(
        self,
        extractor: ContentExtractor,
        config: KnowledgeForgeConfig,
    ) -> None:
        """Test that CSV fallback extraction produces a single text item."""
        parser = DocumentParser(config)
        parse_result = parser.parse(str(INCOME_CSV))
        results = extractor.extract(parse_result)

        assert len(results) == 1
        assert results[0].content_type == ContentType.TEXT
        assert results[0].page_number == 1
        assert results[0].metadata["source"] == "fallback"
        assert len(results[0].content) > 0


@pytest.mark.skipif(not BGF_PDF.exists(), reason="BGF factsheet PDF not found")
class TestBGFFactsheetExtraction:
    """Tests for extraction from the BGF factsheet PDF (tables, pictures, text)."""

    @pytest.fixture
    def bgf_results(
        self, extractor: ContentExtractor, config: KnowledgeForgeConfig
    ) -> list[ExtractedContent]:
        """Parse and extract the BGF factsheet once for all tests."""
        parser = DocumentParser(config)
        parse_result = parser.parse(str(BGF_PDF))
        return extractor.extract(parse_result)

    def test_extract_bgf_pdf_table_count_and_dimensions(
        self, bgf_results: list[ExtractedContent]
    ) -> None:
        """Test that 4 tables are extracted with row/column dimensions."""
        table_items = [
            r for r in bgf_results if r.content_type == ContentType.TABLE
        ]
        assert len(table_items) == 4

        for table in table_items:
            assert "|" in table.content
            assert "num_rows" in table.metadata
            assert "num_cols" in table.metadata
            assert table.metadata["num_rows"] > 0
            assert table.metadata["num_cols"] > 0

    def test_extract_bgf_pdf_image_count(
        self, bgf_results: list[ExtractedContent]
    ) -> None:
        """Test that 10 image items are extracted from the factsheet."""
        image_items = [
            r for r in bgf_results
            if r.content_type == ContentType.IMAGE_DESCRIPTION
        ]
        assert len(image_items) == 10

    def test_extract_bgf_pdf_page_numbers(
        self, bgf_results: list[ExtractedContent]
    ) -> None:
        """Test that extracted content spans all 4 pages."""
        pages = {r.page_number for r in bgf_results}
        assert pages == {1, 2, 3, 4}

    def test_extract_bgf_pdf_text_items(
        self, bgf_results: list[ExtractedContent]
    ) -> None:
        """Test that substantial text content is extracted with header paths."""
        text_items = [
            r for r in bgf_results if r.content_type == ContentType.TEXT
        ]
        assert len(text_items) > 10

        items_with_headers = [r for r in bgf_results if r.header_path]
        assert len(items_with_headers) > 0


class TestPictureImageSaving:
    """Tests for Option A (save_picture_images) and Option B (describe_pictures)."""

    def _make_picture_item(self, caption: str = "", pil_image: object = None) -> MagicMock:
        """Build a minimal mock Docling PictureItem."""
        item = MagicMock()
        item.label = SimpleNamespace(value="picture")
        item.prov = [SimpleNamespace(page_no=1)]
        item.caption_text = MagicMock(return_value=caption)
        item.meta = None
        item.get_image = MagicMock(return_value=pil_image)
        return item

    def _make_parse_result_with_picture(self, picture_item: MagicMock) -> ParseResult:
        """Build a ParseResult whose raw_document yields a single picture."""
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [(picture_item, 1)]
        return ParseResult(
            page_count=1,
            estimated_token_count=10,
            raw_document=mock_doc,
            raw_text="",
        )

    def test_source_path_not_required_when_save_disabled(self) -> None:
        """extract() works without source_path when save_picture_images is False."""
        config = KnowledgeForgeConfig()
        extractor = ContentExtractor(config)
        picture_item = self._make_picture_item(caption="A test chart")
        parse_result = self._make_parse_result_with_picture(picture_item)

        results = extractor.extract(parse_result)  # no source_path

        assert len(results) == 1
        assert results[0].content_type == ContentType.IMAGE_DESCRIPTION
        assert "image_path" not in results[0].metadata

    def test_save_picture_image_writes_file(self, tmp_path: Path) -> None:
        """When save_picture_images=True, a PNG is written and image_path set."""
        from PIL import Image as PILImage

        pil_image = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
        picture_item = self._make_picture_item(caption="Fund NAV chart", pil_image=pil_image)
        parse_result = self._make_parse_result_with_picture(picture_item)

        config = KnowledgeForgeConfig(
            processing={
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": str(tmp_path / "images"),
                }
            }
        )
        extractor = ContentExtractor(config)
        source_path = "/data/source/factsheets/bgf_factsheet.pdf"

        results = extractor.extract(parse_result, source_path=source_path)

        assert len(results) == 1
        result = results[0]
        assert result.content_type == ContentType.IMAGE_DESCRIPTION
        assert "image_path" in result.metadata

        saved_path = Path(result.metadata["image_path"])
        assert saved_path.exists()
        assert saved_path.suffix == ".png"
        # Sub-folder is derived from the source file stem
        assert saved_path.parent.name == "bgf_factsheet"
        assert "Fund NAV chart" in result.content

    def test_save_picture_image_skips_when_get_image_returns_none(
        self, tmp_path: Path
    ) -> None:
        """When get_image() returns None, image_path is not added to metadata."""
        picture_item = self._make_picture_item(pil_image=None)
        parse_result = self._make_parse_result_with_picture(picture_item)

        config = KnowledgeForgeConfig(
            processing={
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": str(tmp_path / "images"),
                }
            }
        )
        extractor = ContentExtractor(config)

        results = extractor.extract(parse_result, source_path="/data/test.pdf")

        assert len(results) == 1
        assert "image_path" not in results[0].metadata

    def test_picture_index_increments_across_pictures(self, tmp_path: Path) -> None:
        """Sequential picture_index produces distinct file names."""
        from PIL import Image as PILImage

        pil_image = PILImage.new("RGB", (10, 10))
        pic1 = self._make_picture_item(pil_image=pil_image)
        pic2 = self._make_picture_item(pil_image=pil_image)

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [(pic1, 1), (pic2, 1)]
        parse_result = ParseResult(
            page_count=1, estimated_token_count=10, raw_document=mock_doc, raw_text=""
        )

        config = KnowledgeForgeConfig(
            processing={
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": str(tmp_path / "images"),
                }
            }
        )
        extractor = ContentExtractor(config)
        results = extractor.extract(parse_result, source_path="/data/report.pdf")

        paths = [r.metadata["image_path"] for r in results]
        assert len(set(paths)) == 2, "Each picture must have a unique file path"
        assert "pic_p1_0.png" in paths[0]
        assert "pic_p1_1.png" in paths[1]

    def test_describe_pictures_calls_llm_and_replaces_content(
        self, tmp_path: Path
    ) -> None:
        """When describe_pictures=True, the LLM description replaces [Image] in content."""
        from PIL import Image as PILImage

        pil_image = PILImage.new("RGB", (10, 10))
        picture_item = self._make_picture_item(pil_image=pil_image)
        parse_result = self._make_parse_result_with_picture(picture_item)

        config = KnowledgeForgeConfig(
            processing={
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": str(tmp_path / "images"),
                    "describe_pictures": True,
                    "vision_api_key_env": "TEST_API_KEY",
                }
            }
        )
        extractor = ContentExtractor(config)

        fake_description = "A bar chart showing quarterly fund returns from 2020 to 2024."

        with patch.dict("os.environ", {"TEST_API_KEY": "fake-key"}):
            with patch("app.services.extraction.ContentExtractor._describe_picture_with_llm") as mock_describe:
                mock_describe.return_value = fake_description
                results = extractor.extract(
                    parse_result, source_path="/data/factsheet.pdf"
                )

        assert len(results) == 1
        assert fake_description in results[0].content
        assert "[Image]" not in results[0].content
        assert "image_path" in results[0].metadata

    def test_describe_pictures_skipped_when_api_key_missing(
        self, tmp_path: Path
    ) -> None:
        """When the API key env var is unset, description falls back gracefully."""
        from PIL import Image as PILImage

        pil_image = PILImage.new("RGB", (10, 10))
        picture_item = self._make_picture_item(pil_image=pil_image)
        parse_result = self._make_parse_result_with_picture(picture_item)

        config = KnowledgeForgeConfig(
            processing={
                "extraction": {
                    "save_picture_images": True,
                    "picture_images_dir": str(tmp_path / "images"),
                    "describe_pictures": True,
                    "vision_api_key_env": "MISSING_KEY_XYZ",
                }
            }
        )
        extractor = ContentExtractor(config)

        # Ensure the env var is definitely not set
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MISSING_KEY_XYZ", None)
            results = extractor.extract(
                parse_result, source_path="/data/factsheet.pdf"
            )

        assert len(results) == 1
        # image_path still saved (Option A succeeded)
        assert "image_path" in results[0].metadata
        # content is fallback "[Image]" since LLM call was skipped
        assert results[0].content in ("[Image]", "")


class TestErrorResilience:
    """Tests for per-item error handling during extraction."""

    def test_extraction_continues_after_item_error(
        self, extractor: ContentExtractor
    ) -> None:
        """Test that extraction continues even if individual items fail."""
        good_item = SimpleNamespace(
            label=SimpleNamespace(value="text"),
            text="Good content",
            prov=[SimpleNamespace(page_no=1)],
        )

        class BrokenItem:
            """Item whose .text property raises to simulate extraction failure."""

            label = SimpleNamespace(value="text")
            prov = [SimpleNamespace(page_no=1)]

            @property
            def text(self) -> str:
                """Raise an error to simulate a broken item."""
                raise RuntimeError("Broken item")

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (good_item, 1),
            (BrokenItem(), 1),
            (good_item, 1),
        ]

        parse_result = ParseResult(
            page_count=1,
            estimated_token_count=100,
            raw_document=mock_doc,
            raw_text="fallback text",
        )

        results = extractor.extract(parse_result)

        assert len(results) == 2
        assert all(r.content == "Good content" for r in results)
