"""Unit tests for the document parsing service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.config import KnowledgeForgeConfig, ParsingConfig
from app.services.parsing import DocumentParser, ParseResult, _estimate_tokens

# Paths to reference files
REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "reference"
INCOME_PDF = REFERENCE_DIR / "household_income_survey(2015).pdf"
INCOME_CSV = REFERENCE_DIR / "household_income_survey(2015).csv"
BGF_PDF = REFERENCE_DIR / "bgf_factsheet.pdf"


@pytest.fixture
def config() -> KnowledgeForgeConfig:
    """Create a default test configuration."""
    return KnowledgeForgeConfig()


@pytest.fixture
def parser(config: KnowledgeForgeConfig) -> DocumentParser:
    """Create a DocumentParser instance."""
    return DocumentParser(config)


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
<p>This is the second paragraph with more content to test parsing.</p>
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


class TestTokenEstimation:
    """Tests for the token estimation function."""

    def test_empty_and_simple_text(self) -> None:
        """Test token estimation for empty and simple strings."""
        assert _estimate_tokens("") == 0

        tokens = _estimate_tokens("Hello world, this is a test.")
        assert 0 < tokens < 20

        short = _estimate_tokens("Hello")
        long = _estimate_tokens("Hello " * 100)
        assert long > short


class TestParsingFallbackAndErrors:
    """Tests for fallback parsing and error handling."""

    def test_file_not_found_raises(self, parser: DocumentParser) -> None:
        """Test that a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse("/nonexistent/file.pdf")

    @pytest.mark.skipif(not INCOME_CSV.exists(), reason="CSV file not found")
    def test_csv_fallback_parsing(self, parser: DocumentParser) -> None:
        """Test that CSV triggers fallback parsing with correct structure."""
        result = parser.parse(str(INCOME_CSV))

        assert isinstance(result, ParseResult)
        assert result.page_count == 1
        assert result.estimated_token_count > 0
        assert result.content_types == {"text": 1}
        assert result.raw_document is None
        assert len(result.raw_text) > 0

    def test_fallback_on_docling_failure(
        self, parser: DocumentParser, sample_html_file: Path
    ) -> None:
        """Test that Docling failure gracefully falls back to text extraction."""
        with patch.object(
            parser, "_docling_parse", side_effect=Exception("Docling error")
        ):
            result = parser.parse(str(sample_html_file))

        assert isinstance(result, ParseResult)
        assert result.page_count == 1
        assert result.estimated_token_count > 0
        assert result.raw_document is None


class TestHTMLParsing:
    """Tests for Docling-based HTML document parsing."""

    def test_parse_html_document(
        self, parser: DocumentParser, sample_html_file: Path
    ) -> None:
        """Test parsing an HTML document produces a valid result with content types."""
        result = parser.parse(str(sample_html_file))

        assert isinstance(result, ParseResult)
        assert result.page_count >= 1
        assert result.estimated_token_count > 0
        assert result.raw_document is not None
        assert len(result.raw_text) > 0
        assert len(result.content_types) > 0

    def test_raw_document_is_docling_document(
        self, parser: DocumentParser, sample_html_file: Path
    ) -> None:
        """Test that raw_document is a DoclingDocument instance."""
        from docling_core.types.doc import DoclingDocument

        result = parser.parse(str(sample_html_file))
        assert isinstance(result.raw_document, DoclingDocument)


@pytest.mark.skipif(not INCOME_PDF.exists(), reason="Income survey PDF not found")
class TestIncomeSurveyPDFParsing:
    """Tests for parsing the household income survey PDF (single-page table)."""

    def test_parse_income_pdf_structure(self, parser: DocumentParser) -> None:
        """Test income survey PDF page count and basic structure."""
        result = parser.parse(str(INCOME_PDF))

        assert isinstance(result, ParseResult)
        assert result.raw_document is not None
        assert result.page_count >= 1
        assert result.estimated_token_count > 100
        assert len(result.raw_text) > 100
        assert len(result.structure) >= 1
        assert result.structure[0].page_number >= 1


@pytest.mark.skipif(not BGF_PDF.exists(), reason="BGF factsheet PDF not found")
class TestBGFFactsheetPDFParsing:
    """Tests for parsing the BGF factsheet PDF (multi-page with tables and pictures)."""

    def test_parse_bgf_pdf_pages_and_tokens(self, parser: DocumentParser) -> None:
        """Test BGF factsheet has 4 pages and substantial token count."""
        result = parser.parse(str(BGF_PDF))

        assert isinstance(result, ParseResult)
        assert result.raw_document is not None
        assert result.page_count == 4
        assert result.estimated_token_count > 1000
        assert len(result.raw_text) > 500

    def test_parse_bgf_pdf_content_types(self, parser: DocumentParser) -> None:
        """Test BGF factsheet identifies tables, pictures, and text content."""
        result = parser.parse(str(BGF_PDF))

        assert "table" in result.content_types
        assert result.content_types["table"] == 4
        assert "picture" in result.content_types
        assert result.content_types["picture"] == 10
        assert "text" in result.content_types
        assert result.content_types["text"] > 0
        assert "section_header" in result.content_types


class TestParseResultDataclass:
    """Tests for the ParseResult dataclass."""

    def test_default_and_full_construction(self) -> None:
        """Test ParseResult with defaults and all fields populated."""
        defaults = ParseResult(page_count=0, estimated_token_count=0)
        assert defaults.content_types == {}
        assert defaults.structure == []
        assert defaults.raw_document is None
        assert defaults.raw_text == ""

        full = ParseResult(
            page_count=5,
            estimated_token_count=1500,
            content_types={"text": 8, "table": 2},
            structure=[],
            raw_document=None,
            raw_text="sample text",
        )
        assert full.page_count == 5
        assert full.estimated_token_count == 1500
        assert full.content_types["table"] == 2


class TestVlmPipelineConfig:
    """Tests for VLM pipeline configuration and converter initialisation."""

    def test_standard_pipeline_is_default(self) -> None:
        """Test that the default pipeline is 'standard'."""
        cfg = ParsingConfig()
        assert cfg.pipeline == "standard"

    def test_vlm_model_default(self) -> None:
        """Test that vlm_model defaults to 'granite_docling'."""
        cfg = ParsingConfig(pipeline="vlm")
        assert cfg.vlm_model == "granite_docling"

    def test_invalid_pipeline_raises(self) -> None:
        """Test that an unknown pipeline value raises a ValidationError."""
        with pytest.raises(ValidationError):
            ParsingConfig(pipeline="unknown")

    def test_vlm_converter_called_when_pipeline_vlm(
        self, config: KnowledgeForgeConfig
    ) -> None:
        """Test that _build_vlm_converter is called when pipeline='vlm'."""
        config.processing.parsing.pipeline = "vlm"
        parser = DocumentParser(config)
        with patch.object(
            parser, "_build_vlm_converter", return_value=MagicMock()
        ) as mock_vlm:
            parser._get_converter()
            mock_vlm.assert_called_once()

    def test_standard_converter_called_by_default(
        self, config: KnowledgeForgeConfig
    ) -> None:
        """Test that _build_standard_converter is called for the default pipeline."""
        parser = DocumentParser(config)
        with patch.object(
            parser, "_build_standard_converter", return_value=MagicMock()
        ) as mock_std:
            parser._get_converter()
            mock_std.assert_called_once()

    def test_build_vlm_converter_uses_preset(
        self, config: KnowledgeForgeConfig
    ) -> None:
        """Test that _build_vlm_converter calls VlmConvertOptions.from_preset with the configured model."""
        config.processing.parsing.pipeline = "vlm"
        config.processing.parsing.vlm_model = "granite_docling"
        parser = DocumentParser(config)

        mock_opts = MagicMock()
        with patch(
            "docling.datamodel.pipeline_options.VlmConvertOptions", mock_opts, create=True
        ), patch(
            "docling.datamodel.pipeline_options.VlmPipelineOptions", create=True
        ), patch(
            "docling.datamodel.base_models.InputFormat", create=True
        ), patch(
            "docling.document_converter.DocumentConverter", create=True
        ), patch(
            "docling.document_converter.PdfFormatOption", create=True
        ):
            parser._build_vlm_converter()

        mock_opts.from_preset.assert_called_once_with("granite_docling")
