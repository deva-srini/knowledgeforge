"""Document parsing service using Docling for structure understanding.

Uses Docling's DocumentConverter to parse PDF, DOCX, HTML, PPTX, and XLSX
files into a structured DoclingDocument representation. Extracts page counts,
token estimates, content type breakdowns, and page-level structure information.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import tiktoken

from app.core.config import KnowledgeForgeConfig

logger = logging.getLogger(__name__)

# Extensions that Docling can handle natively
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".pptx", ".xlsx"}


@dataclass
class PageStructure:
    """Structure information for a single page of a document."""

    page_number: int
    content_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a document with structure information.

    Attributes:
        page_count: Total number of pages in the document.
        estimated_token_count: Estimated token count of extracted text.
        content_types: Mapping of content label to occurrence count
            (e.g. {"text": 20, "table": 5, "picture": 3, "section_header": 10}).
        structure: Per-page breakdown of content types.
        raw_document: The Docling DoclingDocument object for downstream use.
        raw_text: Plain text export of the entire document.
    """

    page_count: int
    estimated_token_count: int
    content_types: Dict[str, int] = field(default_factory=dict)
    structure: List[PageStructure] = field(default_factory=list)
    raw_document: Optional[Any] = None
    raw_text: str = ""


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Args:
        text: The text to estimate token count for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def _get_page_no(item: Any) -> Optional[int]:
    """Extract page number from a Docling item's provenance.

    Args:
        item: A Docling content item (TextItem, TableItem, PictureItem).

    Returns:
        The page number if provenance is available, else None.
    """
    if hasattr(item, "prov") and item.prov:
        return item.prov[0].page_no
    return None


class DocumentParser:
    """Parses documents using Docling to understand structure and content.

    Uses lazy initialization for the DocumentConverter to avoid heavy
    import overhead until parsing is actually needed.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the document parser.

        Args:
            config: KnowledgeForge configuration object.
        """
        self.config = config
        self._converter: Optional[Any] = None

    def _get_converter(self) -> Any:
        """Lazily initialize and return the Docling DocumentConverter.

        Dispatches to either the standard TableFormer pipeline or the VLM
        pipeline based on ``config.processing.parsing.pipeline``.

        Returns:
            A Docling DocumentConverter instance configured for all
            supported formats.
        """
        if self._converter is None:
            if self.config.processing.parsing.pipeline == "vlm":
                self._converter = self._build_vlm_converter()
            else:
                self._converter = self._build_standard_converter()
        return self._converter

    def _build_standard_converter(self) -> Any:
        """Build the standard Docling DocumentConverter.

        Configures the PDF pipeline for optimal performance on digital
        (non-scanned) documents: disables OCR, uses native PDF text,
        enables TableFormer structure recognition, and enables GPU
        acceleration when available.

        Returns:
            A Docling DocumentConverter instance using the standard pipeline.
        """
        

        parsing_cfg = self.config.processing.parsing
        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            force_backend_text=True,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.FAST,
            ),
            do_picture_classification=False,
            do_picture_description=False,
            generate_page_images=parsing_cfg.generate_page_images,
            generate_picture_images=parsing_cfg.generate_picture_images,
            accelerator_options=AcceleratorOptions(device="auto"),
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                ),
            },
        )

    def _build_vlm_converter(self) -> Any:
        """Build a VLM-based Docling DocumentConverter.

        Uses a configurable preset (e.g. ``granite_docling``) via
        ``VlmConvertOptions.from_preset()``. The preset handles model
        specification, engine selection, and image scaling automatically.

        Returns:
            A Docling DocumentConverter instance using the VLM pipeline.
        """
        

        parsing_cfg = self.config.processing.parsing
        logger.info("Initialising VLM pipeline with preset '%s'", parsing_cfg.vlm_model)

        vlm_opts = VlmConvertOptions.from_preset(parsing_cfg.vlm_model)
        pipeline_opts = VlmPipelineOptions(vlm_options=vlm_opts)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            },
        )

    def parse(self, file_path: str) -> ParseResult:
        """Parse a document and extract structural information.

        Args:
            file_path: Path to the document file.

        Returns:
            ParseResult containing page count, token count, content types,
            page-level structure, and the raw Docling document object.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Unsupported format %s, falling back to basic text extraction",
                extension,
            )
            return self._fallback_parse(path)

        try:
            return self._docling_parse(path)
        except Exception:
            logger.exception(
                "Docling parsing failed for %s, falling back to basic extraction",
                file_path,
            )
            return self._fallback_parse(path)

    def _docling_parse(self, path: Path) -> ParseResult:
        """Parse a document using Docling's DocumentConverter.

        Converts the document and extracts:
        - Full text via export_to_text()
        - Content type counts from texts, tables, and pictures lists
        - Per-page structure breakdown using provenance data
        - Page count from the pages dict

        Args:
            path: Path object for the document.

        Returns:
            ParseResult with full structure information from Docling.
        """
        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        # Extract full text and estimate tokens
        raw_text = doc.export_to_text() if doc is not None else ""
        token_count = _estimate_tokens(raw_text)

        content_type_counts: Dict[str, int] = {}
        page_content: Dict[int, Counter[str]] = {}

        if doc is not None:
            # Count text items by label and track per-page structure
            for item in doc.texts:
                label = item.label.value
                content_type_counts[label] = content_type_counts.get(label, 0) + 1

                page_no = _get_page_no(item)
                if page_no is not None:
                    if page_no not in page_content:
                        page_content[page_no] = Counter()
                    page_content[page_no][label] += 1

            # Count tables
            if doc.tables:
                content_type_counts["table"] = len(doc.tables)
                for table in doc.tables:
                    page_no = _get_page_no(table)
                    if page_no is not None:
                        if page_no not in page_content:
                            page_content[page_no] = Counter()
                        page_content[page_no]["table"] += 1

            # Count pictures
            if doc.pictures:
                content_type_counts["picture"] = len(doc.pictures)
                for pic in doc.pictures:
                    page_no = _get_page_no(pic)
                    if page_no is not None:
                        if page_no not in page_content:
                            page_content[page_no] = Counter()
                        page_content[page_no]["picture"] += 1

        # Determine page count from Docling's pages dict
        page_count = doc.num_pages() if doc is not None else 0
        # HTML and other pageless formats report 0 pages
        if page_count == 0 and raw_text:
            page_count = 1

        structure = [
            PageStructure(page_number=pg, content_types=dict(types))
            for pg, types in sorted(page_content.items())
        ]

        logger.info(
            "Parsed %s: %d pages, ~%d tokens, content=%s",
            path.name,
            page_count,
            token_count,
            content_type_counts,
        )

        return ParseResult(
            page_count=page_count,
            estimated_token_count=token_count,
            content_types=content_type_counts,
            structure=structure,
            raw_document=doc,
            raw_text=raw_text,
        )

    def _fallback_parse(self, path: Path) -> ParseResult:
        """Basic text extraction fallback for unsupported or failed formats.

        Reads the file as UTF-8 text. Used when the file format is not
        supported by Docling or when Docling conversion fails.

        Args:
            path: Path object for the file.

        Returns:
            ParseResult with basic text content and estimated metrics.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.exception("Fallback text extraction failed for %s", path)
            text = ""

        token_count = _estimate_tokens(text)

        logger.info(
            "Fallback parse for %s: ~%d tokens",
            path.name,
            token_count,
        )

        return ParseResult(
            page_count=1,
            estimated_token_count=token_count,
            content_types={"text": 1},
            structure=[PageStructure(page_number=1, content_types={"text": 1})],
            raw_document=None,
            raw_text=text,
        )
