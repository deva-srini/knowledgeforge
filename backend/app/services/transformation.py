"""Content transformation service for cleaning and normalizing extracted content.

Takes a list of ExtractedContent objects from the extraction step and applies
optional transformations: whitespace cleanup, Unicode encoding normalization,
and markdown table formatting. Controlled by the processing.organisation
config section. When disabled, content passes through unchanged as
TransformedContent objects.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import KnowledgeForgeConfig
from app.services.extraction import ContentType, ExtractedContent

logger = logging.getLogger(__name__)

# Unicode characters to replace with ASCII equivalents.
_UNICODE_REPLACEMENTS: Dict[str, str] = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # byte order mark / zero-width no-break space
}


@dataclass
class TransformedContent:
    """A single unit of content after transformation/cleaning.

    Mirrors ExtractedContent but represents post-transformation state.

    Attributes:
        content: The transformed text content.
        content_type: Type of content (text, table, image_description).
        page_number: Page where this content appears (0 if unknown).
        header_path: Heading hierarchy path (e.g. "Chapter 1 > Section 2").
        metadata: Additional metadata, including transformation details.
    """

    content: str
    content_type: ContentType
    page_number: int = 0
    header_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformResult:
    """Result of the transformation stage.

    Bundles per-item transformed content with an optional full-document
    structured markdown string for downstream structure-aware chunking.

    Attributes:
        items: Per-item transformed content list.
        document_markdown: Full structured markdown of the document,
            or empty string if unavailable or generation is disabled.
    """

    items: List[TransformedContent]
    document_markdown: str = ""


class ContentTransformer:
    """Transforms extracted content by cleaning text and normalizing formatting.

    When enabled, applies whitespace cleanup, encoding normalization, and
    table format cleaning. When disabled, passes content through unchanged.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the content transformer.

        Args:
            config: KnowledgeForge configuration object.
        """
        self.config = config
        self._enabled = config.processing.organisation.enabled
        self._table_format = config.processing.organisation.table_format
        self._markdown_generation_enabled = config.processing.organisation.markdown_generation

    def transform(
        self,
        extracted_items: List[ExtractedContent],
        raw_document: Optional[Any] = None,
    ) -> TransformResult:
        """Transform a list of extracted content items.

        When organisation is enabled, applies cleaning and normalization.
        When disabled, wraps items as TransformedContent unchanged.
        Optionally generates full-document structured markdown from
        the raw Docling document for downstream chunking.

        Args:
            extracted_items: List of ExtractedContent from the extractor.
            raw_document: Optional raw Docling DoclingDocument for
                structured markdown generation.

        Returns:
            TransformResult containing per-item results and optional
            document-level structured markdown.
        """
        results: List[TransformedContent] = []

        for item in extracted_items:
            try:
                if not self._enabled:
                    results.append(self._pass_through(item))
                else:
                    results.append(self._transform_item(item))
            except Exception:
                logger.exception(
                    "Failed to transform item (type=%s, page=%s), passing through",
                    item.content_type.value,
                    item.page_number,
                )
                results.append(self._pass_through(item))

        document_markdown = self._generate_document_markdown(raw_document)

        logger.info(
            "Transformed %d content items (enabled=%s, markdown=%d chars)",
            len(results),
            self._enabled,
            len(document_markdown),
        )
        return TransformResult(items=results, document_markdown=document_markdown)

    def _pass_through(self, item: ExtractedContent) -> TransformedContent:
        """Convert an ExtractedContent to TransformedContent without changes.

        Args:
            item: The extracted content to pass through.

        Returns:
            TransformedContent with identical content and metadata.
        """
        metadata = dict(item.metadata)
        metadata["transformed"] = False
        return TransformedContent(
            content=item.content,
            content_type=item.content_type,
            page_number=item.page_number,
            header_path=item.header_path,
            metadata=metadata,
        )

    def _transform_item(self, item: ExtractedContent) -> TransformedContent:
        """Apply transformations to a single content item.

        Dispatches to type-specific transformation methods based on
        content_type.

        Args:
            item: The extracted content to transform.

        Returns:
            TransformedContent with cleaned content.
        """
        if item.content_type == ContentType.TABLE:
            cleaned = self._transform_table(item.content)
        elif item.content_type == ContentType.TEXT:
            cleaned = self._transform_text(item.content)
        elif item.content_type == ContentType.IMAGE_DESCRIPTION:
            cleaned = self._transform_text(item.content)
        else:
            cleaned = item.content

        metadata = dict(item.metadata)
        metadata["transformed"] = True

        return TransformedContent(
            content=cleaned,
            content_type=item.content_type,
            page_number=item.page_number,
            header_path=item.header_path,
            metadata=metadata,
        )

    def _transform_text(self, content: str) -> str:
        """Clean and normalize text content.

        Applies encoding normalization followed by whitespace cleanup.

        Args:
            content: Raw text content.

        Returns:
            Cleaned text string.
        """
        text = self._normalize_encoding(content)
        text = self._clean_whitespace(text)
        return text

    def _transform_table(self, content: str) -> str:
        """Clean and normalize table content.

        Tables from the extractor are already in markdown format
        (from export_to_markdown()). This method cleans up any
        formatting inconsistencies.

        Args:
            content: Table content (typically markdown format).

        Returns:
            Cleaned markdown table string.
        """
        text = self._normalize_encoding(content)
        text = self._clean_whitespace(text)

        if self._table_format == "markdown":
            text = self._clean_markdown_table(text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize line endings.

        Args:
            text: Text to clean.

        Returns:
            Whitespace-cleaned text.
        """
        # Normalize line endings (CRLF, CR -> LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove leading and trailing whitespace from each line
        text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

        # Collapse multiple blank lines into a single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces/tabs into a single space (within lines)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Strip leading/trailing whitespace from the whole string
        text = text.strip()

        return text

    def _normalize_encoding(self, text: str) -> str:
        """Normalize Unicode encoding to NFC form and replace common artifacts.

        Args:
            text: Text to normalize.

        Returns:
            Encoding-normalized text.
        """
        # NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Replace known Unicode artifacts
        for char, replacement in _UNICODE_REPLACEMENTS.items():
            text = text.replace(char, replacement)

        return text

    def _clean_markdown_table(self, content: str) -> str:
        """Clean markdown table formatting.

        Normalizes cell padding, removes empty rows, and ensures
        consistent separator formatting.

        Args:
            content: Markdown table string.

        Returns:
            Cleaned markdown table string.
        """
        lines = content.split("\n")
        cleaned_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines within the table
            if not stripped:
                continue

            # Only process lines that look like table rows (contain pipes)
            if "|" not in stripped:
                cleaned_lines.append(stripped)
                continue

            # Split on pipes and get inner cells
            cells = stripped.split("|")

            # Remove leading/trailing empty strings from the split
            if cells and cells[0].strip() == "":
                cells = cells[1:]
            if cells and cells[-1].strip() == "":
                cells = cells[:-1]

            if not cells:
                continue

            # Check if this is a separator row (all cells are dashes/colons)
            is_separator = all(
                re.match(r"^[\s\-:]+$", cell) for cell in cells
            )

            if is_separator:
                # Rebuild separator with consistent formatting
                sep_cells = []
                for cell in cells:
                    cell_stripped = cell.strip()
                    sep_cells.append(" " + cell_stripped + " ")
                cleaned_lines.append("|" + "|".join(sep_cells) + "|")
            else:
                # Regular data row: normalize cell padding
                data_cells = []
                for cell in cells:
                    data_cells.append(" " + cell.strip() + " ")
                cleaned_lines.append("|" + "|".join(data_cells) + "|")

        return "\n".join(cleaned_lines)

    def _generate_document_markdown(
        self, raw_document: Optional[Any]
    ) -> str:
        """Generate structured markdown from the raw Docling document.

        Calls the document's export_to_markdown() method and applies
        encoding normalization and whitespace cleanup. Returns an empty
        string when generation is disabled, no document is available,
        or an error occurs.

        Args:
            raw_document: Raw Docling DoclingDocument, or None.

        Returns:
            Structured markdown string, or empty string.
        """
        if not self._enabled or not self._markdown_generation_enabled:
            return ""

        if raw_document is None:
            logger.debug("No raw document available for markdown generation")
            return ""

        try:
            markdown: str = raw_document.export_to_markdown(
                image_placeholder="<!-- image -->",
                page_break_placeholder="<!-- page break -->",
                escape_underscores=False,
                escape_html=False,
            )
        except Exception:
            logger.exception(
                "Failed to generate structured markdown from raw document"
            )
            return ""

        if not markdown.strip():
            return ""

        markdown = self._normalize_encoding(markdown)
        markdown = self._clean_whitespace(markdown)

        logger.info(
            "Generated structured markdown document (%d characters)",
            len(markdown),
        )
        return markdown
