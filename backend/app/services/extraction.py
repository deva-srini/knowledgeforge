"""Content extraction service for transforming parsed documents into structured content.

Takes a ParseResult from the parsing step and extracts all content into a list
of ExtractedContent objects. Each item preserves its content type, page number,
heading hierarchy path, and metadata. Uses Docling's iterate_items() for
document traversal and maintains a heading stack to build header paths.
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import KnowledgeForgeConfig
from app.services.parsing import ParseResult

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of extracted content."""

    TEXT = "text"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"


class ExtractionStrategy(str, Enum):
    """Supported extraction strategies."""

    AUTO = "auto"
    DIRECT = "direct"
    OCR = "ocr"
    AGENTIC = "agentic"


@dataclass
class ExtractedContent:
    """A single unit of extracted content from a document.

    Attributes:
        content: The extracted text content.
        content_type: Type of content (text, table, image_description).
        page_number: Page where this content appears (0 if unknown).
        header_path: Heading hierarchy path (e.g. "Chapter 1 > Section 2").
        metadata: Additional metadata about the extracted content.
    """

    content: str
    content_type: ContentType
    page_number: int = 0
    header_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _get_page_no(item: Any) -> int:
    """Extract page number from a Docling item's provenance.

    Args:
        item: A Docling content item with optional prov attribute.

    Returns:
        The page number if available, else 0.
    """
    if hasattr(item, "prov") and item.prov:
        return item.prov[0].page_no
    return 0


def _build_header_path(header_stack: List[str]) -> str:
    """Build a header hierarchy path string from a stack of headings.

    Args:
        header_stack: List of heading texts from outermost to innermost.

    Returns:
        Joined header path string (e.g. "Chapter 1 > Section 2").
    """
    if not header_stack:
        return ""
    return " > ".join(header_stack)


class ContentExtractor:
    """Extracts structured content from parsed documents.

    Traverses the Docling document tree using iterate_items(), tracking
    the heading hierarchy to provide context for each content item.
    Supports strategy selection via configuration.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the content extractor.

        Args:
            config: KnowledgeForge configuration object.
        """
        self.config = config
        self._strategy = self._resolve_strategy(
            config.processing.extraction.strategy
        )

    def _resolve_strategy(self, strategy: str) -> ExtractionStrategy:
        """Resolve the extraction strategy from config.

        For Phase 1, "auto" defaults to "direct" for digital documents.

        Args:
            strategy: Strategy string from config.

        Returns:
            Resolved ExtractionStrategy enum value.
        """
        if strategy == ExtractionStrategy.AUTO.value:
            return ExtractionStrategy.DIRECT
        return ExtractionStrategy(strategy)

    @property
    def strategy(self) -> ExtractionStrategy:
        """The resolved extraction strategy."""
        return self._strategy

    def extract(
        self,
        parse_result: ParseResult,
        source_path: Optional[str] = None,
    ) -> List[ExtractedContent]:
        """Extract all content from a parsed document.

        Walks the Docling document tree to extract text, tables, and image
        descriptions. Falls back to raw text extraction when no Docling
        document is available.

        Args:
            parse_result: The ParseResult from the document parser.
            source_path: Optional path to the source file. Required for
                picture image saving (Option A) and LLM description (Option B).

        Returns:
            List of ExtractedContent objects preserving document structure.
        """
        if parse_result.raw_document is None:
            return self._extract_from_raw_text(parse_result)

        return self._extract_from_docling(parse_result, source_path=source_path)

    def _extract_from_raw_text(
        self, parse_result: ParseResult
    ) -> List[ExtractedContent]:
        """Fallback extraction from raw text when no Docling document exists.

        Args:
            parse_result: ParseResult with raw_text but no raw_document.

        Returns:
            Single-element list with the full raw text as content.
        """
        if not parse_result.raw_text:
            return []

        return [
            ExtractedContent(
                content=parse_result.raw_text,
                content_type=ContentType.TEXT,
                page_number=1,
                header_path="",
                metadata={"source": "fallback"},
            )
        ]

    def _extract_from_docling(
        self,
        parse_result: ParseResult,
        source_path: Optional[str] = None,
    ) -> List[ExtractedContent]:
        """Extract content by traversing the Docling document tree.

        Uses iterate_items() to walk through the document in reading order,
        maintaining a heading stack indexed by hierarchy level for building
        header paths. Tracks a sequential picture_index for deterministic
        image file naming.

        Args:
            parse_result: ParseResult containing a Docling DoclingDocument.
            source_path: Optional path to the source file, forwarded to
                _extract_picture() for image saving.

        Returns:
            List of ExtractedContent objects for all document content.
        """
        doc = parse_result.raw_document
        results: List[ExtractedContent] = []

        # Header stack: list of (level, text) pairs for building paths
        header_stack: List[tuple[int, str]] = []
        # Sequential counter for naming saved picture files
        picture_index = 0

        for item, level in doc.iterate_items():
            try:
                label = item.label.value if hasattr(item, "label") else ""

                if label in ("section_header", "title"):
                    self._update_header_stack(header_stack, level, item)
                    continue

                header_path = _build_header_path(
                    [text for _, text in header_stack]
                )
                page_no = _get_page_no(item)

                if label == "table":
                    extracted = self._extract_table(
                        item, doc, page_no, header_path
                    )
                    if extracted is not None:
                        results.append(extracted)

                elif label == "picture":
                    extracted = self._extract_picture(
                        item,
                        doc,
                        page_no,
                        header_path,
                        picture_index=picture_index,
                        source_path=source_path,
                    )
                    if extracted is not None:
                        results.append(extracted)
                    picture_index += 1

                elif hasattr(item, "text") and item.text.strip():
                    results.append(
                        ExtractedContent(
                            content=item.text.strip(),
                            content_type=ContentType.TEXT,
                            page_number=page_no,
                            header_path=header_path,
                            metadata={"label": label},
                        )
                    )

            except Exception:
                logger.exception(
                    "Failed to extract item (label=%s, page=%s), skipping",
                    getattr(item, "label", "unknown"),
                    _get_page_no(item),
                )
                continue

        logger.info(
            "Extracted %d content items (strategy=%s)",
            len(results),
            self._strategy.value,
        )
        return results

    def _update_header_stack(
        self,
        header_stack: List[tuple[int, str]],
        level: int,
        item: Any,
    ) -> None:
        """Update the heading stack when a section header is encountered.

        Removes any headings at the same or deeper level, then pushes
        the new heading. This maintains the correct hierarchy for nested
        sections.

        Args:
            header_stack: The current heading stack (modified in place).
            level: The hierarchy level of the current heading.
            item: The Docling heading item with a text attribute.
        """
        text = item.text.strip() if hasattr(item, "text") else ""
        if not text:
            return

        # Pop headings at same or deeper level
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()

        header_stack.append((level, text))

    def _extract_table(
        self,
        table_item: Any,
        doc: Any,
        page_no: int,
        header_path: str,
    ) -> Optional[ExtractedContent]:
        """Extract a table as markdown-formatted structured data.

        Args:
            table_item: Docling TableItem object.
            doc: The DoclingDocument for resolving references.
            page_no: Page number where the table appears.
            header_path: Current heading hierarchy path.

        Returns:
            ExtractedContent with markdown table, or None if empty.
        """
        markdown = ""
        try:
            markdown = table_item.export_to_markdown(doc)
        except Exception:
            logger.warning(
                "Markdown export failed for table on page %d, "
                "falling back to grid extraction",
                page_no,
            )
            markdown = self._table_grid_to_text(table_item)

        if not markdown.strip():
            return None

        metadata: Dict[str, Any] = {"label": "table"}

        if hasattr(table_item, "data") and table_item.data is not None:
            metadata["num_rows"] = table_item.data.num_rows
            metadata["num_cols"] = table_item.data.num_cols

        caption = ""
        try:
            caption = table_item.caption_text(doc)
        except Exception:
            pass

        if caption:
            metadata["caption"] = caption

        return ExtractedContent(
            content=markdown.strip(),
            content_type=ContentType.TABLE,
            page_number=page_no,
            header_path=header_path,
            metadata=metadata,
        )

    def _table_grid_to_text(self, table_item: Any) -> str:
        """Fallback table extraction using the grid data directly.

        Args:
            table_item: Docling TableItem with data.grid attribute.

        Returns:
            Pipe-delimited text representation of the table.
        """
        if not hasattr(table_item, "data") or table_item.data is None:
            return ""

        try:
            lines = []
            for row in table_item.data.grid:
                cells = [cell.text for cell in row]
                lines.append(" | ".join(cells))
            return "\n".join(lines)
        except Exception:
            return ""

    def _extract_picture(
        self,
        picture_item: Any,
        doc: Any,
        page_no: int,
        header_path: str,
        picture_index: int = 0,
        source_path: Optional[str] = None,
    ) -> Optional[ExtractedContent]:
        """Extract a picture as a description string with optional image saving.

        Collects caption, classification, and description metadata. If
        save_picture_images is enabled in config, saves the picture's PIL
        image to disk and stores the file path in metadata (Option A). If
        describe_pictures is also enabled, calls a Claude Vision model to
        generate a rich text description (Option B).

        Args:
            picture_item: Docling PictureItem object.
            doc: The DoclingDocument for resolving references.
            page_no: Page number where the picture appears.
            header_path: Current heading hierarchy path.
            picture_index: Sequential 0-based index of this picture in the
                document, used for deterministic image file naming.
            source_path: Path to the source document file. Required for
                image saving and LLM description.

        Returns:
            ExtractedContent with image description, or None if no info.
        """
        parts: List[str] = []
        metadata: Dict[str, Any] = {"label": "picture"}

        # Get caption text
        caption = ""
        try:
            caption = picture_item.caption_text(doc)
        except Exception:
            pass

        if caption:
            parts.append(caption.strip())
            metadata["caption"] = caption.strip()

        # Get classification if available
        if (
            hasattr(picture_item, "meta")
            and picture_item.meta is not None
            and hasattr(picture_item.meta, "classification")
            and picture_item.meta.classification is not None
        ):
            try:
                predictions = picture_item.meta.classification.predictions
                if predictions:
                    class_name = predictions[0].class_name
                    metadata["classification"] = class_name
                    if not parts:
                        parts.append(f"[{class_name}]")
            except Exception:
                pass

        # Get description if available
        if (
            hasattr(picture_item, "meta")
            and picture_item.meta is not None
            and hasattr(picture_item.meta, "description")
            and picture_item.meta.description is not None
        ):
            try:
                desc = picture_item.meta.description
                if hasattr(desc, "text") and desc.text:
                    parts.append(desc.text.strip())
            except Exception:
                pass

        extraction_cfg = self.config.processing.extraction

        # Option A: save picture image to disk and store path in metadata
        if extraction_cfg.save_picture_images and source_path is not None:
            image_path = self._save_picture_image(
                picture_item, doc, source_path, page_no, picture_index
            )
            if image_path:
                metadata["image_path"] = image_path

                # Option B: replace weak placeholder with LLM-generated description
                if extraction_cfg.describe_pictures:
                    llm_desc = self._describe_picture_with_llm(image_path)
                    if llm_desc:
                        # Keep the caption for context, replace the rest
                        parts = [caption.strip(), llm_desc] if caption else [llm_desc]

        if not parts:
            parts.append("[Image]")

        return ExtractedContent(
            content="\n".join(parts),
            content_type=ContentType.IMAGE_DESCRIPTION,
            page_number=page_no,
            header_path=header_path,
            metadata=metadata,
        )

    def _save_picture_image(
        self,
        picture_item: Any,
        doc: Any,
        source_path: str,
        page_no: int,
        picture_index: int,
    ) -> Optional[str]:
        """Save a picture item's PIL image to disk as a PNG file.

        The output path is:
            {picture_images_dir}/{source_stem}/pic_p{page_no}_{picture_index}.png

        Args:
            picture_item: Docling PictureItem with a get_image() method.
            doc: The DoclingDocument needed by get_image().
            source_path: Path to the source document (used to derive sub-folder).
            page_no: Page number, used in the output file name.
            picture_index: Sequential picture index, used in the output file name.

        Returns:
            Absolute path string of the saved PNG, or None on failure.
        """
        try:
            pil_image = picture_item.get_image(doc)
            if pil_image is None:
                logger.debug(
                    "get_image() returned None for picture %d on page %d",
                    picture_index,
                    page_no,
                )
                return None

            images_dir = Path(self.config.processing.extraction.picture_images_dir)
            doc_slug = Path(source_path).stem
            output_dir = images_dir / doc_slug
            output_dir.mkdir(parents=True, exist_ok=True)

            img_path = output_dir / f"pic_p{page_no}_{picture_index}.png"
            pil_image.save(str(img_path))

            logger.info("Saved picture image: %s", img_path)
            return str(img_path)

        except Exception:
            logger.warning(
                "Failed to save picture image (page=%d, index=%d)",
                page_no,
                picture_index,
                exc_info=True,
            )
            return None

    def _describe_picture_with_llm(self, image_path: str) -> Optional[str]:
        """Call a Claude Vision model to generate a text description of an image.

        Reads the image at image_path, base64-encodes it, and sends it to the
        configured Claude model. The API key is read from the environment
        variable named by vision_api_key_env.

        Args:
            image_path: Absolute path to the PNG file to describe.

        Returns:
            Description string from the model, or None on any failure.
        """
        try:
            import anthropic
        except ImportError:
            logger.warning(
                "anthropic package not installed; cannot describe pictures. "
                "Install it with: python -m pip install anthropic"
            )
            return None

        extraction_cfg = self.config.processing.extraction
        api_key = os.environ.get(extraction_cfg.vision_api_key_env, "")
        if not api_key:
            logger.warning(
                "describe_pictures is enabled but env var '%s' is not set; "
                "skipping LLM description for %s",
                extraction_cfg.vision_api_key_env,
                image_path,
            )
            return None

        try:
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=extraction_cfg.vision_model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this chart or image concisely and "
                                    "factually. Include key values, labels, trends, "
                                    "and what information it conveys. If it is a "
                                    "chart, mention the chart type, axes, and notable "
                                    "data points."
                                ),
                            },
                        ],
                    }
                ],
            )
            description: str = message.content[0].text.strip()
            logger.info(
                "Generated picture description (%d chars) for %s",
                len(description),
                image_path,
            )
            return description

        except Exception:
            logger.warning(
                "Failed to generate LLM description for %s",
                image_path,
                exc_info=True,
            )
            return None
