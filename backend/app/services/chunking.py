"""Structure-aware chunking service for splitting transformed content into chunks.

Takes a list of TransformedContent objects from the transformation step and
produces a list of Chunk objects ready for embedding. Respects document
structure (headers, content types) and applies configurable token limits
with overlap between consecutive text chunks.

Uses semchunk as an inner splitter for oversized text blocks that exceed
the configured chunk_size_tokens.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import semchunk
import tiktoken

from app.core.config import KnowledgeForgeConfig
from app.services.extraction import ContentType
from app.services.transformation import TransformedContent

logger = logging.getLogger(__name__)

# Module-level tiktoken encoding, matching the pattern in parsing.py.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken.

    Args:
        text: The text to count tokens for.

    Returns:
        Token count, or 0 for empty text.
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text))


@dataclass
class Chunk:
    """A single chunk of content ready for embedding.

    Attributes:
        content: The chunk text.
        content_type: Type of content (text, table, image_description).
        chunk_index: Sequential 0-based index within the document.
        header_path: Heading hierarchy path (e.g. "Chapter 1 > Section 2").
        page_number: Source page number (0 if unknown).
        token_count: Accurate token count for this chunk.
        metadata: Arbitrary metadata dictionary.
    """

    content: str
    content_type: ContentType
    chunk_index: int
    header_path: str
    page_number: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """Splits transformed content into chunks respecting document structure.

    Groups consecutive items by header_path and dispatches tables and
    image descriptions as standalone chunks. Text items are accumulated
    and split at token boundaries with configurable overlap.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the chunker from configuration.

        Args:
            config: KnowledgeForge configuration object.
        """
        self._chunk_size = config.processing.chunking.chunk_size_tokens
        self._overlap = config.processing.chunking.chunk_overlap_tokens
        self._skip_threshold = config.processing.chunking.skip_threshold_tokens

    def chunk(self, items: List[TransformedContent]) -> List[Chunk]:
        """Chunk a list of transformed content items.

        Args:
            items: List of TransformedContent from the transformer.

        Returns:
            List of Chunk objects with sequential chunk_index values.
        """
        if not items:
            return []

        total_tokens = sum(_count_tokens(item.content) for item in items)

        if total_tokens < self._skip_threshold:
            chunks = self._build_single_chunk(items)
        else:
            chunks = self._structure_aware_chunk(items)

        logger.info(
            "Chunked %d items into %d chunks (total_tokens=%d, "
            "skip_threshold=%d, chunk_size=%d, overlap=%d)",
            len(items),
            len(chunks),
            total_tokens,
            self._skip_threshold,
            self._chunk_size,
            self._overlap,
        )
        return chunks

    def _build_single_chunk(
        self, items: List[TransformedContent]
    ) -> List[Chunk]:
        """Build a single chunk when total tokens are below skip threshold.

        Concatenates all item contents with double-newline separators.

        Args:
            items: All transformed content items.

        Returns:
            Single-element list with one Chunk.
        """
        combined = "\n\n".join(item.content for item in items)
        # Use content_type from first item; default to TEXT for mixed.
        content_types = {item.content_type for item in items}
        content_type = items[0].content_type if len(content_types) == 1 else ContentType.TEXT
        # Use first non-zero page number.
        page_number = next(
            (item.page_number for item in items if item.page_number > 0), 0
        )
        # Use first non-empty header path.
        header_path = next(
            (item.header_path for item in items if item.header_path), ""
        )
        return [
            self._make_chunk(
                content=combined,
                content_type=content_type,
                header_path=header_path,
                page_number=page_number,
                chunk_index=0,
                metadata={"chunking_strategy": "skip"},
            )
        ]

    def _structure_aware_chunk(
        self, items: List[TransformedContent]
    ) -> List[Chunk]:
        """Chunk items using structure-aware logic.

        Groups by header_path, processes each section, applies overlap,
        and assigns sequential chunk indices.

        Args:
            items: All transformed content items.

        Returns:
            List of Chunk objects.
        """
        sections = self._group_by_header(items)
        raw_chunks: List[Chunk] = []

        for section in sections:
            section_chunks = self._process_section(section)
            raw_chunks.extend(section_chunks)

        # Apply overlap between consecutive text chunks.
        chunks = self._apply_overlap(raw_chunks)

        # Re-assign sequential chunk indices after overlap.
        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    def _group_by_header(
        self, items: List[TransformedContent]
    ) -> List[List[TransformedContent]]:
        """Group consecutive items by header_path.

        Args:
            items: All transformed content items.

        Returns:
            List of groups, each a list of items sharing the same header_path.
        """
        if not items:
            return []

        groups: List[List[TransformedContent]] = []
        current_group: List[TransformedContent] = [items[0]]

        for item in items[1:]:
            if item.header_path == current_group[0].header_path:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]

        groups.append(current_group)
        return groups

    def _process_section(
        self, section: List[TransformedContent]
    ) -> List[Chunk]:
        """Process a section of items sharing the same header_path.

        Tables and image descriptions become standalone chunks.
        Text items are accumulated and split at token boundaries.

        Args:
            section: Items with the same header_path.

        Returns:
            List of Chunk objects for this section.
        """
        chunks: List[Chunk] = []
        text_buffer: List[TransformedContent] = []

        for item in section:
            try:
                if item.content_type == ContentType.TABLE:
                    # Flush accumulated text first.
                    if text_buffer:
                        chunks.extend(self._chunk_text_items(text_buffer))
                        text_buffer = []
                    # Table is always a standalone chunk (never split).
                    chunks.append(
                        self._make_chunk(
                            content=item.content,
                            content_type=ContentType.TABLE,
                            header_path=item.header_path,
                            page_number=item.page_number,
                            chunk_index=0,  # Will be reassigned.
                            metadata=dict(item.metadata),
                        )
                    )
                elif item.content_type == ContentType.IMAGE_DESCRIPTION:
                    # Flush accumulated text first.
                    if text_buffer:
                        chunks.extend(self._chunk_text_items(text_buffer))
                        text_buffer = []
                    # Image description is always a standalone chunk.
                    chunks.append(
                        self._make_chunk(
                            content=item.content,
                            content_type=ContentType.IMAGE_DESCRIPTION,
                            header_path=item.header_path,
                            page_number=item.page_number,
                            chunk_index=0,
                            metadata=dict(item.metadata),
                        )
                    )
                else:
                    # Text item: accumulate.
                    text_buffer.append(item)
            except Exception:
                logger.exception(
                    "Failed to process item (type=%s, page=%s), "
                    "creating fallback chunk",
                    item.content_type.value,
                    item.page_number,
                )
                chunks.append(
                    self._make_chunk(
                        content=item.content,
                        content_type=item.content_type,
                        header_path=item.header_path,
                        page_number=item.page_number,
                        chunk_index=0,
                        metadata={"error": "processing_failed"},
                    )
                )

        # Flush remaining text.
        if text_buffer:
            chunks.extend(self._chunk_text_items(text_buffer))

        return chunks

    def _chunk_text_items(
        self, text_items: List[TransformedContent]
    ) -> List[Chunk]:
        """Chunk a sequence of text items by accumulating into token-bounded buffers.

        If a single text item exceeds chunk_size_tokens, it is split using
        semchunk. Otherwise, items are merged until the buffer would exceed
        the limit.

        Args:
            text_items: Consecutive text items to chunk.

        Returns:
            List of text Chunk objects.
        """
        if not text_items:
            return []

        chunks: List[Chunk] = []
        buffer_parts: List[str] = []
        buffer_tokens = 0
        # Track page_number and header_path from the first item in each buffer.
        buffer_page = text_items[0].page_number
        buffer_header = text_items[0].header_path

        for item in text_items:
            item_tokens = _count_tokens(item.content)

            # If single item exceeds chunk_size, flush buffer then split it.
            if item_tokens > self._chunk_size:
                # Flush current buffer.
                if buffer_parts:
                    chunks.append(
                        self._make_chunk(
                            content="\n\n".join(buffer_parts),
                            content_type=ContentType.TEXT,
                            header_path=buffer_header,
                            page_number=buffer_page,
                            chunk_index=0,
                        )
                    )
                    buffer_parts = []
                    buffer_tokens = 0

                # Split the oversized item.
                split_chunks = self._split_oversized_text(item)
                chunks.extend(split_chunks)

                # Reset buffer tracking for next items.
                buffer_page = item.page_number
                buffer_header = item.header_path
                continue

            # Check if adding this item would exceed chunk_size.
            # Account for the "\n\n" separator between parts.
            separator_tokens = _count_tokens("\n\n") if buffer_parts else 0
            if buffer_parts and (buffer_tokens + separator_tokens + item_tokens) > self._chunk_size:
                # Flush current buffer.
                chunks.append(
                    self._make_chunk(
                        content="\n\n".join(buffer_parts),
                        content_type=ContentType.TEXT,
                        header_path=buffer_header,
                        page_number=buffer_page,
                        chunk_index=0,
                    )
                )
                buffer_parts = []
                buffer_tokens = 0
                buffer_page = item.page_number
                buffer_header = item.header_path

            buffer_parts.append(item.content)
            if buffer_tokens == 0:
                buffer_tokens = item_tokens
            else:
                buffer_tokens += _count_tokens("\n\n") + item_tokens

        # Flush remaining buffer.
        if buffer_parts:
            chunks.append(
                self._make_chunk(
                    content="\n\n".join(buffer_parts),
                    content_type=ContentType.TEXT,
                    header_path=buffer_header,
                    page_number=buffer_page,
                    chunk_index=0,
                )
            )

        return chunks

    def _split_oversized_text(
        self, item: TransformedContent
    ) -> List[Chunk]:
        """Split a single oversized text item using semchunk.

        Args:
            item: A text item whose token count exceeds chunk_size_tokens.

        Returns:
            List of Chunk objects from the split.
        """
        parts = semchunk.chunk(
            text=item.content,
            chunk_size=self._chunk_size,
            token_counter=_count_tokens,
        )
        chunks: List[Chunk] = []
        for part in parts:
            chunks.append(
                self._make_chunk(
                    content=part,
                    content_type=ContentType.TEXT,
                    header_path=item.header_path,
                    page_number=item.page_number,
                    chunk_index=0,
                    metadata={"split_by": "semchunk"},
                )
            )
        return chunks

    def _apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply token overlap between consecutive text chunks.

        Overlap is only applied between adjacent chunks that are both
        TEXT type. Tables and image descriptions do not participate
        in overlap.

        Args:
            chunks: List of chunks before overlap.

        Returns:
            New list of chunks with overlap applied.
        """
        if self._overlap <= 0 or len(chunks) < 2:
            return chunks

        result: List[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = result[-1]
            curr = chunks[i]

            # Only apply overlap between two consecutive TEXT chunks.
            if (
                prev.content_type == ContentType.TEXT
                and curr.content_type == ContentType.TEXT
            ):
                overlap_text = self._get_overlap_text(prev.content)
                if overlap_text:
                    new_content = overlap_text + "\n\n" + curr.content
                    result.append(
                        self._make_chunk(
                            content=new_content,
                            content_type=curr.content_type,
                            header_path=curr.header_path,
                            page_number=curr.page_number,
                            chunk_index=0,
                            metadata=dict(curr.metadata),
                        )
                    )
                else:
                    result.append(curr)
            else:
                result.append(curr)

        return result

    def _get_overlap_text(self, text: str) -> str:
        """Extract the last overlap_tokens worth of text from a chunk.

        Args:
            text: The source chunk text.

        Returns:
            The trailing text corresponding to overlap_tokens, or empty
            string if overlap is zero or text is too short.
        """
        if self._overlap <= 0 or not text:
            return ""

        token_ids = _ENCODING.encode(text)
        if len(token_ids) <= self._overlap:
            return ""

        overlap_ids = token_ids[-self._overlap:]
        return _ENCODING.decode(overlap_ids)

    def _make_chunk(
        self,
        content: str,
        content_type: ContentType,
        header_path: str,
        page_number: int,
        chunk_index: int,
        metadata: Dict[str, Any] | None = None,
    ) -> Chunk:
        """Factory method to create a Chunk with computed token count.

        Args:
            content: The chunk text.
            content_type: Type of content.
            header_path: Heading hierarchy path.
            page_number: Source page number.
            chunk_index: Sequential index (may be reassigned later).
            metadata: Optional metadata dictionary.

        Returns:
            A new Chunk object.
        """
        return Chunk(
            content=content,
            content_type=content_type,
            chunk_index=chunk_index,
            header_path=header_path,
            page_number=page_number,
            token_count=_count_tokens(content),
            metadata=metadata if metadata is not None else {},
        )
