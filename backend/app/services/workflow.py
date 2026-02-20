"""Workflow orchestrator for running the full document processing pipeline.

Connects all six pipeline stages (parse, extract, transform, chunk, embed,
index) into a single sequential workflow. Tracks progress in the database
via WorkflowRun and WorkflowStage records.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from sqlalchemy.orm import Session

from app.core.config import KnowledgeForgeConfig

if TYPE_CHECKING:
    from app.core.workflow_config import ResolvedWorkflowConfig
from app.models.database import Document, WorkflowRun, WorkflowStage
from app.services.chunking import StructureAwareChunker
from app.services.embedding import Embedder
from app.services.extraction import ContentExtractor
from app.services.indexing import ChromaIndexer
from app.observability.tracing import traced
from app.services.parsing import DocumentParser
from app.services.transformation import ContentTransformer

logger = logging.getLogger(__name__)

STAGE_NAMES: List[str] = [
    "parse",
    "extract",
    "transform",
    "chunk",
    "embed",
    "index",
]


class WorkflowOrchestrator:
    """Orchestrates the full document processing pipeline.

    Accepts optional service instances for testability; creates default
    instances from config when None. Each call to process_document opens
    its own database session to avoid long-lived sessions.
    """

    def __init__(
        self,
        config: KnowledgeForgeConfig,
        session_factory: Callable[[], Session],
        parser: Optional[DocumentParser] = None,
        extractor: Optional[ContentExtractor] = None,
        transformer: Optional[ContentTransformer] = None,
        chunker: Optional[StructureAwareChunker] = None,
        embedder: Optional[Embedder] = None,
        indexer: Optional[ChromaIndexer] = None,
        workflow_config: Optional["ResolvedWorkflowConfig"] = None,
    ) -> None:
        """Initialize the workflow orchestrator.

        Args:
            config: KnowledgeForge configuration object.
            session_factory: Callable that returns a new SQLAlchemy session.
            parser: Optional document parser instance.
            extractor: Optional content extractor instance.
            transformer: Optional content transformer instance.
            chunker: Optional structure-aware chunker instance.
            embedder: Optional embedder instance.
            indexer: Optional ChromaDB indexer instance.
            workflow_config: Optional resolved workflow configuration for
                per-workflow overrides and stage toggling.
        """
        self.config = config
        self.session_factory = session_factory
        self.workflow_config = workflow_config

        # When a workflow config is provided, build services from its merged config
        if workflow_config is not None:
            # Build a KnowledgeForgeConfig from the resolved workflow fields
            svc_config = KnowledgeForgeConfig(
                source=workflow_config.source,
                processing=workflow_config.processing,
                indexing=workflow_config.indexing,
                database=config.database,
                observability=config.observability,
            )
        else:
            svc_config = config

        self.parser = parser or DocumentParser(svc_config)
        self.extractor = extractor or ContentExtractor(svc_config)
        self.transformer = transformer or ContentTransformer(svc_config)
        self.chunker = chunker or StructureAwareChunker(svc_config)
        self.embedder = embedder or Embedder(svc_config)
        self.indexer = indexer or ChromaIndexer(svc_config)

    @traced(run_type="chain", name="process_document")
    def process_document(self, document: Document) -> WorkflowRun:
        """Run the full processing pipeline for a document.

        Creates a WorkflowRun with six WorkflowStage records, then
        executes each stage sequentially. On success, marks the document
        as "indexed" and the run as "success". On failure, marks both
        as "failed" and stops processing.

        Args:
            document: The Document ORM instance to process.

        Returns:
            The WorkflowRun record with final status.
        """
        session = self.session_factory()
        try:
            run = self._execute_pipeline(session, document)
            # Refresh then expunge to allow attribute access after close
            session.refresh(run)
            session.expunge(run)
            return run
        finally:
            session.close()

    def _execute_pipeline(
        self, session: Session, document: Document
    ) -> WorkflowRun:
        """Execute the pipeline within a session.

        Args:
            session: Active SQLAlchemy session.
            document: The Document to process.

        Returns:
            The WorkflowRun record.
        """
        # Re-attach document to this session
        document = session.merge(document)

        # Update document status
        document.status = "processing"
        session.commit()

        # Determine workflow name for tagging
        wf_name = (
            self.workflow_config.name if self.workflow_config is not None else None
        )

        # Create workflow run
        run = WorkflowRun(
            document_id=document.id,
            workflow_id=wf_name,
            status="in_progress",
        )
        session.add(run)
        session.flush()

        # Create stage records
        stages: dict[str, WorkflowStage] = {}
        for name in STAGE_NAMES:
            stage = WorkflowStage(
                run_id=run.id,
                stage_name=name,
                status="pending",
            )
            session.add(stage)
            stages[name] = stage

        session.commit()

        # Determine stage toggling
        stages_cfg = (
            self.workflow_config.stages if self.workflow_config is not None else None
        )

        try:
            # Validate: parse is always required
            if stages_cfg is not None and not stages_cfg.is_enabled("parse"):
                raise ValueError("The 'parse' stage cannot be disabled")

            # Stage 1: Parse (always required)
            parse_result = self._run_stage(
                session,
                stages["parse"],
                lambda: self.parser.parse(document.file_path),
            )
            stages["parse"].metadata_json = json.dumps({
                "pages": parse_result.page_count,
                "tokens": parse_result.estimated_token_count,
            })
            session.commit()

            # Stage 2: Extract
            if stages_cfg is not None and not stages_cfg.is_enabled("extract"):
                self._skip_stage(session, stages["extract"])
                extracted = []
            else:
                extracted = self._run_stage(
                    session,
                    stages["extract"],
                    lambda: self.extractor.extract(
                        parse_result, source_path=document.file_path
                    ),
                )
                stages["extract"].metadata_json = json.dumps({
                    "items": len(extracted),
                })
                session.commit()

            # Stage 3: Transform
            if stages_cfg is not None and not stages_cfg.is_enabled("transform"):
                self._skip_stage(session, stages["transform"])
                transform_result = None
            else:
                transform_result = self._run_stage(
                    session,
                    stages["transform"],
                    lambda: self.transformer.transform(
                        extracted, raw_document=parse_result.raw_document
                    ),
                )
                stages["transform"].metadata_json = json.dumps({
                    "items": len(transform_result.items),
                    "markdown_length": len(transform_result.document_markdown),
                })
                session.commit()

            # Stage 4: Chunk
            if stages_cfg is not None and not stages_cfg.is_enabled("chunk"):
                self._skip_stage(session, stages["chunk"])
                chunks = []
            else:
                chunk_input = transform_result.items if transform_result else []
                chunks = self._run_stage(
                    session,
                    stages["chunk"],
                    lambda: self.chunker.chunk(chunk_input),
                )
                stages["chunk"].metadata_json = json.dumps({
                    "chunks": len(chunks),
                    "total_tokens": sum(c.token_count for c in chunks),
                })
                session.commit()

            # Stage 5: Embed (if disabled, index is also skipped)
            embed_disabled = stages_cfg is not None and not stages_cfg.is_enabled("embed")
            if embed_disabled:
                self._skip_stage(session, stages["embed"])
                self._skip_stage(session, stages["index"])
                embed_result = None
                index_result = None
            else:
                embed_result = self._run_stage(
                    session,
                    stages["embed"],
                    lambda: self.embedder.embed(chunks),
                )
                stages["embed"].metadata_json = json.dumps({
                    "embedded": len(embed_result.embedded_chunks),
                    "skipped": embed_result.skipped_count,
                    "dimension": (
                        embed_result.embedded_chunks[0].metadata.get("dimension", 0)
                        if embed_result.embedded_chunks
                        else 0
                    ),
                })
                session.commit()

                # Stage 6: Index
                if stages_cfg is not None and not stages_cfg.is_enabled("index"):
                    self._skip_stage(session, stages["index"])
                    index_result = None
                else:
                    self.indexer.delete_document(document.id)
                    index_result = self._run_stage(
                        session,
                        stages["index"],
                        lambda: self.indexer.index(
                            embed_result.embedded_chunks,
                            document_id=document.id,
                            file_name=document.file_name,
                            version=document.version,
                            file_path=document.file_path,
                        ),
                    )
                    stages["index"].metadata_json = json.dumps({
                        "collection": index_result.collection_name,
                        "indexed": index_result.total_indexed,
                    })
                    session.commit()

            # Success: update run and document
            run.status = "success"
            run.completed_at = datetime.now(timezone.utc)
            run.total_chunks = index_result.total_indexed if index_result else 0
            run.total_tokens = sum(c.token_count for c in chunks) if chunks else 0
            document.status = "indexed"
            session.commit()

            logger.info(
                "Pipeline completed for '%s': %d chunks, %d tokens",
                document.file_name,
                run.total_chunks,
                run.total_tokens,
            )

        except Exception as exc:
            # Mark run and document as failed
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = str(exc)
            document.status = "failed"
            session.commit()

            logger.error(
                "Pipeline failed for '%s': %s",
                document.file_name,
                exc,
                exc_info=True,
            )

        return run

    @traced(run_type="chain", name="run_stage")
    def _run_stage(
        self,
        session: Session,
        stage: WorkflowStage,
        fn: Callable[[], Any],
    ) -> Any:
        """Execute a single pipeline stage with status tracking.

        Sets started_at before calling fn, and completed_at/status after.
        On error, records the error message and re-raises.

        Args:
            session: Active SQLAlchemy session.
            stage: The WorkflowStage record to update.
            fn: Callable that performs the stage work.

        Returns:
            The result of fn().

        Raises:
            Exception: Re-raises any exception from fn after recording it.
        """
        stage.status = "in_progress"
        stage.started_at = datetime.now(timezone.utc)
        session.commit()

        try:
            result = fn()
            stage.status = "success"
            stage.completed_at = datetime.now(timezone.utc)
            session.commit()
            return result
        except Exception as exc:
            stage.status = "failed"
            stage.completed_at = datetime.now(timezone.utc)
            stage.error_message = f"{type(exc).__name__}: {exc}"
            session.commit()
            raise

    def _skip_stage(
        self,
        session: Session,
        stage: WorkflowStage,
    ) -> None:
        """Mark a pipeline stage as skipped.

        Args:
            session: Active SQLAlchemy session.
            stage: The WorkflowStage record to mark as skipped.
        """
        now = datetime.now(timezone.utc)
        stage.status = "skipped"
        stage.started_at = now
        stage.completed_at = now
        session.commit()
