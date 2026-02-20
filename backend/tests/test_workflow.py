"""Unit tests for the WorkflowOrchestrator service.

All tests use mocked pipeline services and an in-memory SQLite database
to verify orchestration logic without real parsing, embedding, or indexing.
"""

import json
from typing import Any, List
from unittest.mock import MagicMock, call, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import KnowledgeForgeConfig
from app.models.database import Base, Document, WorkflowRun, WorkflowStage
from app.services.chunking import Chunk
from app.services.embedding import EmbedResult, EmbeddedChunk
from app.services.extraction import ContentType, ExtractedContent
from app.services.indexing import IndexResult
from app.services.parsing import ParseResult
from app.services.transformation import TransformResult, TransformedContent
from app.core.workflow_config import ResolvedWorkflowConfig, StagesConfig, StageToggle
from app.services.workflow import STAGE_NAMES, WorkflowOrchestrator


@pytest.fixture()
def db_session_factory():
    """Create an in-memory SQLite database and return a session factory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return factory


@pytest.fixture()
def config():
    """Return a default KnowledgeForgeConfig."""
    return KnowledgeForgeConfig()


@pytest.fixture()
def sample_document(db_session_factory):
    """Create and persist a sample Document record."""
    session = db_session_factory()
    doc = Document(
        file_name="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        version=1,
        file_hash="abc123",
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    session.close()
    return doc


def _make_mock_services():
    """Create mocked pipeline services returning realistic results."""
    parser = MagicMock()
    parser.parse.return_value = ParseResult(
        page_count=3,
        estimated_token_count=500,
        content_types={"text": 10},
        raw_document=MagicMock(),
        raw_text="sample text",
    )

    extractor = MagicMock()
    extracted_items = [
        ExtractedContent(
            content="Hello world",
            content_type=ContentType.TEXT,
            page_number=1,
        ),
    ]
    extractor.extract.return_value = extracted_items

    transformer = MagicMock()
    transformed_items = [
        TransformedContent(
            content="Hello world",
            content_type=ContentType.TEXT,
            page_number=1,
        ),
    ]
    transformer.transform.return_value = TransformResult(
        items=transformed_items,
        document_markdown="# Hello\n\nworld",
    )

    chunker = MagicMock()
    chunks = [
        Chunk(
            content="Hello world",
            content_type=ContentType.TEXT,
            chunk_index=0,
            header_path="",
            page_number=1,
            token_count=3,
        ),
        Chunk(
            content="Another chunk",
            content_type=ContentType.TEXT,
            chunk_index=1,
            header_path="",
            page_number=2,
            token_count=5,
        ),
    ]
    chunker.chunk.return_value = chunks

    embedder = MagicMock()
    embedded_chunks = [
        EmbeddedChunk(
            chunk=chunks[0],
            embedding=[0.1, 0.2, 0.3],
            metadata={"model": "test", "dimension": 3},
        ),
        EmbeddedChunk(
            chunk=chunks[1],
            embedding=[0.4, 0.5, 0.6],
            metadata={"model": "test", "dimension": 3},
        ),
    ]
    embedder.embed.return_value = EmbedResult(
        embedded_chunks=embedded_chunks,
        skipped_count=0,
        total_chunks=2,
    )

    indexer = MagicMock()
    indexer.index.return_value = IndexResult(
        collection_name="default",
        indexed_ids=["doc_0", "doc_1"],
        total_indexed=2,
    )
    indexer.delete_document.return_value = 0

    return parser, extractor, transformer, chunker, embedder, indexer


class TestWorkflowOrchestrator:
    """Tests for the WorkflowOrchestrator."""

    def test_process_document_success(
        self, config, db_session_factory, sample_document
    ):
        """Full pipeline succeeds: Document → 'indexed', Run → 'success'."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "success"

        # Verify document status updated
        session = db_session_factory()
        doc = session.query(Document).filter_by(id=sample_document.id).one()
        assert doc.status == "indexed"
        session.close()

    def test_all_stages_created(
        self, config, db_session_factory, sample_document
    ):
        """Six WorkflowStage records are created with correct names."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        stages = (
            session.query(WorkflowStage)
            .filter_by(run_id=run.id)
            .order_by(WorkflowStage.stage_name)
            .all()
        )
        stage_names = sorted([s.stage_name for s in stages])
        assert stage_names == sorted(STAGE_NAMES)
        assert len(stages) == 6
        session.close()

    def test_stage_statuses_on_success(
        self, config, db_session_factory, sample_document
    ):
        """All stages have status 'success' with timestamps after successful run."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        stages = (
            session.query(WorkflowStage).filter_by(run_id=run.id).all()
        )
        for stage in stages:
            assert stage.status == "success"
            assert stage.started_at is not None
            assert stage.completed_at is not None
        session.close()

    def test_parse_failure_marks_run_failed(
        self, config, db_session_factory, sample_document
    ):
        """Parse error → run 'failed', document 'failed'."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        parser.parse.side_effect = RuntimeError("Parse error")

        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "failed"

        session = db_session_factory()
        doc = session.query(Document).filter_by(id=sample_document.id).one()
        assert doc.status == "failed"
        session.close()

    def test_embed_failure_marks_run_failed(
        self, config, db_session_factory, sample_document
    ):
        """Mid-pipeline error (embed) → run 'failed', correct stage 'failed'."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        embedder.embed.side_effect = RuntimeError("Embed error")

        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "failed"

        session = db_session_factory()
        embed_stage = (
            session.query(WorkflowStage)
            .filter_by(run_id=run.id, stage_name="embed")
            .one()
        )
        assert embed_stage.status == "failed"
        session.close()

    def test_failed_stage_has_error_message(
        self, config, db_session_factory, sample_document
    ):
        """Error message is stored in the failed stage record."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        parser.parse.side_effect = FileNotFoundError("File not found: /tmp/test.pdf")

        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        parse_stage = (
            session.query(WorkflowStage)
            .filter_by(run_id=run.id, stage_name="parse")
            .one()
        )
        assert parse_stage.error_message is not None
        assert "File not found" in parse_stage.error_message
        session.close()

    def test_subsequent_stages_stay_pending(
        self, config, db_session_factory, sample_document
    ):
        """After failure, later stages remain 'pending'."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        extractor.extract.side_effect = RuntimeError("Extract error")

        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        stages = (
            session.query(WorkflowStage).filter_by(run_id=run.id).all()
        )
        stage_map = {s.stage_name: s for s in stages}

        # Parse should succeed, extract should fail
        assert stage_map["parse"].status == "success"
        assert stage_map["extract"].status == "failed"

        # Subsequent stages should remain pending
        for name in ["transform", "chunk", "embed", "index"]:
            assert stage_map[name].status == "pending"
        session.close()

    def test_total_chunks_updated(
        self, config, db_session_factory, sample_document
    ):
        """WorkflowRun.total_chunks matches IndexResult.total_indexed."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.total_chunks == 2

    def test_total_tokens_updated(
        self, config, db_session_factory, sample_document
    ):
        """WorkflowRun.total_tokens computed from chunks."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        # Chunks have token_count 3 and 5
        assert run.total_tokens == 8

    def test_stage_metadata_stored(
        self, config, db_session_factory, sample_document
    ):
        """metadata_json is populated per stage."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        stages = (
            session.query(WorkflowStage).filter_by(run_id=run.id).all()
        )
        stage_map = {s.stage_name: s for s in stages}

        # Parse metadata
        parse_meta = json.loads(stage_map["parse"].metadata_json)
        assert parse_meta["pages"] == 3
        assert parse_meta["tokens"] == 500

        # Extract metadata
        extract_meta = json.loads(stage_map["extract"].metadata_json)
        assert extract_meta["items"] == 1

        # Transform metadata
        transform_meta = json.loads(stage_map["transform"].metadata_json)
        assert transform_meta["items"] == 1
        assert "markdown_length" in transform_meta

        # Chunk metadata
        chunk_meta = json.loads(stage_map["chunk"].metadata_json)
        assert chunk_meta["chunks"] == 2
        assert chunk_meta["total_tokens"] == 8

        # Embed metadata
        embed_meta = json.loads(stage_map["embed"].metadata_json)
        assert embed_meta["embedded"] == 2
        assert embed_meta["skipped"] == 0

        # Index metadata
        index_meta = json.loads(stage_map["index"].metadata_json)
        assert index_meta["collection"] == "default"
        assert index_meta["indexed"] == 2

        session.close()

    def test_default_services_created(self, config, db_session_factory):
        """Constructor creates services when None is passed."""
        with patch(
            "app.services.workflow.DocumentParser"
        ) as mock_parser_cls, patch(
            "app.services.workflow.ContentExtractor"
        ) as mock_extractor_cls, patch(
            "app.services.workflow.ContentTransformer"
        ) as mock_transformer_cls, patch(
            "app.services.workflow.StructureAwareChunker"
        ) as mock_chunker_cls, patch(
            "app.services.workflow.Embedder"
        ) as mock_embedder_cls, patch(
            "app.services.workflow.ChromaIndexer"
        ) as mock_indexer_cls:
            orchestrator = WorkflowOrchestrator(config, db_session_factory)

            mock_parser_cls.assert_called_once_with(config)
            mock_extractor_cls.assert_called_once_with(config)
            mock_transformer_cls.assert_called_once_with(config)
            mock_chunker_cls.assert_called_once_with(config)
            mock_embedder_cls.assert_called_once_with(config)
            mock_indexer_cls.assert_called_once_with(config)

    def test_delete_before_reindex(
        self, config, db_session_factory, sample_document
    ):
        """indexer.delete_document is called before indexer.index."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "success"
        # delete_document should be called before index
        indexer.delete_document.assert_called_once_with(sample_document.id)
        indexer.index.assert_called_once()

        # Verify ordering: delete was called before index
        delete_call_order = indexer.delete_document.call_args_list
        index_call_order = indexer.index.call_args_list
        assert len(delete_call_order) == 1
        assert len(index_call_order) == 1

    def test_run_completed_at_set_on_success(
        self, config, db_session_factory, sample_document
    ):
        """WorkflowRun.completed_at is set on success."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        assert run.completed_at is not None


class TestWorkflowStageSkipping:
    """Tests for stage skipping with workflow_config."""

    def test_skip_embed_and_index(
        self, config, db_session_factory, sample_document
    ):
        """Disabling embed also skips index; both have status 'skipped'."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        wf_config = ResolvedWorkflowConfig(
            name="test_wf",
            stages=StagesConfig(embed=StageToggle(enabled=False)),
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            workflow_config=wf_config,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "success"
        # Embed and index should not have been called
        embedder.embed.assert_not_called()
        indexer.index.assert_not_called()

        session = db_session_factory()
        stage_map = {
            s.stage_name: s
            for s in session.query(WorkflowStage).filter_by(run_id=run.id).all()
        }
        assert stage_map["embed"].status == "skipped"
        assert stage_map["index"].status == "skipped"
        assert stage_map["parse"].status == "success"
        assert stage_map["chunk"].status == "success"
        session.close()

    def test_skip_transform(
        self, config, db_session_factory, sample_document
    ):
        """Disabling transform marks it 'skipped', other stages still run."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        wf_config = ResolvedWorkflowConfig(
            name="test_wf",
            stages=StagesConfig(transform=StageToggle(enabled=False)),
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            workflow_config=wf_config,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "success"
        transformer.transform.assert_not_called()

        session = db_session_factory()
        stage_map = {
            s.stage_name: s
            for s in session.query(WorkflowStage).filter_by(run_id=run.id).all()
        }
        assert stage_map["transform"].status == "skipped"
        assert stage_map["parse"].status == "success"
        session.close()

    def test_disable_parse_raises_error(
        self, config, db_session_factory, sample_document
    ):
        """Disabling parse raises ValueError and marks run as failed."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        wf_config = ResolvedWorkflowConfig(
            name="test_wf",
            stages=StagesConfig(parse=StageToggle(enabled=False)),
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            workflow_config=wf_config,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "failed"
        assert "parse" in run.error_message.lower()

    def test_workflow_id_set_on_run(
        self, config, db_session_factory, sample_document
    ):
        """WorkflowRun.workflow_id is set from workflow_config.name."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        wf_config = ResolvedWorkflowConfig(name="my_workflow")
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            workflow_config=wf_config,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        db_run = session.query(WorkflowRun).filter_by(id=run.id).one()
        assert db_run.workflow_id == "my_workflow"
        session.close()

    def test_no_workflow_config_no_workflow_id(
        self, config, db_session_factory, sample_document
    ):
        """Without workflow_config, workflow_id is None."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
        )

        run = orchestrator.process_document(sample_document)

        session = db_session_factory()
        db_run = session.query(WorkflowRun).filter_by(id=run.id).one()
        assert db_run.workflow_id is None
        session.close()

    def test_all_stages_enabled_same_as_default(
        self, config, db_session_factory, sample_document
    ):
        """All stages enabled behaves identically to no workflow_config."""
        parser, extractor, transformer, chunker, embedder, indexer = (
            _make_mock_services()
        )
        wf_config = ResolvedWorkflowConfig(name="full")
        orchestrator = WorkflowOrchestrator(
            config,
            db_session_factory,
            parser=parser,
            extractor=extractor,
            transformer=transformer,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            workflow_config=wf_config,
        )

        run = orchestrator.process_document(sample_document)

        assert run.status == "success"
        assert run.total_chunks == 2
        assert run.total_tokens == 8
