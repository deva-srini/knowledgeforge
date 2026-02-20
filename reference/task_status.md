# KnowledgeForge — Task Status Tracker

> Last updated: 2026-02-17

## Summary

| Status | Count |
|--------|-------|
| Completed | 13 |
| In Progress | 0 |
| Pending | 4 |
| **Total** | **17** |

---

## Phase 0: Project Scaffolding

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 0.1 | Project Setup & Folder Structure | **Completed** | Full directory structure, `__init__.py` files, FastAPI app with lifespan, CLI with argparse, health endpoint, `requirements.txt` |
| 0.2 | Configuration Loader | **Completed** | `kf_config.yaml` with defaults, Pydantic-validated config loader, `load_config()`, 12 unit tests passing |

## Phase 1: Database & Core Models

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | SQLite Database Models & Session Management | **Completed** | 3 ORM models (Document, WorkflowRun, WorkflowStage), session management, Pydantic response schemas, 16 unit tests passing |

## Phase 2: Document Ingestion

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | File Watcher with Versioning | **Completed** | Watchdog-based watcher, SHA-256 hashing, versioning, staging copy, `scan_existing()`, background thread, 13 unit tests passing |

## Phase 3: Document Parsing

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Document Parser | **Completed** | `DocumentParser` with Docling integration, lazy `DocumentConverter` init with optimized `PdfPipelineOptions` (OCR disabled, backend text, FAST table mode, GPU auto-detect), `PageStructure`/`ParseResult` dataclasses, per-page content tracking, fallback text extraction, tiktoken token estimation. 17 unit tests passing (HTML, PDF via `travel.pdf`, fallback, dataclass) |

## Phase 4: Information Extraction

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 4.1 | Content Extractor | **Completed** | `ContentExtractor` class with `ExtractedContent` dataclass, `ContentType` enum, header hierarchy tracking, table export to markdown, picture description extraction, per-item error handling, fallback for non-Docling docs. 32 unit tests passing. |

## Phase 5: Information Organisation

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 5.1 | Content Transformer | **Completed** | `ContentTransformer` class with `TransformedContent` dataclass, `TransformResult` return type, `_generate_document_markdown()` for full-document structured markdown via Docling, `markdown_generation` config field, whitespace cleanup, Unicode NFC encoding normalization, smart quote/dash/ellipsis replacement, markdown table formatting, pass-through mode when disabled, per-item error handling with fallback. 58 unit tests passing. |

## Phase 6: Chunking

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 6.1 | Structure-Aware Chunker | **Completed** | `StructureAwareChunker` class with `Chunk` dataclass, skip-threshold path, header-based section grouping, standalone table/image chunks, text accumulation with token splitting via semchunk, overlap between consecutive text chunks. 53 unit tests passing. Pipeline demo verified on travel.pdf (54 items → 35 chunks, 4796 tokens, 8ms). |

## Phase 7: Embedding

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 7.1 | Embedding Generator | **Completed** | Multi-provider architecture with `EmbeddingClient` ABC, `SentenceTransformerClient` (lazy loading, GPU auto-detect), `Embedder` orchestrator with batching and fallback error handling, `EmbeddedChunk`/`EmbedResult` dataclasses. 13 unit tests passing. Pipeline demo: 35 chunks embedded in 8.3s, 384-dim vectors, L2 norm=1.0. |

## Phase 8: Indexing

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 8.1 | ChromaDB Indexer | **Completed** | `ChromaIndexer` class with `IndexResult` dataclass, PersistentClient storage, file-path pattern matching for collection routing, deterministic chunk IDs, upsert-based indexing, `delete_document()` for re-indexing. 14 unit tests passing. |

## Phase 9: Pipeline Orchestration

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 9.1 | Workflow Orchestrator | **Completed** | `WorkflowOrchestrator` class with DI, 6-stage sequential pipeline (parse→extract→transform→chunk→embed→index), DB tracking via WorkflowRun/WorkflowStage, per-stage metadata JSON, error handling with fail-fast, delete-before-reindex. 13 unit tests passing. |
| 9.2 | Wire Up CLI & FastAPI Lifespan | **Completed** | FastAPI lifespan wires config→DB→orchestrator→FileWatcher with `on_new_document` callback. CLI `process` command creates/finds Document records and runs pipeline. CLI `status` and `metrics` commands query DB for document status and processing stats. |

## Phase 10: Workflow Registry System

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 10.0 | Workflow Registry System | **Completed** | Per-workflow overlay configs, `WorkflowRegistryManager` with hot-reload, stage skipping, `workflow_id` on Document/WorkflowRun, CLI `--workflow` + `workflows` subcommand, API routes (`GET /workflows`, `POST /workflows/sync`), sample `fund_factsheet` workflow, stage-by-stage E2E demo (notebook + script). 55 new tests (29 config + 7 registry + 6 stage-skipping + 13 existing updated). 290 total tests passing. |

## Phase 11: Metrics & Observability

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 11.1 | Metrics Collection & Reporting | Pending | |
| 11.2 | LangSmith Observability Setup | Pending | |

## Phase 12: API Layer

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 12.1 | FastAPI Route Implementations | Pending | |

## Phase 13: Testing & Polish

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 13.1 | End-to-End Integration Test | Pending | |

---

## Snapshot: Task 4.1 Handoff (2026-02-08)

### Files Created/Modified for Task 4.1
- `backend/app/services/extraction.py` — Full implementation complete
- `backend/tests/test_extraction.py` — 32 tests written, 31 passing, 1 minor fix applied (not yet re-run)

### What Was Done
- `ContentExtractor` class with Docling `iterate_items()` traversal
- `ExtractedContent` dataclass: content, content_type, page_number, header_path, metadata
- `ContentType` enum: text, table, image_description
- `ExtractionStrategy` enum: auto (resolves to direct), direct, ocr, agentic
- Header hierarchy tracking via level-indexed stack (`_update_header_stack`)
- Table extraction: `export_to_markdown()` with grid fallback, captures num_rows/num_cols/caption
- Picture extraction: caption + classification + description metadata
- Per-item error handling: individual failures logged and skipped
- Fallback: raw text extraction when no Docling document available

### What Remains for Task 4.1
- Re-run `conda run -n rag python -m pytest tests/test_extraction.py -v` to confirm all 32 tests pass
- The fix was removing a `del bad_item.text` line in `TestPerSectionErrorHandling` (already applied)

### Test Counts by Task
| Task | Tests | Status |
|------|-------|--------|
| 0.2 Configuration | 13 | All passing |
| 1.1 Database | 16 | All passing |
| 2.1 File Watcher | 13 | All passing |
| 3.1 Document Parser | 17 | All passing |
| 4.1 Content Extractor | 32 | All passing |
| 5.1 Content Transformer | 58 | All passing |
| 6.1 Chunker | 53 | All passing |
| 7.1 Embedding | 13 | All passing |
| 8.1 Indexing | 14 | All passing |
| 9.1 Workflow | 19 | All passing (13 + 6 stage-skipping) |
| 10.0 Workflow Config | 29 | All passing |
| 10.0 Workflow Registry | 7 | All passing |
| E2E | 6 | All passing |
| **Total** | **290** | |

---

## Snapshot: Task 5.1 Handoff (2026-02-08)

### Files Created/Modified for Task 5.1
- `backend/app/services/transformation.py` — Full implementation
- `backend/tests/test_transformation.py` — 42 tests, all passing

### What Was Done
- `TransformedContent` dataclass mirroring `ExtractedContent` with `metadata["transformed"]` flag
- `ContentTransformer` class controlled by `config.processing.organisation.enabled`
- Text cleaning: Unicode NFC normalization, smart quote/dash/ellipsis/zero-width char replacement
- Whitespace cleanup: CRLF normalization, leading/trailing line trimming, blank line collapsing, multi-space collapsing
- Markdown table formatting: consistent cell padding, empty row removal, separator normalization
- Pass-through mode: when disabled, content wrapped as `TransformedContent` unchanged
- Per-item error handling: failures fall back to pass-through (never drop content)
- No new dependencies — uses only `re`, `unicodedata`, `dataclasses` from stdlib

---

## Snapshot: Task 5.1 Markdown Generation Enhancement (2026-02-09)

### Files Modified
- `backend/app/core/config.py` — Added `markdown_generation` field to `OrganisationConfig`
- `knowledgeforge/kf_config.yaml` — Added `markdown_generation: true` default
- `backend/app/services/transformation.py` — Added `TransformResult` dataclass, `_generate_document_markdown()` method, updated `transform()` to accept `raw_document` and return `TransformResult`
- `backend/tests/test_transformation.py` — 16 new tests (14 markdown generation + 2 TransformResult dataclass), updated existing tests for `TransformResult` return type
- `backend/tests/test_config.py` — 1 new test for `markdown_generation` config field
- `backend/test_pipeline_demo.py` — Updated to pass `raw_document` to transformer and use `TransformResult`

### What Was Done
- `TransformResult` dataclass bundles `items: List[TransformedContent]` + `document_markdown: str`
- `_generate_document_markdown()` calls Docling `export_to_markdown()` with `image_placeholder`, `page_break_placeholder`, `escape_underscores=False`, `escape_html=False`; applies encoding normalization and whitespace cleaning
- `markdown_generation` config field on `OrganisationConfig` (default `True`) controls generation
- `transform()` now accepts optional `raw_document` parameter and returns `TransformResult` instead of `List[TransformedContent]`
- 16 new tests added (58 total for transformation, 13 total for config)

### Pipeline Data Flow Update
```
Transformer now returns TransformResult:
  - items: List[TransformedContent]  (per-item cleaned content, unchanged)
  - document_markdown: str           (full structured markdown from Docling)
```

---

## Parsing Performance Notes

- **Converter init (model loading)**: ~95s cold start (one-time cost per app lifecycle)
- **First convert (cold)**: ~16s for 7-page PDF (includes GPU kernel warmup)
- **Warm convert**: ~4.5s for 7-page PDF (models already loaded)
- **Optimizations applied**: OCR disabled, force_backend_text=True, FAST table mode, GPU auto-detect
- In production, converter is a singleton — subsequent documents process in ~4-5s

---

## Snapshot: Task 6.1 In Progress (2026-02-10)

### Files Created/Modified for Task 6.1
- `backend/app/services/chunking.py` — Full implementation (was empty placeholder)
- `backend/tests/test_chunking.py` — ~48 tests across 10 test classes

### What Was Done
- `Chunk` dataclass: content, content_type, chunk_index, header_path, page_number, token_count, metadata
- `StructureAwareChunker` class with `chunk(items)` main method
- `_count_tokens()` module-level helper using tiktoken cl100k_base
- **Skip-threshold path**: documents below `skip_threshold_tokens` (1000) → single concatenated chunk with `{"chunking_strategy": "skip"}`
- **Structure-aware path**: groups consecutive items by `header_path`, dispatches tables/images as standalone chunks, accumulates text with token-bounded splitting
- **Oversized text splitting**: delegates to `semchunk.chunk()` for single items exceeding `chunk_size_tokens`
- **Overlap**: prepends last `chunk_overlap_tokens` from previous text chunk to next text chunk (not applied to/from tables or images)
- **Per-item error handling**: fallback chunk created on failure (never drops content)
- Test suite: TestChunkDataclass, TestCountTokens, TestSkipThreshold, TestTableChunking, TestImageDescriptionChunking, TestTextChunking, TestHeaderBoundaries, TestOverlap, TestChunkIndex, TestEdgeCases, TestMixedContent

### What Was Completed
1. Ran 53 unit tests — 52 passed on first run, 1 test fix (overlap test used identical vocabulary for both text blocks)
2. Full test suite: 202 tests passing, zero regressions
3. Updated `test_pipeline_demo.py` with Stage 4 (Chunking) — shows chunk stats and per-chunk details
4. Pipeline demo on `travel.pdf`: 54 items → 35 chunks, 4,796 tokens, 8ms

### Dependencies Used
- `semchunk` — inner text splitter for oversized blocks
- `tiktoken` cl100k_base — token counting
- `ChunkingConfig` from `config.py` (chunk_size_tokens=512, chunk_overlap_tokens=50, skip_threshold_tokens=1000)

### Pipeline Demo Results (travel.pdf)
```
Parse:       7 pages, 5077 tokens  [192s cold start]
Extract:     54 items              [18ms]
Transform:   54 items (9 changed)  [88ms]
Chunk:       35 chunks, 4796 tokens [8ms]
  - text: 16 chunks, 1318 tokens
  - table: 5 chunks, 3436 tokens
  - image_description: 14 chunks, 42 tokens
```

---

## Snapshot: Task 10.0 Workflow Registry System (2026-02-17)

### Files Created
- `knowledgeforge/workflows/registry.yaml` — workflow registry
- `knowledgeforge/workflows/fund_factsheet.yaml` — sample workflow (256 tokens, fund_factsheets collection)
- `backend/app/core/workflow_config.py` — models + loaders (StagesConfig, ResolvedWorkflowConfig, load_workflow, _deep_merge)
- `backend/app/services/workflow_registry.py` — WorkflowRegistryManager (lifecycle, hot-reload sync)
- `backend/app/api/v1/workflows.py` — REST endpoints (list, sync, status)
- `backend/tests/test_workflow_config.py` — 29 tests
- `backend/tests/test_workflow_registry.py` — 7 tests
- `knowledgeforge/notebooks/workflow_e2e_demo.ipynb` — stage-by-stage Jupyter demo
- `knowledgeforge/notebooks/workflow_e2e_demo.py` — script version

### Files Modified
- `database.py`: added nullable `workflow_id` to Document and WorkflowRun
- `schemas.py`: added `workflow_id` to response schemas
- `filewatcher.py`: added `workflow_name`, `force_rerun` params; filter by (file_name, workflow_id)
- `workflow.py`: added `workflow_config` param, stage skipping, `_skip_stage()` method
- `main.py`: integrated WorkflowRegistryManager in lifespan, added workflows router
- `cli.py`: added `--workflow` to process, added `workflows` subcommand
- `test_workflow.py`: added 6 stage-skipping tests (13 → 19)

### E2E Demo Results (bgf_factsheet.pdf, fund_factsheet workflow)
```
Parse:       4 pages, 3373 tokens     [114.9s]
Extract:     98 items (84 text, 4 table, 10 image)  [0.01s]
Transform:   98 items, 12394 char markdown  [0.02s]
Chunk:       33 chunks, 3049 tokens   [0.01s]
Embed:       33 embedded, 384 dim     [2.94s]
Index:       33 into fund_factsheets  [0.28s]
TOTAL:       118.14s
```

---

## Environment Notes

- **Python**: 3.11.14 (venv: `/workspace/rag/knowledgeforge/.venv/`)
- **Activation**: `source /workspace/rag/knowledgeforge/.venv/bin/activate`
- **torch**: 2.6.0+cu124 (CUDA enabled)
- **onnxruntime-gpu**: 1.24.1 (CUDAExecutionProvider + TensorrtExecutionProvider)
- **GPU**: NVIDIA A40 (46GB VRAM)
- **All tests run from**: `knowledgeforge/backend/`
- **Note**: After adding columns to database.py, delete existing SQLite DB files (no Alembic migrations)
