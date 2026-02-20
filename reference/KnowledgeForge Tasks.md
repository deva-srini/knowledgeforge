# KnowledgeForge Backend — Task Breakdown

> Each task is designed to be a self-contained unit that can be given to Claude Code as a prompt. Tasks are ordered by dependency. Each task specifies what to build, inputs, outputs, and acceptance criteria.

---

## Phase 0: Project Scaffolding

### Task 0.1 — Project Setup & Folder Structure
**Prompt for Claude Code:**
> Set up a Python 3.11 project for KnowledgeForge backend following a FastAPI project structure. Create the folder structure as specified below. Initialize a virtual environment, create `requirements.txt` with initial dependencies (langchain, langchain-community, chromadb, sqlalchemy, pyyaml, docling, watchdog, fastapi, uvicorn, python-multipart, langsmith, pydantic, pydantic-settings), and create both `app/main.py` (FastAPI app with lifespan) and `cli.py` (CLI entry point using argparse). CLI subcommands: `start`, `process`, `status`, `metrics`. Each subcommand should print a placeholder message for now. The FastAPI app should have a basic health endpoint returning `{"status": "ok"}`.

```
knowledgeforge/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app + lifespan
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # Config loader + Pydantic validation
│   │   │   └── logging.py         # Logging setup
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── health.py
│   │   │       ├── documents.py
│   │   │       └── metrics.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── database.py        # SQLAlchemy ORM models
│   │   │   └── schemas.py         # Pydantic request/response schemas
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── session.py         # DB engine, session, init_db
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── filewatcher.py
│   │   │   ├── parsing.py
│   │   │   ├── extraction.py
│   │   │   ├── transformation.py
│   │   │   ├── chunking.py
│   │   │   ├── embedding.py
│   │   │   ├── indexing.py
│   │   │   └── workflow.py        # Pipeline orchestration
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   └── collector.py
│   │   └── observability/
│   │       ├── __init__.py
│   │       └── tracing.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_filewatcher.py
│   │   ├── test_parsing.py
│   │   ├── test_extraction.py
│   │   ├── test_chunking.py
│   │   ├── test_embedding.py
│   │   ├── test_indexing.py
│   │   ├── test_workflow.py
│   │   └── test_e2e.py
│   ├── cli.py
│   ├── requirements.txt
│   └── README.md
├── frontend/                      # Out of scope
├── kf_config.yaml                 # Project root config
└── data/
    ├── source/
    └── staging/
```

**Acceptance:**
- `python cli.py start` prints "Starting KnowledgeForge..." and launches uvicorn
- `GET /api/v1/health` returns `{"status": "ok"}`
- All folders and `__init__.py` files exist
- `pip install -r requirements.txt` succeeds

---

### Task 0.2 — Configuration Loader
**Prompt for Claude Code:**
> Create `kf_config.yaml` at the project root with the full default configuration (see below) and build `app/core/config.py` that loads and validates it. Use Pydantic models to define the config schema with sensible defaults. The loader should be importable as `from app.core.config import load_config` and return a typed Pydantic config object. Support passing a custom config path. Add unit tests in `tests/test_config.py`.

**Config schema:**
```yaml
# kf_config.yaml
source:
  watch_folder: "./data/source"
  staging_folder: "./data/staging"
  file_patterns: ["*.pdf", "*.docx", "*.xlsx", "*.html", "*.pptx"]

processing:
  parsing:
    library: "docling"
  extraction:
    strategy: "auto"
  organisation:
    enabled: true
    table_format: "markdown"
  chunking:
    strategy: "structure_aware"
    chunk_size_tokens: 512
    chunk_overlap_tokens: 50
    skip_threshold_tokens: 1000
  embedding:
    model: "sentence-transformers/all-MiniLM-L6-v2"

indexing:
  vector_store: "chromadb"
  chromadb_path: "./data/chromadb"
  default_collection: "default"
  collection_mapping: {}

database:
  url: "sqlite:///./data/knowledgeforge.db"

observability:
  langsmith_enabled: false
  langsmith_project: "knowledgeforge"
```

**Acceptance:**
- `load_config()` returns typed config object
- `load_config("path/to/custom.yaml")` works for custom paths
- Missing fields fall back to defaults
- Invalid values raise clear validation errors
- Unit tests pass

---

## Phase 1: Database & Core Models

### Task 1.1 — SQLite Database Models & Session Management
**Prompt for Claude Code:**
> Create SQLAlchemy ORM models in `app/models/database.py` for three tables: `documents`, `workflow_runs`, `workflow_stages`. Create `app/db/session.py` with session management (get_session, init_db). Use UUID primary keys, proper foreign key relationships, and datetime defaults. The `init_db()` function should create all tables. NOTE: Chunks are stored only in ChromaDB, not in SQLite — no chunks table needed. Create Pydantic response schemas in `app/models/schemas.py` for API responses. Add unit tests that create an in-memory SQLite DB, insert records into all tables, and query them.

**Table schemas:**

**documents**: id (UUID PK), file_name, file_path, file_type, version (int, default 1), file_hash (SHA-256), status (pending/processing/indexed/failed), created_at, updated_at

**workflow_runs**: id (UUID PK), document_id (FK), status (pending/in_progress/success/failed), started_at, completed_at, error_message (nullable), total_chunks (int, default 0), total_tokens (int, default 0)

**workflow_stages**: id (UUID PK), run_id (FK), stage_name (parse/extract/organise/chunk/embed/index), status (pending/in_progress/success/failed/skipped), started_at (nullable), completed_at (nullable), metadata_json (nullable TEXT for JSON), error_message (nullable)

**Acceptance:**
- `init_db()` creates all 3 tables
- CRUD operations work for all tables
- Foreign key constraints enforced
- Pydantic schemas created for Document, WorkflowRun, WorkflowStage responses
- All tests pass

---

## Phase 2: Document Ingestion

### Task 2.1 — File Watcher with Versioning
**Prompt for Claude Code:**
> Build `app/services/filewatcher.py` using the `watchdog` library. It should watch the folder specified in `kf_config.yaml` (`source.watch_folder`) for new or modified files matching the configured patterns (`source.file_patterns`). When a matching file is detected:
> 1. Compute SHA-256 hash of the file.
> 2. Check if a document with the same file_name exists in the DB. If yes and hash matches → skip. If yes and hash differs → create new record with incremented version.
> 3. Copy the file to the staging folder (`source.staging_folder`).
> 4. Create/update the document record in SQLite.
>
> The file watcher must run as a background thread (non-blocking) so it can be started during FastAPI lifespan. Add a `scan_existing()` method that picks up all matching files already in the source folder (called on app startup). Use `app/db/session.py` for DB operations and `app/core/config.py` for configuration. Add unit tests in `tests/test_filewatcher.py`.

**Acceptance:**
- Dropping a file into source folder triggers detection
- File is copied to staging with document record in SQLite
- Same file uploaded twice → skipped (hash match)
- Modified file → new version created (version incremented)
- `scan_existing()` picks up all matching files on startup
- Watcher runs as background thread
- Tests pass

---

## Phase 3: Document Parsing

### Task 3.1 — Document Parser
**Prompt for Claude Code:**
> Build `app/services/parsing.py` using the Docling library for document parsing. Create a `DocumentParser` class with a `parse(file_path: str) -> ParseResult` method. The `ParseResult` dataclass should contain: page_count, estimated_token_count, content_types (dict mapping content type to count, e.g. {"text": 8, "table": 3, "image": 1}), structure (list of page-level structure info), and raw_document (the Docling document object for downstream use). Handle PDF, DOCX, and HTML at minimum. If Docling cannot handle a format, fall back to basic text extraction. Add proper error handling and logging. Add unit tests in `tests/test_parsing.py` with a sample PDF.

**Acceptance:**
- Parses a PDF and returns correct page count and structure
- Identifies tables vs text sections
- Returns estimated token count
- Handles unsupported formats gracefully
- Tests pass

---

## Phase 4: Information Extraction

### Task 4.1 — Content Extractor
**Prompt for Claude Code:**
> Build `app/services/extraction.py` with a `ContentExtractor` class. It takes the `ParseResult` from the parsing step and extracts all content into a list of `ExtractedContent` objects. Each `ExtractedContent` has: content (string), content_type (text/table/image_description), page_number, header_path (the heading hierarchy, e.g. "Chapter 1 > Section 2"), and metadata (dict). For text: extract directly from Docling's output. For tables: extract as structured data. Strategy selection (direct/OCR/agentic) should be based on the config `processing.extraction.strategy`. For Phase 1, implement "direct" and "auto" (which defaults to direct for digital docs). Add unit tests in `tests/test_extraction.py`.

**Acceptance:**
- Extracts text content with correct page numbers
- Extracts tables as structured data
- Preserves header hierarchy
- Handles extraction errors per-section without failing the whole document
- Tests pass

---

## Phase 5: Information Organisation

### Task 5.1 — Content Transformer
**Prompt for Claude Code:**
> Build `app/services/transformation.py` with a `ContentTransformer` class. It takes a list of `ExtractedContent` objects and optionally transforms them. If `processing.organisation.enabled` is true in config: convert tables to markdown format, clean up whitespace and formatting issues, normalise text encoding. If disabled, pass through unchanged. Return a list of `TransformedContent` objects (same structure as ExtractedContent but post-transformation). Add unit tests in `tests/test_transformation.py`.

**Acceptance:**
- Tables converted to clean markdown
- Text content cleaned (extra whitespace removed, encoding normalised)
- When disabled, content passes through unchanged
- Tests pass

---

## Phase 6: Chunking

### Task 6.1 — Structure-Aware Chunker
**Prompt for Claude Code:**
> Build `app/services/chunking.py` with a `StructureAwareChunker` class. It takes a list of `TransformedContent` objects and the chunking config, and produces a list of `Chunk` objects. Each `Chunk` has: content, content_type, chunk_index, header_path, page_number, token_count, metadata (dict). Implement the following logic:
> 1. If total document token count < `skip_threshold_tokens`, return the entire document as a single chunk.
> 2. Otherwise, chunk using structure-aware splitting: never split tables (keep as whole chunks), respect header boundaries (prefer splitting between sections), use `chunk_size_tokens` and `chunk_overlap_tokens` from config.
> 3. Use LangChain's token counting utilities for accurate token counts.
> Add unit tests in `tests/test_chunking.py` including edge cases (very small doc, doc with only tables, very large section).

**Acceptance:**
- Small documents remain as single chunk
- Tables are never split across chunks
- Header boundaries respected
- Chunk sizes within configured limits (with tolerance for table chunks)
- Overlap correctly applied between text chunks
- Token counts accurate
- Tests pass

---

## Phase 7: Embedding

### Task 7.1 — Embedding Generator
**Prompt for Claude Code:**
> Build `app/services/embedding.py` with an `Embedder` class. It takes a list of `Chunk` objects and generates embeddings using the model specified in config (`processing.embedding.model`). Use LangChain's embedding integrations (HuggingFace sentence-transformers). Return a list of `EmbeddedChunk` objects (Chunk + embedding vector). Implement batching for efficiency (batch size configurable, default 32). Add proper error handling — if embedding fails for a chunk, log the error and skip that chunk. Add unit tests in `tests/test_embedding.py`.

**Acceptance:**
- Generates embeddings for all chunks
- Embedding dimensions match the model's output
- Batching works correctly
- Failed chunks are logged and skipped, not crashing the process
- Tests pass

---

## Phase 8: Indexing

### Task 8.1 — ChromaDB Indexer
**Prompt for Claude Code:**
> Build `app/services/indexing.py` with a `ChromaIndexer` class. It takes a list of `EmbeddedChunk` objects and indexes them into ChromaDB. Determine the target collection based on `indexing.collection_mapping` in config (match file path patterns to collections), falling back to `indexing.default_collection`. Store the chunk content, embedding vector, and metadata (document_id, file_name, version, page_number, chunk_index, content_type, header_path). Use persistent ChromaDB storage at `indexing.chromadb_path`. Handle upserts — if a chunk with the same document_id and chunk_index exists, update it. Return the ChromaDB IDs and total count indexed. NOTE: Chunks live ONLY in ChromaDB — no SQLite chunks table. Add unit tests in `tests/test_indexing.py`.

**Acceptance:**
- Chunks indexed into correct ChromaDB collections
- Metadata stored correctly and queryable
- Upserts work (re-indexing same doc doesn't create duplicates)
- Collection mapping works based on file path patterns
- Persistent storage works (data survives restart)
- Tests pass

---

## Phase 9: Pipeline Orchestration

### Task 9.1 — Workflow Orchestrator
**Prompt for Claude Code:**
> Build `app/services/workflow.py` with a `WorkflowOrchestrator` class that runs the full document processing pipeline: parse → extract → transform → chunk → embed → index. For each document:
> 1. Create a `workflow_run` record in SQLite.
> 2. For each stage, create a `workflow_stage` record and update its status (pending → in_progress → success/failed).
> 3. If a stage fails, mark the run as failed, log the error, and continue to the next document.
> 4. On success, update `workflow_runs.total_chunks` with the count from ChromaDB indexer and set document status to "indexed".
>
> The orchestrator should accept a single file path or process all pending documents. Use dependency injection — accept all service handlers (parser, extractor, etc.) as constructor parameters. Add integration tests in `tests/test_workflow.py` that run the full pipeline on a sample document.

**Acceptance:**
- Full pipeline runs end-to-end on a sample PDF
- All workflow_run and workflow_stage records created with correct statuses and timestamps
- Chunks stored in ChromaDB, total_chunks updated in workflow_runs
- Failed stage → run marked as failed, error logged, other documents still processed
- Integration test passes

---

### Task 9.2 — Wire Up CLI & FastAPI Lifespan
**Prompt for Claude Code:**
> Update `cli.py` and `app/main.py` to wire up real implementations:
>
> **`app/main.py` (FastAPI with lifespan):**
> - On startup: load `kf_config.yaml` from project root, init SQLite DB, start file watcher in background thread, scan existing files in source folder and queue unprocessed ones for processing via WorkflowOrchestrator.
> - On shutdown: gracefully stop file watcher.
> - Include all API routers from `api/v1/`.
>
> **`cli.py` commands:**
> - `python cli.py start` — Start FastAPI server via uvicorn (triggers lifespan → auto-starts watcher)
> - `python cli.py start --config path/to/kf_config.yaml` — Start with custom config
> - `python cli.py process --file <path>` — Process a specific file (one-off, no server)
> - `python cli.py process --file <path> --force` — Re-process even if already indexed
> - `python cli.py status` — Print document statuses in a readable table
> - `python cli.py metrics` — Print processing metrics
>
> The primary mode is `cli.py start` — the app starts, watches folders, and processes automatically.

**Acceptance:**
- `cli.py start` launches server AND file watcher
- Files in source folder are auto-detected and processed on startup
- New files dropped during runtime are auto-detected and processed
- `process --file` works for one-off processing
- `status` shows document statuses in a readable format
- Errors handled gracefully

---

## Phase 10: Metrics & Observability

### Task 10.1 — Metrics Collection & Reporting
**Prompt for Claude Code:**
> Build `app/metrics/collector.py` with a `MetricsCollector` class that queries SQLite to compute and return processing metrics: total documents indexed, total documents by status, total chunks generated (sum of workflow_runs.total_chunks), average chunks per document, total processing time, average time per document, time per stage (average), number of ChromaDB collections, documents processed over time (last 7 days). Return as a typed dataclass. Add a `print_metrics()` function for CLI output. Wire into `api/v1/metrics.py` for the GET endpoint. Add unit tests.

**Acceptance:**
- All metrics computed correctly from SQLite data
- CLI output is clean and readable
- API endpoint returns metrics as JSON
- Empty database returns zeros (not errors)
- Tests pass

---

### Task 10.2 — LangSmith Observability Setup
**Prompt for Claude Code:**
> Build `app/observability/tracing.py` that sets up LangSmith tracing when `observability.langsmith_enabled` is true in config. Create a `setup_tracing()` function that configures environment variables (LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT, LANGCHAIN_API_KEY from env). Create a decorator `@traced` that wraps pipeline stages with LangSmith tracing. Integrate into `services/workflow.py` so each stage is traced. Make this completely optional — if disabled or if env vars are missing, everything works normally.

**Acceptance:**
- When enabled + API key set: traces appear in LangSmith
- When disabled: no errors, no tracing overhead
- Each pipeline stage appears as a separate span
- Token usage tracked for any LLM calls

---

## Phase 11: API Layer

### Task 11.1 — FastAPI Route Implementations
**Prompt for Claude Code:**
> Implement all API routes in `app/api/v1/`:
>
> **health.py:**
> - `GET /api/v1/health` — Returns {"status": "ok", "version": "0.1.0", "watcher_active": bool}
>
> **documents.py:**
> - `GET /api/v1/documents` — List all documents with status, version, chunk count (from workflow_runs.total_chunks)
> - `GET /api/v1/documents/{doc_id}/status` — Detailed status including all workflow stages
> - `POST /api/v1/documents/process` — Trigger processing of all pending documents (background task)
> - `POST /api/v1/documents/{doc_id}/reprocess` — Re-trigger processing for a specific document
>
> **metrics.py:**
> - `GET /api/v1/metrics` — Returns processing metrics from MetricsCollector
>
> Use Pydantic response models from `app/models/schemas.py`. Use FastAPI's BackgroundTasks for async processing.

**Acceptance:**
- All endpoints return correct data with proper response models
- Processing triggers work asynchronously
- Document listing includes chunk counts
- Error responses are properly structured

---

## Phase 12: Testing & Polish

### Task 12.1 — End-to-End Integration Test
**Prompt for Claude Code:**
> Create `tests/test_e2e.py` with a comprehensive end-to-end test. The test should: place a sample PDF and a sample DOCX into the source folder, run the full pipeline via WorkflowOrchestrator, verify documents are in SQLite with status "indexed", verify chunks exist in ChromaDB (query to confirm), verify workflow_runs.total_chunks is correct, verify metrics are correct, then place a modified version of the PDF, re-run the pipeline, and verify versioning works (new version created, new chunks indexed). Use pytest fixtures for setup/teardown (clean data directories, fresh DB, fresh ChromaDB).

**Acceptance:**
- Full pipeline works end-to-end for PDF and DOCX
- ChromaDB contains correct chunks with metadata
- Versioning works correctly
- All assertions pass
- Clean setup/teardown

---

### Task 12.2 — README & Documentation
**Prompt for Claude Code:**
> Create a comprehensive `README.md` covering: project overview, architecture diagram (ASCII), setup instructions (venv, install, config), quick start (`cli.py start` and watch it work), usage guide (all CLI commands with examples), kf_config.yaml reference (all options), API reference (all endpoints), development guide (running tests, project structure), and known limitations / future work.

**Acceptance:**
- A new developer can set up and run the project using only the README
- All CLI commands and config options documented
- API endpoints documented

---

## Task Dependency Graph

```
0.1 (Scaffold) → 0.2 (Config)
                      ↓
                 1.1 (Database)
                      ↓
              2.1 (File Watcher + Versioning)
                      ↓
              3.1 (Parser)
                      ↓
              4.1 (Extractor)
                      ↓
              5.1 (Transformer)
                      ↓
              6.1 (Chunker)
                      ↓
              7.1 (Embedder)
                      ↓
              8.1 (Indexer)
                      ↓
         9.1 (Workflow) → 9.2 (CLI + Lifespan)
                      ↓
        10.1 (Metrics) + 10.2 (LangSmith)
                      ↓
              11.1 (API Routes)
                      ↓
         12.1 (E2E Tests) → 12.2 (README)
```

---

## Estimated Effort Per Task

| Task | Estimated Time | Complexity |
|---|---|---|
| 0.1 Scaffold | 15 min | Low |
| 0.2 Config | 20 min | Low |
| 1.1 Database | 30 min | Medium |
| 2.1 File Watcher + Versioning | 35 min | Medium |
| 3.1 Parser | 45 min | High |
| 4.1 Extractor | 45 min | High |
| 5.1 Transformer | 20 min | Low |
| 6.1 Chunker | 45 min | High |
| 7.1 Embedder | 25 min | Medium |
| 8.1 Indexer | 30 min | Medium |
| 9.1 Workflow | 45 min | High |
| 9.2 CLI + Lifespan | 30 min | Medium |
| 10.1 Metrics | 25 min | Medium |
| 10.2 LangSmith | 20 min | Medium |
| 11.1 API Routes | 30 min | Medium |
| 12.1 E2E Test | 30 min | Medium |
| 12.2 README | 20 min | Low |
| **Total** | **~8.5 hours** | |