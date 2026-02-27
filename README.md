# KnowledgeForge

**A production framework built around Docling — bringing workflow orchestration, structure-aware chunking, and automated file watching to document-based RAG systems.**

KnowledgeForge is centred on [Docling](https://github.com/DS4SD/docling)'s deep document understanding — its ability to parse pages, tables, figures, and header hierarchies from PDF, Word, Excel, HTML, and PowerPoint. Around that core, KnowledgeForge layers a full production stack: automated file watching, per-workflow configuration overlays, structure-aware semantic chunking that respects document structure, GPU-accelerated embedding, and ChromaDB indexing — turning raw documents into queryable knowledge assets with full lineage tracking.

It is the *knowledge creation* layer of a larger Agentic RAG system, feeding a downstream Explorer agent.

---

## Why Build This?

- **Docling as the foundation** — rather than treating parsing as a black box, KnowledgeForge is designed to exploit Docling's full structural output: header trees, table grids, figure metadata, and page-level layout — giving downstream chunking and retrieval a structural advantage over naive text splitting.
- **Workflow-first design** — multiple document workflows (e.g. `fund_factsheets`, `annual_reports`) run concurrently with independent watch folders, chunk sizes, and vector collections, hot-reloaded without restarting the service.
- **Full knowledge lineage** — SHA-256 versioning, per-stage run records, and idempotent re-processing give you a complete audit trail from source file to indexed chunk, built in from the start.

---

## Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  File Watcher│ →  │    Parser    │ →  │  Extraction  │ →  │Transformation│
│  (Watchdog)  │    │  (Docling)   │    │              │    │  (optional)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                      │
                    ┌──────────────┐    ┌──────────────┐    ┌────────▼─────┐
                    │   Indexer    │ ←  │   Embedder   │ ←  │   Chunker    │
                    │  (ChromaDB)  │    │(sent-transf.)│    │(struct-aware)│
                    └──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                    ┌───────────▼──────────┐
                                    │  SQLite Metadata DB  │
                                    │ (lineage + workflow   │
                                    │  run tracking)       │
                                    └──────────────────────┘
```

| Stage | Service | What It Does |
|---|---|---|
| **Ingest** | `filewatcher.py` | Monitors a folder, SHA-256 hashes each file, versions duplicates, copies to staging |
| **Parse** | `parsing.py` | Uses Docling to understand document structure: pages, tables, images, text blocks; token count estimation |
| **Extract** | `extraction.py` | Traverses Docling items, maintains header hierarchy, exports tables to markdown, extracts image captions |
| **Transform** | `transformation.py` | Optional: Unicode normalisation, whitespace cleanup, smart punctuation, consistent markdown table formatting |
| **Chunk** | `chunking.py` | Structure-aware token-bounded chunking respecting header boundaries; tables as standalone units; 50-token overlap |
| **Embed** | `embedding.py` | Sentence-transformers (all-MiniLM-L6-v2, 384-dim), batch processing, GPU auto-detect |
| **Index** | `indexing.py` | ChromaDB persistent storage with per-workflow collection routing and deterministic chunk IDs (upsert-safe) |

---

## Key Features

- **Multi-workflow support** — run `fund_factsheets`, `annual_reports`, and any other workflow simultaneously, each with its own watch folder, chunk size, and ChromaDB collection.
- **Workflow hot-reload** — `WorkflowRegistryManager` syncs `workflows/registry.yaml` every 30 seconds; add or deactivate a workflow without restarting the server.
- **Structure-aware chunking** — chunks respect header hierarchy and never split tables. Short documents (< 1000 tokens) are indexed as a single unit.
- **SHA-256 versioning** — identical file content is a no-op; changed content increments the version and triggers a clean re-index.
- **Stage skipping** — any stage (except `parse`) can be disabled per workflow; disabling `embed` automatically skips `index`.
- **Vision / VLM integration** — optionally generate page and picture images during parsing, save them to disk, and describe them via Claude Haiku (requires `ANTHROPIC_API_KEY`).
- **Full lineage tracking** — every `Document`, `WorkflowRun`, and `WorkflowStage` record is persisted in SQLite with timestamps, token counts, and error messages.
- **GPU-accelerated** — Docling and sentence-transformers both auto-detect CUDA; validated on NVIDIA A40 (46 GB VRAM).
- **290 tests passing** — comprehensive unit test coverage across all pipeline stages.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Document parsing | Docling 2.72+ |
| Vector database | ChromaDB 1.4.1 |
| Metadata database | SQLite + SQLAlchemy ORM |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| File watching | Watchdog 6.0 |
| Configuration | Pydantic + YAML |
| Token counting | tiktoken (`cl100k_base`) |
| Text splitting | semchunk 2.2.2 |
| Observability | LangSmith 0.6.9 (placeholder) |
| Vision descriptions | Anthropic Claude Haiku 4.5 (optional) |
| Testing | pytest 9.0.2, pytest-asyncio |
| GPU | torch 2.6.0+cu124, onnxruntime-gpu 1.24.1 |

---

## Project Structure

```
knowledgeforge/
├── kf_config.yaml                    # Base configuration (Pydantic-validated)
├── workflows/
│   ├── registry.yaml                 # Active/inactive workflow listing
│   └── fund_factsheet.yaml           # Sample workflow override
├── backend/
│   ├── cli.py                        # CLI entry point (argparse)
│   ├── requirements.txt
│   └── app/
│       ├── main.py                   # FastAPI app + lifespan + WorkflowRegistryManager
│       ├── core/
│       │   ├── config.py             # Pydantic config loader
│       │   ├── workflow_config.py    # Overlay system: StagesConfig, ResolvedWorkflowConfig
│       │   └── logging.py
│       ├── models/
│       │   ├── database.py           # SQLAlchemy ORM (Document, WorkflowRun, WorkflowStage)
│       │   └── schemas.py            # Pydantic response schemas
│       ├── db/
│       │   └── session.py            # Engine + session management + init_db
│       ├── api/v1/
│       │   ├── health.py
│       │   ├── documents.py          # (placeholder)
│       │   ├── metrics.py            # (placeholder)
│       │   └── workflows.py          # list, sync, status endpoints
│       ├── services/
│       │   ├── filewatcher.py        # Watchdog watcher + SHA-256 versioning
│       │   ├── parsing.py            # DocumentParser (Docling)
│       │   ├── extraction.py         # ContentExtractor
│       │   ├── transformation.py     # ContentTransformer
│       │   ├── chunking.py           # StructureAwareChunker
│       │   ├── embedding.py          # Embedder (sentence-transformers)
│       │   ├── indexing.py           # ChromaIndexer
│       │   ├── workflow.py           # WorkflowOrchestrator (6-stage pipeline)
│       │   └── workflow_registry.py  # WorkflowRegistryManager (hot-reload)
│       ├── metrics/
│       │   └── collector.py          # (placeholder)
│       └── observability/
│           └── tracing.py            # LangSmith (placeholder)
├── tests/
│   ├── test_config.py                # 13 tests
│   ├── test_database.py              # 16 tests
│   ├── test_filewatcher.py           # 13 tests
│   ├── test_parsing.py               # 17 tests
│   ├── test_extraction.py            # 32 tests
│   ├── test_transformation.py        # 58 tests
│   ├── test_chunking.py              # 53 tests
│   ├── test_embedding.py             # 13 tests
│   ├── test_indexing.py              # 14 tests
│   ├── test_workflow.py              # 19 tests
│   ├── test_workflow_config.py       # 29 tests
│   ├── test_workflow_registry.py     # 7 tests
│   └── test_e2e.py                   # 6 E2E tests
├── notebooks/
│   ├── workflow_e2e_demo.ipynb       # Interactive stage-by-stage demo
│   └── workflow_e2e_demo.py          # Script version
└── reference/
    ├── KnowledgeForge PRD.md
    ├── KnowledgeForge Tasks.md
    └── task_status.md
```

---

## Getting Started

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA 12.4+ (optional but strongly recommended — Docling model loading takes ~95 s on CPU cold start vs ~15 s warm on GPU)
- `ANTHROPIC_API_KEY` (optional — only needed for VLM picture descriptions)

### Installation

```bash
git clone <repo-url>
cd knowledgeforge

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
cd backend
python -m pip install -r requirements.txt
```

### Configure

Copy and edit the base config:

```bash
cp kf_config.yaml kf_config.local.yaml
# Edit watch_folder, chromadb_path, database.url, etc.
```

The server loads `kf_config.yaml` from the project root by default. Pass `--config` to override.

### Create data directories

```bash
mkdir -p data/source data/staging data/chromadb data/images
```

### Start the server

```bash
# From backend/
python cli.py start
# FastAPI runs on http://localhost:8000
# File watcher starts automatically on the configured source folder
```

On startup the app: loads config → initialises the SQLite DB → starts the WorkflowRegistryManager → starts per-workflow file watchers.

---

## Usage

### CLI

```bash
# Start server + all workflow watchers
python cli.py start
python cli.py start --config path/to/kf_config.yaml

# Process a single document (one-off)
python cli.py process --file /path/to/document.pdf
python cli.py process --file /path/to/document.pdf --workflow fund_factsheet
python cli.py process --file /path/to/document.pdf --force   # bypass dedup

# Inspect state
python cli.py status     # all documents and their run status
python cli.py metrics    # processing statistics
python cli.py workflows  # list registered workflows
```

### Auto-ingestion (file drop)

Drop any file matching a workflow's `file_patterns` into the configured `watch_folder`. The watcher picks it up automatically, hashes it, stages it, and runs the full pipeline.

### REST API

```
GET  /api/v1/health
GET  /api/v1/workflows                   # list all workflows + active status
POST /api/v1/workflows/sync              # hot-reload registry from disk
GET  /api/v1/workflows/{name}/status     # recent runs for a workflow
GET  /api/v1/metrics                     # processing stats (placeholder)
GET  /api/v1/documents                   # document list (placeholder)
POST /api/v1/documents/process           # trigger processing (placeholder)
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## Workflow System

KnowledgeForge supports multiple concurrent workflows, each with an independent config overlay, watch folder, and ChromaDB collection.

### Registry (`workflows/registry.yaml`)

```yaml
workflows:
  - name: "fund_factsheet"
    config: "fund_factsheet.yaml"
    active: true
    description: "Process fund factsheet PDFs"
```

Set `active: false` to deactivate a workflow without removing it. The registry manager hot-reloads every 30 seconds.

### Workflow config overlay (`workflows/fund_factsheet.yaml`)

Each workflow YAML is deep-merged on top of `kf_config.yaml`. Only specified keys are overridden:

```yaml
source:
  watch_folder: "./data/source/factsheets"
  file_patterns: ["*.pdf"]

processing:
  parsing:
    generate_page_images: true
    generate_picture_images: true
  extraction:
    save_picture_images: true
    # describe_pictures: true  # requires ANTHROPIC_API_KEY
  chunking:
    chunk_size_tokens: 256      # override: smaller chunks for dense factsheets
    chunk_overlap_tokens: 25

indexing:
  default_collection: "fund_factsheets"   # isolated ChromaDB collection

stages:
  parse:   { enabled: true }
  extract: { enabled: true }
  transform: { enabled: true }
  chunk:   { enabled: true }
  embed:   { enabled: true }
  index:   { enabled: true }

force_rerun: false
```

### Stage skipping rules

- `parse` is always required and cannot be disabled.
- Disabling `embed` automatically skips `index`.
- Skipped stages are recorded in `workflow_stages` with `status = "skipped"`.

---

## Configuration Reference

```yaml
# kf_config.yaml

source:
  watch_folder: "./data/source"        # folder to monitor
  staging_folder: "./data/staging"     # internal staging copy
  file_patterns: ["*.pdf", "*.docx", "*.xlsx", "*.html", "*.pptx"]

processing:
  parsing:
    library: "docling"
    generate_page_images: false        # render full page as PNG
    generate_picture_images: false     # extract embedded pictures

  extraction:
    strategy: "auto"                   # auto | direct | ocr | agentic
    save_picture_images: false         # write picture PNGs to disk
    picture_images_dir: "./data/images"
    describe_pictures: false           # call VLM for picture descriptions
    vision_model: "claude-haiku-4-5-20251001"
    vision_api_key_env: "ANTHROPIC_API_KEY"

  organisation:
    enabled: true
    table_format: "markdown"
    markdown_generation: true          # produce full-document structured markdown via Docling

  chunking:
    strategy: "structure_aware"
    chunk_size_tokens: 512
    chunk_overlap_tokens: 50
    skip_threshold_tokens: 1000        # documents below this are indexed whole

  embedding:
    provider: "sentence_transformers"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: 32

indexing:
  vector_store: "chromadb"
  chromadb_path: "./data/chromadb"
  default_collection: "default"
  collection_mapping: {}               # e.g. "reports/*.pdf": "reports_collection"

database:
  url: "sqlite:///./data/knowledgeforge.db"

observability:
  langsmith_enabled: true
  langsmith_project: "knowledgeforge"
```

---

## Development

### Running tests

```bash
# From backend/
source /workspace/rag/knowledgeforge/.venv/bin/activate
pytest                              # all 290 tests
pytest -v tests/test_parsing.py     # single module
pytest -m e2e                       # E2E tests only (slow — require GPU)
```

### Code style

- **Formatter**: Ruff (`pyproject.toml` at repo root)
- **Type checking**: `mypy --strict` — all public functions must have full type hints
- **Docstrings**: PEP 257, triple-quoted `"""`, required on all classes and functions
- **Comments**: `#` only; no inline explanations for self-evident code

### Git workflow

```
main          ← stable releases
  └── develop ← integration branch
        └── feature/your-task ← short-lived feature branches
```

Open PRs against `develop`, not `main`.

### Database schema changes

There are no Alembic migrations yet. After adding or modifying columns in `backend/app/models/database.py`, delete the existing SQLite file and let the app recreate the schema on startup:

```bash
rm data/knowledgeforge.db
python cli.py start
```
