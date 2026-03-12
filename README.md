# KnowledgeForge

**A production framework built around Docling — bringing workflow orchestration, structure-aware chunking, and automated file watching to document-based RAG systems.**

KnowledgeForge is centred on [Docling](https://github.com/DS4SD/docling)'s deep document understanding — its ability to parse pages, tables, figures, and header hierarchies from PDF, Word, Excel, HTML, and PowerPoint. Around that core, KnowledgeForge layers a full production stack: automated file watching, per-workflow configuration overlays, structure-aware semantic chunking that respects document structure, GPU-accelerated embedding, ChromaDB indexing, and post-indexing analysis tools — turning raw documents into queryable knowledge assets with full lineage tracking.

It is the *knowledge creation* layer of a larger Agentic RAG system, feeding a downstream Explorer agent.

---

## Why Build This?

- **Docling as the foundation** — rather than treating parsing as a black box, KnowledgeForge is designed to exploit Docling's full structural output: header trees, table grids, figure metadata, and page-level layout — giving downstream chunking and retrieval a structural advantage over naive text splitting.
- **Workflow-first design** — multiple document workflows (e.g. `fund_factsheets`, `annual_reports`) run concurrently with independent watch folders, chunk sizes, and vector collections, hot-reloaded without restarting the service.
- **Full knowledge lineage** — SHA-256 versioning, per-stage run records, and idempotent re-processing give you a complete audit trail from source file to indexed chunk, built in from the start.
- **Post-indexing analysis** — a vector index summarizer and overlap detector let you audit what's in your knowledge base and catch duplicate content before it affects retrieval quality.

---

## Pipeline

The core is a **six-stage sequential pipeline**, with two post-indexing analysis tools:

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
          ┌────────────────┴──────────────────┐
          ▼                                    ▼
┌──────────────────┐                ┌──────────────────────┐
│  Vector Index    │                │   Overlap Detector   │
│  Summarizer      │                │  (cosine similarity) │
│  (TOML export)   │                │  within / cross-coll │
└──────────────────┘                └──────────────────────┘
                          │
              ┌───────────▼──────────┐
              │  SQLite Metadata DB  │
              │ (lineage + workflow  │
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
| **Summarize** | `vector_index_summarizer.py` | Post-indexing: scans all collections, emits a structured TOML snapshot (chunks, tokens, sections per document) |
| **Overlap** | `overlap_detection.py` | Post-indexing: cosine similarity search for duplicate/near-duplicate chunks within or across collections |

---

## Key Features

- **Automated file watching** — drop a file into a watch folder and the full pipeline triggers automatically; no manual intervention needed.
- **SHA-256 versioning** — identical file content is a no-op; changed content increments the version and triggers a clean re-index, with the full history preserved in SQLite.
- **Multi-workflow support** — run `fund_factsheets`, `annual_reports`, and any other workflow simultaneously, each with its own watch folder, chunk size, and ChromaDB collection.
- **Workflow hot-reload** — `WorkflowRegistryManager` syncs `workflows/registry.yaml` every 30 seconds; add or deactivate a workflow without restarting the server.
- **Structure-aware chunking** — chunks respect header hierarchy and never split tables. Short documents (< 1000 tokens) are indexed as a single unit.
- **Vector index summary** — `python cli.py summary` produces a TOML snapshot of all indexed collections: chunk counts, token totals, content types, and section headings per document.
- **Overlap detection** — `python cli.py overlap --collection <name>` finds semantically duplicate chunks using cosine similarity; supports within-collection and cross-collection comparison.
- **Vision / VLM integration** — optionally generate page and picture images during parsing, save them to disk, and describe them via Claude Haiku (requires `ANTHROPIC_API_KEY`).
- **Full lineage tracking** — every `Document`, `WorkflowRun`, and `WorkflowStage` record is persisted in SQLite with timestamps, token counts, and error messages.
- **GPU-accelerated** — Docling and sentence-transformers both auto-detect CUDA; validated on NVIDIA A40 (46 GB VRAM).
- **321 tests** — comprehensive unit and integration test coverage across all pipeline stages.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Package management | uv |
| Web framework | FastAPI + Uvicorn |
| Document parsing | Docling 2.73+ |
| Vector database | ChromaDB 1.4+ |
| Metadata database | SQLite + SQLAlchemy ORM |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| File watching | Watchdog 6.0 |
| Configuration | Pydantic + YAML |
| Token counting | tiktoken (`cl100k_base`) |
| Text splitting | semchunk 2.2+ |
| Observability | LangSmith (optional) |
| Vision descriptions | Anthropic Claude Haiku 4.5 (optional) |
| Testing | pytest 9.0+, pytest-asyncio |
| GPU | torch 2.6.0+cu128, onnxruntime 1.24+ |

---

## Project Structure

```
knowledgeforge/
├── pyproject.toml                    # Dependencies + pytest config (uv managed)
├── uv.lock                           # Cross-platform lock file (Linux + macOS x86_64)
├── kf_config.yaml                    # Base configuration (Pydantic-validated)
├── workflows/
│   ├── registry.yaml                 # Active/inactive workflow listing
│   └── fund_factsheet.yaml           # Sample workflow override
├── backend/
│   ├── cli.py                        # CLI entry point (start, process, status, metrics,
│   │                                 #   workflows, summary, overlap)
│   └── app/
│       ├── main.py                   # FastAPI app + lifespan + WorkflowRegistryManager
│       ├── core/
│       │   ├── config.py             # Pydantic config loader (16 validated classes)
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
│       │   ├── vector_index_summarizer.py  # VectorIndexSummarizer (TOML export)
│       │   ├── overlap_detection.py  # OverlapDetector (cosine similarity)
│       │   ├── workflow.py           # WorkflowOrchestrator (6-stage pipeline)
│       │   └── workflow_registry.py  # WorkflowRegistryManager (hot-reload)
│       ├── metrics/
│       │   └── collector.py
│       └── observability/
│           └── tracing.py            # LangSmith tracing
│   └── tests/                        # 321 tests across 18 modules
├── notebooks/
│   ├── workflow_e2e_demo.ipynb       # Interactive stage-by-stage demo (all 8 stages)
│   └── workflow_e2e_demo.py          # Script version
└── reference/
    └── bgf_factsheet.pdf             # Sample PDF for testing
```

---

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) — used for dependency management
- NVIDIA GPU with CUDA 12.4+ (optional but strongly recommended — Docling model loading is significantly slower on CPU)
- `ANTHROPIC_API_KEY` in a `.env` file at the project root (optional — only needed for VLM picture descriptions)

### Installation

```bash
git clone <repo-url>
cd knowledgeforge

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into .venv
uv sync
```

> **Note for Linux users**: the lock file is cross-platform (Linux + macOS x86_64). `uv sync` will install the correct wheels for your platform automatically.

### Create data directories

```bash
mkdir -p data/source data/staging data/chromadb data/images
```

### Configure

The server loads `kf_config.yaml` from the project root by default. Edit it directly or pass `--config` to override:

```bash
# Optional: copy and customise for local overrides
cp kf_config.yaml kf_config.local.yaml
```

Key settings to review before first run:

```yaml
source:
  watch_folder: "./data/source"       # where to drop files for auto-ingestion

indexing:
  chromadb_path: "./data/chromadb"    # vector store location

database:
  url: "sqlite:///./data/knowledgeforge.db"
```

### Start the server

```bash
# From backend/
.venv/bin/python cli.py start
# FastAPI runs on http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

On startup: loads config → initialises SQLite DB → starts WorkflowRegistryManager → starts per-workflow file watchers.

### Run the end-to-end demo

The fastest way to see the full pipeline in action:

```bash
# Script version — runs all 8 stages and prints output for each
.venv/bin/python notebooks/workflow_e2e_demo.py

# Or open the notebook in VS Code / JupyterLab
# First register the Jupyter kernel (one-time setup):
.venv/bin/python -m ipykernel install --user --name knowledgeforge --display-name "KnowledgeForge (.venv)"
sed -i 's|/usr/bin/python|'$(pwd)'/.venv/bin/python|g' \
  ~/.local/share/jupyter/kernels/knowledgeforge/kernel.json
# Then select "KnowledgeForge (.venv)" as the kernel before running
```

---

## Usage

### CLI

All `cli.py` commands run from `backend/`. `pytest` runs from the project root.

```bash
# Start server + all workflow watchers
python cli.py start
python cli.py start --config path/to/kf_config.yaml

# Process a single document (one-off, bypasses file watcher)
python cli.py process --file /path/to/document.pdf
python cli.py process --file /path/to/document.pdf --workflow fund_factsheet
python cli.py process --file /path/to/document.pdf --force   # bypass dedup check

# Inspect state
python cli.py status     # all documents and their processing status
python cli.py metrics    # aggregate processing statistics
python cli.py workflows  # list registered workflows

# Post-indexing analysis
python cli.py summary                              # TOML snapshot of all collections
python cli.py summary --output ./reports/out.toml # write to a specific path

python cli.py overlap --collection fund_factsheets            # within-collection
python cli.py overlap --source fund_factsheets --target annual_reports  # cross-collection
python cli.py overlap --collection fund_factsheets --threshold 0.9      # stricter threshold
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

Interactive API docs at `http://localhost:8000/docs`.

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
  chunking:
    chunk_size_tokens: 256      # override: smaller chunks for dense factsheets
    chunk_overlap_tokens: 25

indexing:
  default_collection: "fund_factsheets"   # isolated ChromaDB collection

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
    pipeline: "standard"               # "standard" | "vlm"
    generate_page_images: false        # render full page as PNG
    generate_picture_images: false     # extract embedded pictures

  extraction:
    strategy: "auto"
    save_picture_images: false         # write picture PNGs to disk
    picture_images_dir: "./data/images"
    describe_pictures: false           # call VLM for picture descriptions
    vision_model: "claude-haiku-4-5-20251001"
    vision_api_key_env: "ANTHROPIC_API_KEY"

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
  summary_path: "./data/vector_index_summary.toml"
  document_descriptions: {}           # e.g. "report.pdf": "Annual report 2025"

database:
  url: "sqlite:///./data/knowledgeforge.db"

observability:
  langsmith_enabled: false
  langsmith_project: "knowledgeforge"
```