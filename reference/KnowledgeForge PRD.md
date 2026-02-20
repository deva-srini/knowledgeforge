# KnowledgeForge Backend — Product Requirements Document (PRD)

## 1. Overview

### 1.1 What is KnowledgeForge?
KnowledgeForge is the knowledge ingestion, processing, and indexing module of a larger Agentic RAG system. It handles the entire lifecycle of turning raw documents (PDF, Word, Excel, HTML, PPT) into searchable, retrievable knowledge chunks stored in a vector database. It is the "knowledge creation" layer — the Explorer module (out of scope) is the "knowledge consumption" layer.

### 1.2 Why Build This (vs. Copilot / Off-the-Shelf)?
- **Full control** over the end-to-end Agentic RAG pipeline — from how knowledge is stored, processed, accessed, and presented.
- **Production-grade**: Designed for deploying customer-facing applications, not just internal research.
- **Governance & Curation**: Ability to control answer quality, knowledge base curation, lineage tracking, and observability — none of which are possible with generic copilot solutions.

### 1.3 Scope — Phase 1
- **In scope**: Documented internal knowledge (personal/enterprise documents) — backend only.
- **Out of scope**: Relational database knowledge (Phase 2), online/runtime knowledge (Explorer), frontend UI.

### 1.4 Learning Objectives (Personal Context)
This project also serves as a hands-on portfolio piece to gain deep practical understanding of: building RAG systems end-to-end, observability tooling, evaluation & feedback loops, prompt engineering cycles, and tool calling — with the intent to share learnings via GitHub and technical writing.

### 1.5 Timeline
Feb 1–15, 2026 (backend MVP)

---

## 2. Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangChain |
| Observability & Evaluation | LangSmith (+ Arize as optional) |
| Vector Store | ChromaDB |
| Metadata / Workflow DB | SQLite |
| Document Parsing | Docling |
| Prompt Versioning | LangSmith / LangChain Hub |

---

## 3. Architecture Overview

```
Source Folder (watched)
        │
        ▼
  ┌─────────────┐
  │  Ingestion   │  ← File watcher picks up new/updated docs
  │   Layer      │     based on regex patterns from kb_config.yml
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │   Parsing    │  ← Understand doc structure (tables, charts, text, pages)
  │   Layer      │     using Docling
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Extraction  │  ← Extract content per format (OCR, agentic, direct)
  │   Layer      │
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │ Organisation │  ← Optional transformation (e.g. table → markdown)
  │   Layer      │
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Chunking    │  ← Structure-aware chunking (respects headers, tables)
  │   Layer      │
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Embedding   │  ← Generate vector embeddings
  │   Layer      │
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Indexing    │  ← Store in ChromaDB with metadata;
  │   Layer      │     separate indexes per product/topic
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  SQLite      │  ← Track workflow runs, metrics, document lineage
  │  Metadata DB │
  └─────────────┘
```

---

## 4. Functional Requirements

### FR-1: Document Ingestion & Source Management

**FR-1.1: Folder Watching**
- KnowledgeForge maintains its own source folder structure.
- A file watcher monitors a configurable source folder using regex patterns.
- Patterns and folder paths are defined in `kf_config.yaml`.
- When a file matching a pattern appears (or is updated), it is picked up for processing.

**FR-1.2: Document Versioning**
- If a document with the same name already exists in the source folder, it must be versioned (not overwritten).
- Version history is tracked in SQLite metadata DB.

**FR-1.3: Supported Formats**
- PDF, Word (.docx), Excel (.xlsx), HTML, PowerPoint (.pptx).

**FR-1.4: Source Layer Copy**
- Documents are copied from the watched source folder into a KnowledgeForge internal source layer (staging area) before processing begins.

---

### FR-2: Document Parsing

**FR-2.1: Structure Understanding**
- Determine the document's internal structure: number of pages, token count, presence of tables, charts, images, plain text.
- Use Docling as the primary parsing library.

**FR-2.2: Metadata Extraction**
- Extract document-level metadata: file name, file type, page count, estimated token size, content type breakdown (% tables, % text, % charts).

**FR-2.3: Routing Decision**
- Based on parsing output, determine the appropriate extraction strategy for each section/page (e.g., direct text extraction vs. OCR vs. agentic extraction for complex layouts).

---

### FR-3: Information Extraction

**FR-3.1: Content Extraction**
- Extract all textual content from the document.
- Handle different content types appropriately: plain text (direct extraction), tables (structured extraction preserving rows/columns), charts/images (OCR or description generation — stretch goal).

**FR-3.2: Extraction Strategy Selection**
- Classic OCR for scanned documents or image-based content.
- Direct text extraction for native digital documents.
- Agentic extraction (LLM-assisted) for complex or ambiguous layouts — as a configurable option.

---

### FR-4: Information Organisation / Transformation

**FR-4.1: Format Normalisation**
- Optional post-extraction step to transform content into standardised formats.
- Example: Convert extracted tables into markdown format for better downstream chunking and retrieval.

**FR-4.2: Configurable**
- This step is optional and should be togglable per document type or globally in `kf_config.yaml`.

---

### FR-5: Chunking

**FR-5.1: Structure-Aware Chunking**
- Chunking must respect document structure — headers, paragraphs, and tables should not be arbitrarily split.
- Tables should be chunked as complete units where possible.
- Header hierarchy should be preserved as chunk metadata.

**FR-5.2: Token Threshold**
- If a document's total token count is below a configurable threshold, skip chunking entirely — index the full document as a single unit.

**FR-5.3: Configurable Parameters**
- Chunk size (in tokens), chunk overlap, chunking strategy — all configurable in `kb_config.yml`.

---

### FR-6: Embedding

**FR-6.1: Vector Embedding Generation**
- Generate embeddings for each chunk using a configurable embedding model.

**FR-6.2: Model Comparison (Stretch)**
- Support swapping embedding models to compare retrieval performance.
- Leverage LangSmith evaluation framework for A/B comparison of embedding + retrieval strategies.

---

### FR-7: Indexing

**FR-7.1: ChromaDB Storage**
- Store embeddings + chunk text + metadata in ChromaDB.

**FR-7.2: Separate Indexes**
- Support creating separate ChromaDB collections per product, topic, or logical grouping.
- Index assignment is configurable per document or folder in `kb_config.yml`.

**FR-7.3: Metadata Storage**
- Each indexed chunk must carry metadata: source document name, version, page number, chunk index, content type (text/table), header path, timestamp.

---

### FR-8: Workflow Orchestration & Tracking

**FR-8.1: Pipeline Execution**
- The full pipeline (ingest → parse → extract → organise → chunk → embed → index) runs as a single orchestrated workflow per document.

**FR-8.2: Workflow State Tracking (SQLite)**
Track the following per document per run:
- Document ID, document name, version
- Workflow run ID, start time, end time, status (pending / in-progress / success / failed)
- Stage-level status and timestamps (per stage: parse, extract, organise, chunk, embed, index)
- Error messages if any stage fails

**FR-8.3: Re-trigger (Optional)**
- Ability to re-trigger the full processing pipeline for a specific document or all documents via a CLI command or API endpoint.

---

### FR-9: Metrics & Observability

**FR-9.1: Processing Metrics (SQLite)**
- Number of documents indexed (total and per run)
- Number of chunks generated (total and per document)
- Time taken per document and per stage
- Number of indexes created / updated over time

**FR-9.2: LangSmith Integration**
- All LLM calls (if any, e.g., agentic extraction) traced via LangSmith.
- Token usage tracked per call.
- Prompt versions tracked.

**FR-9.3: Metrics API**
- Expose a simple API endpoint (or CLI command) to retrieve processing metrics as JSON.

---

### FR-10: Configuration Management

**FR-10.1: `kb_config.yml`**
Central YAML configuration file controlling:
```yaml
# kf_config.yaml
source:
  watch_folder: "./data/source"
  file_patterns: ["*.pdf", "*.docx", "*.xlsx", "*.html", "*.pptx"]

processing:
  parsing:
    library: "docling"
  extraction:
    strategy: "auto"  # auto | ocr | direct | agentic
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
  default_collection: "default"
  collection_mapping:
    "product_a/*.pdf": "product_a_index"
    "product_b/*.pdf": "product_b_index"

database:
  metadata_db: "sqlite:///knowledgeforge.db"

observability:
  langsmith_project: "knowledgeforge"
```

---

## 5. Non-Functional Requirements

| Requirement | Detail |
|---|---|
| **Modularity** | Each pipeline stage is a standalone, testable module. Stages can be swapped or extended independently. |
| **Idempotency** | Re-running the pipeline on the same document version should not create duplicate entries. |
| **Error Handling** | Stage-level error handling with proper logging. A failure in one document should not block others. |
| **Logging** | Structured logging (Python `logging` module) with configurable log levels. |
| **Testability** | Each module must have unit tests. Integration test for full pipeline. |
| **CLI Interface** | Primary interface is CLI for triggering processing, checking status, and viewing metrics. |

---

## 6. Folder Structure

```
knowledgeforge/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app entry point + lifespan
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # YAML config loader + Pydantic validation
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
│   │   │   └── session.py         # DB engine, session management, init_db
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── filewatcher.py     # Folder watcher + versioning
│   │   │   ├── parsing.py         # Docling-based document parser
│   │   │   ├── extraction.py      # Content extraction (text, tables, OCR)
│   │   │   ├── transformation.py  # Optional content transformation
│   │   │   ├── chunking.py        # Structure-aware chunking
│   │   │   ├── embedding.py       # Embedding generation
│   │   │   ├── indexing.py        # ChromaDB indexing
│   │   │   └── workflow.py        # End-to-end pipeline orchestration
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   └── collector.py       # Metrics collection & reporting
│   │   └── observability/
│   │       ├── __init__.py
│   │       └── tracing.py         # LangSmith tracing setup
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
│   ├── cli.py                     # CLI entry point
│   ├── requirements.txt
│   └── README.md
├── frontend/                      # Out of scope for Phase 1
├── kf_config.yaml                 # Central configuration (project root)
└── data/
    ├── source/                    # Watched source folder
    └── staging/                   # KnowledgeForge internal staging
```

---

## 7. API / CLI Interface (Phase 1)

### Application Startup Behavior
When the app starts (either via CLI or FastAPI server), it **automatically**:
1. Loads `kf_config.yaml`
2. Initialises the SQLite database
3. Starts the file watcher on all configured source folders
4. Scans for any existing unprocessed files and queues them for processing

This means the app is always "live" — no manual trigger needed for new files.

### CLI (via `cli.py`)

```bash
# Start the app (server + file watcher — primary mode)
python cli.py start

# Start with a specific config file
python cli.py start --config path/to/kf_config.yaml

# Process a specific document manually (one-off)
python cli.py process --file "path/to/doc.pdf"

# Re-process a specific document (force re-run)
python cli.py process --file "path/to/doc.pdf" --force

# Check processing status
python cli.py status

# View metrics
python cli.py metrics
```

### FastAPI Server (starts automatically with `cli.py start`)

```
GET  /api/v1/health
GET  /api/v1/metrics
GET  /api/v1/documents
GET  /api/v1/documents/{doc_id}/status
POST /api/v1/documents/process
POST /api/v1/documents/{doc_id}/reprocess
```

The FastAPI app uses **lifespan events** to start/stop the file watcher:
- **on_startup**: init DB, load config, start file watcher in background thread
- **on_shutdown**: gracefully stop file watcher

---

## 8. Data Models (SQLite)

### documents
| Column | Type | Description |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| file_name | TEXT | Original file name |
| file_path | TEXT | Path in staging |
| file_type | TEXT | pdf / docx / xlsx / html / pptx |
| version | INTEGER | Document version |
| file_hash | TEXT | SHA-256 hash for dedup |
| created_at | DATETIME | Upload timestamp |
| updated_at | DATETIME | Last update |

### workflow_runs
| Column | Type | Description |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| document_id | TEXT (FK) | Reference to document |
| status | TEXT | pending / in_progress / success / failed |
| started_at | DATETIME | Run start |
| completed_at | DATETIME | Run end |
| error_message | TEXT | Error details if failed |
| total_chunks | INTEGER | Chunks generated |
| total_tokens | INTEGER | Tokens processed |

### workflow_stages
| Column | Type | Description |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| run_id | TEXT (FK) | Reference to workflow run |
| stage_name | TEXT | parse / extract / organise / chunk / embed / index |
| status | TEXT | pending / in_progress / success / failed / skipped |
| started_at | DATETIME | Stage start |
| completed_at | DATETIME | Stage end |
| metadata_json | TEXT | Stage-specific metadata (JSON) |
| error_message | TEXT | Error details if failed |

### chunks — REMOVED
> Chunks are stored only in ChromaDB. No separate SQLite table needed. ChromaDB stores: chunk content, embedding vector, and metadata (document_id, file_name, version, page_number, chunk_index, content_type, header_path, collection_name). The `workflow_runs.total_chunks` field tracks count for metrics.
