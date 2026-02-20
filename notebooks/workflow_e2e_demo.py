"""KnowledgeForge Workflow E2E Demo — script version (stage-by-stage).

Runs each pipeline stage individually and inspects the intermediate
outputs so you can see exactly what happens under the hood:

  Parse -> Extract -> Transform -> Chunk -> Embed -> Index

Uses bgf_factsheet.pdf from the reference directory.

Run from knowledgeforge/backend/:
    python ../notebooks/workflow_e2e_demo.py
"""

import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 0: Setup and imports
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 0  Setup and imports")
print("=" * 70)

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.core.config import KnowledgeForgeConfig, load_config  # noqa: E402
from app.core.workflow_config import load_workflow  # noqa: E402
from app.db.session import get_engine, get_session_factory, init_db, reset_globals  # noqa: E402
from app.models.database import Document, WorkflowRun, WorkflowStage  # noqa: E402
from app.services.chunking import StructureAwareChunker  # noqa: E402
from app.services.embedding import Embedder  # noqa: E402
from app.services.extraction import ContentExtractor, ContentType  # noqa: E402
from app.services.filewatcher import compute_file_hash  # noqa: E402
from app.services.indexing import ChromaIndexer  # noqa: E402
from app.services.parsing import DocumentParser  # noqa: E402
from app.services.transformation import ContentTransformer  # noqa: E402

print("Imports successful\n")

# ---------------------------------------------------------------------------
# Step 1: Load configs
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1  Load global config + workflow overlay")
print("=" * 70)

reset_globals()
config = load_config()
wf_config = load_workflow("fund_factsheet", config)

# Build the merged service config (same logic as WorkflowOrchestrator)
svc_config = KnowledgeForgeConfig(
    source=wf_config.source,
    processing=wf_config.processing,
    indexing=wf_config.indexing,
    database=config.database,
    observability=config.observability,
)

print("Global config")
print(f"  chunk_size_tokens : {config.processing.chunking.chunk_size_tokens}")
print(f"  default_collection: {config.indexing.default_collection}")

print(f"\nWorkflow '{wf_config.name}' overlay")
print(f"  chunk_size_tokens : {wf_config.processing.chunking.chunk_size_tokens}")
print(f"  chunk_overlap     : {wf_config.processing.chunking.chunk_overlap_tokens}")
print(f"  default_collection: {wf_config.indexing.default_collection}")
print(f"  watch_folder      : {wf_config.source.watch_folder}")
print(f"  file_patterns     : {wf_config.source.file_patterns}")
print(f"  force_rerun       : {wf_config.force_rerun}")

print("\nMerged service config (used by all services)")
print(f"  chunk_size_tokens : {svc_config.processing.chunking.chunk_size_tokens}")
print(f"  default_collection: {svc_config.indexing.default_collection}\n")

# ---------------------------------------------------------------------------
# Step 2: Initialise database + prepare document
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 2  Initialise database + prepare document")
print("=" * 70)

engine = get_engine(config.database.url)
init_db(engine)
session_factory = get_session_factory(engine)

# Copy reference PDF into the factsheets folder
reference_pdf = Path(__file__).resolve().parent.parent / "reference" / "bgf_factsheet.pdf"
factsheets_dir = Path("../data/source/factsheets").resolve()
factsheets_dir.mkdir(parents=True, exist_ok=True)

if not reference_pdf.exists():
    print(f"ERROR: Reference PDF not found at {reference_pdf}")
    sys.exit(1)

target_pdf = factsheets_dir / reference_pdf.name
shutil.copy2(str(reference_pdf), str(target_pdf))

file_hash = compute_file_hash(str(target_pdf))
print(f"File        : {target_pdf.name}")
print(f"Size        : {target_pdf.stat().st_size:,} bytes")
print(f"SHA-256     : {file_hash[:24]}...")

session = session_factory()
existing = (
    session.query(Document)
    .filter_by(file_name=target_pdf.name, workflow_id="fund_factsheet")
    .order_by(Document.version.desc())
    .first()
)
new_version = (existing.version + 1) if existing else 1

doc = Document(
    file_name=target_pdf.name,
    file_path=str(target_pdf),
    file_type="pdf",
    version=new_version,
    file_hash=file_hash,
    workflow_id="fund_factsheet",
    status="pending",
)
session.add(doc)
session.commit()
session.refresh(doc)
session.close()
print(f"Document id : {doc.id[:12]}...")
print(f"Version     : {doc.version}\n")

# ============================= PIPELINE =====================================

# ---------------------------------------------------------------------------
# Stage 1: PARSE
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 1  PARSE  (Docling document conversion)")
print("=" * 70)

parser = DocumentParser(svc_config)
t0 = time.time()
parse_result = parser.parse(doc.file_path)
parse_time = time.time() - t0

print(f"Time          : {parse_time:.1f}s")
print(f"Pages         : {parse_result.page_count}")
print(f"Est. tokens   : {parse_result.estimated_token_count:,}")
print(f"Content types : {dict(parse_result.content_types)}")
print(f"Raw text len  : {len(parse_result.raw_text):,} chars")

print("\nPer-page breakdown:")
for ps in parse_result.structure:
    print(f"  Page {ps.page_number}: {dict(ps.content_types)}")

print(f"\nRaw text preview (first 500 chars):")
print("-" * 50)
print(parse_result.raw_text[:500])
print("-" * 50)
print()

# ---------------------------------------------------------------------------
# Stage 2: EXTRACT
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 2  EXTRACT  (structured content extraction)")
print("=" * 70)

extractor = ContentExtractor(svc_config)
t0 = time.time()
extracted = extractor.extract(parse_result)
extract_time = time.time() - t0

print(f"Time            : {extract_time:.2f}s")
print(f"Items extracted : {len(extracted)}")

# Breakdown by content type
type_counts = Counter(item.content_type.value for item in extracted)
print(f"By type         : {dict(type_counts)}")

# Breakdown by page
page_counts = Counter(item.page_number for item in extracted)
print(f"By page         : { {f'p{k}': v for k, v in sorted(page_counts.items())} }")

# Show unique header paths
headers = sorted(set(item.header_path for item in extracted if item.header_path))
print(f"\nUnique header paths ({len(headers)}):")
for h in headers:
    print(f"  {h}")

# Show a few sample items
print(f"\nSample extracted items (first 5):")
print("-" * 50)
for i, item in enumerate(extracted[:5]):
    text_preview = item.content[:120].replace("\n", " ")
    if len(item.content) > 120:
        text_preview += "..."
    print(f"  [{i}] type={item.content_type.value:<18} page={item.page_number}  header='{item.header_path}'")
    print(f"       text: {text_preview}")
print("-" * 50)

# Show tables
tables = [item for item in extracted if item.content_type == ContentType.TABLE]
if tables:
    print(f"\nTables found: {len(tables)}")
    for i, tbl in enumerate(tables[:2]):
        print(f"\n  Table {i} (page {tbl.page_number}, header: '{tbl.header_path}'):")
        # Show first 300 chars of table content
        preview = tbl.content[:300]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(tbl.content) > 300:
            print(f"    ... ({len(tbl.content)} chars total)")

# Show images
images = [item for item in extracted if item.content_type == ContentType.IMAGE_DESCRIPTION]
if images:
    print(f"\nImage descriptions found: {len(images)}")
    for i, img in enumerate(images[:3]):
        print(f"  [{i}] page={img.page_number}  header='{img.header_path}'  text='{img.content[:80]}'")

print()

# ---------------------------------------------------------------------------
# Stage 3: TRANSFORM
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 3  TRANSFORM  (cleaning, normalisation, markdown)")
print("=" * 70)

transformer = ContentTransformer(svc_config)
t0 = time.time()
transform_result = transformer.transform(extracted, raw_document=parse_result.raw_document)
transform_time = time.time() - t0

print(f"Time              : {transform_time:.2f}s")
print(f"Items transformed : {len(transform_result.items)}")
print(f"Markdown length   : {len(transform_result.document_markdown):,} chars")

# Compare an original vs transformed item
if extracted and transform_result.items:
    print("\nBefore/After comparison (item 1):")
    orig = extracted[1] if len(extracted) > 1 else extracted[0]
    trns = transform_result.items[1] if len(transform_result.items) > 1 else transform_result.items[0]
    print(f"  Original  : '{orig.content[:100]}'")
    print(f"  Transformed: '{trns.content[:100]}'")

# Show the generated markdown (first 1000 chars)
if transform_result.document_markdown:
    print(f"\nGenerated document markdown (first 1000 chars):")
    print("-" * 50)
    print(transform_result.document_markdown[:1000])
    print("-" * 50)
    if len(transform_result.document_markdown) > 1000:
        print(f"  ... ({len(transform_result.document_markdown):,} chars total)")

print()

# ---------------------------------------------------------------------------
# Stage 4: CHUNK
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 4  CHUNK  (structure-aware chunking)")
print("=" * 70)

chunker = StructureAwareChunker(svc_config)
t0 = time.time()
chunks = chunker.chunk(transform_result.items)
chunk_time = time.time() - t0

total_tokens = sum(c.token_count for c in chunks)
print(f"Time          : {chunk_time:.2f}s")
print(f"Input items   : {len(transform_result.items)}")
print(f"Output chunks : {len(chunks)}")
print(f"Total tokens  : {total_tokens:,}")
print(f"Config        : size={svc_config.processing.chunking.chunk_size_tokens}, overlap={svc_config.processing.chunking.chunk_overlap_tokens}")

# Token distribution
token_counts = [c.token_count for c in chunks]
if token_counts:
    print(f"Token range   : min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)/len(token_counts):.0f}")

# Breakdown by type
chunk_type_counts = Counter(c.content_type.value for c in chunks)
print(f"By type       : {dict(chunk_type_counts)}")

# Show all chunks
print(f"\nAll {len(chunks)} chunks:")
print(f"{'Idx':>4}  {'Type':<18}  {'Page':>4}  {'Tokens':>6}  {'Header Path':<40}  Content preview")
print("-" * 140)
for c in chunks:
    text_preview = c.content[:60].replace("\n", " ")
    if len(c.content) > 60:
        text_preview += "..."
    print(
        f"{c.chunk_index:>4}  {c.content_type.value:<18}  {c.page_number:>4}  "
        f"{c.token_count:>6}  {c.header_path:<40}  {text_preview}"
    )
print()

# ---------------------------------------------------------------------------
# Stage 5: EMBED
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 5  EMBED  (sentence-transformers embedding)")
print("=" * 70)

embedder = Embedder(svc_config)
t0 = time.time()
embed_result = embedder.embed(chunks)
embed_time = time.time() - t0

print(f"Time          : {embed_time:.2f}s")
print(f"Model         : {svc_config.processing.embedding.model}")
print(f"Batch size    : {svc_config.processing.embedding.batch_size}")
print(f"Embedded      : {len(embed_result.embedded_chunks)}")
print(f"Skipped       : {embed_result.skipped_count}")

if embed_result.embedded_chunks:
    first_ec = embed_result.embedded_chunks[0]
    dim = len(first_ec.embedding)
    print(f"Dimension     : {dim}")
    print(f"\nSample embedding (chunk 0, first 10 values):")
    print(f"  {first_ec.embedding[:10]}")
    print(f"\nSample embedding (chunk 1, first 10 values):")
    if len(embed_result.embedded_chunks) > 1:
        print(f"  {embed_result.embedded_chunks[1].embedding[:10]}")

print()

# ---------------------------------------------------------------------------
# Stage 6: INDEX
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 6  INDEX  (ChromaDB upsert)")
print("=" * 70)

indexer = ChromaIndexer(svc_config)

# Delete old chunks for this document (idempotent re-indexing)
indexer.delete_document(doc.id)

t0 = time.time()
index_result = indexer.index(
    embed_result.embedded_chunks,
    document_id=doc.id,
    file_name=doc.file_name,
    version=doc.version,
    file_path=doc.file_path,
)
index_time = time.time() - t0

print(f"Time          : {index_time:.2f}s")
print(f"Collection    : {index_result.collection_name}")
print(f"Indexed       : {index_result.total_indexed} chunks")
print(f"IDs           : {index_result.indexed_ids[:5]}{'...' if len(index_result.indexed_ids) > 5 else ''}")

# Verify via ChromaDB client
import chromadb  # noqa: E402

chroma_path = Path(svc_config.indexing.chromadb_path).resolve()
client = chromadb.PersistentClient(path=str(chroma_path))
coll = client.get_collection(index_result.collection_name)
print(f"\nChromaDB verification:")
print(f"  Collection '{coll.name}' has {coll.count()} total chunks")

# Sample query
results = coll.peek(limit=3)
print(f"\n  Peek at first 3 stored chunks:")
for i, (text, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"    [{i}] type={meta.get('content_type')}, page={meta.get('page_number')}, "
          f"header='{meta.get('header_path', '')}'")
    text_preview = text[:80].replace("\n", " ")
    print(f"        {text_preview}...")

print()

# ========================= POST-PIPELINE ====================================

# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("TIMING SUMMARY")
print("=" * 70)
total_time = parse_time + extract_time + transform_time + chunk_time + embed_time + index_time
print(f"  {'Parse':<12} {parse_time:>7.2f}s  {'*' * int(parse_time / total_time * 40)}")
print(f"  {'Extract':<12} {extract_time:>7.2f}s  {'*' * max(1, int(extract_time / total_time * 40))}")
print(f"  {'Transform':<12} {transform_time:>7.2f}s  {'*' * max(1, int(transform_time / total_time * 40))}")
print(f"  {'Chunk':<12} {chunk_time:>7.2f}s  {'*' * max(1, int(chunk_time / total_time * 40))}")
print(f"  {'Embed':<12} {embed_time:>7.2f}s  {'*' * max(1, int(embed_time / total_time * 40))}")
print(f"  {'Index':<12} {index_time:>7.2f}s  {'*' * max(1, int(index_time / total_time * 40))}")
print(f"  {'TOTAL':<12} {total_time:>7.2f}s")

# ---------------------------------------------------------------------------
# Final pipeline summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"  Document        : {doc.file_name} (v{doc.version})")
print(f"  Workflow        : {wf_config.name}")
print(f"  Pages           : {parse_result.page_count}")
print(f"  Extracted items : {len(extracted)}")
print(f"    text          : {type_counts.get('text', 0)}")
print(f"    table         : {type_counts.get('table', 0)}")
print(f"    image_desc    : {type_counts.get('image_description', 0)}")
print(f"  Chunks          : {len(chunks)}")
print(f"  Total tokens    : {total_tokens:,}")
print(f"  Embedding dim   : {dim}")
print(f"  Collection      : {index_result.collection_name}")
print(f"  Indexed         : {index_result.total_indexed}")
print(f"\nDemo complete!")
