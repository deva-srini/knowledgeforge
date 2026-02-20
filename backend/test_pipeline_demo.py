"""Demo script: Run the implemented pipeline stages on travel.pdf.

Runs Parse → Extract → Transform → Chunk → Embed → Index with timing,
and shows detailed table before/after comparison, chunking statistics,
embedding output, ChromaDB indexing results, and a WorkflowOrchestrator
end-to-end run with DB tracking.

Usage:
    source /workspace/uv_env/rag/bin/activate
    cd /workspace/rag/knowledgeforge/backend
    python test_pipeline_demo.py
"""

import json
import time
from pathlib import Path

from app.core.config import KnowledgeForgeConfig
from app.db.session import get_engine, get_session_factory, init_db
from app.models.database import Document, WorkflowRun, WorkflowStage
from app.services.chunking import StructureAwareChunker
from app.services.embedding import Embedder
from app.services.extraction import ContentExtractor, ContentType
from app.services.filewatcher import compute_file_hash
from app.services.indexing import ChromaIndexer
from app.services.parsing import DocumentParser
from app.services.transformation import ContentTransformer
from app.services.workflow import WorkflowOrchestrator


def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def main() -> None:
    """Run parse → extract → transform → chunk → embed → index on travel.pdf and print results."""
    pdf_path = Path(__file__).parent.parent / "reference" / "travel.pdf"
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return

    config = KnowledgeForgeConfig()
    timings: dict[str, float] = {}

    # =====================================================================
    # STAGE 1: PARSING
    # =====================================================================
    print("=" * 70)
    print("STAGE 1: PARSING")
    print("=" * 70)

    parser = DocumentParser(config)
    t0 = time.perf_counter()
    parse_result = parser.parse(str(pdf_path))
    timings["parse"] = time.perf_counter() - t0

    print(f"  Time:         {fmt_time(timings['parse'])}")
    print(f"  Pages:        {parse_result.page_count}")
    print(f"  Tokens (est): {parse_result.estimated_token_count}")
    print(f"  Content types: {parse_result.content_types}")
    print(f"  Has raw doc:  {parse_result.raw_document is not None}")
    print(f"  Structure:    {len(parse_result.structure)} pages with breakdown")
    for ps in parse_result.structure:
        print(f"    Page {ps.page_number}: {dict(ps.content_types)}")
    print()

    # =====================================================================
    # STAGE 2: EXTRACTION
    # =====================================================================
    print("=" * 70)
    print("STAGE 2: EXTRACTION")
    print("=" * 70)

    extractor = ContentExtractor(config)
    t0 = time.perf_counter()
    extracted = extractor.extract(parse_result)
    timings["extract"] = time.perf_counter() - t0

    print(f"  Time:           {fmt_time(timings['extract'])}")
    print(f"  Total items:    {len(extracted)}")

    by_type: dict[str, list] = {}
    for item in extracted:
        by_type.setdefault(item.content_type.value, []).append(item)
    for ctype, items in sorted(by_type.items()):
        print(f"    {ctype}: {len(items)}")
    print()

    # Show first 5 items
    print("  --- Sample extracted items (first 5) ---")
    for i, item in enumerate(extracted[:5]):
        preview = item.content[:120].replace("\n", "\\n")
        print(f"  [{i}] {item.content_type.value:18s} page={item.page_number}  "
              f"header={item.header_path!r}")
        print(f"       {preview}")
        if item.metadata:
            meta_keys = ", ".join(f"{k}={v}" for k, v in item.metadata.items())
            print(f"       meta: {meta_keys}")
        print()

    # =====================================================================
    # STAGE 3: TRANSFORMATION
    # =====================================================================
    print("=" * 70)
    print("STAGE 3: TRANSFORMATION")
    print("=" * 70)

    transformer = ContentTransformer(config)
    t0 = time.perf_counter()
    transform_result = transformer.transform(
        extracted, raw_document=parse_result.raw_document
    )
    timings["transform"] = time.perf_counter() - t0

    transformed = transform_result.items
    doc_markdown = transform_result.document_markdown

    print(f"  Time:              {fmt_time(timings['transform'])}")
    print(f"  Total items:       {len(transformed)}")
    print(f"  All transformed:   "
          f"{all(t.metadata.get('transformed') for t in transformed)}")

    # Count how many items actually changed content
    changed_count = sum(
        1 for e, t in zip(extracted, transformed) if e.content != t.content
    )
    print(f"  Content changed:   {changed_count}/{len(extracted)} items")
    print(f"  Document markdown: {len(doc_markdown)} chars")
    print()

    # =====================================================================
    # STRUCTURED MARKDOWN PREVIEW
    # =====================================================================
    if doc_markdown:
        print("=" * 70)
        print("STRUCTURED MARKDOWN (first 80 lines)")
        print("=" * 70)
        md_lines = doc_markdown.split("\n")
        for line in md_lines[:80]:
            print(f"  {line}")
        if len(md_lines) > 80:
            print(f"  ... ({len(md_lines)} total lines)")
        print()

    # =====================================================================
    # TABLE BEFORE/AFTER COMPARISON
    # =====================================================================
    tables_ext = [e for e in extracted if e.content_type == ContentType.TABLE]
    tables_trn = [t for t in transformed if t.content_type == ContentType.TABLE]

    if tables_ext:
        print("=" * 70)
        print(f"TABLE COMPARISON ({len(tables_ext)} table(s) found)")
        print("=" * 70)

        for idx, (ext, trn) in enumerate(zip(tables_ext, tables_trn)):
            changed = ext.content != trn.content
            print(f"\n  --- Table {idx + 1} (page {ext.page_number}) "
                  f"{'[CHANGED]' if changed else '[UNCHANGED]'} ---")

            if ext.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in ext.metadata.items())
                print(f"  Metadata: {meta_str}")

            print(f"\n  BEFORE (extraction output):")
            print("  " + "-" * 50)
            for line in ext.content.split("\n")[:15]:
                print(f"  {line}")
            if ext.content.count("\n") > 15:
                print(f"  ... ({ext.content.count(chr(10)) + 1} total lines)")

            print(f"\n  AFTER (transformation output):")
            print("  " + "-" * 50)
            for line in trn.content.split("\n")[:15]:
                print(f"  {line}")
            if trn.content.count("\n") > 15:
                print(f"  ... ({trn.content.count(chr(10)) + 1} total lines)")

            if changed:
                # Show specific differences
                ext_lines = ext.content.split("\n")
                trn_lines = trn.content.split("\n")
                print(f"\n  DIFF (line-by-line changes):")
                print("  " + "-" * 50)
                max_lines = max(len(ext_lines), len(trn_lines))
                diffs_shown = 0
                for li in range(min(max_lines, 20)):
                    el = ext_lines[li] if li < len(ext_lines) else "<missing>"
                    tl = trn_lines[li] if li < len(trn_lines) else "<missing>"
                    if el != tl:
                        print(f"  Line {li + 1}:")
                        print(f"    - {el!r}")
                        print(f"    + {tl!r}")
                        diffs_shown += 1
                if diffs_shown == 0:
                    print("  (differences only in removed blank lines)")
            print()

    # =====================================================================
    # TEXT BEFORE/AFTER SAMPLES
    # =====================================================================
    text_ext = [e for e in extracted if e.content_type == ContentType.TEXT]
    text_trn = [t for t in transformed if t.content_type == ContentType.TEXT]
    text_changed = [
        (e, t) for e, t in zip(text_ext, text_trn) if e.content != t.content
    ]

    if text_changed:
        print("=" * 70)
        print(f"TEXT ITEMS THAT CHANGED ({len(text_changed)} of {len(text_ext)})")
        print("=" * 70)
        for e, t in text_changed[:5]:
            print(f"\n  Page {e.page_number}, header={e.header_path!r}")
            print(f"    Before: {e.content[:100]!r}")
            print(f"    After:  {t.content[:100]!r}")
        if len(text_changed) > 5:
            print(f"\n  ... and {len(text_changed) - 5} more")
        print()

    # =====================================================================
    # STAGE 4: CHUNKING
    # =====================================================================
    print("=" * 70)
    print("STAGE 4: CHUNKING")
    print("=" * 70)

    chunker = StructureAwareChunker(config)
    t0 = time.perf_counter()
    chunks = chunker.chunk(transformed)
    timings["chunk"] = time.perf_counter() - t0

    print(f"  Time:           {fmt_time(timings['chunk'])}")
    print(f"  Total chunks:   {len(chunks)}")
    total_chunk_tokens = sum(c.token_count for c in chunks)
    print(f"  Total tokens:   {total_chunk_tokens}")

    chunk_by_type: dict[str, list] = {}
    for c in chunks:
        chunk_by_type.setdefault(c.content_type.value, []).append(c)
    for ctype, clist in sorted(chunk_by_type.items()):
        token_sum = sum(c.token_count for c in clist)
        print(f"    {ctype}: {len(clist)} chunks, {token_sum} tokens")
    print()

    # Show chunk details
    print("  --- All chunks ---")
    for c in chunks:
        preview = c.content[:100].replace("\n", "\\n")
        strategy = c.metadata.get("chunking_strategy", "")
        split_by = c.metadata.get("split_by", "")
        extra = ""
        if strategy:
            extra += f" strategy={strategy}"
        if split_by:
            extra += f" split_by={split_by}"
        print(f"  [{c.chunk_index:2d}] {c.content_type.value:18s} "
              f"page={c.page_number}  tokens={c.token_count:4d}  "
              f"header={c.header_path!r}{extra}")
        print(f"       {preview}")
        print()

    # =====================================================================
    # STAGE 5: EMBEDDING
    # =====================================================================
    print("=" * 70)
    print("STAGE 5: EMBEDDING")
    print("=" * 70)

    embedder = Embedder(config)
    t0 = time.perf_counter()
    embed_result = embedder.embed(chunks)
    timings["embed"] = time.perf_counter() - t0

    print(f"  Time:              {fmt_time(timings['embed'])}")
    print(f"  Total chunks:      {embed_result.total_chunks}")
    print(f"  Embedded:          {len(embed_result.embedded_chunks)}")
    print(f"  Skipped:           {embed_result.skipped_count}")

    if embed_result.embedded_chunks:
        first_ec = embed_result.embedded_chunks[0]
        dim = len(first_ec.embedding)
        print(f"  Embedding dim:     {dim}")
        print(f"  Model:             {first_ec.metadata.get('model', 'unknown')}")
        print()

        # Show embedding samples
        print("  --- Sample embeddings (first 5 chunks) ---")
        for ec in embed_result.embedded_chunks[:5]:
            preview = ec.chunk.content[:80].replace("\n", "\\n")
            vec_preview = str(ec.embedding[:5])[:-1] + ", ...]"
            print(f"  [{ec.chunk.chunk_index:2d}] {ec.chunk.content_type.value:18s} "
                  f"tokens={ec.chunk.token_count:4d}")
            print(f"       text: {preview}")
            print(f"       vec:  {vec_preview}")
            print()

        # Embedding statistics
        import statistics
        all_norms = []
        for ec in embed_result.embedded_chunks:
            norm = sum(v * v for v in ec.embedding) ** 0.5
            all_norms.append(norm)
        print(f"  --- Embedding vector statistics ---")
        print(f"  L2 norm min:  {min(all_norms):.4f}")
        print(f"  L2 norm max:  {max(all_norms):.4f}")
        print(f"  L2 norm mean: {statistics.mean(all_norms):.4f}")
    print()

    # =====================================================================
    # STAGE 6: INDEXING
    # =====================================================================
    print("=" * 70)
    print("STAGE 6: INDEXING")
    print("=" * 70)

    indexer = ChromaIndexer(config)
    document_id = "demo_travel_pdf"
    t0 = time.perf_counter()
    index_result = indexer.index(
        embedded_chunks=embed_result.embedded_chunks,
        document_id=document_id,
        file_name="travel.pdf",
        version=1,
        file_path=str(pdf_path),
    )
    timings["index"] = time.perf_counter() - t0

    print(f"  Time:              {fmt_time(timings['index'])}")
    print(f"  Collection:        {index_result.collection_name}")
    print(f"  Total indexed:     {index_result.total_indexed}")
    if index_result.indexed_ids:
        print(f"  Sample IDs:        {index_result.indexed_ids[:5]}")
        if len(index_result.indexed_ids) > 5:
            print(f"                     ... and {len(index_result.indexed_ids) - 5} more")
    print()

    # Verify retrieval from ChromaDB
    if index_result.total_indexed > 0 and embed_result.embedded_chunks:
        print("  --- ChromaDB retrieval verification ---")
        collection = indexer._client.get_collection(index_result.collection_name)
        print(f"  Collection count:  {collection.count()}")

        # Query with the first chunk's embedding
        query_vec = embed_result.embedded_chunks[0].embedding
        query_results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
        )
        print(f"  Query top-3 IDs:   {query_results['ids'][0]}")
        if query_results["documents"]:
            for rank, doc in enumerate(query_results["documents"][0][:3]):
                preview = doc[:80].replace("\n", "\\n")
                print(f"    [{rank}] {preview}")
    print()

    # =====================================================================
    # TIMING SUMMARY
    # =====================================================================
    total = sum(timings.values())
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Parse:       {parse_result.page_count} pages, "
          f"{parse_result.estimated_token_count} tokens  [{fmt_time(timings['parse'])}]")
    print(f"  Extract:     {len(extracted)} items                  "
          f"[{fmt_time(timings['extract'])}]")
    print(f"  Transform:   {len(transformed)} items "
          f"({changed_count} changed), "
          f"markdown={len(doc_markdown)} chars  [{fmt_time(timings['transform'])}]")
    print(f"  Chunk:       {len(chunks)} chunks, "
          f"{total_chunk_tokens} tokens  [{fmt_time(timings['chunk'])}]")
    print(f"  Embed:       {len(embed_result.embedded_chunks)} embedded, "
          f"{embed_result.skipped_count} skipped  [{fmt_time(timings['embed'])}]")
    print(f"  Index:       {index_result.total_indexed} indexed in "
          f"'{index_result.collection_name}'  [{fmt_time(timings['index'])}]")
    print(f"  " + "-" * 50)
    print(f"  Total:       {fmt_time(total)}")
    print()

    # Timing breakdown bar
    print("  Timing breakdown:")
    bar_width = 40
    for stage, t in timings.items():
        pct = (t / total) * 100 if total > 0 else 0
        filled = int(bar_width * t / total) if total > 0 else 0
        bar = "#" * filled + "." * (bar_width - filled)
        print(f"    {stage:12s} [{bar}] {pct:5.1f}%  {fmt_time(t)}")

    # =====================================================================
    # STAGE 7: WORKFLOW ORCHESTRATOR (end-to-end with DB tracking)
    # =====================================================================
    print()
    print("=" * 70)
    print("STAGE 7: WORKFLOW ORCHESTRATOR (end-to-end with DB tracking)")
    print("=" * 70)

    # Set up DB
    engine = get_engine(config.database.url)
    init_db(engine)
    session_factory = get_session_factory(engine)

    # Create a Document record
    session = session_factory()
    file_hash = compute_file_hash(str(pdf_path))
    doc = Document(
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        file_type="pdf",
        version=1,
        file_hash=file_hash,
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    doc_id = doc.id
    session.close()

    # Run orchestrator
    orchestrator = WorkflowOrchestrator(config, session_factory)
    t0 = time.perf_counter()
    run = orchestrator.process_document(doc)
    orchestrator_time = time.perf_counter() - t0

    print(f"  Time:            {fmt_time(orchestrator_time)}")
    print(f"  Run status:      {run.status}")
    print(f"  Total chunks:    {run.total_chunks}")
    print(f"  Total tokens:    {run.total_tokens}")
    if run.error_message:
        print(f"  Error:           {run.error_message}")
    print()

    # Show stage details from DB
    session = session_factory()
    db_run = session.query(WorkflowRun).filter_by(id=run.id).one()
    stages = (
        session.query(WorkflowStage)
        .filter_by(run_id=run.id)
        .all()
    )

    # Verify document status
    db_doc = session.query(Document).filter_by(id=doc_id).one()
    print(f"  Document status: {db_doc.status}")
    print()

    # Stage breakdown
    print("  --- Stage Details ---")
    print(f"  {'Stage':<12s} {'Status':<10s} {'Duration':>10s}  Metadata")
    print("  " + "-" * 65)

    for stage in sorted(stages, key=lambda s: s.started_at or s.completed_at or db_run.started_at):
        duration = ""
        if stage.started_at and stage.completed_at:
            dur_secs = (stage.completed_at - stage.started_at).total_seconds()
            duration = fmt_time(dur_secs)

        meta_str = ""
        if stage.metadata_json:
            meta = json.loads(stage.metadata_json)
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())

        error_str = ""
        if stage.error_message:
            error_str = f"  ERROR: {stage.error_message}"

        print(f"  {stage.stage_name:<12s} {stage.status:<10s} {duration:>10s}  {meta_str}{error_str}")

    session.close()
    print()

    # =====================================================================
    # STAGE 8: METRICS COLLECTION
    # =====================================================================
    print("=" * 70)
    print("STAGE 8: METRICS COLLECTION")
    print("=" * 70)

    from app.metrics.collector import MetricsCollector

    collector = MetricsCollector(session_factory, config)
    t0 = time.perf_counter()
    metrics = collector.collect()
    metrics_time = time.perf_counter() - t0

    print(f"  Time:            {fmt_time(metrics_time)}")
    collector.print_metrics(metrics)
    print()

    # =====================================================================
    # TRACING STATUS
    # =====================================================================
    print("=" * 70)
    print("TRACING STATUS")
    print("=" * 70)

    from app.observability.tracing import is_tracing_enabled, setup_tracing

    tracing_result = setup_tracing(config)
    print(f"  Config enabled:  {config.observability.langsmith_enabled}")
    print(f"  Setup result:    {tracing_result}")
    print(f"  Tracing active:  {is_tracing_enabled()}")
    print()

    print("=" * 70)
    print("END-TO-END DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
