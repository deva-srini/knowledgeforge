# Semantic Overlap Detection - Implementation Plan

## Summary
Add a minimal overlap detection feature that runs at ingestion time (during the indexing stage) to identify semantically similar chunks within and across documents in the same ChromaDB collection. Report-only -- no auto-dedup.

## Architecture Decision
- **Sub-step of indexing stage** (not a new 7th stage). Avoids changing `STAGE_NAMES`, stage creation loops, and all tests that assert on 6 stages.
- Overlap detection runs AFTER `delete_document()` but BEFORE `collection.upsert()` in `ChromaIndexer.index()`.
- Results stored in the index stage's existing `metadata_json` field.

---

## Files to Create

### 1. `backend/app/services/overlap_detection.py` (NEW)

**Data structures:**
- `OverlapMatch` dataclass: `source_chunk_index`, `source_content_preview`, `similar_chunk_id`, `similar_document_id`, `similar_file_name`, `similar_content_preview`, `distance`, `similarity`
- `OverlapReport` dataclass: `total_chunks_analyzed`, `overlaps_found`, `cross_document_overlaps`, `within_document_overlaps`, `matches: List[OverlapMatch]`, `threshold_used`, `max_similar_chunks` + `to_dict()` method

**Class `OverlapDetector`:**
- `__init__(similarity_threshold: float, max_similar_chunks: int)`
- `detect(embedded_chunks, collection, document_id) -> OverlapReport`
- `_detect_cross_document(...)` -- batch query ChromaDB (`collection.query(query_embeddings=...)`) for nearest neighbors, filter out same-document results, convert L2 distance to cosine similarity
- `_detect_within_document(...)` -- numpy pairwise cosine similarity matrix among new chunks

**Key details:**
- Distance threshold conversion: `distance_threshold = 2.0 * (1.0 - similarity_threshold)` (ChromaDB L2 on normalized vectors)
- Guard: skip cross-doc check if `collection.count() == 0`
- Batch all embeddings into a single `collection.query()` call
- numpy is already a transitive dep (via torch/sentence-transformers)

### 2. `backend/tests/test_overlap_detection.py` (NEW)

Tests using real ChromaDB (via `tmp_path`) and `_make_chunk`/`_make_embedded_chunk` helpers, following `test_indexing.py` patterns.

**Test classes:**
- `TestOverlapReport` -- serialization, empty report
- `TestOverlapDetectorCrossDocument` -- empty collection, identical embeddings detected, same-doc filtered, threshold respected, max_similar limit
- `TestOverlapDetectorWithinDocument` -- single chunk no overlap, identical chunks detected, dissimilar chunks not flagged
- `TestOverlapDetectorIntegration` -- both phases combined, empty input

---

## Files to Modify

### 3. `backend/app/core/config.py` (lines ~193-213)

Add `OverlapDetectionConfig` model (after `EmbeddingConfig`, line 182):
```python
class OverlapDetectionConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_similar_chunks: int = Field(default=5, gt=0)
```

Add field to `IndexingConfig` (line 209):
```python
overlap_detection: OverlapDetectionConfig = Field(default_factory=OverlapDetectionConfig)
```

Backward compatible -- existing configs without this key get defaults.

### 4. `backend/app/services/indexing.py`

**`IndexResult` (line 23):** Add optional field:
```python
overlap_report: Optional[dict] = None   # Serialized OverlapReport
```
Using `dict` (not `OverlapReport`) to avoid coupling the dataclass to the overlap module.

**`ChromaIndexer.__init__` (line 46):** Store overlap config:
```python
self._overlap_cfg = config.indexing.overlap_detection
```

**`ChromaIndexer.index()` (line 58):** After `get_or_create_collection` (line 86) but before upsert (line 112), insert:
```python
overlap_report = None
if self._overlap_cfg.enabled:
    from app.services.overlap_detection import OverlapDetector
    detector = OverlapDetector(
        similarity_threshold=self._overlap_cfg.similarity_threshold,
        max_similar_chunks=self._overlap_cfg.max_similar_chunks,
    )
    report = detector.detect(embedded_chunks, collection, document_id)
    overlap_report = report.to_dict()
    if report.overlaps_found > 0:
        logger.info("Overlap detection: %d overlaps for doc '%s'", ...)
```

Add `overlap_report=overlap_report` to the returned `IndexResult`.

### 5. `backend/app/services/workflow.py` (line 285)

Update index stage metadata to include overlap report when present:
```python
index_meta = {
    "collection": index_result.collection_name,
    "indexed": index_result.total_indexed,
}
if index_result.overlap_report:
    index_meta["overlap_report"] = index_result.overlap_report
stages["index"].metadata_json = json.dumps(index_meta)
```

### 6. `backend/cli.py` (after line 146)

Add overlap summary output after the existing process summary:
```python
# After "print(f'  Error: {run.error_message}')"
# Query the index stage metadata_json for overlap_report and print summary
```

Print: chunks analyzed, overlaps found (cross-doc / within-doc), top 3 matches with similarity scores.

### 7. `kf_config.yaml` (after line 39)

Add default config section:
```yaml
  overlap_detection:
    enabled: false
    similarity_threshold: 0.85
    max_similar_chunks: 5
```

---

## Data Flow

```
EmbeddedChunks (from embed stage)
    |
    v
ChromaIndexer.index()
    |
    +-- delete_document() (remove old version)
    +-- get_or_create_collection()
    +-- OverlapDetector.detect()          <-- NEW
    |     |-- _detect_cross_document()    query ChromaDB
    |     |-- _detect_within_document()   numpy pairwise
    |     └-- return OverlapReport
    +-- collection.upsert()               (existing)
    └-- return IndexResult(overlap_report=report.to_dict())
              |
              v
        WorkflowStage.metadata_json       (stored in DB)
              |
              v
        GET /documents/{id}/status        (already surfaced)
        CLI process output                (new print block)
```

## API Surfacing
No new endpoints needed. The existing `GET /documents/{doc_id}/status` endpoint returns `WorkflowStageResponse.metadata_json` which will now contain the overlap report as a nested JSON object in the index stage.

---

## Verification Plan
1. **Unit tests**: `pytest backend/tests/test_overlap_detection.py` -- all overlap detector tests
2. **Existing tests**: `pytest backend/tests/test_indexing.py` -- verify no regressions (overlap disabled by default)
3. **Config test**: `pytest backend/tests/test_config.py` -- verify config loads with new fields
4. **Full suite**: `pytest backend/tests/` -- all 290+ tests pass
5. **Manual E2E**: Enable overlap detection in `kf_config.yaml`, process two similar PDFs via CLI, verify overlap report in output

## Implementation Order
1. `config.py` -- add OverlapDetectionConfig
2. `overlap_detection.py` -- core service (new file)
3. `kf_config.yaml` -- add default config
4. `indexing.py` -- wire detector into ChromaIndexer
5. `workflow.py` -- store report in metadata_json
6. `cli.py` -- print overlap summary
7. `test_overlap_detection.py` -- tests (new file)
8. Run full test suite
