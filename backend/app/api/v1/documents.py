"""Document API endpoints for KnowledgeForge."""

import hashlib
import os
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.models.database import Document
from app.models.schemas import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    ProcessRequest,
    WorkflowRunResponse,
    WorkflowStageResponse,
)

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """List all documents ordered by updated_at descending."""
    session = request.app.state.session_factory()
    try:
        docs = (
            session.query(Document)
            .order_by(Document.updated_at.desc())
            .all()
        )
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(d) for d in docs],
            total=len(docs),
        )
    finally:
        session.close()


@router.get("/documents/{doc_id}/status", response_model=DocumentDetailResponse)
async def document_status(doc_id: str, request: Request) -> DocumentDetailResponse:
    """Get detailed status for a specific document including workflow runs and stages."""
    session = request.app.state.session_factory()
    try:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        runs_response = []
        for run in doc.workflow_runs:
            stages_response = [
                WorkflowStageResponse.model_validate(stage)
                for stage in run.stages
            ]
            run_data = WorkflowRunResponse.model_validate(run)
            run_data.stages = stages_response
            runs_response.append(run_data)

        detail = DocumentDetailResponse.model_validate(doc)
        detail.workflow_runs = runs_response
        return detail
    finally:
        session.close()


@router.post("/documents/process")
async def process_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ProcessRequest = ProcessRequest(),
) -> Dict[str, Any]:
    """Trigger document processing.

    If file_path is provided, process that specific file.
    Otherwise, process all pending/failed documents.
    """
    orchestrator = request.app.state.orchestrator
    session = request.app.state.session_factory()
    try:
        if body.file_path:
            # Find or create a document record for the given file
            doc = (
                session.query(Document)
                .filter(Document.file_path == body.file_path)
                .first()
            )
            if doc is None:
                file_name = os.path.basename(body.file_path)
                _, ext = os.path.splitext(file_name)
                file_hash = ""
                if os.path.exists(body.file_path):
                    with open(body.file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()

                doc = Document(
                    file_name=file_name,
                    file_path=body.file_path,
                    file_type=ext.lstrip(".").lower(),
                    file_hash=file_hash,
                    status="pending",
                )
                session.add(doc)
                session.commit()
                session.refresh(doc)

            # Expunge so we can use it outside the session
            session.expunge(doc)
            background_tasks.add_task(orchestrator.process_document, doc)
            return {"message": "Processing started", "document_count": 1}
        else:
            # Process all pending/failed documents
            docs = (
                session.query(Document)
                .filter(Document.status.in_(["pending", "failed"]))
                .all()
            )
            for doc in docs:
                session.expunge(doc)

            for doc in docs:
                background_tasks.add_task(orchestrator.process_document, doc)

            return {"message": "Processing started", "document_count": len(docs)}
    finally:
        session.close()


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Reprocess a specific document by ID."""
    orchestrator = request.app.state.orchestrator
    session = request.app.state.session_factory()
    try:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        session.expunge(doc)
        background_tasks.add_task(orchestrator.process_document, doc)
        return {"message": "Reprocessing started", "document_id": doc_id}
    finally:
        session.close()
