"""Workflow API routes for KnowledgeForge."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/workflows")
async def list_workflows(request: Request) -> List[Dict[str, Any]]:
    """List all active workflows with their status.

    Returns:
        List of workflow status dicts.
    """
    registry_manager = getattr(request.app.state, "registry_manager", None)
    if registry_manager is None:
        return []
    return registry_manager.list_workflows()


@router.post("/workflows/sync")
async def sync_workflows(request: Request) -> Dict[str, str]:
    """Trigger immediate registry re-read and hot-reload.

    Returns:
        Confirmation message.
    """
    registry_manager = getattr(request.app.state, "registry_manager", None)
    if registry_manager is None:
        raise HTTPException(
            status_code=503, detail="Registry manager not initialised"
        )
    registry_manager.sync()
    return {"status": "synced"}


@router.get("/workflows/{name}/status")
async def get_workflow_status(
    name: str, request: Request
) -> Dict[str, Any]:
    """Get detailed status of a specific workflow.

    Args:
        name: Workflow name.

    Returns:
        Workflow status dict.
    """
    registry_manager = getattr(request.app.state, "registry_manager", None)
    if registry_manager is None:
        raise HTTPException(
            status_code=503, detail="Registry manager not initialised"
        )

    workflows = registry_manager.list_workflows()
    for wf in workflows:
        if wf["name"] == name:
            return wf

    raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
