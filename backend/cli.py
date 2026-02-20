"""CLI entry point for KnowledgeForge.

Provides commands to start the server, process individual documents,
check document status, and view processing metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from app.core.config import load_config
from app.db.session import get_engine, get_session_factory, init_db
from app.models.database import Document
from app.services.filewatcher import compute_file_hash
from app.services.workflow import WorkflowOrchestrator


def cmd_start(args: Any) -> None:
    """Start the KnowledgeForge FastAPI server.

    Args:
        args: Parsed CLI arguments.
    """
    print("Starting KnowledgeForge...")
    import uvicorn

    config_path = getattr(args, "config", None)
    if config_path:
        print(f"Using config: {config_path}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


def cmd_process(args: Any) -> None:
    """Process a specific document through the full pipeline.

    Args:
        args: Parsed CLI arguments with --file, optional --force and --workflow.
    """
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Processing file: {file_path}")

    # Setup
    config = load_config()
    engine = get_engine(config.database.url)
    init_db(engine)
    session_factory = get_session_factory(engine)

    # Load workflow config if specified
    workflow_config = None
    workflow_name = getattr(args, "workflow", None)
    if workflow_name:
        from app.core.workflow_config import load_workflow

        print(f"Using workflow: {workflow_name}")
        workflow_config = load_workflow(workflow_name, config)

    # Find or create document record
    session = session_factory()
    try:
        file_name = file_path.name
        file_hash = compute_file_hash(str(file_path))
        file_type = file_path.suffix.lstrip(".").lower()

        query = session.query(Document).filter_by(file_name=file_name)
        if workflow_name:
            query = query.filter_by(workflow_id=workflow_name)
        else:
            query = query.filter(Document.workflow_id.is_(None))
        existing = query.order_by(Document.version.desc()).first()

        if existing is not None and not args.force:
            if existing.status == "indexed":
                print(
                    f"Document '{file_name}' already indexed "
                    f"(version {existing.version}). "
                    f"Use --force to re-process."
                )
                return
            if existing.file_hash == file_hash and existing.status != "failed":
                print(
                    f"Document '{file_name}' unchanged "
                    f"(version {existing.version}, status={existing.status})."
                )
                return

        if existing is not None and existing.file_hash == file_hash:
            # Re-process existing version
            doc = existing
        elif existing is not None:
            # New version
            doc = Document(
                file_name=file_name,
                file_path=str(file_path),
                file_type=file_type,
                version=existing.version + 1,
                file_hash=file_hash,
                workflow_id=workflow_name,
                status="pending",
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
        else:
            # First time
            doc = Document(
                file_name=file_name,
                file_path=str(file_path),
                file_type=file_type,
                version=1,
                file_hash=file_hash,
                workflow_id=workflow_name,
                status="pending",
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)

        if args.force:
            print("Force re-processing enabled")

    finally:
        session.close()

    # Process
    orchestrator = WorkflowOrchestrator(
        config, session_factory, workflow_config=workflow_config
    )
    run = orchestrator.process_document(doc)

    # Print summary
    print(f"\nResult: {run.status}")
    print(f"  Chunks: {run.total_chunks}")
    print(f"  Tokens: {run.total_tokens}")
    if run.error_message:
        print(f"  Error: {run.error_message}")


def cmd_status(args: Any) -> None:
    """Display the processing status of all documents.

    Args:
        args: Parsed CLI arguments.
    """
    config = load_config()
    engine = get_engine(config.database.url)
    init_db(engine)
    session_factory = get_session_factory(engine)

    session = session_factory()
    try:
        docs = session.query(Document).order_by(Document.updated_at.desc()).all()

        if not docs:
            print("No documents found.")
            return

        print(f"\n{'File Name':<30} {'Version':>7} {'Status':<12} {'Updated At'}")
        print("-" * 75)

        for doc in docs:
            updated = doc.updated_at.strftime("%Y-%m-%d %H:%M:%S") if doc.updated_at else "N/A"
            print(f"{doc.file_name:<30} {doc.version:>7} {doc.status:<12} {updated}")

        print(f"\nTotal: {len(docs)} document(s)")

    finally:
        session.close()


def cmd_metrics(args: Any) -> None:
    """Display processing metrics and statistics.

    Args:
        args: Parsed CLI arguments.
    """
    from app.metrics.collector import MetricsCollector

    config = load_config()
    engine = get_engine(config.database.url)
    init_db(engine)
    session_factory = get_session_factory(engine)

    collector = MetricsCollector(session_factory, config)
    metrics = collector.collect()
    collector.print_metrics(metrics)


def cmd_workflows(args: Any) -> None:
    """List all registered workflows from the registry.

    Args:
        args: Parsed CLI arguments.
    """
    from app.core.workflow_config import load_registry

    try:
        registry = load_registry()
    except FileNotFoundError:
        print("No workflows directory found. Create knowledgeforge/workflows/registry.yaml to use workflows.")
        return

    if not registry.workflows:
        print("No workflows registered.")
        return

    print(f"\n{'Name':<25} {'Active':<8} {'Config':<25} {'Description'}")
    print("-" * 85)

    for entry in registry.workflows:
        active_str = "yes" if entry.active else "no"
        print(f"{entry.name:<25} {active_str:<8} {entry.config:<25} {entry.description}")

    print(f"\nTotal: {len(registry.workflows)} workflow(s)")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="KnowledgeForge CLI - Knowledge ingestion and indexing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start KnowledgeForge server")
    start_parser.add_argument(
        "--config", type=str, help="Path to kf_config.yaml", default=None
    )
    start_parser.set_defaults(func=cmd_start)

    # process command
    process_parser = subparsers.add_parser(
        "process", help="Process a specific document"
    )
    process_parser.add_argument(
        "--file", type=str, required=True, help="Path to document file"
    )
    process_parser.add_argument(
        "--force", action="store_true", help="Force re-processing"
    )
    process_parser.add_argument(
        "--workflow", type=str, default=None,
        help="Workflow name to use from the registry",
    )
    process_parser.set_defaults(func=cmd_process)

    # status command
    status_parser = subparsers.add_parser(
        "status", help="Check document processing status"
    )
    status_parser.set_defaults(func=cmd_status)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="View processing metrics")
    metrics_parser.set_defaults(func=cmd_metrics)

    # workflows command
    workflows_parser = subparsers.add_parser(
        "workflows", help="List registered workflows"
    )
    workflows_parser.set_defaults(func=cmd_workflows)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
