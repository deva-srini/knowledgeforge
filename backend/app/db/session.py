"""Database engine, session management, and initialization for KnowledgeForge."""

from collections.abc import Generator
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import Base

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine(database_url: str = "sqlite:///./data/knowledgeforge.db") -> Engine:
    """Get or create the SQLAlchemy engine.

    Args:
        database_url: SQLAlchemy database connection URL.

    Returns:
        The SQLAlchemy engine instance.
    """
    global _engine
    if _engine is None:
        connect_args = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(database_url, connect_args=connect_args)
    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
    """Get or create the session factory.

    Args:
        engine: SQLAlchemy engine to bind to. Uses the global engine if None.

    Returns:
        A sessionmaker instance bound to the engine.
    """
    global _session_factory
    if _session_factory is None:
        if engine is None:
            engine = get_engine()
        _session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _session_factory


def get_session() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use.

    Yields:
        A SQLAlchemy Session instance.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all database tables.

    Args:
        engine: SQLAlchemy engine to use. Uses the global engine if None.
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)


def reset_globals() -> None:
    """Reset global engine and session factory. Used for testing."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
