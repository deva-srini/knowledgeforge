"""Embedding generator service with multi-provider support.

Takes a list of Chunk objects from the chunking step and produces
EmbeddedChunk objects containing the original chunk plus its vector
embedding. Uses a pluggable provider architecture — currently supports
sentence-transformers locally, with an ABC for future API providers
(OpenAI, Cohere, etc.).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from app.core.config import KnowledgeForgeConfig
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector.

    Attributes:
        chunk: The original Chunk object.
        embedding: The embedding vector as a list of floats.
        metadata: Additional metadata about the embedding.
    """

    chunk: Chunk
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResult:
    """Result of an embedding operation over a set of chunks.

    Attributes:
        embedded_chunks: Successfully embedded chunks.
        skipped_count: Number of chunks that failed and were skipped.
        total_chunks: Total number of input chunks.
    """

    embedded_chunks: List[EmbeddedChunk]
    skipped_count: int
    total_chunks: int


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers.

    Each provider must implement embed_batch, dimension, and model_name.
    Adding a new provider means subclassing this ABC and registering it
    in the Embedder factory.
    """

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension of the embedding output.
        """

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string.

        Returns:
            Model name or identifier.
        """


class SentenceTransformerClient(EmbeddingClient):
    """Embedding client using sentence-transformers locally.

    Lazily loads the model on first call to embed_batch, enabling fast
    initialisation and deferring GPU memory allocation until needed.
    Automatically detects CUDA availability.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize with the model identifier.

        Args:
            model_id: HuggingFace model identifier for sentence-transformers.
        """
        self._model_id = model_id
        self._model: Optional[Any] = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Load the SentenceTransformer model lazily.

        Imports sentence_transformers here to avoid import cost at init.
        Detects CUDA and moves the model to GPU if available.
        """
        from sentence_transformers import SentenceTransformer

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading SentenceTransformer model '%s' on device '%s'",
            self._model_id,
            device,
        )
        self._model = SentenceTransformer(self._model_id, device=device)
        # Determine dimension from a probe embedding.
        probe = self._model.encode(["probe"])
        self._dimension = len(probe[0])

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the sentence-transformers model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors as lists of floats.
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None  # noqa: S101
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Integer dimension of embedding vectors.
        """
        if self._dimension is None:
            self._load_model()
        assert self._dimension is not None  # noqa: S101
        return self._dimension

    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            The HuggingFace model identifier string.
        """
        return self._model_id


# Registry of provider name -> client class.
_PROVIDER_REGISTRY: Dict[str, Type[EmbeddingClient]] = {
    "sentence_transformers": SentenceTransformerClient,
}


class Embedder:
    """Orchestrator that embeds chunks using a configured provider.

    Handles batching, error recovery (batch fallback to individual),
    and result aggregation.
    """

    def __init__(self, config: KnowledgeForgeConfig) -> None:
        """Initialize the embedder from configuration.

        Args:
            config: KnowledgeForge configuration object.

        Raises:
            ValueError: If the configured provider is unknown or a required
                API key environment variable is not set.
        """
        self._batch_size = config.processing.embedding.batch_size
        self._client = self._create_client(config)

    def _create_client(self, config: KnowledgeForgeConfig) -> EmbeddingClient:
        """Factory method to create the appropriate EmbeddingClient.

        Args:
            config: KnowledgeForge configuration object.

        Returns:
            An EmbeddingClient instance for the configured provider.

        Raises:
            ValueError: If the provider is unknown or API key is missing.
        """
        embedding_config = config.processing.embedding
        provider = embedding_config.provider

        if provider not in _PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Available providers: {', '.join(sorted(_PROVIDER_REGISTRY))}"
            )

        # Validate API key for providers that need one.
        if embedding_config.api_key_env:
            api_key = os.environ.get(embedding_config.api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"Embedding provider '{provider}' requires API key "
                    f"but environment variable '{embedding_config.api_key_env}' "
                    f"is not set or empty."
                )

        client_class = _PROVIDER_REGISTRY[provider]

        if provider == "sentence_transformers":
            return client_class(model_id=embedding_config.model)

        # Generic fallback for future providers.
        return client_class(model_id=embedding_config.model)  # type: ignore[call-arg]

    def embed(self, chunks: List[Chunk]) -> EmbedResult:
        """Embed a list of chunks, returning an EmbedResult.

        Chunks are processed in batches. If a batch fails, each chunk in
        that batch is retried individually. Individual failures are logged
        and skipped.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            EmbedResult with embedded chunks, skip count, and total.
        """
        if not chunks:
            return EmbedResult(
                embedded_chunks=[],
                skipped_count=0,
                total_chunks=0,
            )

        embedded_chunks: List[EmbeddedChunk] = []
        skipped_count = 0

        # Split into batches.
        batches = self._make_batches(chunks)

        for batch in batches:
            try:
                texts = [c.content for c in batch]
                embeddings = self._client.embed_batch(texts)
                for chunk, embedding in zip(batch, embeddings):
                    embedded_chunks.append(
                        EmbeddedChunk(
                            chunk=chunk,
                            embedding=embedding,
                            metadata={
                                "model": self._client.model_name(),
                                "dimension": self._client.dimension(),
                            },
                        )
                    )
            except Exception:
                logger.warning(
                    "Batch embedding failed for %d chunks, "
                    "falling back to individual processing",
                    len(batch),
                    exc_info=True,
                )
                # Fallback: try each chunk individually.
                for chunk in batch:
                    try:
                        embeddings = self._client.embed_batch([chunk.content])
                        embedded_chunks.append(
                            EmbeddedChunk(
                                chunk=chunk,
                                embedding=embeddings[0],
                                metadata={
                                    "model": self._client.model_name(),
                                    "dimension": self._client.dimension(),
                                },
                            )
                        )
                    except Exception:
                        logger.error(
                            "Failed to embed chunk (index=%d, "
                            "header_path='%s'), skipping",
                            chunk.chunk_index,
                            chunk.header_path,
                            exc_info=True,
                        )
                        skipped_count += 1

        logger.info(
            "Embedding complete: %d/%d chunks embedded, %d skipped",
            len(embedded_chunks),
            len(chunks),
            skipped_count,
        )

        return EmbedResult(
            embedded_chunks=embedded_chunks,
            skipped_count=skipped_count,
            total_chunks=len(chunks),
        )

    def _make_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Split chunks into batches of configured size.

        Args:
            chunks: All chunks to batch.

        Returns:
            List of chunk lists, each at most batch_size long.
        """
        return [
            chunks[i : i + self._batch_size]
            for i in range(0, len(chunks), self._batch_size)
        ]
