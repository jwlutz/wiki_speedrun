"""
Embedding agent with HuggingFace API or local fallback.

Uses HuggingFace Inference API when API key is available,
otherwise falls back to local sentence-transformers.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from src.agents.base import Agent, AgentContext
from src.config import HF_API_KEY

logger = logging.getLogger(__name__)

# Model name mapping (short name -> HF model ID)
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
}


class EmbeddingClient:
    """
    Client for getting embeddings via HuggingFace API or local models.

    Automatically uses HF API if API key is available, otherwise local.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        api_key: str | None = None,
        on_status: Callable[[str], None] | None = None,
        force_local: bool = False,
    ) -> None:
        """
        Initialize the embedding client.

        Args:
            model_name: Short name or full HF model ID
            api_key: HuggingFace API key (uses env var if not provided)
            on_status: Callback for status messages (warmup, loading, etc.)
            force_local: Force local model even if API key is available
        """
        # Resolve model name
        if model_name in EMBEDDING_MODELS:
            self._model_id = EMBEDDING_MODELS[model_name]
        else:
            self._model_id = model_name

        self._short_name = model_name.split("/")[-1]
        self._api_key = api_key or HF_API_KEY
        self._on_status = on_status
        self._force_local = force_local
        self._is_warm = False

        # Will be initialized lazily
        self._hf_client = None
        self._local_model = None
        self._use_api = bool(self._api_key) and not force_local

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_warm(self) -> bool:
        return self._is_warm

    @property
    def using_api(self) -> bool:
        return self._use_api

    def _notify(self, message: str) -> None:
        """Send status notification."""
        if self._on_status:
            self._on_status(message)
        logger.info(message)

    def _init_hf_client(self) -> bool:
        """Initialize HuggingFace API client."""
        if self._hf_client is not None:
            return True

        try:
            from huggingface_hub import InferenceClient
            self._hf_client = InferenceClient(token=self._api_key)
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed, falling back to local")
            self._use_api = False
            return False
        except Exception as e:
            logger.warning(f"Failed to init HF client: {e}, falling back to local")
            self._use_api = False
            return False

    def _init_local_model(self) -> bool:
        """Initialize local sentence-transformer model."""
        if self._local_model is not None:
            return True

        try:
            self._notify(f"Loading model locally ({self._short_name})...")
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._model_id)
            self._notify("Model loaded!")
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False

    def _embed_via_api(self, texts: list[str]) -> np.ndarray | None:
        """Get embeddings via HuggingFace API."""
        if not self._init_hf_client():
            return None

        try:
            self._notify("Getting embeddings from HuggingFace API...")

            # HF API handles batching internally
            embeddings = []
            for text in texts:
                result = self._hf_client.feature_extraction(
                    text,
                    model=self._model_id,
                )
                # Result may be nested, flatten if needed
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Token-level embeddings, take mean
                        embeddings.append(np.mean(result, axis=0))
                    else:
                        embeddings.append(result)
                else:
                    embeddings.append(result)

            self._is_warm = True
            return np.array(embeddings)

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "loading" in error_str.lower():
                self._notify("Model warming up on HuggingFace (~20s)...")
                time.sleep(20)
                return self._embed_via_api(texts)  # Retry
            elif "401" in error_str or "unauthorized" in error_str.lower():
                logger.warning("HF API unauthorized, falling back to local")
                self._use_api = False
                return self._embed_via_local(texts)
            else:
                logger.error(f"HF API error: {e}")
                # Fall back to local
                self._use_api = False
                return self._embed_via_local(texts)

    def _embed_via_local(self, texts: list[str]) -> np.ndarray | None:
        """Get embeddings via local model."""
        if not self._init_local_model():
            return None

        try:
            embeddings = self._local_model.encode(texts, convert_to_numpy=True)
            self._is_warm = True
            return embeddings
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return None

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """
        Get embeddings for a list of texts.

        Automatically uses HF API if available, otherwise local model.

        Args:
            texts: Texts to embed

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim) or None
        """
        if not texts:
            return np.array([])

        if self._use_api:
            return self._embed_via_api(texts)
        else:
            return self._embed_via_local(texts)

    def warmup(self) -> bool:
        """
        Warm up the model with a test query.

        Returns:
            True if model is ready, False otherwise
        """
        try:
            self._notify("Warming up embedding model...")
            result = self.embed(["test warmup query"])
            if result is not None and len(result) > 0:
                self._notify("Model ready!")
                return True
            return False
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return False


class HFEmbeddingAgent(Agent):
    """
    Greedy agent that uses embeddings for link selection.

    Automatically uses HuggingFace API when available, otherwise local models.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        avoid_revisits: bool = True,
        on_status: Callable[[str], None] | None = None,
        force_local: bool = False,
    ) -> None:
        """
        Initialize embedding agent.

        Args:
            model_name: Model name (short name or full HF model ID)
            avoid_revisits: If True, penalize revisiting pages
            on_status: Callback for status messages
            force_local: Force local model even if API key available
        """
        self._model_name = model_name
        self._avoid_revisits = avoid_revisits
        self._client = EmbeddingClient(
            model_name,
            on_status=on_status,
            force_local=force_local,
        )

    @property
    def name(self) -> str:
        short_name = self._model_name.split("/")[-1]
        prefix = "hf" if self._client.using_api else "local"
        return f"{prefix}-{short_name}"

    @property
    def description(self) -> str:
        mode = "HuggingFace API" if self._client.using_api else "local"
        return f"Greedy embedding ({mode}: {self._model_name})"

    def warmup(self) -> bool:
        """Warm up the model before gameplay."""
        return self._client.warmup()

    def _compute_similarities(
        self, candidates: list[str], target: str
    ) -> list[tuple[str, float]]:
        """Compute cosine similarities between candidates and target."""
        if not candidates:
            return []

        # Encode target and all candidates
        all_texts = [target] + candidates
        embeddings = self._client.embed(all_texts)

        if embeddings is None or len(embeddings) == 0:
            logger.warning("Failed to get embeddings")
            return [(c, 0.0) for c in candidates]

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings = embeddings / norms

        # Target is first embedding
        target_emb = embeddings[0]
        candidate_embs = embeddings[1:]

        # Compute similarities
        similarities = np.dot(candidate_embs, target_emb)

        # Return sorted by similarity
        results = list(zip(candidates, similarities.tolist(), strict=True))
        results.sort(key=lambda x: x[1], reverse=True)
        return [(title, sim) for title, sim in results]

    def choose_link(self, context: AgentContext) -> str:
        """Pick the link most similar to target using embeddings."""
        # Always click target if available
        if context.target_title in context.available_links:
            return context.target_title

        # Compute similarities
        ranked = self._compute_similarities(
            candidates=context.available_links,
            target=context.target_title,
        )

        if not ranked:
            return context.available_links[0]

        # Filter out revisits if enabled
        if self._avoid_revisits:
            visited = set(context.path_so_far)
            for title, _sim in ranked:
                if title not in visited:
                    return title

        return ranked[0][0]
