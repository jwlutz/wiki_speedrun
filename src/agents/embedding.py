"""
Embedding-based agents using pre-computed or live embeddings.

PrecomputedEmbeddingAgent: Uses pre-computed title embeddings from wiki_data
LiveEmbeddingAgent: Computes embeddings on-the-fly with sentence-transformers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.agents.base import Agent, AgentContext

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingAgent(Agent):
    """
    Greedy agent that uses pre-computed title embeddings.

    Picks the link with highest cosine similarity to the target.
    Uses the wiki_data FAISS index for fast similarity lookup.
    """

    def __init__(self, avoid_revisits: bool = True) -> None:
        """
        Initialize with pre-computed embeddings.

        Args:
            avoid_revisits: If True, penalize revisiting pages
        """
        self._avoid_revisits = avoid_revisits
        self._wiki_data = None  # Lazy load

    def _ensure_loaded(self) -> None:
        """Lazy load wiki_data to avoid slow import."""
        if self._wiki_data is None:
            from src.data.loader import wiki_data

            self._wiki_data = wiki_data

    @property
    def name(self) -> str:
        return "precomputed"

    @property
    def description(self) -> str:
        return "Greedy embedding similarity (pre-computed title embeddings)"

    def choose_link(self, context: AgentContext) -> str:
        """Pick the link most similar to target."""
        self._ensure_loaded()

        # Always click target if available
        if context.target_title in context.available_links:
            return context.target_title

        # Rank by similarity to target
        ranked = self._wiki_data.rank_by_similarity(
            candidates=context.available_links,
            target=context.target_title,
        )

        if not ranked:
            # Fallback: no embeddings found, pick first link
            logger.warning("No embeddings found for candidates, using fallback")
            return context.available_links[0]

        # Filter out revisits if enabled
        if self._avoid_revisits:
            visited = set(context.path_so_far)
            for title, _sim in ranked:
                if title not in visited:
                    return title

        # Return best match (or first if all visited)
        return ranked[0][0]


class LiveEmbeddingAgent(Agent):
    """
    Greedy agent that computes embeddings on-the-fly.

    Uses sentence-transformers to embed article titles in real-time.
    More accurate than pre-computed (can use better models) but slower.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        avoid_revisits: bool = True,
    ) -> None:
        """
        Initialize with a sentence-transformer model.

        Args:
            model_name: HuggingFace model name for sentence-transformers
            avoid_revisits: If True, penalize revisiting pages
        """
        self._model_name = model_name
        self._avoid_revisits = avoid_revisits
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformer model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

    @property
    def name(self) -> str:
        # Short name for the model
        short_name = self._model_name.split("/")[-1]
        return f"live-{short_name}"

    @property
    def description(self) -> str:
        return f"Greedy embedding similarity (live: {self._model_name})"

    def _compute_similarities(
        self, candidates: list[str], target: str
    ) -> list[tuple[str, float]]:
        """Compute cosine similarities between candidates and target."""
        self._ensure_loaded()

        if not candidates:
            return []

        # Encode target and all candidates
        all_texts = [target] + candidates
        embeddings = self._model.encode(all_texts, convert_to_numpy=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

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
        """Pick the link most similar to target using live embeddings."""
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


class HybridEmbeddingAgent(Agent):
    """
    Agent that combines pre-computed and live embeddings.

    Uses pre-computed for quick filtering, then live for final ranking.
    Good balance of speed and accuracy.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 20,
        avoid_revisits: bool = True,
    ) -> None:
        """
        Initialize hybrid agent.

        Args:
            model_name: Model for live embedding
            top_k: Number of candidates to re-rank with live embeddings
            avoid_revisits: Penalize revisits
        """
        self._precomputed = PrecomputedEmbeddingAgent(avoid_revisits=False)
        self._live = LiveEmbeddingAgent(model_name, avoid_revisits=False)
        self._top_k = top_k
        self._avoid_revisits = avoid_revisits

    @property
    def name(self) -> str:
        return f"hybrid-{self._top_k}"

    @property
    def description(self) -> str:
        return f"Hybrid: pre-computed filter ({self._top_k}) + live re-rank"

    def choose_link(self, context: AgentContext) -> str:
        """Two-stage selection: pre-computed filter, then live re-rank."""
        # Always click target if available
        if context.target_title in context.available_links:
            return context.target_title

        self._precomputed._ensure_loaded()

        # Stage 1: Get top-k candidates from pre-computed
        ranked = self._precomputed._wiki_data.rank_by_similarity(
            candidates=context.available_links,
            target=context.target_title,
        )

        if not ranked:
            return context.available_links[0]

        # Take top-k for re-ranking
        top_candidates = [title for title, _ in ranked[: self._top_k]]

        # Stage 2: Re-rank with live embeddings
        reranked = self._live._compute_similarities(
            candidates=top_candidates,
            target=context.target_title,
        )

        # Filter revisits
        if self._avoid_revisits:
            visited = set(context.path_so_far)
            for title, _sim in reranked:
                if title not in visited:
                    return title

        return reranked[0][0] if reranked else top_candidates[0]
