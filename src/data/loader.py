"""
WikiData singleton for accessing Wikipedia embeddings, link graph, and titles.

Usage:
    from src.data.loader import wiki_data

    # First access triggers lazy loading (~30-60 seconds)
    wiki_data.get_title(1234)
    wiki_data.get_embedding("Albert Einstein")
    wiki_data.get_links("Python (programming language)")
    wiki_data.similarity("Physics", "Mathematics")
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import faiss
import msgpack
import numpy as np

from src.config import (
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    LINK_GRAPH_PATH,
    TITLE_TO_IDX_PATH,
    TITLES_PATH,
)

# Path for persisted FAISS index (avoids rebuilding on each run)
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

logger = logging.getLogger(__name__)


class WikiData:
    """
    Lazy-loading singleton for Wikipedia data.

    Loads data on first access to any method. All data is shared across
    the application via the module-level `wiki_data` instance.

    Attributes:
        titles: List of all Wikipedia article titles (indexed)
        title_to_idx: Dict mapping title string to index
        link_graph: Dict mapping source index to list of target indices
        embeddings: Memory-mapped numpy array of embeddings
        faiss_index: FAISS index for fast similarity search
    """

    _instance: WikiData | None = None

    def __new__(cls) -> WikiData:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Load all data on first access."""
        if self._initialized:
            return

        logger.info("Loading Wikipedia data (this may take a minute)...")

        self._load_titles()
        self._load_title_to_idx()
        self._load_link_graph()
        self._load_embeddings()
        self._build_faiss_index()

        self._initialized = True
        logger.info("Wikipedia data loaded successfully")

    def _load_titles(self) -> None:
        """Load titles.json into list."""
        logger.info(f"Loading titles from {TITLES_PATH}...")
        with open(TITLES_PATH, encoding="utf-8") as f:
            self._titles: list[str] = json.load(f)
        logger.info(f"Loaded {len(self._titles):,} titles")

    def _load_title_to_idx(self) -> None:
        """Load title_to_idx.json into dict."""
        logger.info(f"Loading title->index mapping from {TITLE_TO_IDX_PATH}...")
        with open(TITLE_TO_IDX_PATH, encoding="utf-8") as f:
            self._title_to_idx: dict[str, int] = json.load(f)
        logger.info(f"Loaded {len(self._title_to_idx):,} mappings")

    def _load_link_graph(self) -> None:
        """Load link_graph.msgpack into dict."""
        logger.info(f"Loading link graph from {LINK_GRAPH_PATH}...")
        with open(LINK_GRAPH_PATH, "rb") as f:
            self._link_graph: dict[int, list[int]] = msgpack.load(f)
        logger.info(f"Loaded {len(self._link_graph):,} articles with outgoing links")

    def _load_embeddings(self) -> None:
        """Memory-map embeddings.npy for efficient access."""
        logger.info(f"Memory-mapping embeddings from {EMBEDDINGS_PATH}...")
        self._embeddings: np.ndarray = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        logger.info(f"Embeddings shape: {self._embeddings.shape}, dtype: {self._embeddings.dtype}")

    def _build_faiss_index(self) -> None:
        """Load or build FAISS index (memory-efficient for 32GB RAM systems)."""
        # Try to load pre-built index from disk
        if FAISS_INDEX_PATH.exists():
            logger.info(f"Loading pre-built FAISS index from {FAISS_INDEX_PATH}...")
            self._faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal:,} vectors")
            return

        # Build IVF index (approximate but memory-efficient)
        # IndexIVFFlat uses centroids + inverted lists, much lower memory than IndexFlatIP
        logger.info("Building FAISS IVF index (this may take a while)...")

        total = len(self._embeddings)
        nlist = 4096  # Number of clusters (sqrt(n) is typical)

        # Create IVF index with inner product (for cosine similarity on normalized vectors)
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._faiss_index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train on a sample (IVF requires training)
        logger.info("  Training index on sample...")
        sample_size = min(500_000, total)
        sample_indices = np.random.choice(total, sample_size, replace=False)
        sample = self._embeddings[sample_indices].astype(np.float32)
        faiss.normalize_L2(sample)
        self._faiss_index.train(sample)
        logger.info("  Training complete")

        # Add vectors in batches
        batch_size = 100_000
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)

            batch = self._embeddings[start_idx:end_idx].astype(np.float32)
            faiss.normalize_L2(batch)
            self._faiss_index.add(batch)

            if (start_idx // batch_size) % 10 == 0:
                logger.info(f"  Added {end_idx:,}/{total:,} vectors ({100*end_idx/total:.1f}%)")

        # Set search parameters (higher nprobe = more accurate but slower)
        self._faiss_index.nprobe = 64

        # Save to disk for future runs
        logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
        faiss.write_index(self._faiss_index, str(FAISS_INDEX_PATH))

        logger.info(f"FAISS index built with {self._faiss_index.ntotal:,} vectors")

    # =========================================================================
    # Core Accessors
    # =========================================================================

    def get_title(self, idx: int) -> str:
        """Get article title by index."""
        self._ensure_loaded()
        if 0 <= idx < len(self._titles):
            return self._titles[idx]
        raise IndexError(f"Index {idx} out of range [0, {len(self._titles)})")

    def get_index(self, title: str) -> int | None:
        """Get index for article title, or None if not found."""
        self._ensure_loaded()
        return self._title_to_idx.get(title)

    def has_article(self, title: str) -> bool:
        """Check if article exists in the dataset."""
        self._ensure_loaded()
        return title in self._title_to_idx

    def is_traversable(self, title: str) -> bool:
        """Check if article has outgoing links (can be navigated from)."""
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return False
        return idx in self._link_graph

    def article_count(self) -> int:
        """Total number of articles with embeddings."""
        self._ensure_loaded()
        return len(self._titles)

    def traversable_count(self) -> int:
        """Number of articles with outgoing links."""
        self._ensure_loaded()
        return len(self._link_graph)

    # =========================================================================
    # Embedding Accessors
    # =========================================================================

    def get_embedding(self, title: str) -> np.ndarray | None:
        """Get embedding for article by title, or None if not found."""
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return None
        return self._embeddings[idx].astype(np.float32)

    def get_embedding_by_idx(self, idx: int) -> np.ndarray:
        """Get embedding by index (faster, no lookup)."""
        self._ensure_loaded()
        return self._embeddings[idx].astype(np.float32)

    def get_normalized_embedding(self, title: str) -> np.ndarray | None:
        """Get L2-normalized embedding for article (for cosine similarity)."""
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return None
        return self._normalize_embedding(self._embeddings[idx])

    def _normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Normalize a single embedding to unit length (for cosine similarity)."""
        emb_f32 = emb.astype(np.float32)
        norm = np.linalg.norm(emb_f32)
        if norm > 0:
            emb_f32 /= norm
        return emb_f32

    def get_embeddings_batch(self, titles: list[str]) -> np.ndarray:
        """Get embeddings for multiple titles as a batch."""
        self._ensure_loaded()
        indices = [self._title_to_idx.get(t) for t in titles]
        valid_indices = [i for i in indices if i is not None]
        if not valid_indices:
            return np.array([])
        return self._embeddings[valid_indices].astype(np.float32)

    # =========================================================================
    # Similarity Functions (FAISS-accelerated)
    # =========================================================================

    def similarity(self, title_a: str, title_b: str) -> float | None:
        """
        Compute cosine similarity between two articles.

        Returns None if either article is not found.
        """
        self._ensure_loaded()
        idx_a = self._title_to_idx.get(title_a)
        idx_b = self._title_to_idx.get(title_b)
        if idx_a is None or idx_b is None:
            return None

        # Normalize embeddings on-demand for cosine similarity
        emb_a = self._normalize_embedding(self._embeddings[idx_a])
        emb_b = self._normalize_embedding(self._embeddings[idx_b])
        return float(np.dot(emb_a, emb_b))

    def rank_by_similarity(
        self, candidates: list[str], target: str
    ) -> list[tuple[str, float]]:
        """
        Rank candidate articles by similarity to target.

        Returns list of (title, similarity) tuples, sorted by similarity descending.
        """
        self._ensure_loaded()
        target_idx = self._title_to_idx.get(target)
        if target_idx is None:
            return []

        target_emb = self._normalize_embedding(self._embeddings[target_idx]).reshape(1, -1)

        # Get candidate indices
        candidate_indices = []
        valid_candidates = []
        for c in candidates:
            idx = self._title_to_idx.get(c)
            if idx is not None:
                candidate_indices.append(idx)
                valid_candidates.append(c)

        if not candidate_indices:
            return []

        # Compute similarities - normalize candidates on-demand
        candidate_embs = np.array([
            self._normalize_embedding(self._embeddings[i])
            for i in candidate_indices
        ])
        similarities = np.dot(candidate_embs, target_emb.T).flatten()

        # Sort by similarity descending
        ranked = sorted(
            zip(valid_candidates, similarities, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(title, float(sim)) for title, sim in ranked]

    def nearest_neighbors(self, title: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Find k nearest neighbors to an article using FAISS.

        Returns list of (title, similarity) tuples.
        """
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return []

        query = self._normalize_embedding(self._embeddings[idx]).reshape(1, -1)
        similarities, indices = self._faiss_index.search(query, k + 1)

        # Skip the first result (the article itself)
        results = []
        for sim, neighbor_idx in zip(similarities[0], indices[0], strict=True):
            if neighbor_idx != idx and neighbor_idx >= 0:
                results.append((self._titles[neighbor_idx], float(sim)))
            if len(results) >= k:
                break

        return results

    # =========================================================================
    # Graph Accessors
    # =========================================================================

    def get_links(self, title: str) -> list[str]:
        """Get outgoing links from an article as titles."""
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None or idx not in self._link_graph:
            return []

        link_indices = self._link_graph[idx]
        return [self._titles[i] for i in link_indices if 0 <= i < len(self._titles)]

    def get_links_by_idx(self, idx: int) -> list[int]:
        """Get outgoing link indices for an article."""
        self._ensure_loaded()
        return self._link_graph.get(idx, [])

    @lru_cache(maxsize=10000)  # noqa: B019 - singleton pattern mitigates leak
    def get_popularity(self, title: str) -> int:
        """
        Get popularity score (number of inbound links) for an article.

        Computed on-demand and cached. First call for a new title requires
        iterating through the link graph, which may be slow.
        """
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return 0

        # Count how many articles link to this one
        count = 0
        for links in self._link_graph.values():
            if idx in links:
                count += 1

        return count

    def get_inbound_links(self, title: str) -> list[str]:
        """
        Get all articles that link to this article.

        Warning: This is slow (O(n) where n = number of articles).
        """
        self._ensure_loaded()
        idx = self._title_to_idx.get(title)
        if idx is None:
            return []

        inbound = []
        for source_idx, links in self._link_graph.items():
            if idx in links:
                inbound.append(self._titles[source_idx])

        return inbound

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def validate(self) -> dict[str, bool]:
        """Run validation checks on loaded data."""
        self._ensure_loaded()
        return {
            "titles_loaded": len(self._titles) > 0,
            "title_to_idx_loaded": len(self._title_to_idx) > 0,
            "link_graph_loaded": len(self._link_graph) > 0,
            "embeddings_loaded": self._embeddings is not None,
            "faiss_index_built": self._faiss_index is not None,
            "faiss_index_count_correct": self._faiss_index.ntotal == len(self._embeddings),
            "counts_match": len(self._titles) == len(self._title_to_idx),
            "embedding_shape_correct": self._embeddings.shape[1] == EMBEDDING_DIM,
        }

    def stats(self) -> dict:
        """Get statistics about the loaded data."""
        self._ensure_loaded()
        return {
            "total_articles": len(self._titles),
            "traversable_articles": len(self._link_graph),
            "embedding_dim": self._embeddings.shape[1],
            "embedding_dtype": str(self._embeddings.dtype),
            "faiss_vectors": self._faiss_index.ntotal,
        }


# Module-level singleton instance
wiki_data = WikiData()
