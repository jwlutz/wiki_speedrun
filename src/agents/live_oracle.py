"""
Live Oracle agent that uses guided BFS on live Wikipedia to find paths.

Unlike the regular OracleAgent which uses pre-computed link graph,
this agent scrapes Wikipedia in real-time. Uses embedding similarity
to prune the search space (only expands top-N links per page).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from src.agents.base import Agent, AgentContext

logger = logging.getLogger(__name__)


class LiveOracleAgent(Agent):
    """
    Oracle agent that finds paths via embedding-guided BFS on live Wikipedia.

    At each BFS level, only expands the top-N links by embedding similarity
    to the target. This dramatically reduces search space while still finding
    good (often optimal) paths.

    Trade-off: May miss the true optimal path if it goes through low-similarity
    intermediate articles, but finds paths much faster than pure BFS.
    """

    def __init__(
        self,
        max_depth: int = 6,
        beam_width: int = 15,
        verbose: bool = True,
    ) -> None:
        """
        Initialize live oracle agent.

        Args:
            max_depth: Maximum BFS depth (default 6)
            beam_width: Number of top links to expand per page (default 15)
            verbose: Print progress during BFS search
        """
        self._max_depth = max_depth
        self._beam_width = beam_width
        self._verbose = verbose
        self._scraper = None
        self._model = None
        self._optimal_path: list[str] | None = None
        self._current_step: int = 0
        self._pages_scraped: int = 0

    def _ensure_loaded(self) -> None:
        """Lazy load the scraper and embedding model."""
        if self._scraper is None:
            from src.wikipedia.scraper import WikiScraper
            self._scraper = WikiScraper()

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model for guided BFS...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def name(self) -> str:
        return "live-oracle"

    @property
    def description(self) -> str:
        return f"Guided BFS (live scraping, beam={self._beam_width}, depth={self._max_depth})"

    def on_game_start(self, start_title: str, target_title: str) -> None:
        """Pre-compute the path via embedding-guided BFS."""
        self._ensure_loaded()
        self._pages_scraped = 0

        if self._verbose:
            print(f"Live Oracle: Searching for path from '{start_title}' to '{target_title}'...")
            print(f"  Max depth: {self._max_depth}, beam width: {self._beam_width}")

        self._optimal_path = self._guided_bfs(start_title, target_title)
        self._current_step = 0

        if self._optimal_path:
            clicks = len(self._optimal_path) - 1
            if self._verbose:
                print(f"Live Oracle: Found path ({clicks} clicks, {self._pages_scraped} pages scraped)")
                print(f"  Path: {' -> '.join(self._optimal_path)}")
            logger.info(
                f"Live Oracle found path ({clicks} clicks): "
                f"{' -> '.join(self._optimal_path)}"
            )
        else:
            if self._verbose:
                print(f"Live Oracle: No path found within {self._max_depth} clicks ({self._pages_scraped} pages scraped)")
            logger.warning(
                f"Live Oracle: No path found from '{start_title}' to '{target_title}' "
                f"within {self._max_depth} clicks"
            )

    def _rank_links_by_similarity(self, links: list[str], target: str) -> list[str]:
        """Rank links by embedding similarity to target, return top beam_width."""
        if not links:
            return []

        # Encode target and all links
        all_texts = [target] + links
        embeddings = self._model.encode(all_texts, convert_to_numpy=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Target is first embedding
        target_emb = embeddings[0]
        link_embs = embeddings[1:]

        # Compute similarities
        similarities = np.dot(link_embs, target_emb)

        # Sort by similarity and take top beam_width
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[: self._beam_width]

        return [links[i] for i in top_indices]

    def _guided_bfs(self, start: str, target: str) -> list[str] | None:
        """
        Find path using embedding-guided BFS.

        At each level, only expands top-N links by embedding similarity.

        Returns:
            List of titles from start to target, or None if no path found
        """
        if start == target:
            return [start]

        # BFS with parent tracking
        # visited maps title -> parent title
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        visited: dict[str, str | None] = {start: None}

        while queue:
            current_title, depth = queue.popleft()

            if depth >= self._max_depth:
                continue

            # Scrape the current page for links
            try:
                all_links = self._scraper.get_links(current_title)
                self._pages_scraped += 1

                if self._verbose:
                    print(f"  [{self._pages_scraped}] Scraped '{current_title}' (depth {depth}, {len(all_links)} links)")

            except Exception as e:
                logger.warning(f"Failed to scrape '{current_title}': {e}")
                continue

            # Check if target is directly reachable
            if target in all_links:
                visited[target] = current_title
                # Found! Reconstruct path
                path = []
                title: str | None = target
                while title is not None:
                    path.append(title)
                    title = visited[title]
                return list(reversed(path))

            # Filter out already-visited links
            unvisited_links = [link for link in all_links if link not in visited]

            if not unvisited_links:
                continue

            # Rank by embedding similarity and take top beam_width
            top_links = self._rank_links_by_similarity(unvisited_links, target)

            # Add to queue
            for link_title in top_links:
                if link_title not in visited:
                    visited[link_title] = current_title
                    queue.append((link_title, depth + 1))

        return None

    def choose_link(self, context: AgentContext) -> str:
        """Follow the pre-computed path."""
        # Always click target if directly available
        if context.target_title in context.available_links:
            return context.target_title

        # If we have a path and are on it, follow it
        if self._optimal_path and self._current_step < len(self._optimal_path) - 1:
            expected_current = self._optimal_path[self._current_step]
            next_step = self._optimal_path[self._current_step + 1]

            if context.current_title == expected_current:
                if next_step in context.available_links:
                    self._current_step += 1
                    return next_step
                else:
                    logger.warning(
                        f"Live Oracle: Expected link '{next_step}' not available on page"
                    )

        # No path or off-path: fall back to first available link
        logger.warning("Live Oracle: Off path or no path, using fallback")
        return context.available_links[0]

    def on_game_end(self, won: bool, path: list[str]) -> None:
        """Clean up and report stats."""
        if self._verbose:
            status = "WON" if won else "LOST"
            print(f"Live Oracle: Game {status}, scraped {self._pages_scraped} pages total")
        self._optimal_path = None
        self._current_step = 0

    def get_optimal_path(self) -> list[str] | None:
        """Return the computed path (for testing)."""
        return self._optimal_path

    def get_stats(self) -> dict:
        """Return search statistics."""
        return {
            "pages_scraped": self._pages_scraped,
            "optimal_path": self._optimal_path,
            "path_length": len(self._optimal_path) - 1 if self._optimal_path else None,
        }
