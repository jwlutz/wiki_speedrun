"""
Oracle agent that uses BFS to find the optimal path.

Has "cheating" knowledge of the full link graph, so it always
finds the shortest path (if one exists in the pre-computed graph).
"""

from __future__ import annotations

import logging
from collections import deque

from src.agents.base import Agent, AgentContext

logger = logging.getLogger(__name__)


class OracleAgent(Agent):
    """
    Oracle agent that knows the optimal path via BFS.

    Uses the pre-computed link graph to find the shortest path.
    Serves as an upper bound on performance (optimal play).

    Note: The oracle uses the pre-computed graph which may differ
    from live Wikipedia. It recalculates if it gets off-path.
    """

    def __init__(self, max_depth: int = 15) -> None:
        """
        Initialize oracle agent.

        Args:
            max_depth: Maximum BFS depth for pathfinding
        """
        self._max_depth = max_depth
        self._wiki_data = None
        self._optimal_path: list[str] | None = None
        self._current_step: int = 0

    def _ensure_loaded(self) -> None:
        """Lazy load wiki_data."""
        if self._wiki_data is None:
            from src.data.loader import wiki_data

            self._wiki_data = wiki_data

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def description(self) -> str:
        return "BFS optimal path (uses pre-computed link graph)"

    def on_game_start(self, start_title: str, target_title: str) -> None:
        """Pre-compute the optimal path."""
        self._ensure_loaded()
        self._optimal_path = self._bfs(start_title, target_title)
        self._current_step = 0

        if self._optimal_path:
            logger.info(
                f"Oracle found path ({len(self._optimal_path) - 1} clicks): "
                f"{' -> '.join(self._optimal_path)}"
            )
        else:
            logger.warning(
                f"Oracle: No path found from '{start_title}' to '{target_title}'"
            )

    def _bfs(self, start: str, target: str) -> list[str] | None:
        """
        Find shortest path using BFS on the pre-computed graph.

        Returns:
            List of titles from start to target, or None if no path exists
        """
        start_idx = self._wiki_data.get_index(start)
        target_idx = self._wiki_data.get_index(target)

        if start_idx is None:
            logger.warning(f"Start '{start}' not in graph")
            return None
        if target_idx is None:
            logger.warning(f"Target '{target}' not in graph")
            return None

        if start_idx == target_idx:
            return [start]

        # BFS with parent tracking
        queue = deque([(start_idx, 0)])
        visited = {start_idx: None}  # Maps idx to parent idx

        while queue:
            current_idx, depth = queue.popleft()

            if depth >= self._max_depth:
                continue

            # Get outgoing links
            for neighbor_idx in self._wiki_data.get_links_by_idx(current_idx):
                if neighbor_idx in visited:
                    continue

                visited[neighbor_idx] = current_idx

                if neighbor_idx == target_idx:
                    # Found! Reconstruct path
                    path = []
                    idx = neighbor_idx
                    while idx is not None:
                        path.append(self._wiki_data.get_title(idx))
                        idx = visited[idx]
                    return list(reversed(path))

                queue.append((neighbor_idx, depth + 1))

        return None

    def choose_link(self, context: AgentContext) -> str:
        """Follow the pre-computed optimal path, or adapt if off-path."""
        # Always click target if directly available
        if context.target_title in context.available_links:
            return context.target_title

        # If we have a pre-computed path and are on it, follow it
        if self._optimal_path and self._current_step < len(self._optimal_path) - 1:
            expected_current = self._optimal_path[self._current_step]
            next_step = self._optimal_path[self._current_step + 1]

            if context.current_title == expected_current:
                if next_step in context.available_links:
                    self._current_step += 1
                    return next_step
                else:
                    # Link exists in graph but not on live page
                    logger.warning(
                        f"Oracle: Expected link '{next_step}' not on live page"
                    )

        # Off-path or no path: recompute from current position
        self._ensure_loaded()
        logger.info(f"Oracle: Recomputing path from '{context.current_title}'")
        new_path = self._bfs(context.current_title, context.target_title)

        if new_path and len(new_path) > 1:
            # Filter to only links available on this page
            next_in_path = new_path[1]
            if next_in_path in context.available_links:
                self._optimal_path = new_path
                self._current_step = 1
                return next_in_path

        # Last resort: use embedding similarity
        logger.warning("Oracle: Falling back to embedding similarity")
        ranked = self._wiki_data.rank_by_similarity(
            candidates=context.available_links,
            target=context.target_title,
        )

        if ranked:
            visited = set(context.path_so_far)
            for title, _ in ranked:
                if title not in visited:
                    return title
            return ranked[0][0]

        return context.available_links[0]

    def on_game_end(self, won: bool, path: list[str]) -> None:
        """Clean up."""
        self._optimal_path = None
        self._current_step = 0
