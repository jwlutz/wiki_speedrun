"""
Random agent baseline - picks links uniformly at random.
"""

from __future__ import annotations

import random

from src.agents.base import Agent, AgentContext


class RandomAgent(Agent):
    """
    Baseline agent that picks a random link on each step.

    Used to establish a lower bound on performance.
    Can optionally exclude links already in the path to avoid loops.
    """

    def __init__(self, avoid_revisits: bool = True, seed: int | None = None) -> None:
        """
        Initialize the random agent.

        Args:
            avoid_revisits: If True, don't revisit pages already in path
            seed: Random seed for reproducibility
        """
        self._avoid_revisits = avoid_revisits
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        suffix = "-noloop" if self._avoid_revisits else ""
        return f"random{suffix}"

    @property
    def description(self) -> str:
        if self._avoid_revisits:
            return "Random selection, avoiding revisits"
        return "Uniformly random link selection"

    def choose_link(self, context: AgentContext) -> str:
        """Pick a random link from available options."""
        candidates = context.available_links

        # Optionally filter out visited pages
        if self._avoid_revisits:
            visited = set(context.path_so_far)
            unvisited = [link for link in candidates if link not in visited]
            if unvisited:
                candidates = unvisited

        # Always prefer target if directly available
        if context.target_title in candidates:
            return context.target_title

        return self._rng.choice(candidates)
