"""
Agent base class and protocol for Wikipedia speedrun players.

All agents must implement choose_link() to decide which link to click next.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.state import GameState


@dataclass
class AgentContext:
    """
    Context passed to agents for decision making.

    Attributes:
        current_title: Title of the current Wikipedia article
        target_title: Title of the target article we're trying to reach
        available_links: List of clickable link titles on current page
        path_so_far: List of titles visited (including current)
        step_count: Number of clicks made so far
    """

    current_title: str
    target_title: str
    available_links: list[str]
    path_so_far: list[str]
    step_count: int


class Agent(ABC):
    """
    Abstract base class for Wikipedia speedrun agents.

    Agents receive the current game state and must choose which link to click.
    Different agents use different strategies (embeddings, LLMs, BFS, human input).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the agent (e.g., 'greedy', 'llm', 'human')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the agent's strategy."""
        ...

    @property
    def requires_visualization(self) -> bool:
        """Whether this agent requires the browser to be visible (e.g., HumanAgent)."""
        return False

    @abstractmethod
    def choose_link(self, context: AgentContext) -> str:
        """
        Choose which link to click next.

        Args:
            context: Current game context with available links

        Returns:
            The title of the link to click. Must be in context.available_links.

        Raises:
            ValueError: If returned link is not in available_links
        """
        ...

    def on_game_start(self, start_title: str, target_title: str) -> None:
        """Called when a new game begins. Override for setup."""
        pass

    def on_game_end(self, won: bool, path: list[str]) -> None:
        """Called when game ends. Override for cleanup/learning."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
