"""
Game state dataclasses for tracking Wikipedia speedrun progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class GameStep:
    """
    Records a single step (click) in the game.

    Attributes:
        from_title: Article we clicked from
        to_title: Article we clicked to
        available_links: Links that were available on the page
        decision_time_ms: Time agent took to decide (milliseconds)
        step_number: 1-indexed step number
    """

    from_title: str
    to_title: str
    available_links: list[str]
    decision_time_ms: float
    step_number: int


@dataclass
class GameResult:
    """
    Complete record of a finished game.

    Attributes:
        start_title: Starting article
        target_title: Target article
        path: Full path taken (including start and end)
        steps: Detailed record of each step
        won: Whether target was reached
        total_clicks: Number of clicks (len(path) - 1)
        total_time_ms: Total game time in milliseconds
        agent_name: Name of the agent that played
        timestamp: When the game was played
    """

    start_title: str
    target_title: str
    path: list[str]
    steps: list[GameStep]
    won: bool
    total_clicks: int
    total_time_ms: float
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def efficiency(self) -> float | None:
        """
        Ratio of optimal path length to actual path length.

        Returns None if optimal path unknown or game lost.
        Higher is better (1.0 = optimal).
        """
        # Will be computed by benchmark when optimal path is known
        return None


@dataclass
class GameState:
    """
    Mutable state during an active game.

    Attributes:
        start_title: Starting article
        target_title: Target article
        current_title: Current article
        path: Path taken so far (including current)
        steps: Steps recorded so far
        start_time_ms: Game start timestamp (epoch ms)
    """

    start_title: str
    target_title: str
    current_title: str
    path: list[str] = field(default_factory=list)
    steps: list[GameStep] = field(default_factory=list)
    start_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [self.start_title]

    @property
    def click_count(self) -> int:
        """Number of clicks made so far."""
        return len(self.path) - 1

    @property
    def is_won(self) -> bool:
        """Whether we've reached the target."""
        return self.current_title == self.target_title

    def record_step(
        self,
        to_title: str,
        available_links: list[str],
        decision_time_ms: float,
    ) -> None:
        """Record a step and update state."""
        step = GameStep(
            from_title=self.current_title,
            to_title=to_title,
            available_links=available_links,
            decision_time_ms=decision_time_ms,
            step_number=len(self.steps) + 1,
        )
        self.steps.append(step)
        self.path.append(to_title)
        self.current_title = to_title

    def to_result(self, agent_name: str, total_time_ms: float) -> GameResult:
        """Convert to immutable GameResult."""
        return GameResult(
            start_title=self.start_title,
            target_title=self.target_title,
            path=list(self.path),
            steps=list(self.steps),
            won=self.is_won,
            total_clicks=self.click_count,
            total_time_ms=total_time_ms,
            agent_name=agent_name,
        )
