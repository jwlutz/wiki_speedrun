"""
Game engine module.

Provides game state management and execution:
- GameState: Tracks current game state
- GameStep: Records a single step
- GameResult: Complete game record
- GameEngine: Runs games with agents
"""

from src.game.engine import GameEngine
from src.game.state import GameResult, GameState, GameStep

__all__ = [
    "GameEngine",
    "GameState",
    "GameStep",
    "GameResult",
]
