"""
Agents module.

Provides various agents for playing the Wikipedia game:
- HumanAgent: Human player via stdin
- RandomAgent: Random baseline
- PrecomputedEmbeddingAgent: Uses pre-computed title embeddings
- LiveEmbeddingAgent: Computes embeddings on-the-fly
- HybridEmbeddingAgent: Pre-computed filter + live re-rank
- OracleAgent: BFS optimal path (uses pre-computed graph)
- LiveOracleAgent: BFS optimal path (live Wikipedia scraping)
- LLMAgent: LLM-based agent using OpenRouter
"""

from src.agents.base import Agent, AgentContext
from src.agents.embedding import (
    HybridEmbeddingAgent,
    LiveEmbeddingAgent,
    PrecomputedEmbeddingAgent,
)
from src.agents.human import HumanAgent
from src.agents.live_oracle import LiveOracleAgent
from src.agents.llm import LLMAgent, get_available_models
from src.agents.oracle import OracleAgent
from src.agents.random_agent import RandomAgent

__all__ = [
    "Agent",
    "AgentContext",
    "HumanAgent",
    "RandomAgent",
    "PrecomputedEmbeddingAgent",
    "LiveEmbeddingAgent",
    "HybridEmbeddingAgent",
    "OracleAgent",
    "LiveOracleAgent",
    "LLMAgent",
    "get_available_models",
]


def get_agent(name: str, **kwargs) -> Agent:
    """
    Get an agent by name.

    Args:
        name: Agent identifier (human, random, precomputed, live, hybrid, oracle, live-oracle, llm)
        **kwargs: Additional arguments passed to agent constructor (e.g., model, prefilter)

    Returns:
        Instantiated agent

    Raises:
        ValueError: If agent name is unknown
    """
    agents = {
        "human": HumanAgent,
        "random": RandomAgent,
        "precomputed": PrecomputedEmbeddingAgent,
        "live": LiveEmbeddingAgent,
        "hybrid": HybridEmbeddingAgent,
        "oracle": OracleAgent,
        "live-oracle": LiveOracleAgent,
        "llm": LLMAgent,
    }

    if name not in agents:
        available = ", ".join(agents.keys())
        raise ValueError(f"Unknown agent '{name}'. Available: {available}")

    # These agents accept kwargs
    if name == "llm":
        return LLMAgent(**kwargs)
    if name == "live-oracle":
        return LiveOracleAgent(**kwargs)
    if name == "live":
        return LiveEmbeddingAgent(**kwargs)

    return agents[name]()
