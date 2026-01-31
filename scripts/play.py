#!/usr/bin/env python3
"""
Wikipedia Speedrun CLI - Play the Wikipedia game with any agent.

Usage:
    python scripts/play.py --start "Python (programming language)" --target "Philosophy"
    python scripts/play.py --start "Cat" --target "Dog" --agent precomputed
    python scripts/play.py --start "Albert Einstein" --target "Pizza" --agent human --visualize
    python scripts/play.py --start "Random" --target "Random" --agent live
    python scripts/play.py --start "Potato" --target "Barack Obama" --agent llm --model google/gemini-2.0-flash-exp:free

Agents:
    human       - You play! Pick links manually (requires --visualize)
    random      - Random baseline
    precomputed - Greedy using pre-computed title embeddings
    live        - Greedy using on-the-fly sentence-transformers
    hybrid      - Pre-computed filter + live re-rank
    oracle      - BFS optimal path (uses pre-computed graph)
    live-oracle - BFS optimal path (live Wikipedia scraping, slow but complete)
    llm         - LLM via OpenRouter (use --model to specify)

LLM Models:
    Use any OpenRouter model ID directly (e.g., "anthropic/claude-haiku-4.5").

    Full model list with pricing: data/openrouter_models.json
    Analyze models: python scripts/analyze_models.py

    Examples:
        google/gemini-2.0-flash-exp:free  - FREE Google model (default)
        meta-llama/llama-3.3-70b-instruct:free - FREE Llama model
        openai/gpt-4o-mini                - Budget ($0.15/1M input)
        anthropic/claude-haiku-4.5        - Fast ($0.80/1M input)
        deepseek/deepseek-chat            - Value ($0.14/1M input)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path

# Suppress TensorFlow and protobuf warnings before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN messages
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import get_agent  # noqa: E402
from src.game import GameEngine  # noqa: E402


def get_random_article() -> str:
    """Get a random traversable article from the dataset."""
    from src.data.loader import wiki_data

    # Get a random article that has outgoing links
    while True:
        idx = random.randint(0, wiki_data.article_count() - 1)
        title = wiki_data.get_title(idx)
        if wiki_data.is_traversable(title):
            return title


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play the Wikipedia Speedrun game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Starting article title (or 'Random' for random)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target article title (or 'Random' for random)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="precomputed",
        choices=["human", "random", "precomputed", "live", "hybrid", "oracle", "live-oracle", "llm"],
        help="Agent to use (default: precomputed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp:free",
        help="LLM model for --agent llm (default: gemini-2.0-flash-exp:free)",
    )
    parser.add_argument(
        "--prefilter",
        type=int,
        default=None,
        help="Pre-filter links to top N using embeddings (default: no filter)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show browser visualization with Playwright",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps before game is lost (default: 50)",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Slow down visualization by this many ms per action",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle random article selection
    start = args.start
    target = args.target

    if start.lower() == "random":
        print("Selecting random start article...")
        start = get_random_article()
        print(f"  Start: {start}")

    if target.lower() == "random":
        print("Selecting random target article...")
        target = get_random_article()
        # Make sure it's different from start
        while target == start:
            target = get_random_article()
        print(f"  Target: {target}")

    # Get agent
    try:
        agent_kwargs = {}
        if args.agent == "llm":
            agent_kwargs["model"] = args.model
            if args.prefilter:
                agent_kwargs["prefilter"] = args.prefilter
        agent = get_agent(args.agent, **agent_kwargs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Pre-warm the agent (load models, data) before starting the game timer
    print("Warming up agent...")
    if hasattr(agent, "_ensure_loaded"):
        agent._ensure_loaded()
    print("Ready!")

    print("\n" + "=" * 60)
    print("Wikipedia Speedrun")
    print("=" * 60)
    print(f"  Start:  {start}")
    print(f"  Target: {target}")
    print(f"  Agent:  {agent.name} - {agent.description}")
    print(f"  Visualize: {args.visualize or agent.requires_visualization}")
    print("=" * 60 + "\n")

    # Create engine and run game
    visualize = args.visualize or agent.requires_visualization

    with GameEngine(visualize=visualize, slow_mo=args.slow_mo) as engine:
        try:
            result = engine.run(
                agent=agent,
                start=start,
                target=target,
                max_steps=args.max_steps,
            )
        except KeyboardInterrupt:
            print("\n\nGame interrupted by user")
            return 130  # Standard exit code for Ctrl+C

    # Print results
    print("\n" + "=" * 60)
    if result.won:
        print(f"Victory! Reached '{result.target_title}' in {result.total_clicks} clicks")
    else:
        print(f"Game Over. Did not reach '{result.target_title}'")
    print("=" * 60)

    print("\nPath taken:")
    for i, title in enumerate(result.path):
        marker = " (START)" if i == 0 else " (TARGET)" if title == result.target_title else ""
        print(f"  {i}. {title}{marker}")

    print(f"\nTotal time: {result.total_time_ms / 1000:.2f} seconds")
    print(f"Average decision time: {sum(s.decision_time_ms for s in result.steps) / len(result.steps):.0f}ms" if result.steps else "")

    # Print LLM stats if applicable
    if hasattr(agent, "get_stats"):
        stats = agent.get_stats()
        if stats.get("total_requests", 0) > 0:
            print(f"\nLLM Stats:")
            print(f"  Model: {stats['model']}")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  API time: {stats['total_time_seconds']}s")

    return 0 if result.won else 1


if __name__ == "__main__":
    sys.exit(main())
