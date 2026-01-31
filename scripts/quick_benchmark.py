#!/usr/bin/env python3
"""
Quick benchmark to compare agents on a few examples.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.WARNING)  # Quiet mode

from src.agents import get_agent
from src.game import GameEngine

# Test cases: (start, target)
TEST_CASES = [
    ("Potato", "Barack Obama"),
    ("Cat", "Philosophy"),
    ("Pizza", "Mathematics"),
    ("Guitar", "World War II"),
    ("Coffee", "Albert Einstein"),
    ("Tennis", "Computer"),
    ("Moon", "Shakespeare"),
    ("Banana", "Japan"),
    ("Chess", "Music"),
    ("Water", "History"),
]

# Agents to test (precomputed was removed in favor of live embeddings)
AGENTS = ["random", "live"]


def run_benchmark():
    print("=" * 70)
    print("Wikipedia Speedrun - Agent Comparison")
    print("=" * 70)
    print(f"\nTesting {len(AGENTS)} agents on {len(TEST_CASES)} problems...")
    print("(First run loads data - subsequent games are fast)\n")

    results = {agent: [] for agent in AGENTS}

    # Pre-load data with live agent
    print("Loading Wikipedia data (this takes ~60 seconds)...")
    start_load = time.time()
    _ = get_agent("live")
    from src.data.loader import wiki_data
    _ = wiki_data.article_count()  # Trigger load
    print(f"Data loaded in {time.time() - start_load:.1f}s")

    # Pre-load sentence-transformers model
    if "live" in AGENTS:
        print("Loading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.\n")

    for i, (start, target) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {start} -> {target}")
        print("-" * 50)

        for agent_name in AGENTS:
            agent = get_agent(agent_name)

            with GameEngine(visualize=False) as engine:
                try:
                    start_time = time.time()
                    result = engine.run(
                        agent=agent,
                        start=start,
                        target=target,
                        max_steps=25,
                    )
                    elapsed = time.time() - start_time

                    status = "WIN" if result.won else "LOST"
                    clicks = result.total_clicks
                    results[agent_name].append((result.won, clicks))

                    print(f"  {agent_name:15} : {status:4} in {clicks:2} clicks ({elapsed:.1f}s)")

                except Exception as e:
                    print(f"  {agent_name:15} : ERROR - {e}")
                    results[agent_name].append((False, -1))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for agent_name in AGENTS:
        wins = sum(1 for won, _ in results[agent_name] if won)
        total = len(results[agent_name])
        avg_clicks = sum(c for won, c in results[agent_name] if won and c > 0)
        win_count = sum(1 for won, c in results[agent_name] if won and c > 0)
        avg_clicks = avg_clicks / win_count if win_count > 0 else 0

        print(f"  {agent_name:15} : {wins}/{total} wins, avg {avg_clicks:.1f} clicks (when won)")


if __name__ == "__main__":
    run_benchmark()
