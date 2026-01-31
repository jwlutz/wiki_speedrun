#!/usr/bin/env python3
"""
Test script: Jeffrey Epstein <-> Severna Park, Maryland (both directions)

Tests precomputed embedding, oracle (BFS), and claude-haiku-4.5 agents.
"""

from __future__ import annotations

import os
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress TensorFlow and protobuf warnings before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.WARNING)

from src.agents import get_agent
from src.game import GameEngine

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CASES = [
    ("Jeffrey Epstein", "Severna Park, Maryland"),
    ("Severna Park, Maryland", "Jeffrey Epstein"),
]

AGENTS = [
    ("precomputed", {}),  # Embedding similarity baseline
    ("oracle", {}),       # BFS optimal path
    ("llm", {"model": "anthropic/claude-haiku-4.5"}),
]


def run_game(agent_name: str, start: str, target: str, max_steps: int = 30, **kwargs) -> dict:
    """Run a single game and return result dict."""
    try:
        agent = get_agent(agent_name, **kwargs)

        # Warmup: load models/data before timing (follows project pattern)
        if hasattr(agent, "_ensure_loaded"):
            agent._ensure_loaded()

        with GameEngine(visualize=False) as engine:
            start_time = time.time()
            result = engine.run(agent=agent, start=start, target=target, max_steps=max_steps)
            elapsed = time.time() - start_time

        return {
            "agent": agent.name,
            "won": result.won,
            "clicks": result.total_clicks,
            "time": round(elapsed, 2),
            "path": result.path,
        }
    except Exception as e:
        import traceback
        return {
            "agent": agent_name,
            "won": False,
            "clicks": -1,
            "time": 0,
            "path": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    print("=" * 80)
    print("EPSTEIN TEST: Comparing agents on a tricky path")
    print("=" * 80)
    print()

    total_tests = len(TEST_CASES) * len(AGENTS)
    current_test = 0

    for case_idx, (start, target) in enumerate(TEST_CASES, 1):
        print("=" * 80)
        print(f"TEST {case_idx}/{len(TEST_CASES)}: {start} -> {target}")
        print("=" * 80)

        for agent_name, kwargs in AGENTS:
            current_test += 1
            agent_display = kwargs.get("model", agent_name).split("/")[-1]
            print(f"\n[{current_test}/{total_tests}] Running {agent_display}...", flush=True)

            result = run_game(agent_name, start, target, **kwargs)

            if "error" in result:
                print(f"  Result: ERROR - {result['error']}")
                if result.get("traceback"):
                    # Print last few lines of traceback
                    tb_lines = result["traceback"].strip().split("\n")
                    for line in tb_lines[-3:]:
                        print(f"  {line}")
            else:
                status = "WIN" if result["won"] else "LOST"
                clicks = result["clicks"] if result["won"] else ">30"
                print(f"  Model:  {result['agent']}")
                print(f"  Result: {status}")
                print(f"  Clicks: {clicks}")
                print(f"  Time:   {result['time']:.1f}s")
                print(f"  Path:   {' -> '.join(result['path'])}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())