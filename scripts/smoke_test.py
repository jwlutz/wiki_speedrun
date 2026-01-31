#!/usr/bin/env python3
"""
Smoke test: 3 cases (easy/medium/hard) to verify all models work.

Shows optimal path vs what each model gets + timing.
"""

from __future__ import annotations

import os
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.WARNING)

from src.agents import get_agent
from src.game import GameEngine

# =============================================================================
# 3 TEST CASES
# =============================================================================

TEST_CASES = [
    ("Easy", "Espresso", "Leonardo da Vinci", "3-5 clicks expected"),
    ("Medium", "IKEA", "Cleopatra", "8-12 clicks expected"),
    ("Hard", "Fortnite", "Bubonic plague", "10-20 clicks expected"),
    ("Tricky", "Jeffrey Epstein", "Severna Park, Maryland", "unknown"),
    ("Tricky", "Severna Park, Maryland", "Jeffrey Epstein", "unknown"),
]

# =============================================================================
# MODELS TO TEST
# =============================================================================

LLM_MODELS = [
    # === SMALL/FAST MODELS (2026) ===
    "openai/gpt-4o-mini",           # Budget baseline
    "openai/gpt-5-mini",            # GPT-5 small
    "deepseek/deepseek-chat",       # DeepSeek budget
    "google/gemini-2.0-flash-lite-001",  # Gemini 2.0 lite
    "google/gemini-3-flash-preview",     # Gemini 3 flash
    "anthropic/claude-haiku-4.5",   # Claude 4.5 haiku

    # === BIG/SMART MODELS (2026) ===
    "openai/gpt-5.2-chat",          # GPT-5.2 (Dec 2025)
    "deepseek/deepseek-v3.2",       # DeepSeek V3.2
    "google/gemini-3-pro-preview",  # Gemini 3 Pro (Nov 2025)
    "anthropic/claude-sonnet-4.5",  # Claude Sonnet 4.5
]


def run_game(agent_name: str, start: str, target: str, max_steps: int = 30, **kwargs) -> dict:
    """Run a single game and return result dict."""
    try:
        agent = get_agent(agent_name, **kwargs)

        with GameEngine(visualize=False) as engine:
            start_time = time.time()
            result = engine.run(agent=agent, start=start, target=target, max_steps=max_steps)
            elapsed = time.time() - start_time

        # Get LLM stats if available
        stats = agent.get_stats() if hasattr(agent, "get_stats") else {}
        fallbacks = stats.get("fallback_count", 0)
        llm_requests = stats.get("total_requests", 0)

        return {
            "agent": agent.name,
            "won": result.won,
            "clicks": result.total_clicks,
            "time": round(elapsed, 2),
            "path": result.path,
            "llm_requests": llm_requests,
            "fallbacks": fallbacks,
            "used_fallback": fallbacks > 0,
        }
    except Exception as e:
        return {
            "agent": agent_name,
            "won": False,
            "clicks": -1,
            "time": 0,
            "path": [],
            "error": str(e)[:80],
        }


def main():
    print("=" * 90)
    print("SMOKE TEST: 5 Cases × 10 LLM Models (Small vs Big)")
    print("=" * 90)
    print("Comparing small/fast models vs big/smart models (Jan 2026)")
    print()

    all_results = {}

    for difficulty, start, target, expected in TEST_CASES:
        print("=" * 90)
        print(f"{difficulty.upper()}: {start} → {target}")
        print(f"Expected: {expected}")
        print("=" * 90)

        case_key = f"{start} → {target}"
        all_results[case_key] = {"difficulty": difficulty, "results": []}

        print(f"\n{'Agent':<40} {'Result':>8} {'Clicks':>7} {'LLM?':>6} {'Time':>8}")
        print("-" * 75)

        # Run each LLM model
        for model in LLM_MODELS:
            model_short = model.split("/")[-1][:38]

            result = run_game("llm", start, target, model=model)
            all_results[case_key]["results"].append(result)

            if "error" in result:
                print(f"{model_short:<40} {'ERROR':>8} {'-':>7} {'-':>6} {'-':>8}")
                print(f"  Error: {result['error'][:60]}")
            else:
                status = "WIN" if result["won"] else "LOST"
                clicks_str = str(result["clicks"]) if result["won"] else ">30"
                # Show if result is from LLM or fallback
                llm_indicator = "Yes" if not result.get("used_fallback", False) else "FALL"
                print(f"{model_short:<40} {status:>8} {clicks_str:>7} {llm_indicator:>6} {result['time']:>7.1f}s")
                if result.get("used_fallback"):
                    print(f"  (Used embedding fallback - LLM rate limited)")

            # Small delay to avoid rate limits
            time.sleep(1.0)

        print()

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 90)
    print("SUMMARY BY MODEL")
    print("=" * 90)

    # Aggregate results by agent
    agent_stats = {}
    for case_key, case_data in all_results.items():
        for r in case_data["results"]:
            agent = r["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {"wins": 0, "total": 0, "clicks": [], "times": []}
            agent_stats[agent]["total"] += 1
            if r["won"]:
                agent_stats[agent]["wins"] += 1
                agent_stats[agent]["clicks"].append(r["clicks"])
                agent_stats[agent]["times"].append(r["time"])

    print(f"\n{'Agent':<45} {'Wins':>8} {'AvgClicks':>10} {'AvgTime':>10}")
    print("-" * 75)

    # Sort by wins, then avg clicks
    sorted_agents = sorted(
        agent_stats.items(),
        key=lambda x: (-x[1]["wins"], sum(x[1]["clicks"])/len(x[1]["clicks"]) if x[1]["clicks"] else 999)
    )

    for agent, stats in sorted_agents:
        wins = f"{stats['wins']}/{stats['total']}"
        avg_clicks = f"{sum(stats['clicks'])/len(stats['clicks']):.1f}" if stats['clicks'] else "-"
        avg_time = f"{sum(stats['times'])/len(stats['times']):.1f}s" if stats['times'] else "-"
        print(f"{agent:<45} {wins:>8} {avg_clicks:>10} {avg_time:>10}")

    # ==========================================================================
    # TOP 5 RECOMMENDATIONS
    # ==========================================================================

    print("\n" + "=" * 90)
    print("TOP 5 LLM RECOMMENDATIONS")
    print("=" * 90)

    llm_agents = [(a, s) for a, s in sorted_agents if "llm-" in a]

    for i, (agent, stats) in enumerate(llm_agents[:5], 1):
        model_name = agent.replace("llm-", "")
        wins = stats["wins"]
        avg_clicks = sum(stats['clicks'])/len(stats['clicks']) if stats['clicks'] else 0
        avg_time = sum(stats['times'])/len(stats['times']) if stats['times'] else 0
        print(f"{i}. {model_name}")
        print(f"   Wins: {wins}/{len(TEST_CASES)}, Avg clicks: {avg_clicks:.1f}, Avg time: {avg_time:.1f}s")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    import json
    from datetime import datetime

    output_file = project_root / "data" / "smoke_test_results.json"
    output_file.parent.mkdir(exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": [{"difficulty": d, "start": s, "target": t} for d, s, t, _ in TEST_CASES],
        "models": LLM_MODELS,
        "results": all_results,
        "summary": {
            agent: {
                "wins": stats["wins"],
                "total": stats["total"],
                "avg_clicks": sum(stats['clicks'])/len(stats['clicks']) if stats['clicks'] else None,
                "avg_time": sum(stats['times'])/len(stats['times']) if stats['times'] else None,
            }
            for agent, stats in agent_stats.items()
        }
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
