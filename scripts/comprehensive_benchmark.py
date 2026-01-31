#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all agents on curated test cases.

Compares:
- Live embedding agent (greedy similarity)
- LLM agents (multiple models via OpenRouter)

Features:
- Incremental save: Results saved after each game (survives crashes)
- Resume support: Automatically skips already-completed games
- Use --fresh to start over and clear previous results
- Use --rerun-errors to rerun games that had API/network errors
- Use --rerun-losses to rerun games that hit max steps (didn't win)

Usage:
    python scripts/comprehensive_benchmark.py
    python scripts/comprehensive_benchmark.py --llm-only
    python scripts/comprehensive_benchmark.py --free-only
    python scripts/comprehensive_benchmark.py --fresh  # Clear cache and start over
    python scripts/comprehensive_benchmark.py --rerun-errors  # Retry failed API calls
    python scripts/comprehensive_benchmark.py --rerun-losses  # Retry games that lost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.WARNING)

from src.agents import get_agent
from src.game import GameEngine

# =============================================================================
# LOAD PROBLEMS FROM JSON
# =============================================================================

def load_benchmark_problems() -> list[tuple[str, str]]:
    """Load benchmark problems from JSON file."""
    problems_path = project_root / "data" / "benchmark_problems.json"
    if problems_path.exists():
        with open(problems_path) as f:
            data = json.load(f)
        return [(p["start"], p["target"]) for p in data["problems"]]
    else:
        # Fallback to hardcoded if JSON doesn't exist
        print(f"Warning: {problems_path} not found, using fallback problems")
        return [
            ("Potato", "Barack Obama"),
            ("Chess", "Mathematics"),
            ("Moon", "Shakespeare"),
        ]


def load_model_tiers() -> dict[str, list[str]]:
    """Load model tiers from JSON file."""
    problems_path = project_root / "data" / "benchmark_problems.json"
    if problems_path.exists():
        with open(problems_path) as f:
            data = json.load(f)
        return data.get("model_tiers", {})
    return {}


def load_embedding_models() -> list[str]:
    """Load embedding models from JSON file."""
    problems_path = project_root / "data" / "benchmark_problems.json"
    if problems_path.exists():
        with open(problems_path) as f:
            data = json.load(f)
        return data.get("embedding_models", ["all-MiniLM-L6-v2"])
    return ["all-MiniLM-L6-v2"]


# Load from JSON
TEST_CASES = load_benchmark_problems()
MODEL_TIERS = load_model_tiers()
EMBEDDING_MODELS = load_embedding_models()

# =============================================================================
# LLM MODELS TO TEST (from JSON or fallback)
# =============================================================================

# Free models (test first to avoid costs)
FREE_MODELS = MODEL_TIERS.get("free", [
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
])

# Paid models (budget tier)
BUDGET_MODELS = MODEL_TIERS.get("budget", [
    "openai/gpt-5-nano",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
])

# Paid models (fast tier)
FAST_MODELS = MODEL_TIERS.get("fast", [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
])

# Paid models (premium tier)
PREMIUM_MODELS = MODEL_TIERS.get("premium", [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "google/gemini-3-pro-preview",
])

# =============================================================================
# RESULT TRACKING
# =============================================================================

@dataclass
class GameResult:
    agent: str
    start: str
    target: str
    won: bool
    clicks: int
    time_seconds: float
    path: list[str]
    error: str = ""
    error_type: str = ""  # "http_error", "timeout", "api_error", "game_error", "unknown"
    traceback: str = ""   # Full traceback for debugging
    tokens_used: int = 0  # LLM token usage (0 for embedding agents)


@dataclass
class AgentSummary:
    agent: str
    games_played: int
    wins: int
    win_rate: float
    avg_clicks_when_won: float
    avg_time_seconds: float
    total_time: float


def classify_error(e: Exception) -> tuple[str, str, str]:
    """Classify an exception into (error_type, error_message, traceback)."""
    import traceback as tb
    import requests

    error_msg = str(e)
    traceback_str = tb.format_exc()

    # Classify by exception type
    if isinstance(e, requests.exceptions.Timeout):
        return "timeout", error_msg, traceback_str
    elif isinstance(e, requests.exceptions.HTTPError):
        return "http_error", error_msg, traceback_str
    elif isinstance(e, requests.exceptions.RequestException):
        return "network_error", error_msg, traceback_str
    elif "API" in error_msg or "api" in error_msg or "OpenRouter" in error_msg:
        return "api_error", error_msg, traceback_str
    elif "rate limit" in error_msg.lower() or "429" in error_msg:
        return "rate_limit", error_msg, traceback_str
    elif "402" in error_msg or "credits" in error_msg.lower():
        return "insufficient_credits", error_msg, traceback_str
    elif "404" in error_msg or "not found" in error_msg.lower():
        return "not_found", error_msg, traceback_str
    elif isinstance(e, (ValueError, KeyError, IndexError)):
        return "game_error", error_msg, traceback_str
    else:
        return "unknown", error_msg, traceback_str


# =============================================================================
# INCREMENTAL SAVE / RESUME
# =============================================================================

def get_jsonl_path(output_path: Path) -> Path:
    """Get the JSONL checkpoint file path from the final output path."""
    return output_path.with_suffix(".jsonl")


def load_completed_games(jsonl_path: Path) -> tuple[set[tuple[str, str, str]], list[GameResult]]:
    """
    Load previously completed games from JSONL checkpoint.

    Returns:
        (completed_keys, results) where completed_keys is set of (agent, start, target) tuples
    """
    completed: set[tuple[str, str, str]] = set()
    results: list[GameResult] = []

    if not jsonl_path.exists():
        return completed, results

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result = GameResult(**data)
                results.append(result)
                completed.add((result.agent, result.start, result.target))
            except (json.JSONDecodeError, TypeError):
                continue  # Skip malformed lines

    return completed, results


def append_result(jsonl_path: Path, result: GameResult) -> None:
    """Append a single result to the JSONL checkpoint file."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def run_game(agent_name: str, start: str, target: str, max_steps: int = 25, **kwargs) -> GameResult:
    """Run a single game and return result with detailed error tracking."""
    try:
        agent = get_agent(agent_name, **kwargs)

        with GameEngine(visualize=False) as engine:
            start_time = time.time()
            result = engine.run(
                agent=agent,
                start=start,
                target=target,
                max_steps=max_steps,
            )
            elapsed = time.time() - start_time

        # Get token usage from LLM agents
        tokens_used = 0
        if hasattr(agent, "get_stats"):
            stats = agent.get_stats()
            tokens_used = stats.get("total_tokens", 0)

        return GameResult(
            agent=agent.name,
            start=start,
            target=target,
            won=result.won,
            clicks=result.total_clicks,
            time_seconds=round(elapsed, 2),
            path=result.path,
            tokens_used=tokens_used,
        )
    except Exception as e:
        error_type, error_msg, traceback_str = classify_error(e)
        return GameResult(
            agent=agent_name,
            start=start,
            target=target,
            won=False,
            clicks=-1,
            time_seconds=0,
            path=[],
            error=error_msg,
            error_type=error_type,
            traceback=traceback_str,
        )


def summarize_results(results: list[GameResult], agent_name: str) -> AgentSummary:
    """Summarize results for an agent."""
    agent_results = [r for r in results if r.agent == agent_name or agent_name in r.agent]

    if not agent_results:
        return AgentSummary(
            agent=agent_name,
            games_played=0,
            wins=0,
            win_rate=0,
            avg_clicks_when_won=0,
            avg_time_seconds=0,
            total_time=0,
        )

    wins = [r for r in agent_results if r.won]
    total_time = sum(r.time_seconds for r in agent_results)

    return AgentSummary(
        agent=agent_name,
        games_played=len(agent_results),
        wins=len(wins),
        win_rate=len(wins) / len(agent_results) if agent_results else 0,
        avg_clicks_when_won=sum(r.clicks for r in wins) / len(wins) if wins else 0,
        avg_time_seconds=total_time / len(agent_results) if agent_results else 0,
        total_time=round(total_time, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive agent benchmark")
    parser.add_argument("--llm-only", action="store_true", help="Only test LLM agents")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip oracle (slow)")
    parser.add_argument("--free-only", action="store_true", help="Only test free LLM models")
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per game")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json", help="Output file")
    parser.add_argument("--cases", type=int, default=None, help="Limit number of test cases")
    parser.add_argument("--fresh", action="store_true", help="Clear cached results and start fresh")
    parser.add_argument("--rerun-errors", action="store_true", help="Rerun games that had errors")
    parser.add_argument("--rerun-losses", action="store_true", help="Rerun games that lost (hit max steps)")
    args = parser.parse_args()

    test_cases = TEST_CASES[:args.cases] if args.cases else TEST_CASES

    # Setup paths
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)
    jsonl_path = get_jsonl_path(output_path)

    # Load or clear cached results
    if args.fresh and jsonl_path.exists():
        jsonl_path.unlink()
        print("Cleared cached results (--fresh)")
        completed_games: set[tuple[str, str, str]] = set()
        all_results: list[GameResult] = []
    else:
        completed_games, all_results = load_completed_games(jsonl_path)

        # Filter out errors/losses for rerun
        if args.rerun_errors or args.rerun_losses:
            filtered_results = []
            removed_count = 0
            for r in all_results:
                should_remove = False
                if args.rerun_errors and r.error:
                    should_remove = True
                if args.rerun_losses and not r.won and not r.error:
                    should_remove = True

                if should_remove:
                    completed_games.discard((r.agent, r.start, r.target))
                    removed_count += 1
                else:
                    filtered_results.append(r)

            all_results = filtered_results
            # Rewrite JSONL without the removed results
            if removed_count > 0:
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for r in all_results:
                        f.write(json.dumps(asdict(r)) + "\n")
                print(f"Removed {removed_count} results for rerun")

        if completed_games:
            print(f"Resuming: {len(completed_games)} games already completed")

    print("=" * 80)
    print("COMPREHENSIVE WIKIPEDIA SPEEDRUN BENCHMARK")
    print("=" * 80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Embedding models: {len(EMBEDDING_MODELS)}")
    print(f"Max steps: {args.max_steps}")
    print()

    # ==========================================================================
    # EMBEDDING AGENTS (test multiple models)
    # ==========================================================================

    if not args.llm_only:
        print("-" * 80)
        print("TESTING EMBEDDING AGENTS")
        print("-" * 80)

        for model_name in EMBEDDING_MODELS:
            model_short = model_name.split("/")[-1]
            print(f"\n{model_short}:")
            print("  Loading embedding model...")

            # Pre-warm embedding model and get actual agent name
            agent = get_agent("live", model_name=model_name)
            agent._ensure_loaded()
            # Use the agent's actual name for cache key (e.g., "live-all-MiniLM-L6-v2")
            agent_name = agent.name

            for i, (start, target) in enumerate(test_cases, 1):
                # Skip if already completed
                if (agent_name, start, target) in completed_games:
                    print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : CACHED")
                    continue

                result = run_game("live", start, target, args.max_steps, model_name=model_name)
                all_results.append(result)
                append_result(jsonl_path, result)  # Incremental save

                if result.error:
                    print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : ERROR [{result.error_type}] {result.error[:50]}")
                else:
                    status = "WIN" if result.won else "LOST"
                    clicks = result.clicks if result.won else f">{args.max_steps}"
                    print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : {status} ({clicks} clicks, {result.time_seconds:.1f}s)")

    # ==========================================================================
    # LLM AGENTS
    # ==========================================================================

    print("\n" + "-" * 80)
    print("TESTING LLM AGENTS")
    print("-" * 80)

    if args.free_only:
        llm_models = FREE_MODELS
    else:
        # Skip free models by default (rate limits make them unreliable)
        llm_models = BUDGET_MODELS + FAST_MODELS + PREMIUM_MODELS

    for model in llm_models:
        model_short = model.split("/")[-1]
        # Agent name uses short model name (after slash), truncated to 25 chars
        if len(model_short) > 25:
            agent_name_for_cache = f"llm-{model_short[:22]}..."
        else:
            agent_name_for_cache = f"llm-{model_short}"
        print(f"\n{model_short[:30]}:")

        for i, (start, target) in enumerate(test_cases, 1):
            # Skip if already completed (check both possible agent name formats)
            if (agent_name_for_cache, start, target) in completed_games:
                print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : CACHED")
                continue

            result = run_game("llm", start, target, args.max_steps, model=model)
            all_results.append(result)
            append_result(jsonl_path, result)  # Incremental save

            if result.error:
                print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : ERROR [{result.error_type}] {result.error[:50]}")
            else:
                status = "WIN" if result.won else "LOST"
                clicks = result.clicks if result.won else f">{args.max_steps}"
                print(f"  [{i:2}/{len(test_cases)}] {start:20} → {target:20} : {status} ({clicks} clicks, {result.time_seconds:.1f}s)")

            # Small delay between LLM calls to avoid rate limits
            time.sleep(0.5)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Get unique agent names
    agent_names = sorted(set(r.agent for r in all_results))

    summaries = []
    for agent_name in agent_names:
        summary = summarize_results(all_results, agent_name)
        summaries.append(summary)

    # Sort by win rate, then by avg clicks
    summaries.sort(key=lambda s: (-s.win_rate, s.avg_clicks_when_won))

    print(f"\n{'Agent':<45} {'Wins':>8} {'Win%':>8} {'AvgClicks':>10} {'AvgTime':>10}")
    print("-" * 85)

    for s in summaries:
        win_pct = f"{s.win_rate*100:.0f}%"
        avg_clicks = f"{s.avg_clicks_when_won:.1f}" if s.avg_clicks_when_won > 0 else "-"
        avg_time = f"{s.avg_time_seconds:.1f}s"
        print(f"{s.agent:<45} {s.wins:>3}/{s.games_played:<3} {win_pct:>8} {avg_clicks:>10} {avg_time:>10}")

    # ==========================================================================
    # TOP 5 LLM RECOMMENDATIONS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("TOP 5 LLM RECOMMENDATIONS")
    print("=" * 80)

    llm_summaries = [s for s in summaries if "llm-" in s.agent]
    llm_summaries.sort(key=lambda s: (-s.win_rate, s.avg_clicks_when_won, s.avg_time_seconds))

    for i, s in enumerate(llm_summaries[:5], 1):
        model_name = s.agent.replace("llm-", "")
        print(f"{i}. {model_name}")
        print(f"   Win rate: {s.win_rate*100:.0f}%, Avg clicks: {s.avg_clicks_when_won:.1f}, Avg time: {s.avg_time_seconds:.1f}s")

    # ==========================================================================
    # ERROR SUMMARY
    # ==========================================================================

    errors = [r for r in all_results if r.error]
    if errors:
        print("\n" + "=" * 80)
        print("ERROR SUMMARY")
        print("=" * 80)

        # Count errors by type
        error_counts: dict[str, int] = {}
        for r in errors:
            error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count} errors")

        print(f"\nTotal errors: {len(errors)} / {len(all_results)} games")

    # ==========================================================================
    # SAVE FINAL RESULTS (consolidate from JSONL)
    # ==========================================================================

    # Compute error summary for JSON
    error_summary = {}
    for r in all_results:
        if r.error:
            error_summary[r.error_type] = error_summary.get(r.error_type, 0) + 1

    output_data = {
        "test_cases": [{"start": s, "target": t} for s, t in test_cases],
        "results": [asdict(r) for r in all_results],
        "summaries": [asdict(s) for s in summaries],
        "error_summary": error_summary,
        "total_errors": len(errors) if errors else 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Checkpoint file: {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
