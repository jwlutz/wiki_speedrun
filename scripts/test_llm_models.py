#!/usr/bin/env python3
"""
Parallel test of LLM models on Wikipedia speedrun.

Tests multiple models simultaneously on the same problem to compare:
- Response time
- Link choice quality
- Token usage

Usage:
    python scripts/test_llm_models.py
    python scripts/test_llm_models.py --models "gpt-4o-mini,claude-3-haiku" --start "Cat" --target "Dog"
    python scripts/test_llm_models.py --free-only
    python scripts/test_llm_models.py --full-game --models "gemini-2.0-flash-exp:free"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to test
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
]

BUDGET_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct",
    "openai/gpt-5-nano",
    "meta-llama/llama-3.2-3b-instruct",
]

VALUE_MODELS = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
]

FAST_MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-haiku",
    "google/gemini-3-flash-preview",
]

SMART_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5",
]

# Sample prompt for testing
SAMPLE_LINKS = [
    "United States",
    "Agriculture",
    "South America",
    "Food",
    "Vegetable",
    "Plant",
    "Europe",
    "History",
    "World War II",
    "International trade",
    "Science",
    "Economy",
    "Government",
    "Politics",
    "Democratic Party (United States)",
    "President of the United States",
    "Washington, D.C.",
    "American",
    "English language",
    "Human",
]


@dataclass
class TestResult:
    model: str
    success: bool
    response: str = ""
    parsed_link: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def build_prompt(current: str, target: str, links: list[str]) -> str:
    """Build test prompt."""
    links_str = "\n".join(f"- {link}" for link in links)
    return f"""You are playing Wikipedia Speedrun. Navigate from one article to another by clicking links only.

Current page: "{current}"
Target page: "{target}"

Available links on this page ({len(links)} total):
{links_str}

Which link should you click to reach "{target}"?
Reply with ONLY the exact link name from the list above, nothing else."""


def test_model(model: str, prompt: str, timeout: int = 60) -> TestResult:
    """Test a single model."""
    start_time = time.time()

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/jacklutz/wiki_speedrun",
                "X-Title": "Wikipedia Speedrun Benchmark",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
            },
            timeout=timeout,
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            return TestResult(
                model=model,
                success=False,
                elapsed_seconds=elapsed,
                error=f"HTTP {response.status_code}: {response.text[:100]}",
            )

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        # Parse link
        parsed = content.strip().strip('"').strip("'")
        if "\n" in parsed:
            parsed = parsed.split("\n")[0].strip()

        return TestResult(
            model=model,
            success=True,
            response=content,
            parsed_link=parsed,
            elapsed_seconds=round(elapsed, 3),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    except requests.exceptions.Timeout:
        return TestResult(
            model=model,
            success=False,
            elapsed_seconds=60,
            error="Timeout",
        )
    except Exception as e:
        return TestResult(
            model=model,
            success=False,
            elapsed_seconds=time.time() - start_time,
            error=str(e)[:100],
        )


def run_full_game(model: str, start: str, target: str, max_steps: int = 25) -> dict:
    """Run a full game with a model."""
    from src.agents import get_agent
    from src.game import GameEngine

    agent = get_agent("llm", model=model)

    with GameEngine(visualize=False) as engine:
        start_time = time.time()
        result = engine.run(
            agent=agent,
            start=start,
            target=target,
            max_steps=max_steps,
        )
        elapsed = time.time() - start_time

    stats = agent.get_stats()

    return {
        "model": model,
        "won": result.won,
        "clicks": result.total_clicks,
        "path": result.path,
        "elapsed_seconds": round(elapsed, 2),
        "total_tokens": stats.get("total_tokens", 0),
        "api_time": stats.get("total_time_seconds", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM models for Wikipedia speedrun")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to test (default: free models)",
    )
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="Only test free models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all model tiers",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="Potato",
        help="Start article for test",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Barack Obama",
        help="Target article for test",
    )
    parser.add_argument(
        "--full-game",
        action="store_true",
        help="Run full games instead of single prompts",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel requests (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return 1

    # Determine models to test
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.all:
        models = FREE_MODELS + BUDGET_MODELS + VALUE_MODELS + FAST_MODELS + SMART_MODELS
    elif args.free_only:
        models = FREE_MODELS
    else:
        models = FREE_MODELS  # Default to free

    print("=" * 80)
    print("LLM Model Comparison for Wikipedia Speedrun")
    print("=" * 80)
    print(f"Testing {len(models)} models")
    print(f"Problem: {args.start} → {args.target}")
    print(f"Mode: {'Full game' if args.full_game else 'Single prompt'}")
    print(f"Parallel: {args.parallel}")
    print("=" * 80)

    if args.full_game:
        # Run full games (sequential to avoid overwhelming Wikipedia)
        print("\nRunning full games (sequential)...")
        results = []
        for model in models:
            print(f"\n  Testing {model}...")
            try:
                result = run_full_game(model, args.start, args.target)
                results.append(result)
                status = "WIN" if result["won"] else "LOST"
                print(f"    {status} in {result['clicks']} clicks, {result['elapsed_seconds']}s")
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"model": model, "error": str(e)})
    else:
        # Single prompt test (parallel)
        prompt = build_prompt(args.start, args.target, SAMPLE_LINKS)

        print(f"\nTesting with sample prompt (parallel={args.parallel})...")
        results = []

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(test_model, model, prompt): model
                for model in models
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result.success:
                    print(f"  {result.model:<50} {result.elapsed_seconds:>6.2f}s  → {result.parsed_link}")
                else:
                    print(f"  {result.model:<50} FAILED: {result.error[:30]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if args.full_game:
        print(f"{'Model':<50} {'Result':>8} {'Clicks':>8} {'Time':>10}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: (not x.get("won", False), x.get("clicks", 999))):
            if "error" in r:
                print(f"{r['model']:<50} {'ERROR':>8}")
            else:
                status = "WIN" if r["won"] else "LOST"
                print(f"{r['model']:<50} {status:>8} {r['clicks']:>8} {r['elapsed_seconds']:>9.2f}s")
    else:
        print(f"{'Model':<50} {'Time':>8} {'Tokens':>8} {'Choice'}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x.elapsed_seconds if x.success else 999):
            if r.success:
                tokens = r.prompt_tokens + r.completion_tokens
                print(f"{r.model:<50} {r.elapsed_seconds:>7.2f}s {tokens:>8} {r.parsed_link}")
            else:
                print(f"{r.model:<50} {'FAILED':>8} {r.error[:30]}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            if args.full_game:
                json.dump(results, f, indent=2)
            else:
                json.dump(
                    [
                        {
                            "model": r.model,
                            "success": r.success,
                            "elapsed_seconds": r.elapsed_seconds,
                            "parsed_link": r.parsed_link,
                            "tokens": r.prompt_tokens + r.completion_tokens,
                            "error": r.error,
                        }
                        for r in results
                    ],
                    f,
                    indent=2,
                )
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
