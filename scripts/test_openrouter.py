#!/usr/bin/env python3
"""
Test OpenRouter API and compare models for Wikipedia speedrun.

This script:
1. Fetches all available models with pricing/specs from OpenRouter
2. Tests a sample Wikipedia speedrun prompt across selected models
3. Saves model comparison data to data/openrouter_models.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models we want to test for Wikipedia speedrun
# Fast models (good for speedrun time)
FAST_MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-5-haiku",
    "google/gemini-flash-1.5",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "mistralai/mistral-small-24b-instruct-2501",
]

# Smart models (good for fewest clicks)
SMART_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-5-sonnet-20241022",
    "openai/gpt-4o",
    "google/gemini-pro-1.5",
    "google/gemini-2.0-pro-exp-02-05",
    "deepseek/deepseek-chat",
]

# Sample prompt for testing
SAMPLE_PROMPT = """You are playing Wikipedia Speedrun. Navigate from one article to another by clicking links.

Current page: "Potato"
Target page: "Barack Obama"
Path so far: Potato

Available links on this page (showing top 20 by relevance):
- United States
- Agriculture
- South America
- Food
- Vegetable
- Plant
- Europe
- History
- World War II
- International trade
- Science
- Economy
- Government
- Politics
- Democratic Party
- President
- Washington, D.C.
- American
- English language
- Human

Which link should you click to get closer to "Barack Obama"?
Reply with ONLY the link name, nothing else."""


def fetch_all_models() -> list[dict]:
    """Fetch all available models from OpenRouter."""
    print("Fetching model list from OpenRouter...")

    response = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    models = data.get("data", [])
    print(f"Found {len(models)} models")
    return models


def extract_model_info(models: list[dict]) -> list[dict]:
    """Extract relevant info for each model."""
    extracted = []

    for m in models:
        info = {
            "id": m.get("id"),
            "name": m.get("name"),
            "context_length": m.get("context_length"),
            "pricing": {
                "prompt": m.get("pricing", {}).get("prompt"),  # per token
                "completion": m.get("pricing", {}).get("completion"),
            },
            "top_provider": m.get("top_provider", {}).get("max_completion_tokens"),
            "architecture": m.get("architecture", {}).get("modality"),
        }
        extracted.append(info)

    return extracted


def test_model(model_id: str) -> dict:
    """Test a single model with the sample prompt."""
    print(f"\nTesting {model_id}...")

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
                "model": model_id,
                "messages": [{"role": "user", "content": SAMPLE_PROMPT}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=60,
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            return {
                "model": model_id,
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "elapsed_seconds": elapsed,
            }

        data = response.json()

        # Extract response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        # Parse the link choice
        parsed_link = content.strip().strip('"').strip("'")
        # Remove any explanation if present
        if "\n" in parsed_link:
            parsed_link = parsed_link.split("\n")[0].strip()

        return {
            "model": model_id,
            "success": True,
            "raw_response": content,
            "parsed_link": parsed_link,
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    except requests.exceptions.Timeout:
        return {
            "model": model_id,
            "success": False,
            "error": "Timeout after 60 seconds",
            "elapsed_seconds": 60,
        }
    except Exception as e:
        return {
            "model": model_id,
            "success": False,
            "error": str(e),
            "elapsed_seconds": time.time() - start_time,
        }


def calculate_cost_per_game(model_info: dict, avg_requests: int = 16) -> float | None:
    """Estimate cost per game based on pricing."""
    pricing = model_info.get("pricing", {})
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")

    if not prompt_price or not completion_price:
        return None

    # Estimate: ~500 tokens prompt, ~10 tokens completion per request
    prompt_tokens = 500 * avg_requests
    completion_tokens = 10 * avg_requests

    # OpenRouter prices are per token (not per 1M)
    cost = (float(prompt_price) * prompt_tokens) + (float(completion_price) * completion_tokens)
    return round(cost, 6)


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        print("Make sure .env file exists with OPENROUTER_API_KEY=...")
        return 1

    print("=" * 70)
    print("OpenRouter Model Test for Wikipedia Speedrun")
    print("=" * 70)

    # Fetch all models
    all_models = fetch_all_models()
    model_info = extract_model_info(all_models)

    # Save model list
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    models_file = output_dir / "openrouter_models.json"
    with open(models_file, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\nSaved {len(model_info)} models to {models_file}")

    # Create lookup dict
    model_lookup = {m["id"]: m for m in model_info}

    # Test selected models
    print("\n" + "=" * 70)
    print("Testing Fast Models (for speed)")
    print("=" * 70)

    fast_results = []
    for model_id in FAST_MODELS:
        if model_id in model_lookup or True:  # Try anyway
            result = test_model(model_id)
            fast_results.append(result)

            if result["success"]:
                print(f"  Response: {result['raw_response'][:60]}...")
                print(f"  Parsed: '{result['parsed_link']}'")
                print(f"  Time: {result['elapsed_seconds']}s, Tokens: {result.get('total_tokens')}")
            else:
                print(f"  ERROR: {result['error'][:80]}")

    print("\n" + "=" * 70)
    print("Testing Smart Models (for fewest clicks)")
    print("=" * 70)

    smart_results = []
    for model_id in SMART_MODELS:
        result = test_model(model_id)
        smart_results.append(result)

        if result["success"]:
            print(f"  Response: {result['raw_response'][:60]}...")
            print(f"  Parsed: '{result['parsed_link']}'")
            print(f"  Time: {result['elapsed_seconds']}s, Tokens: {result.get('total_tokens')}")
        else:
            print(f"  ERROR: {result['error'][:80]}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Time':>8} {'Link Choice':<20}")
    print("-" * 75)

    all_results = fast_results + smart_results
    for r in all_results:
        if r["success"]:
            model_short = r["model"].split("/")[-1][:40]
            print(f"{model_short:<45} {r['elapsed_seconds']:>7.2f}s {r['parsed_link']:<20}")
        else:
            model_short = r["model"].split("/")[-1][:40]
            print(f"{model_short:<45} {'FAILED':>8} {r['error'][:20]}")

    # Save test results
    results_file = output_dir / "openrouter_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "fast_models": fast_results,
            "smart_models": smart_results,
            "sample_prompt": SAMPLE_PROMPT,
        }, f, indent=2)
    print(f"\nSaved test results to {results_file}")

    # Cost estimates
    print("\n" + "=" * 70)
    print("COST ESTIMATES (per 16-click game)")
    print("=" * 70)

    for model_id in FAST_MODELS + SMART_MODELS:
        if model_id in model_lookup:
            cost = calculate_cost_per_game(model_lookup[model_id])
            if cost:
                print(f"  {model_id:<45} ${cost:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
