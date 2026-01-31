"""
Data loader for benchmark results.

Supports both JSON and JSONL formats.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class BenchmarkResult:
    agent: str
    start: str
    target: str
    won: bool
    clicks: int
    time_seconds: float
    path: list[str]
    error: str
    tokens_used: int


@dataclass
class AgentSummary:
    agent: str
    games_played: int
    wins: int
    win_rate: float
    avg_clicks: float
    avg_time: float
    total_tokens: int
    agent_type: str  # "llm" or "embedding"


def get_data_path() -> Path:
    """Get path to data directory."""
    return Path(__file__).parent.parent.parent / "data"


def _load_jsonl_results() -> list[dict]:
    """Load benchmark results from JSONL file."""
    jsonl_path = get_data_path() / "benchmark_results.jsonl"
    if not jsonl_path.exists():
        return []

    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


@lru_cache(maxsize=1)
def load_benchmark_results() -> dict:
    """Load benchmark results from JSON or JSONL file."""
    # Try JSONL first (more likely to have recent data)
    jsonl_results = _load_jsonl_results()
    if jsonl_results:
        return {"results": jsonl_results, "summaries": [], "test_cases": []}

    # Fall back to JSON
    results_path = get_data_path() / "benchmark_results.json"
    if not results_path.exists():
        return {"results": [], "summaries": [], "test_cases": []}

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
        # Handle both formats
        if "results" in data:
            return data
        elif "test_cases" in data:
            return {"results": [], "summaries": [], "test_cases": data["test_cases"]}
        else:
            return {"results": [], "summaries": [], "test_cases": []}


@lru_cache(maxsize=1)
def load_benchmark_problems() -> dict:
    """Load benchmark problems from JSON file."""
    problems_path = get_data_path() / "benchmark_problems.json"
    if not problems_path.exists():
        return {"problems": []}

    with open(problems_path, encoding="utf-8") as f:
        data = json.load(f)
        # Handle different formats
        if "problems" in data:
            return data
        elif isinstance(data, list):
            return {"problems": data}
        else:
            return {"problems": []}


def get_results() -> list[BenchmarkResult]:
    """Get all benchmark results as dataclass instances."""
    data = load_benchmark_results()
    results = []
    for r in data.get("results", []):
        results.append(BenchmarkResult(
            agent=r.get("agent", ""),
            start=r.get("start", ""),
            target=r.get("target", ""),
            won=r.get("won", False),
            clicks=r.get("clicks", 0),
            time_seconds=r.get("time_seconds", 0),
            path=r.get("path", []),
            error=r.get("error", ""),
            tokens_used=r.get("tokens_used", 0),
        ))
    return results


def get_agent_summaries() -> list[AgentSummary]:
    """Compute agent summaries from results."""
    results = get_results()

    # Group by agent
    agent_data: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.agent not in agent_data:
            agent_data[r.agent] = []
        agent_data[r.agent].append(r)

    summaries = []
    for agent, games in agent_data.items():
        wins = [g for g in games if g.won]
        win_clicks = [g.clicks for g in wins] if wins else [0]

        # Determine agent type
        agent_type = "embedding" if agent.startswith("live-") else "llm"

        summaries.append(AgentSummary(
            agent=agent,
            games_played=len(games),
            wins=len(wins),
            win_rate=len(wins) / len(games) * 100 if games else 0,
            avg_clicks=sum(win_clicks) / len(win_clicks) if win_clicks else 0,
            avg_time=sum(g.time_seconds for g in games) / len(games) if games else 0,
            total_tokens=sum(g.tokens_used for g in games),
            agent_type=agent_type,
        ))

    # Sort by win rate, then avg clicks
    summaries.sort(key=lambda s: (-s.win_rate, s.avg_clicks))
    return summaries


def get_problems() -> list[dict]:
    """Get all benchmark problems."""
    data = load_benchmark_problems()
    return data.get("problems", [])


def get_results_by_difficulty() -> dict[str, list[BenchmarkResult]]:
    """Group results by problem difficulty."""
    problems = {p["start"] + "→" + p["target"]: p["difficulty"]
                for p in get_problems()}

    results = get_results()
    by_difficulty: dict[str, list[BenchmarkResult]] = {
        "easy": [], "medium": [], "hard": []
    }

    for r in results:
        key = r.start + "→" + r.target
        difficulty = problems.get(key, "unknown")
        if difficulty in by_difficulty:
            by_difficulty[difficulty].append(r)

    return by_difficulty


def get_path_data() -> list[dict]:
    """Get path data for network visualization."""
    results = get_results()
    paths = []

    for r in results:
        if r.won and r.path:
            paths.append({
                "agent": r.agent,
                "start": r.start,
                "target": r.target,
                "path": r.path,
                "clicks": r.clicks,
            })

    return paths


@lru_cache(maxsize=1)
def load_model_pricing() -> dict[str, float]:
    """Load OpenRouter model pricing (cost per 1M input tokens)."""
    models_path = get_data_path() / "openrouter_models.json"
    if not models_path.exists():
        return {}

    with open(models_path, encoding="utf-8") as f:
        models = json.load(f)

    # Map model ID to price per 1M input tokens
    pricing = {}
    for m in models:
        price_str = m.get("pricing", {}).get("prompt", "0")
        pricing[m["id"]] = float(price_str) * 1_000_000
    return pricing


def _agent_to_model_id(agent: str) -> str | None:
    """Map agent name to OpenRouter model ID."""
    if not agent.startswith("llm-"):
        return None

    model_part = agent.replace("llm-", "")
    pricing = load_model_pricing()

    # Try exact match first
    for mid in pricing:
        # Handle truncated agent names (e.g., "llm-llama-3.3-70b-instruct...")
        if model_part.rstrip(".") in mid or mid.endswith(model_part.rstrip(".")):
            return mid
        # Handle common patterns
        short_mid = mid.split("/")[-1] if "/" in mid else mid
        if model_part.startswith(short_mid) or short_mid.startswith(model_part.rstrip(".")):
            return mid

    return None


@dataclass
class AgentCostSummary:
    """Agent summary with cost information for Pareto analysis."""
    agent: str
    agent_type: str
    win_rate: float
    avg_clicks: float
    avg_time: float
    total_tokens: int
    total_cost: float
    cost_per_game: float
    games_played: int
    wins: int


def get_agent_cost_summaries() -> list[AgentCostSummary]:
    """Get agent summaries with cost data for Pareto frontier analysis."""
    summaries = get_agent_summaries()
    pricing = load_model_pricing()

    cost_summaries = []
    for s in summaries:
        model_id = _agent_to_model_id(s.agent)
        price_per_1m = pricing.get(model_id, 0) if model_id else 0

        # Cost = (tokens / 1M) * price_per_1M
        total_cost = (s.total_tokens / 1_000_000) * price_per_1m
        cost_per_game = total_cost / s.games_played if s.games_played > 0 else 0

        cost_summaries.append(AgentCostSummary(
            agent=s.agent,
            agent_type=s.agent_type,
            win_rate=s.win_rate,
            avg_clicks=s.avg_clicks,
            avg_time=s.avg_time,
            total_tokens=s.total_tokens,
            total_cost=total_cost,
            cost_per_game=cost_per_game,
            games_played=s.games_played,
            wins=s.wins,
        ))

    return cost_summaries


def get_problem_difficulty() -> list[dict]:
    """
    Compute empirical difficulty for each problem based on benchmark results.

    Difficulty score combines:
    - Failure rate: what % of agents failed this problem
    - Average clicks: more clicks = harder (even for successful runs)
    - Click variance: high variance suggests tricky navigation
    """
    results = get_results()

    # Group results by problem
    problem_stats: dict[str, dict] = {}
    for r in results:
        key = f"{r.start} -> {r.target}"
        if key not in problem_stats:
            problem_stats[key] = {
                "start": r.start,
                "target": r.target,
                "attempts": 0,
                "wins": 0,
                "clicks": [],
                "times": [],
                "failed_agents": [],
            }
        problem_stats[key]["attempts"] += 1
        if r.won:
            problem_stats[key]["wins"] += 1
            problem_stats[key]["clicks"].append(r.clicks)
            problem_stats[key]["times"].append(r.time_seconds)
        else:
            problem_stats[key]["failed_agents"].append(r.agent)

    # Compute difficulty metrics
    difficulties = []
    for key, stats in problem_stats.items():
        attempts = stats["attempts"]
        wins = stats["wins"]
        clicks = stats["clicks"]

        # Failure rate (0-1, higher = harder)
        failure_rate = 1 - (wins / attempts) if attempts > 0 else 0

        # Average clicks for wins (higher = harder)
        avg_clicks = sum(clicks) / len(clicks) if clicks else 25  # max if no wins

        # Click variance (higher variance = trickier)
        if len(clicks) > 1:
            mean_clicks = avg_clicks
            variance = sum((c - mean_clicks) ** 2 for c in clicks) / len(clicks)
            click_std = variance ** 0.5
        else:
            click_std = 0

        # Combined difficulty score (0-100)
        # Weight: 50% failure rate, 35% avg clicks (normalized), 15% variance
        normalized_clicks = min(avg_clicks / 25, 1)  # 25 is max
        normalized_std = min(click_std / 10, 1)  # cap at 10
        difficulty_score = (
            failure_rate * 50 +
            normalized_clicks * 35 +
            normalized_std * 15
        )

        difficulties.append({
            "start": stats["start"],
            "target": stats["target"],
            "attempts": attempts,
            "wins": wins,
            "failure_rate": failure_rate * 100,
            "avg_clicks": avg_clicks,
            "click_std": click_std,
            "difficulty_score": difficulty_score,
            "failed_agents": stats["failed_agents"],
            "difficulty_label": (
                "Easy" if difficulty_score < 20 else
                "Medium" if difficulty_score < 40 else
                "Hard" if difficulty_score < 60 else
                "Very Hard"
            ),
        })

    # Sort by difficulty score (hardest first)
    difficulties.sort(key=lambda x: -x["difficulty_score"])
    return difficulties


def get_failure_analysis() -> list[dict]:
    """Get analysis of failed games grouped by problem (uses difficulty calculation)."""
    difficulties = get_problem_difficulty()

    # Filter to only problems with failures
    failures = [d for d in difficulties if d["failure_rate"] > 0]

    # Convert to expected format
    return [
        {
            "start": d["start"],
            "target": d["target"],
            "failure_count": len(d["failed_agents"]),
            "failed_agents": d["failed_agents"],
            "difficulty_score": d["difficulty_score"],
            "difficulty_label": d["difficulty_label"],
            "avg_clicks": d["avg_clicks"],
        }
        for d in failures
    ]


def get_dashboard_stats() -> dict:
    """Get summary statistics for dashboard cards."""
    results = get_results()
    summaries = get_agent_summaries()
    cost_summaries = get_agent_cost_summaries()

    total_games = len(results)
    total_wins = sum(1 for r in results if r.won)

    # Best agent by win rate
    best_agent = summaries[0] if summaries else None

    # Cheapest agent with 95%+ win rate
    high_performers = [s for s in cost_summaries if s.win_rate >= 95]
    cheapest_good = min(high_performers, key=lambda s: s.cost_per_game) if high_performers else None

    # Hardest problem (most failures)
    failures = get_failure_analysis()
    hardest = failures[0] if failures else None

    return {
        "total_games": total_games,
        "total_wins": total_wins,
        "win_rate": total_wins / total_games * 100 if total_games > 0 else 0,
        "agents_tested": len(summaries),
        "best_agent": best_agent.agent if best_agent else "N/A",
        "best_win_rate": best_agent.win_rate if best_agent else 0,
        "cheapest_good_agent": cheapest_good.agent if cheapest_good else "N/A",
        "cheapest_good_cost": cheapest_good.cost_per_game if cheapest_good else 0,
        "hardest_problem": f"{hardest['start']} -> {hardest['target']}" if hardest else "N/A",
        "hardest_failure_count": hardest["failure_count"] if hardest else 0,
    }
