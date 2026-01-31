"""
Wikipedia Speedrun - Flask app for Hugging Face Spaces.

Features:
- Dashboard: Benchmark results with Pareto chart and leaderboard
- Play: Navigate Wikipedia with inline clickable links
- Watch AI: Spectate an AI agent playing live
"""

import os
import time
import json
import threading
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify
import requests
from bs4 import BeautifulSoup
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "wiki-speedrun-dev-key")

# ====================
# Data Loading
# ====================

def get_data_path() -> Path:
    return Path(__file__).parent / "data"


def load_benchmark_data() -> dict:
    data_path = get_data_path()
    jsonl_path = data_path / "benchmark_results.jsonl"
    if jsonl_path.exists():
        results = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if results:
            return {"results": results}

    json_path = data_path / "benchmark_results.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {"results": []}


def load_model_pricing() -> dict[str, float]:
    models_path = get_data_path() / "openrouter_models.json"
    if not models_path.exists():
        return {}
    with open(models_path, encoding="utf-8") as f:
        models = json.load(f)
    return {m["id"]: float(m.get("pricing", {}).get("prompt", "0")) * 1_000_000 for m in models}


@dataclass
class AgentStats:
    agent: str
    agent_type: str
    games: int
    wins: int
    win_rate: float
    avg_clicks: float
    avg_time: float
    cost_per_game: float


def compute_agent_stats() -> list[AgentStats]:
    data = load_benchmark_data()
    pricing = load_model_pricing()
    agent_data: dict[str, list] = {}
    for r in data.get("results", []):
        agent = r.get("agent", "")
        agent_data.setdefault(agent, []).append(r)

    stats = []
    for agent, games in agent_data.items():
        wins = [g for g in games if g.get("won", False)]
        win_clicks = [g.get("clicks", 0) for g in wins] if wins else [0]
        agent_type = "embedding" if agent.startswith("live-") else "llm"
        total_tokens = sum(g.get("tokens_used", 0) for g in games)

        model_id = None
        if agent.startswith("llm-"):
            model_part = agent.replace("llm-", "")
            for mid in pricing:
                short_mid = mid.split("/")[-1] if "/" in mid else mid
                if model_part.startswith(short_mid) or short_mid.startswith(model_part.rstrip(".")):
                    model_id = mid
                    break

        price = pricing.get(model_id, 0) if model_id else 0
        total_cost = (total_tokens / 1_000_000) * price
        cost_per_game = total_cost / len(games) if games else 0

        stats.append(AgentStats(
            agent=agent, agent_type=agent_type, games=len(games), wins=len(wins),
            win_rate=len(wins) / len(games) * 100 if games else 0,
            avg_clicks=sum(win_clicks) / len(win_clicks) if win_clicks else 0,
            avg_time=sum(g.get("time_seconds", 0) for g in games) / len(games) if games else 0,
            cost_per_game=cost_per_game,
        ))
    stats.sort(key=lambda s: (-s.win_rate, s.avg_clicks))
    return stats


# ====================
# Wikipedia Scraping
# ====================

_wiki_cache = {}


def fetch_wiki_page(title: str) -> tuple[str, list[str]]:
    if title in _wiki_cache:
        return _wiki_cache[title]

    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded}"
    resp = requests.get(url, headers={"User-Agent": "WikiSpeedrun/1.0"}, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for sel in [".navbox", ".infobox", ".sidebar", ".reference", ".reflist",
                ".mw-editsection", "figure", ".thumb", ".ambox", ".mbox", ".toc",
                ".hatnote", ".shortdescription", "style", "script", "base"]:
        for el in soup.select(sel):
            el.decompose()

    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        # Skip red links (non-existent pages) - they have class="new"
        if "new" in a.get("class", []):
            continue
        if href.startswith("./") and ":" not in href:
            path = href.replace("./", "").split("#")[0].split("?")[0]
            decoded = urllib.parse.unquote(path).replace("_", " ")
            if decoded and decoded not in links:
                links.append(decoded)

    result = (str(soup), links)
    _wiki_cache[title] = result
    return result


def process_wiki_html(html: str, available_links: list[str], target: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    available_lower = {l.lower(): l for l in available_links}
    target_lower = target.lower()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.startswith("./"):
            a["class"] = a.get("class", []) + ["disabled"]
            if "href" in a.attrs:
                del a["href"]
            continue

        path = href.replace("./", "").split("#")[0].split("?")[0]
        decoded = urllib.parse.unquote(path).replace("_", " ")
        norm = decoded.lower()

        if norm == target_lower:
            a["class"] = a.get("class", []) + ["target-link"]
            a["href"] = f"/navigate?title={urllib.parse.quote(decoded)}"
        elif norm in available_lower:
            a["class"] = a.get("class", []) + ["available"]
            a["href"] = f"/navigate?title={urllib.parse.quote(available_lower[norm])}"
        else:
            a["class"] = a.get("class", []) + ["disabled"]
            if "href" in a.attrs:
                del a["href"]

    return str(soup)


# ====================
# AI Agent (Embedding-based)
# ====================

_embedding_model = None
_model_ready = False
_model_warming_up = False


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None
    return _embedding_model


def warmup_model():
    """Warmup embedding model in background thread."""
    global _model_ready, _model_warming_up
    _model_warming_up = True
    print("Warming up embedding model...")
    model = get_embedding_model()
    if model:
        # Do a test encode to fully initialize
        model.encode(["warmup test"], convert_to_numpy=True)
        _model_ready = True
        print("Model ready!")
    else:
        print("Warning: sentence-transformers not available")
    _model_warming_up = False


def is_model_ready() -> bool:
    return _model_ready


def ai_choose_link(available_links: list[str], target: str, path_so_far: list[str]) -> tuple[str, float]:
    """AI agent picks best link using embeddings. Returns (link, similarity_score)."""
    if target in available_links:
        return target, 1.0

    model = get_embedding_model()
    if model is None:
        # Fallback: random choice
        import random
        return random.choice(available_links), 0.0

    # Avoid revisits
    visited = set(path_so_far)
    candidates = [l for l in available_links if l not in visited]
    if not candidates:
        candidates = available_links

    # Encode target + candidates
    all_texts = [target] + candidates
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    target_emb = embeddings[0]
    candidate_embs = embeddings[1:]
    similarities = np.dot(candidate_embs, target_emb)

    best_idx = int(np.argmax(similarities))
    return candidates[best_idx], float(similarities[best_idx])


# ====================
# AI Spectator State
# ====================

@dataclass
class AIGame:
    start: str
    target: str
    path: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    html: str = ""
    won: bool = False
    start_time: float = 0
    last_choice: str = ""
    last_score: float = 0
    pending_choice: str = ""
    pending_score: float = 0
    error: str = ""
    running: bool = False

_ai_games: dict[str, AIGame] = {}


def ai_game_choose(game_id: str) -> tuple[str, float]:
    """AI chooses next link but doesn't navigate yet. Returns (choice, score)."""
    game = _ai_games.get(game_id)
    if not game or not game.running:
        return "", 0.0

    current = game.path[-1]

    # Check win
    if current.lower() == game.target.lower():
        game.won = True
        game.running = False
        return "", 0.0

    # Choose next link
    choice, score = ai_choose_link(game.links, game.target, game.path)
    game.pending_choice = choice
    game.pending_score = score
    return choice, score


def ai_game_advance(game_id: str):
    """Navigate to the pending choice. Skips invalid links and retries."""
    game = _ai_games.get(game_id)
    if not game or not game.running or not game.pending_choice:
        return

    choice = game.pending_choice
    game.last_choice = choice
    game.last_score = game.pending_score
    game.pending_choice = ""
    game.pending_score = 0

    # Navigate - retry on 404
    max_retries = 5
    for attempt in range(max_retries):
        try:
            html, links = fetch_wiki_page(choice)
            game.path.append(choice)
            game.html = html
            game.links = links

            # Check win after navigate
            if choice.lower() == game.target.lower():
                game.won = True
                game.running = False
            return  # Success
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "Not Found" in error_str:
                # Remove invalid link and choose another
                if choice in game.links:
                    game.links.remove(choice)
                if not game.links:
                    game.error = "No valid links remaining"
                    game.running = False
                    return
                # Choose a new link
                choice, score = ai_choose_link(game.links, game.target, game.path)
                game.last_choice = choice
                game.last_score = score
            else:
                game.error = error_str
                game.running = False
                return

    game.error = f"Failed after {max_retries} retries"
    game.running = False


# ====================
# HTML Templates
# ====================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Wikipedia Speedrun</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; min-height: 100vh; }
        .header { background: #1a1a2e; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.5rem; }
        .header a { color: #88c0d0; text-decoration: none; margin-left: 20px; }
        .header a:hover { text-decoration: underline; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 20px; }
        h2 { margin-bottom: 20px; color: #1a1a2e; }
        input[type="text"], select { width: 100%; padding: 12px 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; margin-bottom: 15px; }
        input:focus, select:focus { outline: none; border-color: #4ecdc4; }
        button, .btn { background: #4ecdc4; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #45b7aa; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-secondary { background: #95a5a6; }
        .btn-secondary:hover { background: #7f8c8d; }
        .game-status { background: #1a1a2e; color: white; padding: 15px 30px; position: sticky; top: 0; z-index: 100; display: flex; gap: 30px; align-items: center; flex-wrap: wrap; }
        .game-status .target { background: #f1c40f; color: #1a1a2e; padding: 5px 15px; border-radius: 6px; font-weight: 600; }
        .game-status .stat { display: flex; align-items: center; gap: 8px; }
        .game-status .stat-value { font-size: 1.3rem; font-weight: 600; color: #4ecdc4; }
        .path { background: #ecf0f1; padding: 10px 20px; font-size: 14px; color: #666; border-bottom: 1px solid #ddd; }
        .wiki-content { background: white; padding: 30px; line-height: 1.8; }
        .wiki-content h1, .wiki-content h2, .wiki-content h3 { margin: 20px 0 10px; color: #1a1a2e; }
        .wiki-content p { margin-bottom: 15px; }
        .wiki-content a.available { background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px; text-decoration: none; }
        .wiki-content a.available:hover { background: #28a745; color: white; }
        .wiki-content a.target-link { background: linear-gradient(90deg, #fff3cd, #ffeaa7); border: 2px solid #f1c40f; padding: 3px 10px; border-radius: 6px; color: #856404; font-weight: 600; text-decoration: none; animation: glow 1.5s ease-in-out infinite; }
        @keyframes glow { 0%,100% { box-shadow: 0 0 5px #f1c40f; } 50% { box-shadow: 0 0 20px #f1c40f; } }
        .wiki-content a.disabled { color: #aaa; pointer-events: none; }
        .wiki-content img { max-width: 300px; height: auto; }
        .legend { display: flex; gap: 20px; padding: 15px 30px; background: #f8f9fa; border-bottom: 1px solid #ddd; font-size: 14px; flex-wrap: wrap; }
        .legend span { padding: 4px 12px; border-radius: 4px; }
        .legend .available { background: #d4edda; color: #155724; }
        .legend .target { background: #fff3cd; border: 1px solid #f1c40f; color: #856404; }
        .win-screen { text-align: center; padding: 60px 30px; }
        .win-screen h1 { font-size: 3rem; color: #27ae60; margin-bottom: 20px; }
        .form-row { display: flex; gap: 20px; margin-bottom: 20px; }
        .form-row > div { flex: 1; }
        .form-row label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .preset-grid { display: flex; flex-direction: column; gap: 10px; margin-top: 20px; }
        .preset-btn { color: white; border: none; padding: 15px 20px; border-radius: 8px; cursor: pointer; text-align: left; }
        .preset-btn:hover { opacity: 0.85; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .stat-card .value { font-size: 2rem; font-weight: 700; }
        .stat-card .label { color: #666; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; color: white; }
        .badge-emb { background: #2ecc71; }
        .badge-llm { background: #3498db; }
        .ai-info { background: #e8f4fd; border: 1px solid #bee5eb; border-radius: 8px; padding: 15px; margin: 15px 0; }
        .ai-info strong { color: #0c5460; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #ddd; border-top-color: #4ecdc4; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.9); display: flex; justify-content: center; align-items: center; z-index: 9999; flex-direction: column; gap: 15px; }
        .loading-overlay .spinner { width: 40px; height: 40px; border-width: 4px; }
        .loading-text { font-size: 1.2rem; color: #666; }
    </style>
</head>
<body>
    {{ content | safe }}
    <script>
        // Live timer
        const timerEl = document.getElementById('timer');
        const startTime = {{ start_time or 0 }};
        if (timerEl && startTime) {
            setInterval(() => {
                const elapsed = (Date.now() / 1000) - startTime;
                timerEl.textContent = elapsed.toFixed(1) + 's';
            }, 100);
        }
    </script>
</body>
</html>
"""


# ====================
# Routes - Dashboard
# ====================

@app.route("/")
def dashboard():
    """Dashboard with full Plotly charts."""
    NAV_BAR = '<div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>'

    try:
        from ui.components.data_loader import (
            get_agent_summaries,
            get_results,
            get_agent_cost_summaries,
            get_failure_analysis,
            get_dashboard_stats,
            get_problems,
        )
        from ui.components.charts import (
            create_win_rate_chart,
            create_clicks_boxplot,
            create_time_scatter,
            create_difficulty_heatmap,
            create_pareto_chart,
            create_leaderboard_table,
            create_failure_table,
        )

        # Load data
        stats = get_dashboard_stats()
        summaries = get_agent_summaries()
        cost_summaries = get_agent_cost_summaries()
        results = get_results()
        problems = get_problems()
        failures = get_failure_analysis()

        if not summaries:
            content = f"""
            <div class="header">
                <h1>Wikipedia Speedrun</h1>
                {NAV_BAR}
            </div>
            <div class="container">
                <div class="card">
                    <h2>No Benchmark Data</h2>
                    <p>Add results to <code>data/benchmark_results.jsonl</code> to see the dashboard.</p>
                    <p style="margin-top:20px;"><a href="/play" class="btn">Play Now</a></p>
                </div>
            </div>
            """
            return render_template_string(BASE_TEMPLATE, content=content, start_time=0)

        # Generate charts as HTML
        pareto_html = create_pareto_chart(cost_summaries).to_html(full_html=False, include_plotlyjs="cdn")
        win_rate_html = create_win_rate_chart(summaries).to_html(full_html=False, include_plotlyjs=False)
        clicks_html = create_clicks_boxplot(results).to_html(full_html=False, include_plotlyjs=False)
        time_html = create_time_scatter(summaries).to_html(full_html=False, include_plotlyjs=False)

        # Try difficulty heatmap (may fail if no difficulty data)
        try:
            difficulty_html = create_difficulty_heatmap(results, problems).to_html(full_html=False, include_plotlyjs=False)
        except Exception:
            difficulty_html = "<p style='color:#666;'>No difficulty data available.</p>"

        # Generate tables
        leaderboard_html = create_leaderboard_table(cost_summaries)
        failure_html = create_failure_table(failures)

        # Format stats for display
        cheapest_cost = f"${stats['cheapest_good_cost']:.4f}" if stats['cheapest_good_cost'] > 0 else "FREE"

        content = f"""
        <div class="header">
            <h1>Wikipedia Speedrun Benchmark</h1>
            {NAV_BAR}
        </div>
        <div class="container" style="max-width:1200px;">
            <!-- Summary Cards -->
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px;">
                <div class="card" style="padding:20px;text-align:center;">
                    <div style="font-size:2rem;font-weight:700;color:#3498db;">{stats['total_games']}</div>
                    <div style="color:#666;font-size:14px;">Total Games</div>
                </div>
                <div class="card" style="padding:20px;text-align:center;">
                    <div style="font-size:2rem;font-weight:700;color:#2ecc71;">{stats['agents_tested']}</div>
                    <div style="color:#666;font-size:14px;">Agents Tested</div>
                </div>
                <div class="card" style="padding:20px;text-align:center;">
                    <div style="font-size:1rem;font-weight:700;color:#9b59b6;">{stats['best_agent'].replace('llm-','').replace('live-','')}</div>
                    <div style="color:#666;font-size:14px;">Best Agent ({stats['best_win_rate']:.0f}%)</div>
                </div>
                <div class="card" style="padding:20px;text-align:center;">
                    <div style="font-size:1rem;font-weight:700;color:#f39c12;">{stats['cheapest_good_agent'].replace('llm-','').replace('live-','')}</div>
                    <div style="color:#666;font-size:14px;">Best Value ({cheapest_cost}/game)</div>
                </div>
            </div>

            <!-- Pareto Frontier (Hero Chart) -->
            <div class="card">
                <h2>Cost vs Performance</h2>
                <p style="color:#666;margin-bottom:15px;">Points on the yellow frontier represent optimal cost-performance tradeoffs. Green = embedding models, Blue = LLM models.</p>
                {pareto_html}
            </div>

            <!-- Leaderboard -->
            <div class="card">
                <h2>Agent Leaderboard</h2>
                {leaderboard_html}
            </div>

            <!-- Charts Row 1 -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div class="card">
                    <h2>Win Rate by Agent</h2>
                    {win_rate_html}
                </div>
                <div class="card">
                    <h2>Click Distribution (Wins Only)</h2>
                    {clicks_html}
                </div>
            </div>

            <!-- Charts Row 2 -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div class="card">
                    <h2>Efficiency: Clicks vs Time</h2>
                    {time_html}
                </div>
                <div class="card">
                    <h2>Win Rate by Difficulty</h2>
                    {difficulty_html}
                </div>
            </div>

            <!-- Failure Analysis -->
            <div class="card">
                <h2>Hardest Problems (Most Failures)</h2>
                <p style="color:#666;margin-bottom:15px;">All failures are timeouts (25 clicks max). These problems defeated the most agents.</p>
                {failure_html}
            </div>

            <!-- Try It Section -->
            <div class="card" style="text-align:center;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;">
                <h2 style="color:white;">Think You Can Beat the AI?</h2>
                <p style="margin:15px 0;">Try the Wikipedia Speedrun yourself and see how you compare to our AI agents.</p>
                <a href="/play" class="btn" style="background:white;color:#667eea;font-weight:600;margin-right:10px;">Play Now</a>
                <a href="/watch" class="btn" style="background:rgba(255,255,255,0.2);border:2px solid white;">Watch AI</a>
            </div>
        </div>
        """

    except Exception as e:
        content = f"""
        <div class="header">
            <h1>Wikipedia Speedrun</h1>
            <div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>
        </div>
        <div class="container">
            <div class="card">
                <h2>Error Loading Dashboard</h2>
                <p style="color:#e74c3c;">{e}</p>
                <p>Make sure benchmark data exists in <code>data/benchmark_results.jsonl</code></p>
                <p style="margin-top:20px;"><a href="/play" class="btn">Play Now</a></p>
            </div>
        </div>
        """

    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


# ====================
# Routes - Play
# ====================

@app.route("/play")
def play_start():
    content = """
    <div class="header">
        <h1>Wikipedia Speedrun</h1>
        <div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>
    </div>
    <div class="container">
        <div class="card">
            <h2>Start a New Game</h2>
            <p style="color:#666;margin-bottom:20px;">Navigate from one Wikipedia article to another using only the links on each page.</p>
            <form method="POST" action="/start">
                <div class="form-row">
                    <div><label>Start Article</label><input type="text" name="start" value="Emperor Penguin"></div>
                    <div><label>Target Article</label><input type="text" name="target" value="iPhone"></div>
                </div>
                <div style="margin-bottom:15px;">
                    <label style="display:flex;align-items:center;gap:10px;cursor:pointer;">
                        <input type="checkbox" name="visualize" value="1" checked style="width:18px;height:18px;">
                        <span><strong>Visualize</strong> - Show full Wikipedia pages (slower but prettier)</span>
                    </label>
                </div>
                <button type="submit">Start Game</button>
            </form>
        </div>
        <div class="card">
            <h2>Difficulty Presets</h2>
            <div class="preset-grid">
                <button type="button" onclick="setGame('Cat','Dog')" class="preset-btn" style="background:#2ecc71;"><strong>Easy:</strong> Cat → Dog</button>
                <button type="button" onclick="setGame('Coffee','Minecraft')" class="preset-btn" style="background:#3498db;"><strong>Medium:</strong> Coffee → Minecraft</button>
                <button type="button" onclick="setGame('Lamprey','Costco hot dog')" class="preset-btn" style="background:#e67e22;"><strong>Hard:</strong> Lamprey → Costco hot dog</button>
            </div>
        </div>
    </div>
    <div id="loading-overlay" class="loading-overlay" style="display:none;">
        <div class="spinner"></div>
        <div class="loading-text">Loading Wikipedia pages...</div>
    </div>
    <script>
    function setGame(start, target) {
        document.querySelector('input[name="start"]').value = start;
        document.querySelector('input[name="target"]').value = target;
    }
    document.querySelector('form').addEventListener('submit', function() {
        document.getElementById('loading-overlay').style.display = 'flex';
    });
    </script>
    """
    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


@app.route("/start", methods=["POST"])
def start_game():
    start = request.form.get("start", "").strip()
    target = request.form.get("target", "").strip()
    visualize = request.form.get("visualize") == "1"

    if not start or not target:
        return redirect(url_for("play_start"))

    try:
        fetch_wiki_page(start)
        fetch_wiki_page(target)
    except Exception as e:
        return render_template_string(BASE_TEMPLATE, content=f"""
            <div class="header"><h1>Wikipedia Speedrun</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
            <div class="container"><div class="card"><h2>Error</h2><p>Could not find article: {e}</p><a href="/play" class="btn">Back</a></div></div>
        """, start_time=0)

    session["start"] = start
    session["target"] = target
    session["current"] = start
    session["path"] = [start]
    session["start_time"] = time.time()
    session["visualize"] = visualize
    return redirect(url_for("game"))


@app.route("/game")
def game():
    if "current" not in session:
        return redirect(url_for("play_start"))

    current = session["current"]
    target = session["target"]
    path = session["path"]
    start_time = session["start_time"]
    visualize = session.get("visualize", True)

    if current.lower() == target.lower():
        return redirect(url_for("win"))

    try:
        html, links = fetch_wiki_page(current)
    except Exception as e:
        html = f"<p>Error: {e}</p>"
        links = []

    clicks = len(path) - 1

    # Check if target is in available links
    target_available = target.lower() in [l.lower() for l in links]

    if visualize:
        # Full visual mode with Wikipedia content
        processed = process_wiki_html(html, links, target)
        content = f"""
        <div class="game-status">
            <div class="stat"><span>Target:</span> <span class="target">{target}</span></div>
            <div class="stat"><span>Clicks:</span> <span class="stat-value">{clicks}</span></div>
            <div class="stat"><span>Time:</span> <span class="stat-value" id="timer">0.0s</span></div>
            <a href="/give-up" class="btn btn-danger" style="margin-left:auto;">Give Up</a>
        </div>
        <div class="path">📍 {' → '.join(path)}</div>
        <div class="legend">
            <span class="available">Green = Click to navigate ({len(links)} links)</span>
            <span class="target">Gold = TARGET (click to win!)</span>
        </div>
        <div class="wiki-content">
            <h1>{current}</h1>
            {processed}
        </div>
        <div id="loading-overlay" class="loading-overlay" style="display:none;">
            <div class="spinner"></div>
            <div class="loading-text">Loading...</div>
        </div>
        <script>
        document.querySelectorAll('.wiki-content a.available, .wiki-content a.target-link').forEach(function(link) {{
            link.addEventListener('click', function() {{
                document.getElementById('loading-overlay').style.display = 'flex';
            }});
        }});
        </script>
        """
    else:
        # Minimal mode - just timer, path, and link dropdown
        sorted_links = sorted(links, key=str.lower)
        options = "".join(f'<option value="{urllib.parse.quote(l)}">{l}</option>' for l in sorted_links)

        target_btn = ""
        if target_available:
            target_btn = f'<a href="/navigate?title={urllib.parse.quote(target)}" class="btn" style="background:#f1c40f;color:#1a1a2e;font-weight:600;">🎯 Click to WIN: {target}</a>'

        content = f"""
        <div class="header">
            <h1>Wikipedia Speedrun</h1>
            <div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>
        </div>
        <div class="container">
            <div class="card" style="text-align:center;">
                <div style="display:flex;justify-content:center;gap:40px;margin-bottom:30px;">
                    <div>
                        <div style="font-size:3rem;font-weight:700;color:#4ecdc4;" id="timer">0.0s</div>
                        <div style="color:#666;">Time</div>
                    </div>
                    <div>
                        <div style="font-size:3rem;font-weight:700;color:#3498db;">{clicks}</div>
                        <div style="color:#666;">Clicks</div>
                    </div>
                </div>
                <div style="margin-bottom:20px;">
                    <span style="background:#f1c40f;color:#1a1a2e;padding:8px 20px;border-radius:6px;font-weight:600;font-size:1.2rem;">Target: {target}</span>
                </div>
            </div>
            <div class="card">
                <h2>Current: {current}</h2>
                <div style="background:#ecf0f1;padding:15px;border-radius:8px;margin-bottom:20px;font-size:14px;word-wrap:break-word;">
                    📍 {' → '.join(path)}
                </div>
                {target_btn}
                <div style="margin-top:20px;">
                    <label style="font-weight:600;margin-bottom:10px;display:block;">Choose next article ({len(links)} links available):</label>
                    <div style="display:flex;gap:10px;">
                        <select id="link-select" style="flex:1;">
                            <option value="">-- Select a link --</option>
                            {options}
                        </select>
                        <button onclick="navigateToSelected()" class="btn">Go</button>
                    </div>
                </div>
                <div style="margin-top:20px;text-align:center;">
                    <a href="/give-up" class="btn btn-danger">Give Up</a>
                </div>
            </div>
        </div>
        <div id="loading-overlay" class="loading-overlay" style="display:none;">
            <div class="spinner"></div>
            <div class="loading-text">Loading...</div>
        </div>
        <script>
        function navigateToSelected() {{
            const sel = document.getElementById('link-select');
            if (sel.value) {{
                document.getElementById('loading-overlay').style.display = 'flex';
                window.location.href = '/navigate?title=' + sel.value;
            }}
        }}
        document.getElementById('link-select').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') navigateToSelected();
        }});
        // Also show loading when clicking the win button
        const winBtn = document.querySelector('a[href*="/navigate"]');
        if (winBtn) {{
            winBtn.addEventListener('click', function() {{
                document.getElementById('loading-overlay').style.display = 'flex';
            }});
        }}
        </script>
        """
    return render_template_string(BASE_TEMPLATE, content=content, start_time=start_time)


@app.route("/navigate")
def navigate():
    if "current" not in session:
        return redirect(url_for("play_start"))
    title = request.args.get("title", "")
    if title:
        session["current"] = title
        session["path"] = session.get("path", []) + [title]
    return redirect(url_for("game"))


@app.route("/win")
def win():
    if "current" not in session:
        return redirect(url_for("play_start"))

    path = session["path"]
    clicks = len(path) - 1
    elapsed = time.time() - session["start_time"]
    session.clear()

    content = f"""
    <div class="header"><h1>Wikipedia Speedrun</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
    <div class="container">
        <div class="card win-screen">
            <h1>You Won!</h1>
            <p style="font-size:1.5rem;"><strong>{clicks}</strong> clicks in <strong>{elapsed:.1f}</strong> seconds</p>
            <div style="background:#ecf0f1;padding:20px;border-radius:8px;margin:20px 0;text-align:left;">
                <strong>Your path:</strong><br>{' → '.join(path)}
            </div>
            <a href="/play" class="btn">Play Again</a>
            <a href="/watch" class="btn btn-secondary" style="margin-left:10px;">Watch AI</a>
        </div>
    </div>
    """
    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


@app.route("/give-up")
def give_up():
    session.clear()
    return redirect(url_for("play_start"))


# ====================
# Routes - Watch AI
# ====================

@app.route("/watch")
def watch_start():
    model_status = ""
    if not is_model_ready():
        model_status = """
        <div class="ai-info" style="background:#fff3cd;border-color:#ffc107;">
            <span class="spinner" style="border-top-color:#f1c40f;"></span>
            <strong>Warming up model...</strong> The embedding model is loading. This may take a moment on first run.
        </div>
        """

    content = f"""
    <div class="header">
        <h1>Watch AI Play</h1>
        <div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>
    </div>
    <div class="container">
        <div class="card">
            <h2>Watch an AI Agent Play</h2>
            <p style="color:#666;margin-bottom:20px;">Watch a live embedding agent navigate Wikipedia in real-time!</p>
            {model_status}
            <form method="POST" action="/watch/start" id="watch-form">
                <div class="form-row">
                    <div><label>Start Article</label><input type="text" name="start" value="Cat"></div>
                    <div><label>Target Article</label><input type="text" name="target" value="Dog"></div>
                </div>
                <button type="submit" id="start-btn">Start Watching</button>
            </form>
        </div>
        <div class="card">
            <h2>About the AI Agent</h2>
            <p>The AI uses <strong>sentence-transformers</strong> to compute semantic similarity between article titles and the target.</p>
            <p>At each step, it chooses the link that is most similar to the target article's title.</p>
            <p style="margin-top:15px;"><strong>Model:</strong> all-MiniLM-L6-v2 (384 dimensions)</p>
        </div>
    </div>
    <div id="loading-overlay" class="loading-overlay" style="display:none;">
        <div class="spinner"></div>
        <div class="loading-text">Loading Wikipedia pages...</div>
    </div>
    <script>
        // Poll for model ready status
        function checkModelStatus() {{
            fetch('/watch/status')
                .then(r => r.json())
                .then(data => {{
                    if (data.ready) {{
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('start-btn').textContent = 'Start Watching';
                        const info = document.querySelector('.ai-info');
                        if (info) info.remove();
                    }} else {{
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('start-btn').textContent = 'Model warming up...';
                        setTimeout(checkModelStatus, 500);
                    }}
                }});
        }}
        checkModelStatus();

        document.getElementById('watch-form').addEventListener('submit', function() {{
            document.getElementById('loading-overlay').style.display = 'flex';
        }});
    </script>
    """
    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


@app.route("/watch/start", methods=["POST"])
def watch_start_game():
    start = request.form.get("start", "").strip()
    target = request.form.get("target", "").strip()

    if not start or not target:
        return redirect(url_for("watch_start"))

    # Wait for model to be ready
    if not is_model_ready():
        return render_template_string(BASE_TEMPLATE, content="""
            <div class="header"><h1>Watch AI</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
            <div class="container"><div class="card"><h2>Model Not Ready</h2><p>Please wait for the model to finish warming up.</p><a href="/watch" class="btn">Back</a></div></div>
        """, start_time=0)

    try:
        html, links = fetch_wiki_page(start)
        fetch_wiki_page(target)
    except Exception as e:
        return render_template_string(BASE_TEMPLATE, content=f"""
            <div class="header"><h1>Watch AI</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
            <div class="container"><div class="card"><h2>Error</h2><p>Could not find article: {e}</p><a href="/watch" class="btn">Back</a></div></div>
        """, start_time=0)

    game_id = f"{start}_{target}_{time.time()}"
    game = AIGame(
        start=start, target=target, path=[start], links=links,
        html=html, start_time=time.time(), running=True
    )
    _ai_games[game_id] = game

    session["ai_game_id"] = game_id

    return redirect(url_for("watch_game"))


@app.route("/watch/game")
def watch_game():
    game_id = session.get("ai_game_id")
    if not game_id or game_id not in _ai_games:
        return redirect(url_for("watch_start"))

    game = _ai_games[game_id]

    current = game.path[-1]
    clicks = len(game.path) - 1
    elapsed = time.time() - game.start_time

    if game.won:
        content = f"""
        <div class="header"><h1>AI Won!</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
        <div class="container">
            <div class="card win-screen">
                <h1>AI Won!</h1>
                <p style="font-size:1.5rem;"><strong>{clicks}</strong> clicks in <strong>{elapsed:.1f}</strong> seconds</p>
                <div style="background:#ecf0f1;padding:20px;border-radius:8px;margin:20px 0;text-align:left;">
                    <strong>AI's path:</strong><br>{' → '.join(game.path)}
                </div>
                <a href="/watch" class="btn">Watch Again</a>
                <a href="/play" class="btn btn-secondary" style="margin-left:10px;">Try It Yourself</a>
            </div>
        </div>
        """
        del _ai_games[game_id]
        return render_template_string(BASE_TEMPLATE, content=content, start_time=0)

    if game.error:
        content = f"""
        <div class="header"><h1>AI Error</h1><div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div></div>
        <div class="container"><div class="card"><h2>Error</h2><p>{game.error}</p><a href="/watch" class="btn">Try Again</a></div></div>
        """
        del _ai_games[game_id]
        return render_template_string(BASE_TEMPLATE, content=content, start_time=0)

    # Process HTML for display
    processed = process_wiki_html(game.html, game.links, game.target)

    ai_info = ""
    if game.last_choice:
        ai_info = f"""
        <div class="ai-info">
            <strong>AI's Analysis:</strong> Chose "<strong>{game.last_choice}</strong>" (similarity: {game.last_score:.3f})
        </div>
        """

    content = f"""
    <div class="header">
        <h1>Watch AI Play</h1>
        <div><a href="/">Dashboard</a><a href="/play">Play</a><a href="/watch">Watch AI</a></div>
    </div>
    <div class="game-status">
        <div class="stat"><span>AI Playing</span></div>
        <div class="stat"><span>Target:</span> <span class="target">{game.target}</span></div>
        <div class="stat"><span>Clicks:</span> <span class="stat-value">{clicks}</span></div>
        <div class="stat"><span>Time:</span> <span class="stat-value" id="timer">{elapsed:.1f}s</span></div>
        <a href="/watch/stop" class="btn btn-danger" style="margin-left:auto;">Stop</a>
    </div>
    <div class="path">{' -> '.join(game.path)}</div>
    {ai_info}
    <div class="legend">
        <span class="available">Green = Available links ({len(game.links)})</span>
        <span class="target">Gold = TARGET</span>
    </div>
    <div class="wiki-content">
        <h1>{current}</h1>
        {processed}
    </div>
    <style>
        .ai-chosen {{
            background: #e74c3c !important;
            color: white !important;
            outline: 3px solid #c0392b !important;
            animation: pulse 0.3s ease-in-out 3;
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
    </style>
    <script>
        const startTime = {game.start_time};

        // Update timer
        setInterval(() => {{
            const elapsed = (Date.now() / 1000) - startTime;
            document.getElementById('timer').textContent = elapsed.toFixed(1) + 's';
        }}, 100);

        // AI step: get choice, scroll to it, highlight, then advance
        fetch('/watch/step')
            .then(r => r.json())
            .then(data => {{
                if (data.done || data.error) {{
                    location.reload();
                    return;
                }}

                const choice = data.choice;
                if (!choice) {{
                    location.reload();
                    return;
                }}

                // Find the link in the page
                const hrefPart = choice.replace(/ /g, '_');
                const links = document.querySelectorAll('.wiki-content a.available, .wiki-content a.target-link');
                let targetLink = null;

                for (const link of links) {{
                    const href = link.getAttribute('href') || '';
                    if (href.includes(encodeURIComponent(choice)) || href.includes(hrefPart)) {{
                        targetLink = link;
                        break;
                    }}
                    // Also check link text
                    if (link.textContent.trim().toLowerCase() === choice.toLowerCase()) {{
                        targetLink = link;
                        break;
                    }}
                }}

                if (targetLink) {{
                    // Scroll to the link
                    targetLink.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                    // After scroll completes, highlight and click
                    setTimeout(() => {{
                        targetLink.classList.add('ai-chosen');

                        // After highlight animation, advance
                        setTimeout(() => {{
                            fetch('/watch/advance').then(() => location.reload());
                        }}, 800);
                    }}, 500);
                }} else {{
                    // Link not found visually, just advance
                    fetch('/watch/advance').then(() => location.reload());
                }}
            }})
            .catch(() => {{
                // On error, just reload
                setTimeout(() => location.reload(), 1000);
            }});
    </script>
    """
    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


@app.route("/watch/step")
def watch_step():
    """AI chooses a link, returns it for scroll animation."""
    game_id = session.get("ai_game_id")
    if not game_id or game_id not in _ai_games:
        return jsonify({"error": "no game"}), 404

    game = _ai_games[game_id]
    if game.won or game.error:
        return jsonify({"done": True})

    choice, score = ai_game_choose(game_id)
    return jsonify({"choice": choice, "score": score})


@app.route("/watch/advance")
def watch_advance():
    """Navigate to the pending choice after animation."""
    game_id = session.get("ai_game_id")
    if game_id and game_id in _ai_games:
        ai_game_advance(game_id)
    return "", 204


@app.route("/watch/stop")
def watch_stop():
    game_id = session.get("ai_game_id")
    if game_id and game_id in _ai_games:
        del _ai_games[game_id]
    session.pop("ai_game_id", None)
    return redirect(url_for("watch_start"))


@app.route("/watch/status")
def watch_status():
    """Return model warmup status."""
    return jsonify({"ready": is_model_ready(), "warming_up": _model_warming_up})


# ====================
# Main
# ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n=== Wikipedia Speedrun ===")
    print(f"Open http://localhost:{port} in your browser\n")

    # Start model warmup in background thread
    warmup_thread = threading.Thread(target=warmup_model, daemon=True)
    warmup_thread.start()

    app.run(host="0.0.0.0", port=port, debug=False)
