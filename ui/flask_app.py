"""
Flask-based Wikipedia Speedrun game with clickable inline links.
"""

import time
import urllib.parse
from flask import Flask, render_template_string, request, session, redirect, url_for
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = "wiki-speedrun-secret-key-change-in-production"

# HTML Templates
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
        .header a { color: #88c0d0; text-decoration: none; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 20px; }
        h2 { margin-bottom: 20px; color: #1a1a2e; }
        input[type="text"] { width: 100%; padding: 12px 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; margin-bottom: 15px; }
        input[type="text"]:focus { outline: none; border-color: #4ecdc4; }
        button, .btn { background: #4ecdc4; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #45b7aa; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .game-status { background: #1a1a2e; color: white; padding: 15px 30px; position: sticky; top: 0; z-index: 100; display: flex; gap: 40px; align-items: center; flex-wrap: wrap; }
        .game-status .target { background: #f1c40f; color: #1a1a2e; padding: 5px 15px; border-radius: 6px; font-weight: 600; }
        .game-status .stat { display: flex; align-items: center; gap: 8px; }
        .game-status .stat-value { font-size: 1.3rem; font-weight: 600; color: #4ecdc4; }
        .path { background: #ecf0f1; padding: 10px 20px; font-size: 14px; color: #666; border-bottom: 1px solid #ddd; }
        .wiki-content { background: white; padding: 30px; line-height: 1.8; }
        .wiki-content h1, .wiki-content h2, .wiki-content h3 { margin: 20px 0 10px; color: #1a1a2e; }
        .wiki-content p { margin-bottom: 15px; }
        .wiki-content a { text-decoration: none; }
        .wiki-content a.available { background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px; transition: all 0.2s; }
        .wiki-content a.available:hover { background: #28a745; color: white; }
        .wiki-content a.target-link { background: linear-gradient(90deg, #fff3cd, #ffeaa7); border: 2px solid #f1c40f; padding: 3px 10px; border-radius: 6px; color: #856404; font-weight: 600; animation: glow 1.5s ease-in-out infinite; }
        @keyframes glow { 0%,100% { box-shadow: 0 0 5px #f1c40f; } 50% { box-shadow: 0 0 20px #f1c40f, 0 0 30px #f39c12; } }
        .wiki-content a.disabled { color: #aaa; pointer-events: none; }
        .wiki-content img { max-width: 300px; height: auto; }
        .wiki-content table { border-collapse: collapse; margin: 15px 0; }
        .wiki-content th, .wiki-content td { border: 1px solid #ddd; padding: 8px 12px; }
        .legend { display: flex; gap: 20px; padding: 15px 30px; background: #f8f9fa; border-bottom: 1px solid #ddd; font-size: 14px; }
        .legend span { padding: 4px 12px; border-radius: 4px; }
        .legend .leg-available { background: #d4edda; color: #155724; }
        .legend .leg-target { background: #fff3cd; border: 1px solid #f1c40f; color: #856404; }
        .win-screen { text-align: center; padding: 60px 30px; }
        .win-screen h1 { font-size: 3rem; color: #27ae60; margin-bottom: 20px; }
        .win-screen .stats { font-size: 1.5rem; margin: 30px 0; }
        .win-screen .path-display { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: left; }
        .form-row { display: flex; gap: 20px; margin-bottom: 20px; }
        .form-row > div { flex: 1; }
        .form-row label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
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

PLAY_CONTENT = """
<div class="header">
    <h1>Wikipedia Speedrun</h1>
    <a href="/">Dashboard</a>
</div>
<div class="container">
    <div class="card">
        <h2>Start a New Game</h2>
        <p style="color:#666;margin-bottom:20px;">Navigate from one Wikipedia article to another using only the links on each page. Can you beat the AI?</p>
        <form method="POST" action="/start">
            <div class="form-row">
                <div>
                    <label>Start Article</label>
                    <input type="text" name="start" id="start" value="Emperor Penguin" placeholder="Starting Wikipedia article">
                </div>
                <div>
                    <label>Target Article</label>
                    <input type="text" name="target" id="target" value="iPhone" placeholder="Target Wikipedia article">
                </div>
            </div>
            <button type="submit">Start Game</button>
        </form>
    </div>
    <div class="card">
        <h2>Example Challenges</h2>
        <p style="color:#666;margin-bottom:15px;">Try these curated challenges at different difficulty levels:</p>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:15px;">
            <button type="button" onclick="setGame('Cat','Dog')" class="preset-btn" style="background:#2ecc71;">
                <strong>Easy:</strong> Cat → Dog
            </button>
            <button type="button" onclick="setGame('Coffee','Minecraft')" class="preset-btn" style="background:#3498db;">
                <strong>Medium:</strong> Coffee → Minecraft
            </button>
            <button type="button" onclick="setGame('Lamprey','Costco hot dog')" class="preset-btn" style="background:#e67e22;">
                <strong>Hard:</strong> Lamprey → Costco Hot Dog
            </button>
            <button type="button" onclick="setGame('Emperor Penguin','iPhone')" class="preset-btn" style="background:#9b59b6;">
                <strong>Default:</strong> Emperor Penguin → iPhone
            </button>
        </div>
    </div>
</div>
<style>
.preset-btn { color:white; border:none; padding:15px 20px; border-radius:8px; cursor:pointer; text-align:left; transition:opacity 0.2s; }
.preset-btn:hover { opacity:0.85; }
</style>
<script>
function setGame(start, target) {
    document.getElementById('start').value = start;
    document.getElementById('target').value = target;
}
</script>
"""

# Simple cache for Wikipedia pages
_wiki_cache = {}

def fetch_wiki_html(title: str) -> tuple[str, list[str]]:
    """Fetch Wikipedia article HTML and extract available links."""
    # Check cache first
    if title in _wiki_cache:
        return _wiki_cache[title]

    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded}"
    headers = {"User-Agent": "WikiSpeedrun/1.0 (educational)"}

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove base tag (causes links to go to Wikipedia instead of our app)
    for base in soup.find_all("base"):
        base.decompose()

    # Remove unwanted elements
    for sel in [".navbox", ".infobox", ".sidebar", ".reference", ".reflist",
                ".mw-editsection", "figure", ".thumb", ".ambox", ".mbox", ".toc",
                ".hatnote", ".shortdescription", "style", "script"]:
        for el in soup.select(sel):
            el.decompose()

    # Extract available links
    available_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if href.startswith("./") and ":" not in href:
            path = href.replace("./", "").split("#")[0].split("?")[0]
            decoded = urllib.parse.unquote(path).replace("_", " ")
            if decoded and decoded not in available_links:
                available_links.append(decoded)

    result = (str(soup), available_links)
    _wiki_cache[title] = result  # Cache the result
    return result


def process_wiki_html(html: str, available_links: list[str], target: str) -> str:
    """Process Wikipedia HTML to add link styling and navigation."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove any base tags that might redirect links
    for base in soup.find_all("base"):
        base.decompose()

    available_lower = {l.lower() for l in available_links}
    available_map = {l.lower(): l for l in available_links}
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

        is_target = norm == target_lower
        is_available = norm in available_lower

        if is_target:
            a["class"] = a.get("class", []) + ["target-link"]
            # Use absolute URL to ensure it goes to our app
            a["href"] = f"/navigate?title={urllib.parse.quote(decoded)}"
        elif is_available:
            a["class"] = a.get("class", []) + ["available"]
            link_title = available_map.get(norm, decoded)
            a["href"] = f"/navigate?title={urllib.parse.quote(link_title)}"
        else:
            a["class"] = a.get("class", []) + ["disabled"]
            if "href" in a.attrs:
                del a["href"]

    return str(soup)


@app.route("/play")
def play_start():
    """Show game start page with difficulty presets."""
    return render_template_string(BASE_TEMPLATE, content=PLAY_CONTENT, start_time=0)


@app.route("/start", methods=["POST"])
def start_game():
    start = request.form.get("start", "").strip()
    target = request.form.get("target", "").strip()

    if not start or not target:
        return redirect(url_for("play_start"))

    # Validate articles exist
    try:
        fetch_wiki_html(start)
        fetch_wiki_html(target)
    except Exception as e:
        return render_template_string(BASE_TEMPLATE, content=f"""
            <div class="header"><h1>Wikipedia Speedrun</h1><a href="/">Dashboard</a></div>
            <div class="container">
                <div class="card">
                    <h2>Error</h2>
                    <p>Could not find article: {e}</p>
                    <a href="/play" class="btn">Back</a>
                </div>
            </div>
        """, start_time=0)

    session["start"] = start
    session["target"] = target
    session["current"] = start
    session["path"] = [start]
    session["start_time"] = time.time()

    return redirect(url_for("game"))


@app.route("/game")
def game():
    """Active game page - shows Wikipedia content with navigation."""
    if "current" not in session:
        return redirect(url_for("play_start"))

    current = session["current"]
    target = session["target"]
    path = session["path"]
    start_time = session["start_time"]

    # Check for win
    if current.lower() == target.lower():
        return redirect(url_for("win"))

    try:
        html, available_links = fetch_wiki_html(current)
        processed_html = process_wiki_html(html, available_links, target)
    except Exception as e:
        processed_html = f"<p>Error loading article: {e}</p>"
        available_links = []

    clicks = len(path) - 1

    content = f"""
    <div class="game-status">
        <div class="stat"><span>Target:</span> <span class="target">{target}</span></div>
        <div class="stat"><span>Clicks:</span> <span class="stat-value">{clicks}</span></div>
        <div class="stat"><span>Time:</span> <span class="stat-value" id="timer">0.0s</span></div>
        <a href="/give-up" class="btn btn-danger" style="margin-left: auto;">Give Up</a>
    </div>
    <div class="path">📍 {' → '.join(path)}</div>
    <div class="legend">
        <span class="leg-available">Green = Click to navigate ({len(available_links)} links)</span>
        <span class="leg-target">Gold = TARGET (click to win!)</span>
    </div>
    <div class="wiki-content">
        <h1>{current}</h1>
        {processed_html}
    </div>
    """

    return render_template_string(BASE_TEMPLATE, content=content, start_time=start_time)


@app.route("/navigate")
def navigate():
    if "current" not in session:
        return redirect(url_for("play_start"))

    title = request.args.get("title", "")
    if not title:
        return redirect(url_for("game"))

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

    content = f"""
    <div class="header"><h1>Wikipedia Speedrun</h1><a href="/">Dashboard</a></div>
    <div class="container">
        <div class="card win-screen">
            <h1>You Won!</h1>
            <div class="stats">
                <p><strong>{clicks}</strong> clicks in <strong>{elapsed:.1f}</strong> seconds</p>
            </div>
            <div class="path-display">
                <strong>Your path:</strong><br>
                {' → '.join(path)}
            </div>
            <a href="/play" class="btn">Play Again</a>
        </div>
    </div>
    """

    # Clear session
    session.clear()

    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


@app.route("/give-up")
def give_up():
    session.clear()
    return redirect(url_for("play_start"))


@app.route("/")
def home():
    """Dashboard homepage showing benchmark results."""
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

        # Generate charts as HTML
        # Pareto is rendered first in HTML, so it needs the Plotly CDN
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
            <a href="/play">Try It Yourself</a>
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
                <h2>Cost vs Performance (Pareto Frontier)</h2>
                <p style="color:#666;margin-bottom:15px;">Points on the yellow frontier represent optimal cost-performance tradeoffs. Green = embedding models (free), Blue = LLM models.</p>
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
                <a href="/play" class="btn" style="background:white;color:#667eea;font-weight:600;">Play Now</a>
            </div>
        </div>
        """

    except Exception as e:
        content = f"""
        <div class="header">
            <h1>Wikipedia Speedrun</h1>
            <a href="/play">Play</a>
        </div>
        <div class="container">
            <div class="card">
                <h2>Error Loading Dashboard</h2>
                <p style="color:#e74c3c;">{e}</p>
                <p>Make sure benchmark data exists in <code>data/benchmark_results.json</code></p>
            </div>
        </div>
        """

    return render_template_string(BASE_TEMPLATE, content=content, start_time=0)


if __name__ == "__main__":
    print("\n=== Wikipedia Speedrun ===")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
