"""
Plotly chart components for dashboard.
"""

import plotly.graph_objects as go

from ui.components.data_loader import (
    get_agent_summaries,
    get_results,
    get_agent_cost_summaries,
    AgentSummary,
    AgentCostSummary,
    BenchmarkResult,
)


def create_win_rate_chart(summaries: list[AgentSummary]) -> go.Figure:
    """Bar chart of win rates."""
    sorted_summaries = sorted(summaries, key=lambda s: s.win_rate, reverse=True)[:10]
    colors = ["#2ecc71" if s.agent_type == "embedding" else "#3498db" for s in sorted_summaries]
    labels = [s.agent.replace("llm-", "").replace("live-", "")[:12] for s in sorted_summaries]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=[s.win_rate for s in sorted_summaries], marker_color=colors)
    ])

    fig.update_layout(
        title="Win Rate",
        yaxis_title="Win %",
        yaxis_range=[0, 105],
        showlegend=False,
        height=280,
        margin=dict(t=35, b=70, l=45, r=15),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
    return fig


def create_clicks_boxplot(results: list[BenchmarkResult]) -> go.Figure:
    """Box plot of clicks by agent."""
    wins = [r for r in results if r.won]

    agent_clicks: dict[str, list[int]] = {}
    for r in wins:
        name = r.agent.replace("llm-", "").replace("live-", "")[:12]
        if name not in agent_clicks:
            agent_clicks[name] = []
        agent_clicks[name].append(r.clicks)

    sorted_agents = sorted(agent_clicks.keys(), key=lambda a: sorted(agent_clicks[a])[len(agent_clicks[a])//2])[:8]

    fig = go.Figure()
    for agent in sorted_agents:
        fig.add_trace(go.Box(y=agent_clicks[agent], name=agent, marker_color="#3498db", boxmean=True))

    fig.update_layout(
        title="Clicks (Wins)",
        yaxis_title="Clicks",
        showlegend=False,
        height=280,
        margin=dict(t=35, b=70, l=45, r=15),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
    return fig


def create_time_scatter(summaries: list[AgentSummary]) -> go.Figure:
    """Scatter: clicks vs time."""
    fig = go.Figure()

    for s in summaries:
        color = "#2ecc71" if s.agent_type == "embedding" else "#3498db"
        name = s.agent.replace("llm-", "").replace("live-", "")[:10]

        fig.add_trace(go.Scatter(
            x=[s.avg_time],
            y=[s.avg_clicks],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.8),
            name=name,
            hovertemplate=f"<b>{name}</b><br>{s.avg_time:.1f}s, {s.avg_clicks:.1f} clicks<extra></extra>",
        ))

    fig.update_layout(
        title="Clicks vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Clicks",
        showlegend=False,
        height=280,
        margin=dict(t=35, b=45, l=45, r=15),
    )
    return fig


def create_difficulty_heatmap(results: list[BenchmarkResult], problems: list[dict]) -> go.Figure:
    """Heatmap: win rate by agent and difficulty."""
    problem_diff = {p["start"] + "→" + p["target"]: p.get("difficulty", "unknown") for p in problems}

    agent_diff_wins: dict[str, dict[str, list[bool]]] = {}
    for r in results:
        key = r.start + "→" + r.target
        diff = problem_diff.get(key, "unknown")
        if diff == "unknown":
            continue

        name = r.agent.replace("llm-", "").replace("live-", "")[:12]
        if name not in agent_diff_wins:
            agent_diff_wins[name] = {"easy": [], "medium": [], "hard": []}
        agent_diff_wins[name][diff].append(r.won)

    agents = sorted(agent_diff_wins.keys())[:8]
    difficulties = ["easy", "medium", "hard"]

    z = []
    for agent in agents:
        row = []
        for diff in difficulties:
            wins = agent_diff_wins[agent].get(diff, [])
            rate = sum(wins) / len(wins) * 100 if wins else 0
            row.append(rate)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=difficulties, y=agents,
        colorscale="RdYlGn", zmin=0, zmax=100,
        text=[[f"{v:.0f}" for v in row] for row in z],
        texttemplate="%{text}%",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Win% by Difficulty",
        height=280,
        margin=dict(t=35, b=35, l=100, r=15),
    )
    return fig


def create_pareto_chart(cost_summaries: list[AgentCostSummary]) -> go.Figure:
    """Create Pareto frontier chart: Cost vs Win Rate with step-line frontier."""
    fig = go.Figure()

    # Prepare data points
    points = []
    for s in cost_summaries:
        name = s.agent.replace("llm-", "").replace("live-", "")
        points.append({
            "name": name,
            "cost": s.cost_per_game,
            "win_rate": s.win_rate,
            "clicks": s.avg_clicks,
            "type": s.agent_type,
        })

    # Sort by cost for Pareto frontier calculation
    points.sort(key=lambda p: p["cost"])

    # Find Pareto frontier (non-dominated points)
    pareto_points = []
    max_win_rate_seen = -1
    for p in points:
        if p["win_rate"] > max_win_rate_seen:
            pareto_points.append(p)
            max_win_rate_seen = p["win_rate"]

    # Get max cost for extending the frontier line
    max_cost = max(p["cost"] for p in points) * 1.1

    # Draw Pareto frontier as filled step area (more visible)
    if pareto_points:
        # Build step coordinates: from each point, go right, then up to next
        step_x = []
        step_y = []
        for i, p in enumerate(pareto_points):
            step_x.append(p["cost"])
            step_y.append(p["win_rate"])
            if i < len(pareto_points) - 1:
                # Horizontal line to next point's x
                step_x.append(pareto_points[i + 1]["cost"])
                step_y.append(p["win_rate"])
        # Extend to max cost
        step_x.append(max_cost)
        step_y.append(pareto_points[-1]["win_rate"])

        # Draw filled area under frontier
        fig.add_trace(go.Scatter(
            x=step_x + [max_cost, 0],
            y=step_y + [80, 80],
            fill="toself",
            fillcolor="rgba(241, 196, 15, 0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Draw the frontier step line
        fig.add_trace(go.Scatter(
            x=step_x,
            y=step_y,
            mode="lines",
            line=dict(color="#f1c40f", width=3),
            name="Pareto Frontier",
            hoverinfo="skip",
        ))

    # Plot all points
    for p in points:
        color = "#2ecc71" if p["type"] == "embedding" else "#3498db"
        is_pareto = p in pareto_points

        fig.add_trace(go.Scatter(
            x=[p["cost"]],
            y=[p["win_rate"]],
            mode="markers+text" if is_pareto else "markers",
            marker=dict(
                size=16 if is_pareto else 10,
                color=color,
                opacity=1.0 if is_pareto else 0.6,
                line=dict(width=3, color="#f1c40f") if is_pareto else None,
                symbol="star" if is_pareto else "circle",
            ),
            text=[p["name"][:12]] if is_pareto else None,
            textposition="top right",
            textfont=dict(size=10, color="#333"),
            name=p["name"],
            hovertemplate=(
                f"<b>{p['name']}</b><br>"
                f"Cost: ${p['cost']:.4f}/game<br>"
                f"Win Rate: {p['win_rate']:.1f}%<br>"
                f"Avg Clicks: {p['clicks']:.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="Cost vs Win Rate (Pareto Frontier)",
        xaxis_title="Cost per Game ($)",
        yaxis_title="Win Rate (%)",
        yaxis_range=[80, 102],
        height=320,
        margin=dict(t=40, b=50, l=50, r=20),
        showlegend=True,
        legend=dict(x=0.7, y=0.15),
    )

    return fig


def create_leaderboard_table(cost_summaries: list[AgentCostSummary]) -> str:
    """Create HTML table for agent leaderboard."""
    # Sort by win rate, then by clicks
    sorted_summaries = sorted(cost_summaries, key=lambda s: (-s.win_rate, s.avg_clicks))

    rows = []
    for i, s in enumerate(sorted_summaries, 1):
        name = s.agent.replace("llm-", "").replace("live-", "")
        type_badge = (
            '<span style="background:#2ecc71;color:white;padding:2px 6px;border-radius:4px;font-size:11px;">EMB</span>'
            if s.agent_type == "embedding"
            else '<span style="background:#3498db;color:white;padding:2px 6px;border-radius:4px;font-size:11px;">LLM</span>'
        )
        cost_str = "FREE" if s.cost_per_game == 0 else f"${s.cost_per_game:.4f}"

        rows.append(f"""
            <tr>
                <td style="text-align:center;">{i}</td>
                <td>{name} {type_badge}</td>
                <td style="text-align:center;font-weight:600;">{s.win_rate:.1f}%</td>
                <td style="text-align:center;">{s.avg_clicks:.1f}</td>
                <td style="text-align:center;">{s.avg_time:.1f}s</td>
                <td style="text-align:right;">{cost_str}</td>
            </tr>
        """)

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">
                <th style="padding:10px;text-align:center;">#</th>
                <th style="padding:10px;text-align:left;">Agent</th>
                <th style="padding:10px;text-align:center;">Win Rate</th>
                <th style="padding:10px;text-align:center;">Avg Clicks</th>
                <th style="padding:10px;text-align:center;">Avg Time</th>
                <th style="padding:10px;text-align:right;">Cost/Game</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_failure_table(failures: list[dict]) -> str:
    """Create HTML table for failure analysis with difficulty scores."""
    if not failures:
        return "<p>No failures recorded.</p>"

    rows = []
    for f in failures[:10]:  # Top 10 hardest problems
        agents_str = ", ".join(
            a.replace("llm-", "").replace("live-", "")[:10]
            for a in f["failed_agents"][:3]
        )
        if len(f["failed_agents"]) > 3:
            agents_str += f" +{len(f['failed_agents']) - 3} more"

        # Difficulty badge color
        diff_score = f.get("difficulty_score", 0)
        diff_label = f.get("difficulty_label", "Unknown")
        avg_clicks = f.get("avg_clicks", 0)

        if diff_label == "Very Hard":
            badge_color = "#e74c3c"
        elif diff_label == "Hard":
            badge_color = "#f39c12"
        elif diff_label == "Medium":
            badge_color = "#3498db"
        else:
            badge_color = "#2ecc71"

        rows.append(f"""
            <tr>
                <td style="padding:8px;">{f['start']}</td>
                <td style="padding:8px;">{f['target']}</td>
                <td style="padding:8px;text-align:center;">
                    <span style="background:{badge_color};color:white;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;">{diff_label}</span>
                </td>
                <td style="padding:8px;text-align:center;font-weight:600;">{diff_score:.0f}</td>
                <td style="padding:8px;text-align:center;">{avg_clicks:.1f}</td>
                <td style="padding:8px;text-align:center;color:#e74c3c;font-weight:600;">{f['failure_count']}</td>
            </tr>
        """)

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">
                <th style="padding:10px;text-align:left;">Start</th>
                <th style="padding:10px;text-align:left;">Target</th>
                <th style="padding:10px;text-align:center;">Difficulty</th>
                <th style="padding:10px;text-align:center;">Score</th>
                <th style="padding:10px;text-align:center;">Avg Clicks</th>
                <th style="padding:10px;text-align:center;">Failures</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
