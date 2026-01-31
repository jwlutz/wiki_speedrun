"""
Dashboard - Benchmark results and visualizations.
"""

import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("Dashboard")

try:
    from ui.components.data_loader import get_agent_summaries, get_results, get_problems
    from ui.components.charts import (
        create_win_rate_chart,
        create_clicks_boxplot,
        create_time_scatter,
        create_difficulty_heatmap,
    )

    summaries = get_agent_summaries()
    results = get_results()
    problems = get_problems()

    if not summaries:
        st.warning("No results. Run: `python scripts/comprehensive_benchmark.py`")
        st.stop()

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", len(summaries))
    c2.metric("Games", len(results))
    c3.metric("Wins", sum(s.wins for s in summaries))
    c4.metric("Avg Win%", f"{sum(s.win_rate for s in summaries) / len(summaries):.0f}")

    st.divider()

    # Leaderboard
    rows = []
    for i, s in enumerate(summaries[:8]):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
        rows.append({
            "#": f"{medal}{i+1}",
            "Agent": s.agent.replace("llm-", "").replace("live-", "")[:20],
            "Win%": f"{s.win_rate:.0f}",
            "Clicks": f"{s.avg_clicks:.1f}",
            "Time": f"{s.avg_time:.0f}s",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=180)

    st.divider()

    # Charts - 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_win_rate_chart(summaries), use_container_width=True)

    with col2:
        st.plotly_chart(create_clicks_boxplot(results), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_time_scatter(summaries), use_container_width=True)

    with col2:
        if problems:
            st.plotly_chart(create_difficulty_heatmap(results, problems), use_container_width=True)

except ImportError as e:
    st.error(f"Missing: {e}")
except Exception as e:
    st.error(f"Error: {e}")
