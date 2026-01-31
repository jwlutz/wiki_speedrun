"""
Wikipedia Speedrun Benchmark
"""

import streamlit as st

st.set_page_config(page_title="Wiki Speedrun", page_icon="🔗", layout="wide")

st.title("Wikipedia Speedrun")
st.caption("Navigate Wikipedia using only links. Race AI agents.")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

with col2:
    if st.button("🎮 Play", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Play.py")

st.divider()

try:
    from ui.components.data_loader import get_agent_summaries, get_results

    summaries = get_agent_summaries()
    results = get_results()

    if summaries:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Models", len(summaries))
        c2.metric("Games", len(results))
        best = max(summaries, key=lambda s: s.win_rate)
        c3.metric("Best", f"{best.win_rate:.0f}%", best.agent.replace("llm-", "").replace("live-", ""))
        if any(s.wins > 0 for s in summaries):
            efficient = min([s for s in summaries if s.wins > 0], key=lambda s: s.avg_clicks)
            c4.metric("Fastest", f"{efficient.avg_clicks:.1f}", efficient.agent.replace("llm-", "").replace("live-", ""))
except Exception:
    pass
