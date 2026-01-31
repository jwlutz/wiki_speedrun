"""
Play - Wikipedia Speedrun game.
"""

import time
import urllib.parse
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Play", page_icon="🎮", layout="wide")

# Initialize session state
defaults = {
    "game_active": False,
    "current_title": "",
    "target_title": "",
    "path": [],
    "start_time": None,
    "available_links": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def validate_article(title: str) -> tuple[bool, str, str]:
    from src.wikipedia.scraper import WikiScraper
    scraper = WikiScraper(rate_limit=0.0)
    try:
        page = scraper.get_page(title)
        return True, page.title, ""
    except Exception as e:
        return False, "", str(e)


def start_game(start: str, target: str):
    st.session_state.game_active = True
    st.session_state.current_title = start
    st.session_state.target_title = target
    st.session_state.path = [start]
    st.session_state.start_time = time.time()
    st.session_state.available_links = []


def end_game():
    st.session_state.game_active = False
    st.session_state.path = []
    st.session_state.start_time = None
    st.session_state.available_links = []


@st.cache_data(ttl=300)
def fetch_wiki_html(title: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded}"
    headers = {"User-Agent": "WikiSpeedrun/1.0 (educational)"}

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for sel in [".navbox", ".infobox", ".sidebar", ".reference", ".reflist",
                ".mw-editsection", "figure", ".thumb", ".ambox", ".mbox", ".toc"]:
        for el in soup.select(sel):
            el.decompose()

    return str(soup)


def render_wiki_page(title: str, available_links: list[str], target: str, clicks: int, elapsed: float):
    """Render Wikipedia page with highlighted links (display only)."""
    try:
        html = fetch_wiki_html(title)
    except Exception as e:
        st.error(f"Could not load article: {e}")
        return

    available_lower = {l.lower() for l in available_links}
    available_under = {l.lower().replace(" ", "_") for l in available_links}
    target_lower = target.lower()
    target_under = target.lower().replace(" ", "_")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.startswith("./"):
            a["class"] = a.get("class", []) + ["disabled"]
            a.attrs.pop("href", None)
            continue

        path = href.replace("./", "").split("#")[0].split("?")[0]
        decoded = urllib.parse.unquote(path).replace("_", " ")
        norm = decoded.lower()
        norm_under = norm.replace(" ", "_")

        is_target = norm == target_lower or norm_under == target_under
        is_available = norm in available_lower or norm_under in available_under

        if is_target:
            a["class"] = a.get("class", []) + ["target-link"]
        elif is_available:
            a["class"] = a.get("class", []) + ["available-link"]
        else:
            a["class"] = a.get("class", []) + ["disabled"]
        a.attrs.pop("href", None)

    full_html = f'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; font-size: 15px; line-height: 1.6; margin: 0; padding: 0; background: #fff; color: #222; }}
.status {{ position: sticky; top: 0; background: #f8f9fa; padding: 12px 15px; border-bottom: 2px solid #dee2e6; display: flex; gap: 30px; font-size: 15px; z-index: 1000; }}
.target-badge {{ background: #fff3cd; color: #664d03; padding: 2px 10px; border-radius: 4px; font-weight: 600; }}
.content {{ padding: 15px; }}
a {{ cursor: default; text-decoration: none; }}
a.available-link {{ background: #d1e7dd; border-radius: 3px; padding: 1px 5px; color: #0f5132 !important; }}
a.target-link {{ background: linear-gradient(90deg, #fff3cd, #ffe69c); border: 2px solid #ffca2c; border-radius: 4px; padding: 2px 8px; color: #664d03 !important; font-weight: 600; animation: glow 1.5s ease-in-out infinite; }}
@keyframes glow {{ 0%,100% {{ box-shadow: 0 0 5px #ffca2c; }} 50% {{ box-shadow: 0 0 20px #ffc107; }} }}
a.disabled {{ color: #ccc !important; }}
img {{ max-width: 200px; height: auto; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 10px; }}
.legend {{ background: #fff; padding: 8px 15px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; display: flex; gap: 20px; font-size: 13px; }}
.leg {{ padding: 3px 10px; border-radius: 4px; }}
.leg-a {{ background: #d1e7dd; color: #0f5132; }}
.leg-t {{ background: #fff3cd; border: 1px solid #ffca2c; color: #664d03; }}
</style>
</head>
<body>
<div class="status">
    <div><b>Target:</b> <span class="target-badge">{target}</span></div>
    <div><b>Clicks:</b> {clicks}</div>
    <div><b>Time:</b> {elapsed:.1f}s</div>
</div>
<div class="content">
    <div class="legend">
        <span class="leg leg-a">Green = Available ({len(available_links)} links)</span>
        <span class="leg leg-t">Gold = TARGET</span>
    </div>
    {soup}
</div>
</body>
</html>
'''
    components.html(full_html, height=500, scrolling=True)


st.title("Play")

# GAME SETUP
if not st.session_state.game_active:
    col1, col2 = st.columns(2)
    with col1:
        start_input = st.text_input("Start:", value="Emperor Penguin", key="start")
    with col2:
        target_input = st.text_input("Target:", value="iPhone", key="target")

    if st.button("Start Game", type="primary", use_container_width=True):
        with st.spinner("Loading..."):
            ok1, s, e1 = validate_article(start_input.strip())
            ok2, t, e2 = validate_article(target_input.strip())
            if not ok1:
                st.error(f"Start not found: {e1}")
            elif not ok2:
                st.error(f"Target not found: {e2}")
            else:
                start_game(s, t)
                st.rerun()

# ACTIVE GAME
else:
    won = st.session_state.current_title.lower() == st.session_state.target_title.lower()

    if won:
        clicks = len(st.session_state.path) - 1
        elapsed = time.time() - st.session_state.start_time
        st.balloons()
        st.success(f"**Won in {clicks} clicks!** ({elapsed:.1f}s)")
        st.write(" → ".join(st.session_state.path))
        if st.button("Play Again", type="primary"):
            end_game()
            st.rerun()
    else:
        # Load links if needed
        if not st.session_state.available_links:
            from src.wikipedia.scraper import WikiScraper
            scraper = WikiScraper(rate_limit=0.0)
            try:
                st.session_state.available_links = scraper.get_links(st.session_state.current_title)
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.available_links:
            clicks = len(st.session_state.path) - 1
            elapsed = time.time() - st.session_state.start_time

            # Navigation controls at top
            st.markdown(f"**Current:** {st.session_state.current_title} → **Target:** {st.session_state.target_title}")

            if len(st.session_state.path) > 1:
                st.caption(" → ".join(st.session_state.path))

            # Searchable link selector
            col1, col2 = st.columns([4, 1])
            with col1:
                # Sort links, put target at top if available
                sorted_links = sorted(st.session_state.available_links)
                target_lower = st.session_state.target_title.lower()

                # Check if target is in available links
                target_in_links = None
                for link in sorted_links:
                    if link.lower() == target_lower:
                        target_in_links = link
                        break

                if target_in_links:
                    sorted_links.remove(target_in_links)
                    sorted_links = [f"⭐ {target_in_links} (TARGET)"] + sorted_links

                selected = st.selectbox(
                    "Choose link to follow:",
                    [""] + sorted_links,
                    key="link_select",
                    label_visibility="collapsed",
                    placeholder="Type to search links..."
                )

            with col2:
                go_clicked = st.button("Go", type="primary", use_container_width=True)

            if go_clicked and selected:
                # Clean up selection (remove target marker if present)
                clean_selected = selected.replace("⭐ ", "").replace(" (TARGET)", "")
                st.session_state.path.append(clean_selected)
                st.session_state.current_title = clean_selected
                st.session_state.available_links = []
                st.rerun()

            # Show Wikipedia content below
            st.divider()
            render_wiki_page(
                st.session_state.current_title,
                st.session_state.available_links,
                st.session_state.target_title,
                clicks,
                elapsed
            )

        if st.button("Give Up"):
            end_game()
            st.rerun()
