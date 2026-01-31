"""
Wikipedia page viewer component for Streamlit.

Fetches Wikipedia HTML, highlights available links, and renders inline.
"""

import urllib.parse

import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup


def fetch_wikipedia_html(title: str) -> tuple[str, str]:
    """
    Fetch Wikipedia article HTML content using REST API.

    Returns:
        (html_content, canonical_title)
    """
    # Use Wikipedia REST API (more reliable than action API)
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded_title}"

    headers = {
        "User-Agent": "WikiSpeedrun/1.0 (educational; contact@example.com)",
        "Accept": "text/html; charset=utf-8",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    return response.text, title


def process_wikipedia_html(
    html: str,
    available_links: list[str],
    target_title: str,
) -> str:
    """Process Wikipedia HTML to highlight links."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for selector in [
        ".navbox", ".vertical-navbox", ".sidebar", ".infobox",
        ".mw-editsection", ".reference", ".reflist", ".references",
        ".toc", ".mbox-small", ".ambox", ".dmbox",
        ".portal", ".sistersitebox", ".noprint", "figure",
    ]:
        for el in soup.select(selector):
            el.decompose()

    # Build lookup sets
    available_set = {link.lower().replace(" ", "_") for link in available_links}
    available_set.update({link.lower() for link in available_links})
    target_normalized = target_title.lower().replace(" ", "_")

    # Process all links
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")

        if not href.startswith("./"):
            link["class"] = link.get("class", []) + ["external-link"]
            continue

        path = href.replace("./", "").split("#")[0].split("?")[0]
        try:
            decoded = urllib.parse.unquote(path)
        except Exception:
            decoded = path

        normalized = decoded.lower().replace(" ", "_")
        title_clean = decoded.replace("_", " ")

        if normalized == target_normalized or decoded.lower() == target_title.lower():
            link["class"] = link.get("class", []) + ["wiki-target-link"]
            link["data-wiki-title"] = title_clean
            link["onclick"] = f"window.parent.postMessage({{type:'wiki-click',title:'{title_clean}'}}, '*'); return false;"
            link["href"] = "#"
        elif normalized in available_set or decoded.lower() in available_set:
            link["class"] = link.get("class", []) + ["wiki-available-link"]
            link["data-wiki-title"] = title_clean
            link["onclick"] = f"window.parent.postMessage({{type:'wiki-click',title:'{title_clean}'}}, '*'); return false;"
            link["href"] = "#"
        else:
            link["class"] = link.get("class", []) + ["wiki-disabled-link"]
            link["onclick"] = "return false;"
            link["href"] = "#"

    return str(soup)


def render_wikipedia_page(
    title: str,
    available_links: list[str],
    target_title: str,
    height: int = 600,
) -> None:
    """Render a Wikipedia page with highlighted links."""
    try:
        html, display_title = fetch_wikipedia_html(title)
    except Exception as e:
        st.error(f"Failed to load page: {e}")
        return

    processed_html = process_wikipedia_html(html, available_links, target_title)

    full_html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <base href="https://en.wikipedia.org/wiki/">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            padding: 15px;
            background: white;
            color: #202122;
        }}
        .wiki-available-link {{
            background: rgba(46, 204, 113, 0.25) !important;
            border-bottom: 2px solid #2ecc71 !important;
            color: #1a7f37 !important;
            text-decoration: none !important;
            padding: 1px 3px;
            border-radius: 2px;
            cursor: pointer;
        }}
        .wiki-available-link:hover {{
            background: rgba(46, 204, 113, 0.5) !important;
        }}
        .wiki-target-link {{
            background: rgba(255, 215, 0, 0.5) !important;
            border: 2px solid gold !important;
            color: #856404 !important;
            text-decoration: none !important;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.4); }}
            50% {{ box-shadow: 0 0 0 6px rgba(255, 215, 0, 0); }}
        }}
        .wiki-disabled-link, .external-link {{
            color: #aaa !important;
            text-decoration: none !important;
            pointer-events: none;
        }}
        img {{ max-width: 180px; height: auto; }}
        table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #ddd; padding: 4px 8px; }}
        .legend {{
            position: sticky;
            top: 0;
            background: white;
            padding: 8px 15px;
            margin: -15px -15px 15px -15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            gap: 15px;
            font-size: 12px;
            z-index: 100;
        }}
        .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
        .dot.green {{ background: rgba(46, 204, 113, 0.4); border: 1px solid #2ecc71; }}
        .dot.gold {{ background: rgba(255, 215, 0, 0.5); border: 1px solid gold; }}
    </style>
</head>
<body>
    <div class="legend">
        <span><span class="dot green"></span> Clickable ({len(available_links)})</span>
        <span><span class="dot gold"></span> TARGET</span>
    </div>
    {processed_html}
</body>
</html>
'''

    components.html(full_html, height=height, scrolling=True)
