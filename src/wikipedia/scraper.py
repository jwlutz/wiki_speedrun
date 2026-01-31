"""
Wikipedia scraper for extracting links from live Wikipedia pages.

Uses requests + BeautifulSoup for fast link extraction,
or Playwright when visualization is needed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote, unquote, urljoin

import requests
from bs4 import BeautifulSoup

from src.config import (
    USER_AGENT,
    WIKIPEDIA_BASE_URL,
    WIKIPEDIA_REQUEST_DELAY,
    WIKIPEDIA_TIMEOUT,
)

logger = logging.getLogger(__name__)


@dataclass
class WikiPage:
    """
    Represents a scraped Wikipedia page.

    Attributes:
        title: The article title (from page heading)
        url: Full URL of the page
        links: List of article titles this page links to (main content only)
        html: Raw HTML content (optional, for debugging)
    """

    title: str
    url: str
    links: list[str]
    html: str | None = None


class WikiScraper:
    """
    Scrapes Wikipedia pages to extract article links.

    Only extracts links from the main article content, excluding:
    - Navigation/sidebar links
    - External links
    - Reference/citation links
    - Category links
    - File/image links
    """

    # Pattern for valid Wikipedia article URLs
    ARTICLE_PATTERN = re.compile(r"^/wiki/([^:#?]+)")

    # Namespaces to exclude (Wikipedia:, Help:, etc.)
    # Note: Use spaces not underscores - titles get underscores converted to spaces
    EXCLUDED_PREFIXES = {
        "Wikipedia:",
        "Help:",
        "Template:",
        "Template talk:",
        "Category:",
        "Portal:",
        "File:",
        "Special:",
        "Talk:",
        "User:",
        "User talk:",
        "Module:",
        "MediaWiki:",
        "Draft:",
        "MOS:",  # Manual of Style shortcuts
        "WP:",   # Wikipedia shortcuts
    }

    def __init__(self, rate_limit: float = WIKIPEDIA_REQUEST_DELAY) -> None:
        """
        Initialize the scraper.

        Args:
            rate_limit: Minimum seconds between requests
        """
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._rate_limit = rate_limit
        self._last_request_time: float = 0

    def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def title_to_url(self, title: str) -> str:
        """Convert article title to Wikipedia URL."""
        # Replace spaces with underscores, then URL-encode special chars
        url_title = title.replace(" ", "_")
        # Encode special chars but keep underscores and slashes (for subpages)
        url_title = quote(url_title, safe="_/")
        return urljoin(WIKIPEDIA_BASE_URL, url_title)

    def url_to_title(self, url: str) -> str | None:
        """Extract article title from Wikipedia URL."""
        # Handle both full URLs and /wiki/ paths
        if "/wiki/" in url:
            path = url.split("/wiki/")[-1]
            # Remove any query parameters or anchors
            path = path.split("?")[0].split("#")[0]
            # Decode URL encoding
            title = unquote(path).replace("_", " ")
            return title
        return None

    def _is_valid_article_link(self, href: str | None) -> bool:
        """Check if a link points to a valid Wikipedia article."""
        if not href:
            return False

        match = self.ARTICLE_PATTERN.match(href)
        if not match:
            return False

        # Get the article title part
        title = unquote(match.group(1)).replace("_", " ")

        # Check for excluded namespaces
        for prefix in self.EXCLUDED_PREFIXES:
            if title.startswith(prefix):
                return False

        return True

    def _extract_title_from_href(self, href: str) -> str:
        """Extract clean article title from href."""
        match = self.ARTICLE_PATTERN.match(href)
        if match:
            return unquote(match.group(1)).replace("_", " ")
        return ""

    def get_page(self, title: str) -> WikiPage:
        """
        Fetch and parse a Wikipedia page.

        Args:
            title: Article title to fetch

        Returns:
            WikiPage with extracted links

        Raises:
            requests.RequestException: If fetch fails
        """
        self._wait_for_rate_limit()

        url = self.title_to_url(title)
        logger.debug(f"Fetching: {url}")

        response = self._session.get(url, timeout=WIKIPEDIA_TIMEOUT)
        response.raise_for_status()
        self._last_request_time = time.time()

        return self._parse_page(response.text, url)

    def _parse_page(self, html: str, url: str) -> WikiPage:
        """Parse HTML and extract article links."""
        soup = BeautifulSoup(html, "lxml")

        # Get the actual page title from the heading
        title_elem = soup.find("h1", {"id": "firstHeading"})
        title = title_elem.get_text() if title_elem else self.url_to_title(url) or ""

        # Find the main content div
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            logger.warning(f"No content found for {url}")
            return WikiPage(title=title, url=url, links=[])

        # Find the parser output div (actual article content)
        parser_output = content.find("div", {"class": "mw-parser-output"})
        if not parser_output:
            parser_output = content

        # Extract unique links from paragraphs, lists, tables in main content
        # Exclude infoboxes, navboxes, reference sections
        links: list[str] = []
        seen: set[str] = set()

        # Elements to extract links from
        for elem in parser_output.find_all(["p", "li", "td", "th", "dd"]):
            # Skip if inside excluded containers
            parent_classes = []
            for parent in elem.parents:
                if parent.get("class"):
                    parent_classes.extend(parent.get("class", []))

            # Skip navigation, references, infoboxes
            skip_classes = {
                "navbox",
                "infobox",
                "sidebar",
                "references",
                "reflist",
                "refbegin",
                "mw-references-wrap",
                "toc",
                "vertical-navbox",
                "navigation-not-searchable",
            }
            if skip_classes & set(parent_classes):
                continue

            # Extract links from this element
            for link in elem.find_all("a", href=True):
                href = link.get("href", "")
                if self._is_valid_article_link(href):
                    link_title = self._extract_title_from_href(href)
                    if link_title and link_title not in seen:
                        links.append(link_title)
                        seen.add(link_title)

        logger.debug(f"Found {len(links)} links on '{title}'")
        return WikiPage(title=title, url=url, links=links)

    def get_links(self, title: str) -> list[str]:
        """
        Get just the links from a Wikipedia page.

        Convenience method that returns only the link list.
        """
        return self.get_page(title).links


# Module-level instance for convenience
scraper = WikiScraper()
