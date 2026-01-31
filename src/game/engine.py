"""
Game engine for running Wikipedia speedruns with optional visualization.

Supports both headless execution (fast) and Playwright visualization (shows
the actual Wikipedia page with links highlighted).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from src.agents.base import AgentContext
from src.config import (
    MAX_STEPS,
    PLAYWRIGHT_HEADLESS,
    PLAYWRIGHT_TIMEOUT,
    WIKIPEDIA_BASE_URL,
)
from src.game.state import GameResult, GameState
from src.wikipedia.scraper import WikiScraper

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from src.agents.base import Agent

logger = logging.getLogger(__name__)


class GameEngine:
    """
    Runs Wikipedia speedrun games with optional browser visualization.

    The engine handles:
    - Fetching pages and extracting links
    - Running the agent's decision loop
    - Optional Playwright visualization with link highlighting
    - Recording game state and results
    """

    # CSS for highlighting links
    HIGHLIGHT_CSS = """
        .wiki-speedrun-highlight {
            outline: 3px solid red !important;
            background-color: rgba(255, 0, 0, 0.1) !important;
        }
        .wiki-speedrun-chosen {
            outline: 3px solid green !important;
            background-color: rgba(0, 255, 0, 0.2) !important;
        }
        .wiki-speedrun-target {
            outline: 3px solid gold !important;
            background-color: rgba(255, 215, 0, 0.2) !important;
        }
    """

    def __init__(
        self,
        visualize: bool = False,
        headless: bool = PLAYWRIGHT_HEADLESS,
        slow_mo: int = 0,
    ) -> None:
        """
        Initialize the game engine.

        Args:
            visualize: Whether to show browser visualization
            headless: Run browser in headless mode (only if visualize=True)
            slow_mo: Slow down Playwright operations by this many ms
        """
        self._visualize = visualize
        self._headless = headless
        self._slow_mo = slow_mo
        self._scraper = WikiScraper()

        # Playwright resources (lazy-initialized)
        self._browser = None
        self._context = None
        self._page: Page | None = None

    def _init_browser(self) -> None:
        """Initialize Playwright browser for visualization."""
        if self._page is not None:
            return

        from playwright.sync_api import sync_playwright

        logger.info("Starting Playwright browser...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(PLAYWRIGHT_TIMEOUT)

        # Inject our highlight CSS
        self._page.add_style_tag(content=self.HIGHLIGHT_CSS)

    def _close_browser(self) -> None:
        """Clean up Playwright resources."""
        if self._page:
            self._page.close()
            self._page = None
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if hasattr(self, "_playwright") and self._playwright:
            self._playwright.stop()
            self._playwright = None

    def _navigate_to(self, title: str) -> list[str]:
        """
        Navigate to a Wikipedia page and extract links.

        Uses Playwright if visualizing, otherwise just scrapes.

        Returns:
            List of article titles linked from this page
        """
        # Always use BeautifulSoup for reliable link extraction
        links = self._scraper.get_links(title)

        # If visualizing, also navigate the browser
        if self._visualize and self._page:
            url = self._scraper.title_to_url(title)
            self._page.goto(url, wait_until="load")

            # Re-inject CSS (lost on navigation)
            try:
                self._page.add_style_tag(content=self.HIGHLIGHT_CSS)
            except Exception:
                pass  # Page might not be ready

        return links

    def _extract_links_playwright(self) -> list[str]:
        """Extract article links from current page using Playwright."""
        # JavaScript to extract links from main content
        js_code = """
        () => {
            const content = document.querySelector('.mw-parser-output');
            if (!content) {
                console.log('No .mw-parser-output found');
                return [];
            }

            const links = [];
            const seen = new Set();

            // Get links from paragraphs, lists, tables
            const elements = content.querySelectorAll('p a, li a, td a, th a, dd a');

            for (const link of elements) {
                const href = link.getAttribute('href');
                if (!href || !href.startsWith('/wiki/')) continue;

                // Skip special namespaces
                const excluded = ['Wikipedia:', 'Help:', 'Template:', 'Category:',
                                  'Portal:', 'File:', 'Special:', 'Talk:', 'User:'];
                // Strip anchors and query params, then decode
                const path = href.replace('/wiki/', '').split('#')[0].split('?')[0];
                const title = decodeURIComponent(path).replace(/_/g, ' ');
                if (excluded.some(prefix => title.startsWith(prefix))) continue;

                // Skip if in navbox, infobox, etc.
                let skip = false;
                let parent = link.parentElement;
                while (parent && parent !== content) {
                    const classes = parent.className || '';
                    if (/navbox|infobox|sidebar|reflist|references|toc/.test(classes)) {
                        skip = true;
                        break;
                    }
                    parent = parent.parentElement;
                }
                if (skip) continue;

                if (!seen.has(title)) {
                    seen.add(title);
                    links.push(title);
                }
            }
            return links;
        }
        """
        return self._page.evaluate(js_code)

    def _highlight_links(self, links: list[str], target: str) -> None:
        """Highlight available links on the page."""
        if not self._visualize or not self._page:
            return

        # Convert link titles to href format
        href_parts = [link.replace(" ", "_") for link in links]
        target_href = target.replace(" ", "_") if target in links else None

        # Single JS call to highlight all links
        self._page.evaluate("""
            ({hrefParts, targetHref}) => {
                // Clear previous highlights
                document.querySelectorAll('.wiki-speedrun-highlight, .wiki-speedrun-target')
                    .forEach(el => el.classList.remove('wiki-speedrun-highlight', 'wiki-speedrun-target'));

                // Build a Set for fast lookup
                const hrefSet = new Set(hrefParts);

                // Single pass through all links
                document.querySelectorAll('a').forEach(link => {
                    if (!link.href || !link.href.includes('/wiki/')) return;

                    // Extract the wiki page name from href
                    const match = link.href.match(/\\/wiki\\/([^#?]+)/);
                    if (!match) return;
                    const pageName = decodeURIComponent(match[1]);

                    if (pageName === targetHref) {
                        link.classList.add('wiki-speedrun-target');
                    } else if (hrefSet.has(pageName)) {
                        link.classList.add('wiki-speedrun-highlight');
                    }
                });
            }
        """, {"hrefParts": href_parts, "targetHref": target_href})

    def _show_chosen_link(self, chosen: str) -> None:
        """Briefly highlight the chosen link before clicking."""
        if not self._visualize or not self._page:
            return

        href_chosen = chosen.replace(" ", "_")
        self._page.evaluate("""
            (hrefPart) => {
                document.querySelectorAll('a').forEach(link => {
                    if (!link.href || !link.href.includes('/wiki/')) return;
                    const match = link.href.match(/\\/wiki\\/([^#?]+)/);
                    if (!match) return;
                    const pageName = decodeURIComponent(match[1]);
                    if (pageName === hrefPart) {
                        link.classList.remove('wiki-speedrun-highlight', 'wiki-speedrun-target');
                        link.classList.add('wiki-speedrun-chosen');
                    }
                });
            }
        """, href_chosen)

        # Scroll the chosen link into view
        self._page.evaluate("""
            (hrefPart) => {
                for (const link of document.querySelectorAll('a')) {
                    if (!link.href || !link.href.includes('/wiki/')) continue;
                    const match = link.href.match(/\\/wiki\\/([^#?]+)/);
                    if (!match) continue;
                    const pageName = decodeURIComponent(match[1]);
                    if (pageName === hrefPart) {
                        link.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        break;
                    }
                }
            }
        """, href_chosen)

        # Brief pause to show the selection
        time.sleep(0.5)

    def run(
        self,
        agent: Agent,
        start: str,
        target: str,
        max_steps: int = MAX_STEPS,
    ) -> GameResult:
        """
        Run a complete Wikipedia speedrun game.

        Args:
            agent: The agent to play the game
            start: Starting article title
            target: Target article title
            max_steps: Maximum clicks before game is lost

        Returns:
            GameResult with full game record
        """
        logger.info(f"Starting game: '{start}' -> '{target}' with {agent.name}")

        # Initialize visualization if needed
        if self._visualize or agent.requires_visualization:
            self._visualize = True
            self._headless = False  # Force visible for human agents
            self._init_browser()

        # Initialize game state
        state = GameState(
            start_title=start,
            target_title=target,
            current_title=start,
        )
        state.start_time_ms = time.time() * 1000

        # Notify agent
        agent.on_game_start(start, target)

        try:
            # Main game loop
            while not state.is_won and state.click_count < max_steps:
                # Get available links from current page
                available_links = self._navigate_to(state.current_title)

                if not available_links:
                    logger.warning(f"No links found on '{state.current_title}'")
                    break

                # Highlight links if visualizing
                self._highlight_links(available_links, target)

                # Check if target is directly reachable
                if target in available_links:
                    logger.info(f"Target '{target}' is reachable!")

                # Build context for agent
                context = AgentContext(
                    current_title=state.current_title,
                    target_title=target,
                    available_links=available_links,
                    path_so_far=list(state.path),
                    step_count=state.click_count,
                )

                # Let agent choose
                decision_start = time.time() * 1000
                chosen = agent.choose_link(context)
                decision_time = time.time() * 1000 - decision_start

                # Validate choice
                if chosen not in available_links:
                    logger.error(f"Agent chose invalid link: '{chosen}'")
                    raise ValueError(f"Invalid link choice: '{chosen}'")

                logger.info(
                    f"Step {state.click_count + 1}: '{state.current_title}' -> '{chosen}'"
                )

                # Show the chosen link before navigating
                self._show_chosen_link(chosen)

                # Record step and update state
                state.record_step(chosen, available_links, decision_time)

            # Game finished
            total_time = time.time() * 1000 - state.start_time_ms
            result = state.to_result(agent.name, total_time)

            # Notify agent
            agent.on_game_end(result.won, result.path)

            if result.won:
                logger.info(
                    f"Won! Path ({result.total_clicks} clicks): "
                    f"{' -> '.join(result.path)}"
                )
            else:
                logger.info(f"Lost after {result.total_clicks} clicks")

            return result

        finally:
            # Clean up browser if we own it
            if self._visualize and not agent.requires_visualization:
                self._close_browser()

    def close(self) -> None:
        """Clean up resources."""
        self._close_browser()

    def __enter__(self) -> "GameEngine":
        return self

    def __exit__(self, *args) -> None:
        self.close()
