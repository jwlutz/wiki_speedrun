"""
Wikipedia interaction module.

Provides scraping, API access, and Playwright rendering
for live Wikipedia pages.
"""

from src.wikipedia.scraper import WikiPage, WikiScraper, scraper

__all__ = [
    "WikiScraper",
    "WikiPage",
    "scraper",
]
