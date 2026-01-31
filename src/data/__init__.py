"""
Data loading module.

Provides WikiData singleton for accessing Wikipedia embeddings,
link graph, and title mappings.

Usage:
    from src.data import wiki_data

    wiki_data.get_title(0)
    wiki_data.get_embedding("Albert Einstein")
    wiki_data.get_links("Python (programming language)")
"""

from src.data.loader import WikiData, wiki_data

__all__ = ["WikiData", "wiki_data"]
