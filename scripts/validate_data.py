#!/usr/bin/env python3
"""
Validate Wikipedia data files and WikiData loader.

Usage:
    python scripts/validate_data.py
"""

import logging
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402 - must be after sys.path modification
    EMBEDDINGS_PATH,
    LINK_GRAPH_PATH,
    TITLE_TO_IDX_PATH,
    TITLES_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def check_data_files_exist() -> bool:
    """Check that all data files exist."""
    print("\n=== Checking Data Files ===\n")

    files = {
        "titles.json": TITLES_PATH,
        "title_to_idx.json": TITLE_TO_IDX_PATH,
        "link_graph.msgpack": LINK_GRAPH_PATH,
        "embeddings.npy": EMBEDDINGS_PATH,
    }

    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
        status = f"✓ {name}: {size_mb:,.1f} MB" if exists else f"✗ {name}: NOT FOUND"
        print(status)
        if not exists:
            all_exist = False

    return all_exist


def load_and_validate() -> bool:
    """Load WikiData and run validation checks."""
    print("\n=== Loading WikiData ===\n")

    start_time = time.time()

    # Import here to trigger lazy loading
    from src.data import wiki_data

    # Trigger loading by accessing data
    print("Loading data (this may take a minute)...")
    _ = wiki_data.article_count()

    load_time = time.time() - start_time
    print(f"\nLoad time: {load_time:.1f} seconds")

    # Get stats
    print("\n=== Data Statistics ===\n")
    stats = wiki_data.stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # Run validation
    print("\n=== Validation Checks ===\n")
    validation = wiki_data.validate()
    all_valid = True
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_valid = False

    return all_valid


def test_sample_queries() -> bool:
    """Test sample queries against known articles."""
    print("\n=== Sample Queries ===\n")

    from src.data import wiki_data

    test_articles = [
        "Albert Einstein",
        "Physics",
        "United States",
        "Python (programming language)",
    ]

    all_passed = True

    for title in test_articles:
        print(f"\nTesting: {title}")

        # Check existence
        if not wiki_data.has_article(title):
            print("  ✗ Article not found")
            all_passed = False
            continue
        print("  ✓ Article exists")

        # Check index
        idx = wiki_data.get_index(title)
        print(f"  ✓ Index: {idx:,}")

        # Check embedding
        emb = wiki_data.get_embedding(title)
        if emb is None:
            print("  ✗ No embedding")
            all_passed = False
        else:
            print(f"  ✓ Embedding shape: {emb.shape}")

        # Check links
        links = wiki_data.get_links(title)
        if wiki_data.is_traversable(title):
            print(f"  ✓ Outgoing links: {len(links):,}")
        else:
            print("  ⚠ No outgoing links (not traversable)")

    # Test similarity
    print("\n--- Similarity Tests ---")
    sim1 = wiki_data.similarity("Physics", "Mathematics")
    sim2 = wiki_data.similarity("Physics", "Pizza")

    if sim1 is not None and sim2 is not None:
        print(f"  Physics ↔ Mathematics: {sim1:.4f}")
        print(f"  Physics ↔ Pizza: {sim2:.4f}")

        if sim1 > sim2:
            print("  ✓ Physics is more similar to Mathematics than Pizza")
        else:
            print("  ✗ Unexpected similarity ordering")
            all_passed = False
    else:
        print("  ✗ Similarity calculation failed")
        all_passed = False

    # Test nearest neighbors
    print("\n--- Nearest Neighbors (Physics) ---")
    neighbors = wiki_data.nearest_neighbors("Physics", k=5)
    for i, (neighbor, sim) in enumerate(neighbors, 1):
        print(f"  {i}. {neighbor}: {sim:.4f}")

    # Test ranking
    print("\n--- Ranking Test ---")
    candidates = ["Mathematics", "Pizza", "Chemistry", "Football", "Biology"]
    ranked = wiki_data.rank_by_similarity(candidates, "Physics")
    print("  Candidates ranked by similarity to Physics:")
    for title, sim in ranked:
        print(f"    {title}: {sim:.4f}")

    return all_passed


def main() -> int:
    """Main validation routine."""
    print("=" * 60)
    print("Wikipedia Speedrun Data Validation")
    print("=" * 60)

    # Check files exist
    if not check_data_files_exist():
        print("\n✗ Some data files are missing. Cannot continue.")
        return 1

    # Load and validate
    try:
        if not load_and_validate():
            print("\n✗ Validation checks failed.")
            return 1
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test sample queries
    try:
        if not test_sample_queries():
            print("\n✗ Sample query tests failed.")
            return 1
    except Exception as e:
        print(f"\n✗ Error running sample queries: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✓ All validation checks passed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
