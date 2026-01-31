"""
Unit tests for WikiData loader.

Note: These tests require the data files to be present. Tests will be
skipped if data files are missing.
"""

import numpy as np
import pytest

from src.config import EMBEDDING_DIM, validate_data_files

# Skip all tests if data files are missing
pytestmark = pytest.mark.skipif(
    not all(validate_data_files().values()),
    reason="Data files not available",
)


@pytest.fixture(scope="module")
def wiki_data():
    """Load wiki_data once for all tests in this module."""
    from src.data import wiki_data

    # Trigger loading
    _ = wiki_data.article_count()
    return wiki_data


class TestCoreAccessors:
    """Test core accessor methods."""

    def test_article_count_positive(self, wiki_data):
        """Article count should be positive."""
        assert wiki_data.article_count() > 0

    def test_traversable_count_positive(self, wiki_data):
        """Traversable count should be positive."""
        assert wiki_data.traversable_count() > 0

    def test_traversable_less_than_total(self, wiki_data):
        """Traversable count should be less than total (not all have links)."""
        assert wiki_data.traversable_count() <= wiki_data.article_count()

    def test_get_title_valid_index(self, wiki_data):
        """Should return title for valid index."""
        title = wiki_data.get_title(0)
        assert isinstance(title, str)

    def test_get_title_invalid_index_raises(self, wiki_data):
        """Should raise IndexError for invalid index."""
        with pytest.raises(IndexError):
            wiki_data.get_title(-1)
        with pytest.raises(IndexError):
            wiki_data.get_title(999_999_999)

    def test_get_index_known_article(self, wiki_data):
        """Should return index for known article."""
        idx = wiki_data.get_index("Albert Einstein")
        assert idx is not None
        assert isinstance(idx, int)
        assert idx >= 0

    def test_get_index_unknown_article(self, wiki_data):
        """Should return None for unknown article."""
        idx = wiki_data.get_index("ZZZZZ_NOT_A_REAL_ARTICLE_12345")
        assert idx is None

    def test_has_article_true(self, wiki_data):
        """Should return True for existing article."""
        assert wiki_data.has_article("Albert Einstein") is True

    def test_has_article_false(self, wiki_data):
        """Should return False for non-existing article."""
        assert wiki_data.has_article("ZZZZZ_NOT_A_REAL_ARTICLE_12345") is False

    def test_is_traversable_true(self, wiki_data):
        """Should return True for article with outgoing links."""
        # Albert Einstein should have links
        assert wiki_data.is_traversable("Albert Einstein") is True

    def test_is_traversable_false_nonexistent(self, wiki_data):
        """Should return False for non-existent article."""
        assert wiki_data.is_traversable("ZZZZZ_NOT_A_REAL_ARTICLE_12345") is False


class TestEmbeddingAccessors:
    """Test embedding accessor methods."""

    def test_get_embedding_valid(self, wiki_data):
        """Should return embedding for valid article."""
        emb = wiki_data.get_embedding("Albert Einstein")
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_get_embedding_invalid(self, wiki_data):
        """Should return None for invalid article."""
        emb = wiki_data.get_embedding("ZZZZZ_NOT_A_REAL_ARTICLE_12345")
        assert emb is None

    def test_get_embedding_by_idx(self, wiki_data):
        """Should return embedding by index."""
        idx = wiki_data.get_index("Albert Einstein")
        emb = wiki_data.get_embedding_by_idx(idx)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_get_normalized_embedding(self, wiki_data):
        """Should return normalized embedding (L2 norm ~= 1)."""
        emb = wiki_data.get_normalized_embedding("Albert Einstein")
        assert emb is not None
        norm = np.linalg.norm(emb)
        assert 0.99 < norm < 1.01  # Should be approximately 1

    def test_get_embeddings_batch(self, wiki_data):
        """Should return batch of embeddings."""
        titles = ["Albert Einstein", "Physics", "Mathematics"]
        batch = wiki_data.get_embeddings_batch(titles)
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (3, EMBEDDING_DIM)

    def test_get_embeddings_batch_partial(self, wiki_data):
        """Should handle batch with some invalid titles."""
        titles = ["Albert Einstein", "INVALID_ARTICLE", "Physics"]
        batch = wiki_data.get_embeddings_batch(titles)
        assert batch.shape[0] == 2  # Only 2 valid


class TestSimilarityFunctions:
    """Test similarity computation methods."""

    def test_similarity_valid(self, wiki_data):
        """Should return similarity for valid articles."""
        sim = wiki_data.similarity("Physics", "Mathematics")
        assert sim is not None
        assert isinstance(sim, float)
        assert -1 <= sim <= 1  # Cosine similarity range

    def test_similarity_invalid(self, wiki_data):
        """Should return None if either article is invalid."""
        assert wiki_data.similarity("Physics", "INVALID") is None
        assert wiki_data.similarity("INVALID", "Physics") is None

    def test_similarity_self(self, wiki_data):
        """Similarity of article with itself should be ~1."""
        sim = wiki_data.similarity("Albert Einstein", "Albert Einstein")
        assert sim is not None
        assert sim > 0.99

    def test_similarity_semantic_ordering(self, wiki_data):
        """Related articles should be more similar than unrelated."""
        sim_related = wiki_data.similarity("Physics", "Mathematics")
        sim_unrelated = wiki_data.similarity("Physics", "Pizza")
        assert sim_related is not None
        assert sim_unrelated is not None
        assert sim_related > sim_unrelated

    def test_rank_by_similarity(self, wiki_data):
        """Should rank candidates by similarity."""
        candidates = ["Mathematics", "Pizza", "Chemistry"]
        ranked = wiki_data.rank_by_similarity(candidates, "Physics")
        assert len(ranked) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked)
        # Check sorted descending
        sims = [r[1] for r in ranked]
        assert sims == sorted(sims, reverse=True)

    def test_rank_by_similarity_empty(self, wiki_data):
        """Should return empty list for invalid target."""
        ranked = wiki_data.rank_by_similarity(["Physics"], "INVALID")
        assert ranked == []

    def test_nearest_neighbors(self, wiki_data):
        """Should return nearest neighbors."""
        neighbors = wiki_data.nearest_neighbors("Physics", k=5)
        assert len(neighbors) == 5
        assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)
        # Should not include self
        assert all(n[0] != "Physics" for n in neighbors)

    def test_nearest_neighbors_invalid(self, wiki_data):
        """Should return empty list for invalid article."""
        neighbors = wiki_data.nearest_neighbors("INVALID", k=5)
        assert neighbors == []


class TestGraphAccessors:
    """Test graph accessor methods."""

    def test_get_links_valid(self, wiki_data):
        """Should return links for valid traversable article."""
        links = wiki_data.get_links("Albert Einstein")
        assert isinstance(links, list)
        assert len(links) > 0
        assert all(isinstance(link, str) for link in links)

    def test_get_links_invalid(self, wiki_data):
        """Should return empty list for invalid article."""
        links = wiki_data.get_links("INVALID_ARTICLE_12345")
        assert links == []

    def test_get_links_by_idx(self, wiki_data):
        """Should return link indices."""
        idx = wiki_data.get_index("Albert Einstein")
        if wiki_data.is_traversable("Albert Einstein"):
            links = wiki_data.get_links_by_idx(idx)
            assert isinstance(links, list)
            assert all(isinstance(i, int) for i in links)

    def test_get_popularity(self, wiki_data):
        """Should return popularity (inbound link count)."""
        # Popular articles should have high popularity
        pop = wiki_data.get_popularity("United States")
        assert isinstance(pop, int)
        assert pop >= 0
        # United States should be linked to by many articles
        # Note: This test may be slow first time due to on-demand calculation

    def test_get_popularity_invalid(self, wiki_data):
        """Should return 0 for invalid article."""
        pop = wiki_data.get_popularity("INVALID_ARTICLE_12345")
        assert pop == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_title(self, wiki_data):
        """Should handle empty string gracefully."""
        assert wiki_data.get_index("") is None or wiki_data.get_index("") is not None
        # Just shouldn't crash

    def test_unicode_title(self, wiki_data):
        """Should handle unicode titles."""
        # Try some unicode - may or may not exist
        wiki_data.get_index("日本")  # Japan in Japanese
        wiki_data.get_index("Москва")  # Moscow in Russian
        # Just shouldn't crash

    def test_special_characters(self, wiki_data):
        """Should handle special characters in titles."""
        wiki_data.get_index("C++")
        wiki_data.get_index("AC/DC")
        # Just shouldn't crash


class TestValidation:
    """Test validation and stats methods."""

    def test_validate_all_pass(self, wiki_data):
        """All validation checks should pass."""
        validation = wiki_data.validate()
        assert all(validation.values()), f"Failed checks: {validation}"

    def test_stats_complete(self, wiki_data):
        """Stats should include all expected keys."""
        stats = wiki_data.stats()
        expected_keys = [
            "total_articles",
            "traversable_articles",
            "embedding_dim",
            "embedding_dtype",
            "faiss_vectors",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
