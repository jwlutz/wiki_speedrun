"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides
fixtures available to all test files.
"""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_articles() -> list[str]:
    """Return a list of well-known Wikipedia articles for testing."""
    return [
        "Albert Einstein",
        "Physics",
        "United States",
        "Pizza",
        "Python (programming language)",
        "Mathematics",
        "World War II",
        "New York City",
    ]


@pytest.fixture
def sample_problem() -> tuple[str, str]:
    """Return a sample start/target pair for testing."""
    return ("Albert Einstein", "Pizza")
