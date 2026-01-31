"""
Configuration constants for the Wikipedia Speedrun project.

All paths, settings, and tunable parameters are defined here.
API keys are loaded from environment variables - never hardcode secrets.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Project root is parent of src/
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory (contains embeddings, link graph, titles)
DATA_DIR = PROJECT_ROOT / "data"

# Individual data file paths
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
LINK_GRAPH_PATH = DATA_DIR / "link_graph.msgpack"
TITLES_PATH = DATA_DIR / "titles.json"
TITLE_TO_IDX_PATH = DATA_DIR / "title_to_idx.json"

# Results and cache directories
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

# =============================================================================
# Embedding Configuration
# =============================================================================

# Dimension of title embeddings (from pre-computed data)
EMBEDDING_DIM = 384

# Data type for embeddings (float16 for memory efficiency)
EMBEDDING_DTYPE = "float16"

# =============================================================================
# Game Configuration
# =============================================================================

# Maximum steps before game is considered lost
MAX_STEPS = 50

# Default beam width for beam search agent
DEFAULT_BEAM_WIDTH = 5

# =============================================================================
# Heuristic Configuration
# =============================================================================

# Hybrid heuristic weights:
# score = ALPHA * embedding_sim + BETA * popularity + GAMMA * (1/depth)
HEURISTIC_ALPHA = 0.6   # Embedding similarity weight
HEURISTIC_BETA = 0.2    # Popularity weight
HEURISTIC_GAMMA = 0.2   # Depth penalty weight

# Weighted A* epsilon (< 1 for inadmissible heuristic)
# f(n) = g(n) + EPSILON * h(n)
ASTAR_EPSILON = 0.8

# =============================================================================
# LLM Configuration
# =============================================================================

# OpenRouter API settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# HuggingFace API settings (for hosted embedding inference)
HF_API_KEY = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"

# Default model for LLM agent (cheap, fast)
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"

# Number of links to send to LLM after embedding pre-filtering
# Reduces cost by filtering 500+ links down to top N
LLM_LINK_FILTER_COUNT = 20

# LLM request timeout in seconds
LLM_TIMEOUT = 30

# LLM temperature (0 = deterministic)
LLM_TEMPERATURE = 0.0

# =============================================================================
# Wikipedia Scraping Configuration
# =============================================================================

# Base URL for Wikipedia
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"

# API endpoint for Wikipedia
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Rate limiting: minimum seconds between requests
# Set to 0 for max speed, 1.0 to be polite to Wikipedia
WIKIPEDIA_REQUEST_DELAY = 0.0

# Request timeout in seconds
WIKIPEDIA_TIMEOUT = 10

# User agent for requests (be a good citizen)
USER_AGENT = "WikiSpeedrunBenchmark/0.1 (https://github.com/jacklutz/wiki_speedrun)"

# =============================================================================
# Visualization Configuration
# =============================================================================

# Playwright browser settings
PLAYWRIGHT_HEADLESS = True
PLAYWRIGHT_TIMEOUT = 30000  # milliseconds

# Network graph settings
GRAPH_NODE_SIZE_MIN = 10
GRAPH_NODE_SIZE_MAX = 50
GRAPH_EDGE_WIDTH = 1

# =============================================================================
# Benchmark Configuration
# =============================================================================

# Default number of problems per difficulty level
DEFAULT_PROBLEMS_PER_DIFFICULTY = 10

# BFS maximum depth for pathfinding
BFS_MAX_DEPTH = 15

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# =============================================================================
# Validation Helpers
# =============================================================================

def validate_data_files() -> dict[str, bool]:
    """Check which data files exist."""
    return {
        "embeddings": EMBEDDINGS_PATH.exists(),
        "link_graph": LINK_GRAPH_PATH.exists(),
        "titles": TITLES_PATH.exists(),
        "title_to_idx": TITLE_TO_IDX_PATH.exists(),
    }


def get_missing_data_files() -> list[str]:
    """Return list of missing data file names."""
    status = validate_data_files()
    return [name for name, exists in status.items() if not exists]
