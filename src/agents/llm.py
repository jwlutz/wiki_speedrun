"""
LLM-based agent using OpenRouter API.

Uses any LLM available on OpenRouter to choose which link to click.
Sends full link list by default (or pre-filtered if specified).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import requests
from dotenv import load_dotenv

from src.agents.base import Agent, AgentContext

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Available models organized by category (Updated Jan 2026)
MODELS = {
    # FREE models (may hit rate limits)
    "gemini-2.0-flash-exp:free": "google/gemini-2.0-flash-exp:free",
    "llama-3.3-70b:free": "meta-llama/llama-3.3-70b-instruct:free",

    # Budget models ($0.01-0.10 per 1M input)
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5-nano": "openai/gpt-5-nano",
    "deepseek-chat": "deepseek/deepseek-chat",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite-001",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",

    # Fast models ($0.10-1.00 per 1M input)
    "gpt-5-mini": "openai/gpt-5-mini",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",

    # Smart models ($1.00-5.00 per 1M input)
    "gpt-5.2-chat": "openai/gpt-5.2-chat",
    "gpt-5.2": "openai/gpt-5.2",
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",

    # Frontier models ($5.00+ per 1M input)
    "gpt-5.2-pro": "openai/gpt-5.2-pro",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
}

# Default model (free for testing)
DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"

# Reasoning models need more tokens (they use tokens for internal reasoning)
REASONING_MODELS = {
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.2",  # Also reasoning
    "openai/o1",
    "openai/o1-mini",
    "openai/o1-pro",
    "openai/o3",
    "openai/o3-mini",
}


def get_available_models() -> dict[str, str]:
    """Return dict of short_name -> full_model_id."""
    return MODELS.copy()


def resolve_model_name(name: str) -> str:
    """Resolve short name or full name to OpenRouter model ID."""
    # If it's a short name, look it up
    if name in MODELS:
        return MODELS[name]
    # If it contains a slash, assume it's already a full model ID
    if "/" in name:
        return name
    # Try to find partial match
    for short, full in MODELS.items():
        if name.lower() in short.lower() or name.lower() in full.lower():
            return full
    # Return as-is and let OpenRouter handle it
    return name


class LLMAgent(Agent):
    """
    Agent that uses an LLM via OpenRouter to choose links.

    The LLM receives the current page, target, path history, and available
    links, then chooses which link to click.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prefilter: int | None = None,
        temperature: float = 0.0,
        timeout: int = 60,
        always_call_llm: bool = False,
    ) -> None:
        """
        Initialize LLM agent.

        Args:
            model: OpenRouter model ID or short name from MODELS dict
            prefilter: If set, pre-filter to top N links using embeddings
            temperature: LLM temperature (0 = deterministic)
            timeout: Request timeout in seconds
            always_call_llm: If True, always call LLM even if target is visible
                            (useful for testing/benchmarking)
        """
        self._model = resolve_model_name(model)
        self._prefilter = prefilter
        self._temperature = temperature
        self._timeout = timeout
        self._always_call_llm = always_call_llm
        self._wiki_data = None  # Lazy load for prefiltering

        # Stats tracking
        self._total_tokens = 0
        self._total_requests = 0
        self._total_time = 0.0
        self._fallback_count = 0  # Track how many times we fell back to embeddings

    def _ensure_loaded(self) -> None:
        """Lazy load wiki_data for pre-filtering."""
        if self._prefilter and self._wiki_data is None:
            from src.data.loader import wiki_data
            self._wiki_data = wiki_data

    @property
    def name(self) -> str:
        # Extract short model name
        model_short = self._model.split("/")[-1]
        if len(model_short) > 25:
            model_short = model_short[:22] + "..."
        return f"llm-{model_short}"

    @property
    def description(self) -> str:
        prefilter_str = f", prefilter={self._prefilter}" if self._prefilter else ""
        return f"LLM agent ({self._model}{prefilter_str})"

    def _filter_links(self, context: AgentContext, links: list[str]) -> list[str]:
        """Filter out current page and visited pages to prevent loops."""
        # Remove current page (prevents infinite loops)
        filtered = [link for link in links if link != context.current_title]

        # Remove already visited pages
        visited = set(context.path_so_far)
        unvisited = [link for link in filtered if link not in visited]

        # Return unvisited if any, otherwise return filtered (allows backtracking if stuck)
        return unvisited if unvisited else filtered

    def _build_prompt(self, context: AgentContext, links: list[str], retry: bool = False) -> str:
        """Build the prompt for the LLM."""
        path_str = " → ".join(context.path_so_far)
        # Number links for easier validation
        links_str = "\n".join(f"{i+1}. {link}" for i, link in enumerate(links))

        if retry:
            return f"""Wikipedia Speedrun. Pick ONE number from the list.

Target: {context.target_title}
Current: {context.current_title}

{links_str}

Reply with ONLY a number (1-{len(links)}). Nothing else."""

        return f"""Wikipedia Speedrun: Navigate from "{context.current_title}" to "{context.target_title}".

Path so far: {path_str} ({context.step_count} steps)

Available links:
{links_str}

Which link leads toward "{context.target_title}"? Reply with ONLY the number (1-{len(links)})."""

    def _prefilter_links(self, links: list[str], target: str) -> list[str]:
        """Pre-filter links using embedding similarity."""
        if not self._prefilter or not self._wiki_data:
            return links

        if len(links) <= self._prefilter:
            return links

        # Rank by similarity and take top N
        ranked = self._wiki_data.rank_by_similarity(
            candidates=links,
            target=target,
        )

        if not ranked:
            return links[:self._prefilter]

        return [title for title, _sim in ranked[:self._prefilter]]

    def _call_openrouter(self, prompt: str, max_retries: int = 3) -> tuple[str, dict]:
        """
        Call OpenRouter API and return (response_text, usage_dict).

        Retries on rate limits (429) with exponential backoff.

        Raises:
            RuntimeError: If API call fails after retries
        """
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env file."
            )

        last_error = None
        for attempt in range(max_retries):
            start_time = time.time()

            try:
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/jacklutz/wiki_speedrun",
                        "X-Title": "Wikipedia Speedrun Benchmark",
                    },
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self._temperature,
                        # Reasoning models need more tokens for internal thinking
                        "max_tokens": 8000 if self._model in REASONING_MODELS else 100,
                    },
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout:
                last_error = RuntimeError("Request timed out")
                continue

            elapsed = time.time() - start_time
            self._total_time += elapsed

            # Handle rate limits with backoff
            if response.status_code == 429:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.debug(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                last_error = RuntimeError(f"Rate limited: {response.text[:100]}")
                continue

            if response.status_code != 200:
                error_msg = response.text[:200]
                raise RuntimeError(
                    f"OpenRouter API error {response.status_code}: {error_msg}"
                )

            data = response.json()

            # Check for error in response body (some providers return 200 with error)
            if "error" in data:
                error_info = data["error"]
                error_msg = error_info.get("message", str(error_info))[:200]
                wait_time = 2 ** attempt
                logger.debug(f"Error in response body, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                last_error = RuntimeError(f"OpenRouter API error in body: {error_msg}")
                continue

            # Extract response text
            choices = data.get("choices", [])
            if not choices:
                wait_time = 2 ** attempt
                logger.warning(f"Empty choices array. Full response: {json.dumps(data)[:500]}")
                time.sleep(wait_time)
                last_error = RuntimeError("OpenRouter returned empty choices array")
                continue

            # Try standard OpenAI format first
            content = choices[0].get("message", {}).get("content", "")
            # Some providers use 'text' instead of 'message.content'
            if not content:
                content = choices[0].get("text", "")
            # Log the actual structure if still empty
            if not content or not content.strip():
                wait_time = 2 ** attempt
                logger.warning(f"Empty content. choices[0] structure: {json.dumps(choices[0])[:300]}")
                time.sleep(wait_time)
                last_error = RuntimeError("OpenRouter returned empty content")
                continue

            # Success - break out of retry loop
            break
        else:
            # All retries exhausted
            raise last_error or RuntimeError("All retries exhausted")

        usage = data.get("usage", {})

        self._total_tokens += usage.get("total_tokens", 0)
        self._total_requests += 1

        return content.strip(), usage

    def _parse_response(self, response: str, available_links: list[str]) -> str | None:
        """
        Parse LLM response to extract link choice (expecting a number).

        Returns:
            Link name if valid number found, None otherwise
        """
        # Clean up response
        choice = response.strip().strip('"').strip("'").strip()

        # Take first line only
        if "\n" in choice:
            choice = choice.split("\n")[0].strip()

        # Try to extract a number
        import re
        numbers = re.findall(r'\d+', choice)
        if numbers:
            num = int(numbers[0])
            if 1 <= num <= len(available_links):
                return available_links[num - 1]

        # Fallback: try exact match on link name (in case LLM ignores number instruction)
        if choice in available_links:
            return choice

        # Case-insensitive match
        choice_lower = choice.lower()
        for link in available_links:
            if link.lower() == choice_lower:
                return link

        return None

    def choose_link(self, context: AgentContext) -> str:
        """Choose which link to click using LLM."""
        # Click target if available (unless always_call_llm is set for testing)
        if context.target_title in context.available_links and not self._always_call_llm:
            return context.target_title

        # Filter out current page and visited pages to prevent loops
        links = self._filter_links(context, context.available_links)

        # Pre-filter by embedding similarity if enabled
        if self._prefilter:
            self._ensure_loaded()
            links = self._prefilter_links(links, context.target_title)

        # Build prompt and call LLM (with retry on invalid response)
        max_attempts = 2
        for attempt in range(max_attempts):
            retry = attempt > 0
            prompt = self._build_prompt(context, links, retry=retry)

            try:
                response, usage = self._call_openrouter(prompt)
                logger.debug(f"LLM response (attempt {attempt+1}): {response[:100]}")
                logger.debug(f"Usage: {usage}")

                # Parse response - validate against filtered links only
                chosen = self._parse_response(response, links)

                if chosen:
                    return chosen

                # Invalid response - retry with stricter prompt
                if attempt < max_attempts - 1:
                    logger.debug(f"Invalid response '{response[:30]}', retrying...")
                    continue

                # Final attempt failed - log and fallback
                logger.warning(
                    f"LLM returned invalid link '{response[:50]}' after {max_attempts} attempts, using fallback"
                )
                self._fallback_count += 1

            except Exception as e:
                logger.warning(f"LLM API error: {e}, using fallback")
                self._fallback_count += 1
                break

        # Fallback: simple heuristic (no wiki_data needed)
        return self._embedding_fallback(context)

    def _embedding_fallback(self, context: AgentContext) -> str:
        """Fallback when LLM fails - simple heuristic, no heavy data loading."""
        import random

        # Avoid revisits
        visited = set(context.path_so_far)
        unvisited = [link for link in context.available_links if link not in visited]

        if not unvisited:
            # All links visited, just pick first available
            return context.available_links[0]

        # Simple heuristic: prefer links with words from target title
        target_words = set(context.target_title.lower().split())
        scored = []
        for link in unvisited:
            link_words = set(link.lower().split())
            overlap = len(target_words & link_words)
            scored.append((link, overlap))

        # Sort by overlap (desc), take best or random from ties
        scored.sort(key=lambda x: -x[1])
        best_score = scored[0][1]
        best_links = [link for link, score in scored if score == best_score]

        return random.choice(best_links)

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "model": self._model,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_time_seconds": round(self._total_time, 2),
            "avg_time_per_request": round(
                self._total_time / self._total_requests, 3
            ) if self._total_requests > 0 else 0,
            "fallback_count": self._fallback_count,
        }

    def on_game_end(self, won: bool, path: list[str]) -> None:
        """Log stats at end of game."""
        stats = self.get_stats()
        logger.info(
            f"LLM stats: {stats['total_requests']} requests, "
            f"{stats['total_tokens']} tokens, "
            f"{stats['total_time_seconds']}s total"
        )
