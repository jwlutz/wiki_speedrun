# Wikipedia Speedrun Benchmark

A benchmark comparing AI agents playing the **Wikipedia Speedrun** game - navigating from one Wikipedia article to another using only the links on each page.

## What is Wikipedia Speedrun?

Wikipedia Speedrun (also known as "Wiki Game" or "Wikipedia Golf") is a game where players start on one Wikipedia article and must navigate to a target article by clicking only the links within each page. The goal is to reach the target in as few clicks as possible.

**Example challenges:**
- **Easy:** Cat -> Dog (semantically close, many shared links)
- **Medium:** Coffee -> Minecraft (requires creative intermediate hops)
- **Hard:** Lamprey -> Costco Hot Dog (obscure start, specific target)
- **Default:** Emperor Penguin -> iPhone (classic benchmark problem)

## Try It Yourself

Run the Flask app and play in your browser:

```bash
python -m ui.flask_app
```

Then open http://localhost:5000/play to challenge yourself against our AI agents.

## Benchmark Results

We tested 15 AI agents across 35 unique problems, totaling 420+ games:

| Agent Type | Best Performer | Win Rate | Avg Clicks | Cost/Game |
|------------|----------------|----------|------------|-----------|
| **LLM** | gpt-5-mini | 97% | 3.4 | ~$0.02 |
| **LLM** | gemini-2.0-flash | 97% | 3.5 | ~$0.001 |
| **Embedding** | bge-large | 94% | 4.1 | FREE |

**Key Finding:** Embedding-based agents achieve 94% win rates at zero cost, while LLMs reach 97% for minimal cost. The Pareto-optimal choices are `bge-large` (free, 94%) and `gemini-2.0-flash` ($0.001/game, 97%).

## Requirements

- Python 3.11+
- OpenRouter API key (for LLM agents)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Play with embedding-based agent
python scripts/play.py --start "Cat" --target "Philosophy" --agent precomputed

# Play with LLM agent (any OpenRouter model ID)
python scripts/play.py --start "Potato" --target "Barack Obama" --agent llm --model google/gemini-2.0-flash-001

# Run benchmark
python scripts/quick_benchmark.py

# Start web dashboard
python -m ui.flask_app
```

## Agents

| Agent | Description | Cost |
|-------|-------------|------|
| `precomputed` | Greedy using pre-computed title embeddings | Free |
| `live` | Greedy using on-the-fly sentence-transformers | Free |
| `hybrid` | Pre-computed filter + live re-rank | Free |
| `oracle` | BFS optimal path (uses pre-computed graph) | Free |
| `llm` | LLM via OpenRouter (use `--model` to specify) | Varies |
| `human` | Manual play in browser | Free |
| `random` | Random baseline | Free |

## LLM Models

Use any OpenRouter model ID with `--model`:

```bash
# Free models
--model google/gemini-2.0-flash-exp:free
--model meta-llama/llama-3.3-70b-instruct:free

# Budget models
--model openai/gpt-4o-mini
--model deepseek/deepseek-chat

# Premium models
--model anthropic/claude-sonnet-4
--model openai/gpt-5-mini
```

## Data Requirements

Place these files in the `data/` directory:
- `embeddings.npy` - Article embeddings (18.5M titles, 384-dim, float16)
- `link_graph.msgpack` - Wikipedia link graph (7.1M articles)
- `titles.json` - Article titles
- `title_to_idx.json` - Title to index mapping
- `openrouter_models.json` - OpenRouter model catalog with pricing

## Project Structure

```
wiki_speedrun/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── data/            # Data loading utilities
│   └── game/            # Game logic
├── ui/
│   ├── flask_app.py     # Web interface
│   └── components/      # Dashboard charts
├── scripts/
│   ├── play.py          # CLI game runner
│   └── quick_benchmark.py
└── data/                # Benchmark data and results
```

## License

MIT
