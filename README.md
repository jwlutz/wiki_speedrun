# Wikipedia Speedrun Benchmark

[Try it live](https://huggingface.co/spaces/jwlutz/wiki-speedrun)

A benchmark comparing AI agents playing the **Wikipedia Speedrun** game - navigating from one Wikipedia article to another using only the links on each page.

## What is Wikipedia Speedrun?

Wikipedia Speedrun (also known as "Wiki Game" or "Wikipedia Golf") is a game where players start on one Wikipedia article and must navigate to a target article by clicking only the links within each page. The goal is to reach the target in as few clicks as possible.

**Example challenges:**
- **Easy:** Cat -> Dog (semantically close, many shared links)
- **Medium:** Coffee -> Minecraft (requires creative intermediate hops)
- **Hard:** Lamprey -> Costco Hot Dog (obscure start, specific target)

## Try It Yourself

Run the Flask app and play in your browser:

```bash
python app.py
```

Then open http://localhost:7860 to play or watch AI agents compete.

## Benchmark Results

We tested 15 AI agents across 35 unique problems, totaling 420+ games.

### Cost vs Performance

![Cost vs Performance](docs/images/pareto.png)

Points on the yellow Pareto frontier represent optimal cost-performance tradeoffs. Embedding models (green) are free while LLM models (blue) cost per API call.

**Key Finding:** Embedding-based agents achieve 94% win rates at zero cost, while LLMs reach 97% for minimal cost. The Pareto-optimal choices are `bge-large` (free, 94%) and `gemini-2.0-flash` ($0.001/game, 97%).

### Win Rate by Agent

![Win Rate](docs/images/win_rate.png)

### Click Distribution

![Clicks](docs/images/clicks.png)

### Efficiency: Clicks vs Time

![Efficiency](docs/images/efficiency.png)

### Win Rate by Difficulty

![Difficulty](docs/images/difficulty.png)
