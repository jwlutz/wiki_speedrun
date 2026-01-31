"""Save dashboard charts as PNG images for README."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.components.data_loader import (
    get_agent_summaries,
    get_results,
    get_agent_cost_summaries,
    get_failure_analysis,
    get_problems,
)
from ui.components.charts import (
    create_win_rate_chart,
    create_clicks_boxplot,
    create_time_scatter,
    create_difficulty_heatmap,
    create_pareto_chart,
)

def main():
    output_dir = Path(__file__).parent.parent / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    summaries = get_agent_summaries()
    cost_summaries = get_agent_cost_summaries()
    results = get_results()
    problems = get_problems()

    if not summaries:
        print("No benchmark data found!")
        return

    print("Generating charts...")

    # Pareto chart
    fig = create_pareto_chart(cost_summaries)
    fig.update_layout(width=800, height=500)
    fig.write_image(output_dir / "pareto.png", scale=2)
    print(f"  Saved pareto.png")

    # Win rate chart
    fig = create_win_rate_chart(summaries)
    fig.update_layout(width=700, height=400)
    fig.write_image(output_dir / "win_rate.png", scale=2)
    print(f"  Saved win_rate.png")

    # Clicks boxplot
    fig = create_clicks_boxplot(results)
    fig.update_layout(width=700, height=400)
    fig.write_image(output_dir / "clicks.png", scale=2)
    print(f"  Saved clicks.png")

    # Time scatter
    fig = create_time_scatter(summaries)
    fig.update_layout(width=700, height=400)
    fig.write_image(output_dir / "efficiency.png", scale=2)
    print(f"  Saved efficiency.png")

    # Difficulty heatmap
    try:
        fig = create_difficulty_heatmap(results, problems)
        fig.update_layout(width=600, height=400)
        fig.write_image(output_dir / "difficulty.png", scale=2)
        print(f"  Saved difficulty.png")
    except Exception as e:
        print(f"  Skipped difficulty.png: {e}")

    print(f"\nCharts saved to {output_dir}")


if __name__ == "__main__":
    main()
