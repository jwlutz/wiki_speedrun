"""
Human agent that prompts for user input.

Requires visualization to be useful - the human needs to see the page.
"""

from __future__ import annotations

from src.agents.base import Agent, AgentContext


class HumanAgent(Agent):
    """
    Human-controlled agent that prompts for link selection.

    Best used with --visualize flag so the human can see the Wikipedia page.
    Displays numbered links and accepts input via stdin.
    """

    @property
    def name(self) -> str:
        return "human"

    @property
    def description(self) -> str:
        return "Human player - manually choose links"

    @property
    def requires_visualization(self) -> bool:
        return True

    def choose_link(self, context: AgentContext) -> str:
        """Prompt the human to choose a link."""
        print("\n" + "=" * 60)
        print(f"Current: {context.current_title}")
        print(f"Target:  {context.target_title}")
        print(f"Steps:   {context.step_count}")
        print("=" * 60)

        # Check if target is directly reachable
        if context.target_title in context.available_links:
            print(f"\n*** TARGET '{context.target_title}' IS ON THIS PAGE! ***\n")

        # Show links in columns for easier reading
        print(f"\nAvailable links ({len(context.available_links)} total):")
        print("-" * 60)

        # Display first 50 links numbered, with target highlighted
        display_count = min(50, len(context.available_links))
        for i, link in enumerate(context.available_links[:display_count], 1):
            marker = " <<<" if link == context.target_title else ""
            print(f"  {i:3}. {link}{marker}")

        if len(context.available_links) > display_count:
            print(f"\n  ... and {len(context.available_links) - display_count} more")
            print("  (Type exact title to select unlisted links)")

        print("-" * 60)
        print("Enter number or title (or 'q' to quit):")

        while True:
            try:
                choice = input("> ").strip()

                if choice.lower() == "q":
                    raise KeyboardInterrupt("User quit")

                # Try parsing as number first
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(context.available_links):
                        selected = context.available_links[idx]
                        print(f"Selected: {selected}")
                        return selected
                    else:
                        print(f"Invalid number. Enter 1-{len(context.available_links)}")
                        continue

                # Try matching title (case-insensitive partial match)
                matches = [
                    link
                    for link in context.available_links
                    if choice.lower() in link.lower()
                ]

                if len(matches) == 1:
                    print(f"Selected: {matches[0]}")
                    return matches[0]
                elif len(matches) > 1:
                    print(f"Multiple matches ({len(matches)}). Be more specific:")
                    for m in matches[:10]:
                        print(f"  - {m}")
                else:
                    # Try exact match
                    if choice in context.available_links:
                        return choice
                    print("No match found. Try again.")

            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt("User quit")
