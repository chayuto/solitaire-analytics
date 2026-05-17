"""Example: an agent playing Solitaire through a GameSession.

This demonstrates the interactive game layer that the MCP server is built on.
It shows the core loop an agentic AI player follows -- observe, pick a legal
move, apply it -- and how the ``ObservationConfig`` information level changes
what the agent is allowed to see.

Run it directly:

    python scripts/example_mcp_agent.py
"""

from solitaire_analytics.game import GameSession, ObservationConfig


def play_with_information_level(name: str, config: ObservationConfig) -> None:
    """Play a short, greedy game under a given information level."""
    print("=" * 60)
    print(f"Information level: {name}")
    print("=" * 60)

    session = GameSession.new_game(seed=11, draw_count=1, observation_config=config)

    obs = session.observation()
    print(f"Info level the agent sees: {obs['info_level']}")
    print(f"Pile 7 face-down cards: {obs['tableau'][6]['face_down_count']}")
    if "face_down_cards" in obs["tableau"][6]:
        codes = [c["code"] for c in obs["tableau"][6]["face_down_cards"]]
        print(f"  ...and the agent CAN see their identities: {codes}")
    else:
        print("  ...and the agent CANNOT see their identities (hidden).")
    print(f"Stock view: {obs['stock']}")

    # A trivial greedy agent: prefer real moves, fall back to draw/recycle.
    # (It is deliberately simple -- the point here is the information level,
    # not winning play.)
    move_cap = 120
    for _ in range(move_cap):
        if session.is_won() or session.is_stuck():
            break
        actions = session.legal_actions()
        chosen = next((a for a in actions if a.kind == "move"), actions[0])
        session.apply_action(chosen.index)

    if session.is_won():
        result = "won"
    elif session.is_stuck():
        result = "stuck (no legal moves)"
    else:
        result = f"stopped at the {move_cap}-move demo cap"
    print(f"Result: {result} after {session.state.move_count} moves, "
          f"score {session.state.score}\n")


def main() -> None:
    print("\nAgentic Solitaire -- information level demonstration\n")

    # Realistic: a human's view -- face-down cards and stock stay hidden.
    play_with_information_level("human (imperfect info)", ObservationConfig.human())

    # Perfect information: every card revealed, for solver-style agents.
    play_with_information_level(
        "perfect information", ObservationConfig.perfect_information()
    )

    # Most restrictive: no stock info, only the top waste card.
    play_with_information_level("minimal", ObservationConfig.minimal())

    print("The same engine drives the MCP server. Start it with:")
    print("    python -m solitaire_analytics.mcp_server")


if __name__ == "__main__":
    main()
