# Solitaire MCP Server

The MCP server lets an agentic AI play Klondike Solitaire move by move through
the [Model Context Protocol](https://modelcontextprotocol.io). Its defining
feature is a configurable **information level**: when you start a game you
decide exactly how much the agent is allowed to see.

## Why an information level?

Real Solitaire is a game of *imperfect information* -- you cannot see face-down
tableau cards or the order of the stock. A solver, by contrast, plays with
*perfect information*. The server lets you pick any point on that spectrum so
you can study how an agent plays when it must reason under uncertainty versus
when it can see everything.

Three knobs control what is revealed:

| Knob        | Values                          | Meaning |
|-------------|---------------------------------|---------|
| `face_down` | `count`, `revealed`             | Whether face-down tableau cards' identities are exposed. `count` only reveals *how many* are hidden (what a human sees). |
| `stock`     | `hidden`, `count`, `revealed`   | Whether the draw pile is fully hidden, shown as a size only, or fully revealed in order. |
| `waste`     | `top`, `full`                   | Whether the agent sees only the playable top waste card, or the whole stack of previously drawn cards. |

Convenient presets are available on `ObservationConfig`:

- `ObservationConfig.human()` -- realistic imperfect information.
- `ObservationConfig.perfect_information()` -- everything revealed.
- `ObservationConfig.minimal()` -- the most restrictive level.

## Running the server

```bash
# As a module
python -m solitaire_analytics.mcp_server

# Or, after `pip install -e .`, via the console script
solitaire-mcp
```

The server communicates over stdio. Register it with any MCP client, e.g. in a
client config:

```json
{
  "mcpServers": {
    "solitaire": {
      "command": "python",
      "args": ["-m", "solitaire_analytics.mcp_server"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `new_game` | Deal a fresh game. Arguments: `seed`, `draw_count`, `face_down`, `stock`, `waste`, `include_legal_moves`. Returns the first observation. |
| `get_observation` | The current observation at the configured information level. |
| `get_legal_moves` | The legal actions, each with a stable `index`. |
| `play_move` | Apply the action at `move_index`; returns the result and new observation. |
| `set_information_level` | Change what the agent may see mid-game. |
| `render_board` | A compact, human-readable text board. |
| `game_status` | Won / stuck / score / move and redeal counts. |
| `suggest_move` | A hint from the built-in greedy strategy. |
| `get_game_log` | The full per-game session log (see below). |
| `save_game_log` | Write the session log to a JSON file. |
| `get_server_analytics` | Aggregate analytics across all games this session. |

## Session log

Every session records a log so a game can be reproduced and audited. The log
contains:

- `seed` -- the RNG seed; re-dealing with the same seed reproduces the game
- `draw_count` and `observation_config`
- `initial_state` -- the dealt **starting deck** and full game state
- `actions` -- every action taken, each with its `description`, `move`, and
  the `resulting_state`
- `result` -- final outcome (won / stuck / score / counters)

Logging is on by default; pass `log=False` to `GameSession` to disable it.
The starting deck is always captured up front, so even a game dealt without a
seed can be fully reconstructed from its log.

## Server-side analytics

Separately from the per-game session log, the server records a **cross-game
event stream** for detailed analysis of play across many games. Every game
lifecycle step is recorded as a structured event:

- `game_started` -- seed, draw count, information level
- `move` -- the action kind, description, score, and win/stuck flags
- `game_ended` -- the result (won / stuck), score, and counters
- `game_abandoned` -- a game replaced by `new_game` before it finished

`get_server_analytics` returns aggregates: games started/completed/won/stuck/
abandoned, win rate, actions logged by kind, and average game length.

To persist the raw event stream for offline analysis, set the
`SOLITAIRE_MCP_LOG_FILE` environment variable to a file path -- each event is
appended as one line of JSON (JSON Lines format):

```bash
SOLITAIRE_MCP_LOG_FILE=./solitaire_events.jsonl python -m solitaire_analytics.mcp_server
```

Events are also emitted via Python `logging` to **stderr** (stdout is reserved
for the MCP protocol and is never written to).

## The play loop

An agent plays by repeating three steps:

1. **Observe** -- read the observation (foundations, tableau, stock, waste).
2. **Choose** -- pick a `move_index` from `legal_moves`.
3. **Act** -- call `play_move`; the response includes the new observation.

Each legal action has a `kind`:

- `move` -- a tableau / foundation / waste move.
- `draw` -- deal `draw_count` cards from the stock onto the waste.
- `recycle` -- turn an exhausted waste pile back into the stock (offered only
  when the stock is empty and the waste is not).

Move indices are stable until the next action is applied, so an index from
`get_legal_moves` can safely be passed to `play_move`.

## Programmatic use

The server is a thin layer over `GameSession`, which you can use directly:

```python
from solitaire_analytics.game import GameSession, ObservationConfig

session = GameSession.new_game(
    seed=42,
    draw_count=1,
    observation_config=ObservationConfig.human(),
)

obs = session.observation()
for action in session.legal_actions():
    print(action.index, action.kind, action.description)

session.apply_action(0)
print(session.render())
```

See `scripts/example_mcp_agent.py` for a runnable demonstration.
