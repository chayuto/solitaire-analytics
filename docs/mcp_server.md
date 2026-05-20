# Solitaire MCP Server

The MCP server lets agentic AI players play Klondike Solitaire move by move
through the [Model Context Protocol](https://modelcontextprotocol.io). Each
connected agent plays its own concurrent game; per-session locks keep moves
within one game serial while allowing different sessions to make progress in
parallel.

Two defining features:

1. **Configurable information level** -- choose exactly how much the agent is
   allowed to see, from realistic human-style imperfect information to full
   perfect-information view.
2. **Server-side harvest emitter** -- when every move is made by an AI, the
   server can capture one structured decision record per `play_move` call,
   producing ingest-ready training data for distillation.

## Why an information level?

Real Solitaire is a game of *imperfect information* -- you cannot see face-down
tableau cards or the order of the stock. A solver, by contrast, plays with
*perfect information*. The server lets you pick any point on that spectrum so
you can study how an agent plays under uncertainty versus when it can see
everything.

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

## Multi-session model

Every `new_game` call mints an opaque `session_id` (UUID4). Pass it back on
every subsequent tool call so the server knows which game you're acting on.

* The registry caps simultaneously open sessions (default 100); call
  `end_session` to free a slot when a game ends.
* Per-session locks mean two agents can play in parallel; calls *within* one
  session still serialise.
* `list_sessions` returns a live snapshot of every active session.

```python
# Pseudocode for an agent client
result = call_tool("new_game", agent_id="alice", model="gemma-4-31b-it")
sid = result["session_id"]
while True:
    legal = call_tool("get_legal_moves", session_id=sid)
    chosen = pick_move(legal)
    out = call_tool("play_move", session_id=sid, move_index=chosen,
                    decision_meta={"confidence": 0.9, "reasoning": "..."})
    if out["won"] or out["stuck"]:
        break
call_tool("end_session", session_id=sid)
```

## Tools

| Tool | Description |
|------|-------------|
| `new_game` | Deal a fresh game; returns `{session_id, agent, observation}`. Accepts `seed`, `draw_count`, info-level knobs (`face_down`, `stock`, `waste`, `include_legal_moves`), and agent identity (`agent_id`, `model`, `provider`, `app_commit`). |
| `list_sessions` | Return a summary of every active session on this server. |
| `end_session` | Close a session and free its slot; records `game_abandoned` if unfinished. |
| `get_observation` | The session's current observation at its configured information level. |
| `get_legal_moves` | The legal actions, each with a stable `index`. |
| `play_move` | Apply the action at `move_index`; emits one decision record to the harvest stream. Accepts optional `decision_meta` (`confidence`, `reasoning`, `boardAnalysis`, `alternativeMoveIndex`, `thinkingText`). |
| `set_information_level` | Change what the agent may see mid-game. |
| `render_board` | A compact, human-readable text board. |
| `game_status` | Won / stuck / score / move and redeal counts. |
| `suggest_move` | A hint from the built-in greedy strategy. |
| `get_game_log` | The full per-game session log (see below). |
| `save_game_log` | Write the session log to a JSON file. |
| `get_server_analytics` | Aggregate analytics across all games + harvest stats + active session count. |

Except for `new_game`, `list_sessions`, and `get_server_analytics`, every tool
takes `session_id` as its first argument.

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
lifecycle step is recorded as a structured event, tagged with `session_id`:

- `game_started` -- session_id, seed, draw count, info level, agent metadata
- `move` -- the action kind, description, score, and win/stuck flags
- `game_ended` -- the result (won / stuck), score, and counters
- `game_abandoned` -- a session ended before it finished

`get_server_analytics` returns aggregates: games started/completed/won/stuck/
abandoned, win rate, actions logged by kind, average game length, harvest
totals, and active session count.

To persist the raw event stream for offline analysis, set the
`SOLITAIRE_MCP_LOG_FILE` environment variable to a file path -- each event is
appended as one line of JSON (JSON Lines format):

```bash
SOLITAIRE_MCP_LOG_FILE=./solitaire_events.jsonl python -m solitaire_analytics.mcp_server
```

Events are also emitted via Python `logging` to **stderr** (stdout is reserved
for the MCP protocol and is never written to).

## Decision harvest (training data for distillation)

When every move on the server is made by an AI agent -- e.g. the Gemma 4 E2B
distillation pipeline -- the server doubles as the authoritative training-data
source. Set `SOLITAIRE_MCP_HARVEST_FILE` to a JSONL path and every successful
`play_move` writes one record:

```bash
SOLITAIRE_MCP_HARVEST_FILE=./data/raw/server_harvest.jsonl \
    python -m solitaire_analytics.mcp_server
```

Each record is shaped to align with the fields
`scripts/ingest_exports.py` derives from the external collection harness, so
the file drops into `data/raw/` for ingestion with minimal adaptation. Fields:

| Field | Source |
|---|---|
| `id` | UUID4, fresh per decision |
| `sessionId` | The session that made the call |
| `turnIndex` | `state.move_count` *before* the move (so it can be joined to the log) |
| `timestamp` | Server-side milliseconds since epoch |
| `schemaTier` | `server-harvest-v1` |
| `outcome` | `success` (failures raise, no record is emitted) |
| `agentId` / `model` / `provider` / `appCommit` | From the `new_game` call |
| `drawCount`, `seed`, `infoLevel` | Session configuration |
| `prompt.legalMoves` | The full legal-action list the agent chose from |
| `prompt.observation` | The information-level-correct board view |
| `decision.moveIndex` | What the agent picked |
| `decision.chosenKind`, `chosenDescribe` | The move at that index |
| `decision.confidence`, `reasoning`, `boardAnalysis`, `alternativeMoveIndex` | Agent-supplied via `decision_meta` |
| `thinkingText` | Agent-supplied via `decision_meta` (top-level, like the ingest schema) |

Compared to the external collection harness, the server-side harvest closes
several open data-quality issues by construction:

- **Deck seed / `gameId`** -- always present (registry-assigned session_id, log carries the seed).
- **Auto-played moves untagged** -- not applicable; every move is an explicit `play_move` call.
- **Mixed info modes** -- `infoLevel` is stamped on every record from the session's `ObservationConfig`.

Agent identity (`model`, `provider`, `app_commit`) is the one thing the server
cannot infer -- the agent must declare it once via `new_game`, and the server
stamps every subsequent decision with the same values.

## The play loop

An agent plays by repeating three steps per session:

1. **Observe** -- read the observation (foundations, tableau, stock, waste).
2. **Choose** -- pick a `move_index` from `legal_moves`.
3. **Act** -- call `play_move(session_id, move_index, decision_meta=...)`; the
   response includes the new observation and the harvested `decision_id`.

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
