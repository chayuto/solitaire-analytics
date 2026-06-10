# Eval fidelity and the tourA_v16 re-run: the prior full-game verdicts were measurement artifacts

Date: 2026-06-11. Status: COMPLETE, 48/48 games (both arms 24/24); final
leaderboard in `gemma4_finetune/play_runs/tourA_v16/leaderboard.txt`.
Companion to `docs/reports/20260608_orpo_pilot_and_fullgame_eval.md`
(sections 10 and 10.1); this document supersedes that report's absolute
claims. Headline tables use the final 24/24 figures; two
forensic cuts noted inline were computed at the 22-game interim and do not
move with the last games.

## Abstract

The 2026-06-09 Window A tournament (96 games, 4 model arms, 24 winnable
decks) concluded that the Gemma 4 E2B base and its ORPO checkpoints plateau
at foundation-cards (fc) 2-3, hit a "planning wall" by turn 4-6, and that
the move-contrast ORPO adapter v7-300 holds a small edge over the base
(+0.29 mean fc). An operator question ("did the player have the same game
context and prompt as the teacher?") triggered a fidelity audit of the
evaluation harness. The audit found four infidelities, two of them rule-level
(no recycle existed; forced moves were sent to the model), and two
distributional (a six-versions-stale prompt; duplicate/mis-pileable
foundation moves). After fixing all four and byte-validating the render
against real harvester prompts, a paired re-run (base and v7-300, same 24
decks, cap 100 model turns) overturned both headline conclusions. The base
nearly tripled its mean progress (2.83 to 7.92 fc; paired +5.08, better on
20/24 decks), the turn-4-6 wall moved to turn 68 (it was an artifact), and
the ORPO adapter's edge inverted into a large deficit (base better on 21/24,
v7-300 better on 0; paired +4.42 for base). Move-mix forensics identify the
adapter's failure mechanism: action-distribution collapse. Trained on pairs
whose chosen side was always a foundation/reveal move and whose rejected
side was always a tableau shuffle, the adapter drove draws and recycles to
near zero (0.5% draws vs the base's 23%; zero recycles) and shuffled
tableau cards 93% of the time, exactly the loop behavior the training was
meant to suppress. We conclude that all absolute numbers from the v1.0-era
harness are void, that off-policy move-contrast ORPO as minted is refuted
(not merely parked), and that the untuned base under the production v1.6
prompt is the strongest student configuration measured to date.

## 1. Background

Three instruments have now been shown to mispredict full-game play in this
project: validation loss (v7b-1000 was val-best and play-worst), the
20-state single-turn bench (ranked untuned E2B above the 31B teacher on
oscillation states, opposite of play), and, with this work, the full-game
play harness itself when its prompt and rules drift from production. The
project's standing rule, gate training decisions on graded full-game play,
is necessary but not sufficient: the play harness must also be faithful to
the deployed environment, or the graded numbers are precise measurements of
the wrong thing.

At stake were the Window A conclusions (`20260608` report, section 10):

1. Absolute: every model plateaus at fc 2-3; median loop-onset turn 4-6
   ("the planning wall"); illegal-move grounding deaths dominate.
2. Relative: v7-300 marginally beats base (+0.29 mean fc); scaling ORPO
   (v7b) regresses.

## 2. The four infidelities

Found 2026-06-10 by diffing the harness against real v1.6 harvester prompts
from corpus sessions #57947c, #0b0f2e, #4c73b8 (templateHash `7d2c6cad...`).

**I1. Stale prompt (v1.0-era vs a v1.5/1.6-dominated training mix).** The
harness `STATIC_PROMPT_HEADER` predated six prompt revisions. It asked for
the `confidence` + `alternative_move_index` schema (dropped at v1.1),
rendered a SEEN IN WASTE block (replaced by the DRAW TIMELINE at v1.2) and a
PRIOR REASONING block (absent in v1.6), and lacked: the DRAW TIMELINE block
and its interpreting rules, the CYCLE field, the two v1.5 stall counters on
the PROGRESS line, the v1.6 reveal-tag-anchored STRATEGY GUIDANCE bullets,
and the resign (-1) instruction. The v7b training pairs are 76%
v1.5/1.6-style (1410/1852), 19% v1.0-style (356), 5% other, so tuned
checkpoints were evaluated off their training distribution and the base off
the teacher's working distribution.

**I2. No recycle existed (rules-level).** The engine's `generate_moves`
emits `STOCK_TO_WASTE` only while the stock is non-empty and has no recycle
move type at all. Once the stock emptied, the waste was permanently lost, in
every full-game evaluation ever run on this harness. The teacher's wins
recycle 2-4 times routinely. This single defect makes many deals
mechanically much harder or unwinnable regardless of policy quality.

**I3. Forced positions were sent to the model.** The production harvester
auto-plays single-legal-move positions; the harness sent them to the model,
which sometimes answered with an out-of-range index at `legal=[0..0]` states
the deployed system never shows it. Three consecutive such answers
terminated the game as `illegal_move`. This manufactured a class of early
deaths that does not exist in production (10/24 base games in the old run
ended `illegal_move`).

**I4. Duplicate, mis-pileable foundation moves.** The engine generates one
ace-to-foundation move per empty foundation pile (four duplicates with
identical describe text), and a model picking the first could place an ace
on a wrong-suit pile index, after which every FOUNDATIONS line rendered it
under the wrong suit label.

### Fixes and validation

All four were fixed harness-side (commit `9023054`): a v1.6 renderer whose
static header is loaded byte-verbatim from a corpus extraction
(`gemma4_finetune/prompt_header_v16.txt`, sha-identical across 3 sessions /
3 builds); a synthesized recycle pseudo-move (waste reversed back into the
stock, order-preserving, CYCLE increments); auto-play of forced positions
(default in v1.6 mode, capped, logged); canonicalization of foundation moves
to the card's own suit pile; resign (-1) handled as a terminal outcome.

Validation tiers, all passing before launch:

- Turn-0 byte-equality against real corpus prompts on the two seeds with a
  clean pre-history prompt (77009315, 495097115): pre-LEGAL bytes, legal-move
  multiset (index-stripped; ordering is arbitrary), post-LEGAL bytes.
- Scripted mid-game checks: DRAW TIMELINE token accounting (??? for
  never-drawn during cycle 1; all-known after recycle; `{waste-top}` anchor;
  stock printed bottom-to-top left of the anchor; earlier-drawn waste
  recent-first to its right), CYCLE increments, "recycle stock" RECENT MOVES
  line, adaptive index width, the two PROGRESS stall counters.
- End-to-end smoke: auto-forced firing on the exact deck/state class where
  the old harness died (fc=4 by turn 10 where the old setup died at fc=2 on
  turn 8).

## 3. Method

Paired re-run: arms `base` (untuned Gemma4-E2B-IT-Text-int4) and `v7-300`
(move-contrast ORPO LoRA, play-best checkpoint), each playing the same 24
solver-confirmed-winnable decks (`data/benchmarks/winnable_decks.json`),
v1.6-faithful prompt, cap 100 model-decision turns (auto-forced moves do not
consume the cap; runaway-capped at 400/game), `--max-parse-failures 3`,
`--max-illegal-moves 3` consecutive. Decoding is mlx_lm's default greedy
sampling (temp 0.0), so each game is deterministic. One subprocess per game
(inference peak ~3.3 GB, released between games), sequential on the single
GPU, resumable (4 base games survived a battery pause mid-run and were
skipped on resume). Runner: `gemma4_finetune/tournament_A.py
--arms base,v7-300 --out-name tourA_v16 --prompt-version v1.6
--max-turns 100`. Comparator: the same 24 decks from the 2026-06-09 run
(`tourA`, v1.0 harness), cap 100, same decoding.

Cost: base arm 11.5 h wall-clock (median 30 min/game, median call 18.2 s);
v7-300 arm ~3.2 h (median 7 min/game, 4.2 s/call). The 4x speed difference
is response length: the base emits a ~4.9k-char thinking-channel response
per turn, v7-300 emits short direct JSON.

## 4. Results

### R1. The old absolute ceilings were artifacts

Base arm, paired across the same 24 decks:

| metric | v1.0 harness | v1.6-faithful | note |
|---|---|---|---|
| mean max-fc | 2.83 | 7.92 | paired delta +5.08; better 20, tie 3, worse 1 |
| median max-fc | 2.0 | 6.5 | |
| max fc | 18 | 18 | |
| decks at fc>=8 | 1 | 11 | |
| median loop-onset turn | 4 | 68 | turn at which fc stops rising for good |
| illegal-move deaths | 10/24 | 0/24 | replaced by auto-forced play |
| outcomes | 13 cap, 10 illegal, 1 stalled | 18 cap, 6 parse | |
| recycles used | impossible | 29 across 17/24 games | |
| wins | 0 | 0 | |

The "planning wall at turn 4-6" is gone: with the faithful prompt and rules,
the base makes progress to median turn 68. Pacing (mean fc at model-turn
25/50/75/100): 4.7, 6.4, 7.0, 7.9, slow continued progress, not an early
freeze. The canonical doom-loop deck 3263196305, fc=0 in every prior
measurement across every model and checkpoint, reached fc=10. Best games:
fc=18 (seed 1388178981, 9 face-down remaining, died on parse), fc=17 (seed
495097115, only 3 face-down remaining, died on parse), fc=15 (seed
4250754298, capped, 4 recycles).

Auto-forced moves (median 14/game) contributed zero direct foundation
cards; all 190 fc gains in the base arm occurred on model-decision turns.
Auto-play removes an artificial death mode without scoring for the model.

### R2. The relative verdict inverts: v7-300 is harmful on a fair eval

| metric | base v1.6 | v7-300 v1.6 |
|---|---|---|
| mean max-fc | 7.92 | 3.50 |
| median | 6.5 | 3.0 |
| max | 18 | 17 |
| decks at fc>=8 | 11 | 2 |
| outcomes | 18 cap, 6 parse | 21 cap, 3 parse |
| paired (base minus v7-300) | | +4.42 mean; base better 21, tie 3, v7 better 0 |

The old run's "+0.29 edge for v7-300" inverts to a 4-point deficit. The
explanation for the old edge: v7-300's compact JSON dodged the early-death
modes the broken harness inflicted on the base (CoT-related parse/illegal
deaths at manufactured states), so the adapter looked mildly better by
surviving an artifact, not by playing better. v7-300's own v1.0-to-v1.6
paired delta is +0.41 with 19 ties of 22, i.e. the adapter is nearly
insensitive to the richer, faithful prompt, consistent with its training
having narrowed it to a fixed response pattern.

### R3. Mechanism: action-distribution collapse (draw starvation)

Move mix over model decisions (v1.6 runs, interim cut n=22 for v7-300):

| action | base (n=2179) | v7-300 (n=2082) |
|---|---|---|
| tableau_to_tableau | 60% | 93% |
| draw (stock_to_waste) | 23% | 0.5% (11 draws total) |
| foundation plays | 9% | 4% |
| waste_to_tableau | 7% | 2% |
| recycle_stock | 1% (15) | 0 |

The minter (`mint_move_swap_pairs.py`) builds pairs whose chosen is always a
foundation or reveal move and whose rejected is always a tableau shuffle.
Drawing appears on neither side of any pair, so the odds-ratio gradient that
boosts chosen-over-rejected also implicitly suppresses every action class
absent from the pairs. The trained policy almost never draws and never
recycles; without stock flow it has nothing to do but shuffle the tableau:
93% tableau-to-tableau, immediate-reversal median 88 per game (max 98 of a
possible ~99) versus the base's 30, and a pacing flatline (mean fc 3.4 at
turn 25, 3.5 at turn 100; the base goes 4.7 to 7.9). v7-300's capped games
end with a median plateau of 96 turns (the base: 44). The adapter is not a
weaker player with the same shape; it is a player whose action space
collapsed onto the one move class its preference data never penalized in
isolation, which happens to be the loop move.

This is the cleanest available demonstration that preference data must
cover the full action space: contrast pairs that only ever oppose
"foundation/reveal vs shuffle" teach "shuffle when neither is available",
and they de facto remove drawing from the policy.

### R4. Residual failure modes on the faithful harness (base arm)

With grounding deaths gone, two modes remain:

1. **Cap exhaustion with late-game oscillation** (18/24): median
   plateau-at-end 44 turns; immediate-reversal median 30/game. The loop
   pathology is real but late: it begins after substantive progress, around
   fc 6-18, when the remaining moves require multi-step unburying. Three of
   the 18 capped games were still progressing at the cap (plateau <= 15
   turns), so cap 100 truncates some live games.
2. **Parse deaths** (6/24), and these are not token overruns: failed
   responses are normal-length (median 5130 chars; zero exceeded 7000), and
   zero of 27 failed turns (both arms) are rescuable by a string-aware or
   control-char-tolerant JSON parser. 18/27 show unescaped double quotes
   inside the JSON prose, characteristically the model echoing the prompt's
   own quoted phrases (e.g. tagged "(reveals a hidden card)") verbatim
   inside a JSON string. The production harvester tolerates this error
   class with per-turn retries (observed up to ~17); the harness aborts
   after 3 consecutive failures. Both of the base's best games (fc=18,
   fc=17 with only 3 face-down left) died this way, so the measured base
   numbers are lower bounds on faithful-harness play.

### R5. What survives from the old tournament

Within-run relative orderings on the old harness remain internally
consistent (all arms saw the same broken environment), so "v7b-600 and
v7b-1000 degrade monotonically vs v7-300" likely still holds; given R2-R3,
the more useful statement is that the entire v7/v7b family is refuted at the
formulation level. The overcook/format-bug observations (v7-600's
`8C_to_9H` content-in-integer-field) stand as observations about training
dynamics, but no longer carry decision weight: the track they belonged to is
closed by mechanism, not by checkpoint choice.

## 5. Discussion

**The proxy chain now has three broken links.** Validation loss, the
single-turn bench, and an unfaithful play harness each pointed in the wrong
direction at least once in this project. The only instrument currently
trusted is graded full-game play on a harness whose prompt is byte-validated
against production and whose rules are verified against the production
game's affordances (recycle, forced-move handling, resign). The general
lesson generalizes beyond this project: when the deployed environment
evolves (six prompt versions in five weeks here), the eval harness is a
second deployment surface that drifts silently unless pinned to the same
artifacts.

**The student picture changes materially.** Under faithful conditions the
untuned E2B with the production v1.6 prompt is a weak-but-sane player: it
uses the stock, recycles, makes slow progress to fc ~8 on average and 15-18
at best, and its failures are late-game conversion and JSON discipline, not
turn-4 freezes. The teacher-student gap is therefore smaller and differently
shaped than believed: the deployable v1.6 prompt is doing significant work
for the small model, exactly as it does for the 31B teacher.

**Why this also reframes the training program.** Every prior
student-training verdict (SFT inert, corpus filtering inert, ORPO
"promising") was reached with evaluations on the broken harness, and the
positive ORPO signal specifically is now attributed to an artifact. Training
levers should be re-derived from the new failure inventory: late-game
multi-step conversion, JSON string discipline, and (if pursued) preference
data that spans the full action space including draws and recycles.

## 6. Threats to validity

- Cap 100 truncates: 3 base games were still progressing at the cap, and 6
  base games died on a parse-tolerance (3 consecutive) that is stricter
  than production's retry depth. Both biases push the base DOWN, so R1-R2
  conclusions are conservative.
- Single deterministic run per (deck, model) at temp 0: no sampling
  variance, but also no measurement of the stochastic ceiling
  (best-of-N is future work).
- Deck set is sourced from teacher wins (winnable by construction but
  biased toward deals the 31B could win); 8/24 were dead-for-all on the old
  harness, and the set's discriminating power on the new harness is now
  much better (fc spread 0-18) but unquantified for harder deals.
- The base's long CoT responses sit near a ~4.9k-char norm against a
  2048-token output budget; we did not observe overruns, but max-token
  pressure on longer boards is untested.
- The v1.6 render was byte-validated at turn 0 on two seeds and
  format-validated mid-game; full-trajectory byte-equivalence against a
  live harvester session is not possible offline (move histories diverge).

## 7. Implications and next-window levers (reordered)

1. **Cheap headroom first (hours):** raise `--max-parse-failures` to ~10
   (production-like) and the cap to 150-200, then rerun only the 6
   parse-dead and 3 still-progressing base games. The fc=17/fd=3 game in
   particular may convert; the first full student win under faithful
   conditions is plausibly already inside the base policy.
2. **Best-of-N probe on the faithful harness** (temp ~0.7, N~8, the 2-4
   discriminating decks plus a sample of mid decks): measures the
   stochastic ceiling and gates RFT. The previous probe plan was on the
   broken harness and would have measured noise.
3. **If RFT proceeds:** rewards on graded fc from this harness; rollouts and
   any preference pairs must preserve the full action distribution (draws,
   recycles included). The R3 mechanism is the design constraint.
4. **JSON discipline lever** (prompt-side or decode-side): the parse mode is
   unescaped inner quotes echoing the prompt's quoted phrases; constrained
   decoding / grammar enforcement on the student side would eliminate the
   class without touching the model.
5. **ORPO move-contrast: closed.** Only revisit preference training with
   full-action-space pairs, and only after the best-of-N evidence says
   off-policy data is needed at all.

## 8. Artifacts

- Runs: `gemma4_finetune/play_runs/tourA/` (v1.0 harness, 96 games) and
  `gemma4_finetune/play_runs/tourA_v16/` (this re-run; `leaderboard.txt`,
  `leaderboard.json`, per-game `summary.json` + `turns.jsonl` + raw
  responses).
- Harness: `gemma4_finetune/play_deck_with_student.py` (v1.6 renderer,
  recycle, auto-forced, resign; `--prompt-version v1.0` preserves the old
  behavior), header `gemma4_finetune/prompt_header_v16.txt`, runner
  `gemma4_finetune/tournament_A.py`. Fix commit `9023054`.
- Prior report: `docs/reports/20260608_orpo_pilot_and_fullgame_eval.md`
  (sections 10, 10.1).
