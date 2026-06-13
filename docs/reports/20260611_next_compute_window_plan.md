# Next compute window plan: parse-rescue rerun, then sampling headroom (2026-06-11)

Status: EXECUTING as of late 2026-06-11 (Phase 0 + Phase 1 launched; see
section 11). Written after the tourA_v16 verdict
(`docs/reports/20260611_tourA_v16_eval_fidelity_rerun.md`, commit 4e27b9d).
All numbers in this plan are measured, not estimated, unless marked "est".

## 1. Where we stand (one paragraph)

The faithful re-run overturned both prior conclusions: untuned Gemma 4 E2B int4
under the v1.6 prompt is the strongest student config measured (mean fc 7.92
over 24 winnable decks, loop onset t68, 0 illegal moves), and ORPO move-contrast
is refuted (v7-300 paired -4.42, 0/24 decks better, mechanism = draw
starvation). Still 0 wins, but the two best base games died on a JSON parse
artifact, not on play quality: fc=18 (seed 1388178981) and fc=17 with only 3
face-down cards left (seed 495097115). The single question this window should
answer: **is the first full-game win already inside the base policy, hidden
behind the harness's parse tolerance and the 100-turn cap?**

## 2. New evidence gathered for this plan (offline, zero GPU, 2026-06-11)

Measured against the actual recorded responses in
`gemma4_finetune/play_runs/tourA_v16/`:

1. **Every parse death was deterministic, not stochastic.** The harness calls
   `mlx_lm.generate` with no sampler (temp 0.0 greedy), and on parse failure it
   `continue`s with the state unchanged, so the retry re-renders a
   byte-identical prompt and gets the byte-identical bad response. The 27
   recorded failures are exactly 9 unique events x 3 identical retries.
   Consequence: raising `--max-parse-failures` alone is a no-op at temp 0.
2. **All 6 base-arm parse deaths are rescuable on CPU alone.** Two independent
   rescue tiers were tested against the recorded failed responses:
   - Tier A, field extraction: regex `"move_index"\s*:\s*(-?\d+)`, last match.
   - Tier B, quote-repair: escape unescaped inner double quotes in
     single-line string values, then strict `json.loads`.
   Both rescue 18/27 responses (the same 18), agree 18/18 on the extracted
   move_index, and those 18 are precisely the 6 base events (mi = 0, 1, 1, 2,
   3, 3). The 9 unrescuable responses all belong to the refuted v7-300 arm.
3. **The rescued move is the model's own chosen move** (it is sitting in the
   response text; the JSON wrapper around it is what broke). So the repair tier
   preserves a pure greedy-policy trajectory: no fidelity compromise, unlike a
   temp>0 resample.
4. **Cost basis (base arm, measured):** 2197 calls, mean 18.9 s/call, median
   18.2 s, p90 21.4 s; mean 29 min/game at cap 100, max 54 min. A cap-200 game
   is therefore ~60-65 min (est).
5. **Cap headroom exists.** At cap 100, 5 base games were still gaining
   foundation cards after t75: seeds 3123337720 (fc13 fd2), 3263196305 (the
   doom deck, fc10 fd11), 350743738 (fc10 fd3), 4221577640 (fc9 fd10),
   4161700176 (fc6 fd15).

## 3. Phase 0 (CPU only, ~1 h, can run on battery): the parse-rescue patch

Edit `gemma4_finetune/play_deck_with_student.py`:

1. **Tiered extraction in `extract_decision`** (or a wrapper):
   tier 1 = current strict path; tier 2 = quote-repair then strict
   (the line-level repair validated above); tier 3 = field extraction regex,
   last match. Record which tier fired (`json_ok_via: strict|repair|field`)
   in the turn record so rescued turns are countable.
2. **Temp-rescue backstop** for responses no tier can read (none recorded on
   base so far, but new positions will produce new failure shapes): on a parse
   failure, rebuild the sampler with `make_sampler(temp=0.3)` (production
   teacher temp) for the retry only, reset to greedy on success. This makes
   `--max-parse-failures 10` meaningful instead of 10 identical attempts.
   Add `--parse-retry-temp` (default 0.3).
3. **`--seeds` filter in `tournament_A.py`** (run a named subset instead of all
   24), or alternatively drive `play_deck_with_student.py` directly per seed.
   Prefer the tournament flag: it keeps the resume machinery and leaderboard.
4. **New out-name, never resume the old run with a new cap**: the resume path
   warns on cap mismatch by design. Use `--out-name tourA_v16_rescue`.

Validation before any GPU spend: re-run the offline tier test against the 27
recorded responses (expect 18/27, agreement 18/18), plus one smoke game at
`--max-turns 5`.

## 4. Phase 1 (GPU, ~6 h core + stretch): the rescue rerun

Rerun base arm only, `--prompt-version v1.6`, `--max-turns 200`,
`--max-parse-failures 10`, priority-ordered by win proximity. Greedy replays
the identical prefix, so each rerun reproduces the old trajectory up to the old
death/cap point, then continues.

| pri | seed | old outcome | old fc/fd | why |
|----|------|-------------|-----------|-----|
| 1 | 495097115 | parse_failure t97 | 17/3 | best win candidate; rescuable mi=3; teacher won this deck (#109f85) |
| 2 | 1388178981 | parse_failure t57 | 18/9 | highest fc reached; rescuable mi=0 |
| 3 | 3123337720 | max_turns | 13/2 | only 2 face-down left, still progressing at cap |
| 4 | 4250754298 | max_turns | 15/8 | high fc |
| 5 | 239901548 | max_turns | 13/4 | high fc, low fd |
| 6 | 350743738 | max_turns | 10/3 | progressing at cap |
| 7 | 3263196305 | max_turns | 10/11 | the doom deck; progressing at cap; narrative value |
| stretch | 2703165610, 405489085, 4197389931, 3841057237, 4221577640, 4161700176 | mixed | low | parse-dead or progressing but far from done |

Core set (1-7) at ~60 min/game = ~7 h (est); acceptable overnight. Stop
anywhere; the tournament resume machinery picks up the remainder next window.

Readout discipline: the greedy prefix is identical by construction, so the
result of interest is simply the delta after the unlock point: final fc/fd,
outcome, and whether any game reaches `outcome=win`. Report rescued-turn counts
(`json_ok_via != strict`) and any temp-rescue firings separately, since
temp-rescued turns are the only non-greedy moves in the run.

Safety rails (unchanged from the tournament runs): subprocess-per-game
(~3.3 GB peak), strictly one GPU job at a time, no training concurrently,
resume-safe outputs, machine on mains.

## 5. Phase 2 (GPU, gated): sampling headroom, best-of-N

Run only if Phase 1 completes with window left, or next window.

- **If Phase 1 produced a win**: reframe as win-rate estimation and win
  harvesting. Same mechanics, but the goal becomes collecting student-policy
  win trajectories (the seed data RFT/on-policy distillation needs).
- **If Phase 1 produced no win**: pass@N probe to measure how far greedy is
  from the policy's sampled ceiling. Patch: add `--temp` to the harness via
  `make_sampler` (greedy stays the default). Design: temp 0.7, N=4 per deck,
  decks = the 3-4 nearest misses from Phase 1. Cost ~4 h/deck at cap 200
  (est), so one deck per spare night-slot, more realistically a dedicated
  window. pass@4 > 0 on any deck gates the RFT lever.

Phase 2 exists because greedy is a lower bound on the policy: production runs
temp 0.3, and the teacher itself wins-or-stalls stochastically on the same
deck.

## 6. Phase 3 (CPU, parallel, battery-safe filler)

- **Constrained decoding research**: the right long-term fix for the JSON
  class is grammar/schema-constrained generation in mlx_lm (logit-processor or
  outlines-style). Research + prototype offline; no GPU needed to write it.
  Resurfaces as P0 if new unrescuable parse shapes appear in Phase 1.
- **Ingest** any harvester drops that land overnight (v1.6 sessions keep
  flowing; triage-first per the solitaire-ingest skill).
- **HF dataset re-push** when the user provides/rotates the token
  (`data/publish` is ahead of the 2026-06-08 push).

## 7. Decision gates after Phase 1

| Outcome | Interpretation | Next move |
|---|---|---|
| Any win | First faithful student win; policy can win games | Win-rate estimation (Phase 2 reframed), archive trajectory, start RFT data design; update README/report "0 wins" claims |
| No win, fc gains continue (e.g. several games >fc15 or fd<=2) | Ceiling is cap/tolerance tail, not policy | Raise cap again on the closest games; then best-of-N |
| No win, fc stalls where it stalled before | Ceiling is policy quality at greedy | Best-of-N probe (Phase 2) before touching training |
| New unrescuable parse shapes dominate | Decoding discipline is the binding constraint | Constrained decoding becomes P0 |

## 8. Parked levers (kept alive, evidence-gated, not scheduled)

- **ORPO with full-action-space pairs**: redesign so chosen/rejected sets span
  draw/recycle/foundation/tableau (the v7 failure was pairs that never
  contained draws). Gate: best-of-N shows wins exist to prefer, giving
  on-policy preference data a target.
- **v7b-600 / v7b-1000 faithful re-grade**: same pair-construction as v7-300,
  so the draw-starvation prediction transfers; only worth 3 h/arm if GPU is
  otherwise idle. Low priority.
- **Won-games SFT re-test**: prior HOLD verdicts (v4-A, v2/v3/v5/v5b) were
  graded on the unfaithful harness, so their full-game conclusions are
  formally suspect. Cheap re-grade of one existing checkpoint on the faithful
  harness would tell us whether any old verdict flips. Gate: idle window after
  Phases 1-2, or win-corpus growth (v1.6 keeps producing 31B wins).
- **Teacher-side levers** (remote, not this machine's GPU): resign-lever
  reliability (fires 1/3 on proven-dead v1.6 boards), best-of-N teacher replay
  of winnable decks, the 240 s reasoning-overrun timeout. Tracked in
  `docs/reports/20260607_more_win_data_strategy_research.md`.

## 9. Strategy frame (why this ordering)

Cheapest-information-first: Phase 0 is free and was already de-risked by the
offline measurement above; Phase 1 spends ~7 GPU-hours directly on the single
highest-value open question (first win); Phase 2 spends more only if Phase 1
says the answer is "not at greedy"; training-side levers spend the most and
stay gated until eval-side headroom is exhausted. This is the same discipline
that caught the three broken proxies (val loss, single-turn bench, unfaithful
harness): never buy compute before the measurement layer underneath it is
trusted.

## 10. Strategic review addendum (operator discussion, 2026-06-11)

Recorded from the trajectory discussion held after the plan above was written.
Where this section disagrees with sections 5-8 on ordering, this section wins.

### 10.1 Trajectory verdict

The measurement trajectory is right (three broken proxies caught and fixed;
instruments now trustworthy). The training trajectory as executed is
falsified, not merely "in progress": v2, v3, v4a, v5, v5b, v7, v7b have all
lost to not-training, and the strongest student config ever measured is
untuned base + the v1.6 prompt. The shipped v1.1 (gemma-3n) student was
validated only on proxies since shown broken; its faithful full-game standing
is unmeasured. Three mechanistically different attacks on the corpus
(filtering, won-only, preference-against) all failed, which points at the
corpus itself: the teacher wins ~31%, the corpus is majority pathology, and
imitation cannot exceed its teacher.

Two structural blind spots named in the review:

1. **The teacher does not have to be an LLM.** We own a sound solver and,
   since deck-logging landed, exact board reconstruction. Solver-as-teacher
   (unlimited pathology-free winning trajectories, rendered through the v1.6
   template) is the only training experiment available that bypasses the
   teacher ceiling. It became the designated next *training* experiment.
2. **Product and research goals have diverged.** A product-grade local advisor
   is probably base + v1.6 + a thin engine-side wrapper (anti-undo filter,
   1-ply lookahead), no training at all. The research output (three-proxies
   story + action-distribution collapse) is close to a publishable
   negative-results write-up. These want different windows and should be
   chosen explicitly rather than split implicitly.

### 10.2 The winnable-to-won teacher program

Operator question: instead of (or alongside) replacing the teacher, convert
winnable boards into teacher WINS. Assessment: this is the strongest version
of the teacher track and the only lever that attacks corpus quality while
keeping the dataset's differentiator (real model reasoning on real
decisions). Four mechanisms, in increasing value:

1. **Solver pre-screen of decks** (winnable-only feeding): dead deals are
   ~10% of budget, so per-attempt win rate rises ~31% to ~34% (est). Marginal.
2. **Best-of-N same-deck replay**: teacher at temp 0.3 wins-or-stalls
   stochastically; `solitaire.chayuto.com/?seed=` makes replay a queueing
   decision. At p~0.34: P(win within 2) ~56%, within 3 ~71%, within 5 ~87%.
3. **Early-kill discipline is the real yield multiplier**: wins finish in
   ~200-340 moves, losses burn to the 500-700 cap. Killing confirmed
   loops/proven-dead boards at ~turn 150 cuts move-budget per win from ~1450
   to ~600 (est), a ~2.5x win-trajectory yield per unit of 31B compute.
   Needs the auto-terminator (or formalized operator kills).
4. **Solver post-hoc labeling of deck-logged sessions** (the sleeper):
   `true_world_winnability.py` can answer "still winnable?" at any turn;
   binary search finds the exact fatal move in a lost-on-winnable session.
   This converts every loss into blame-assigned training signal and yields
   same-position preference pairs that span the full action space (draws are
   often win-preserving), fixing the v7 design flaw with real data. Nuance:
   oscillation moves are usually win-preserving, so the solver label does NOT
   filter loops; the behavioural detector still does that. The filters compose.

What it cannot buy: it never exceeds the teacher ceiling (~43-52%
imperfect-info), wins still contain survivor-biased pathology (#3e91a0 won
through a 23x loop), and throughput stays session-bound (~1-2 wins/day even
with all four mechanisms; +100 wins is a 2-3 month program). Complements, not
rivals, with solver-as-teacher: this fixes yield, the solver fixes the ceiling.

### 10.3 The gate experiment (runs BEFORE scaling the win harvest)

We have never validly tested whether teacher wins fix the student: the
won-only verdicts (v5/v5b HOLD) were graded on the broken bench and the
unfaithful harness, so they are void. The corpus now holds roughly two dozen
win trajectories and `build_wononly_corpus.py` exists. The gate:

> Rebuild the won-only corpus from today's wins, one LoRA run, grade on the
> faithful harness against untuned base.

- Beats base: the data consumer is validated; the winnable-to-won program
  becomes the top teacher-side priority.
- Still loses at N~25 wins: scaling the harvest 2.5x pours water into a leaky
  bucket; the binding constraint is win count or quality, and the
  solver-teacher pilot becomes the cleaner test of whether wins help at all.

Validate the consumer before scaling the producer.

### 10.4 Revised lever order (supersedes section 8 ordering)

1. Tonight: Phase 0 + Phase 1 (parse-rescue rerun) as planned.
2. Next window: the won-only faithful re-grade gate (10.3), then Phase 2
   best-of-N if GPU remains.
3. Gated by 10.3's outcome: winnable-to-won teacher program (10.2) vs
   solver-teacher pilot (10.1) as the next training experiment.
4. Teacher prompt polish stays in maintenance mode; the engine-wrapper
   product path and the research write-up await the explicit
   product-vs-research call.

## 11. Execution log (2026-06-11 night)

- Phase 0 patch: tiered extraction (strict / quote-repair / field) with
  `json_ok_via` logging, temp-rescue backstop (`--parse-retry-temp`, default
  0.3, fires only after a strict+repair+field miss), `--seeds` +
  `--max-parse-failures` passthrough in `tournament_A.py`.
- Offline validation target: 18/27 recorded failures rescued, 18/18 agreement
  between repair and field tiers (must reproduce the section 2 measurement
  through the patched code path).
- Phase 1 launch: base arm, cap 200, parse tolerance 10, out-name
  `tourA_v16_rescue`, seeds in section 4 priority order (7 core + 6 stretch).
  Rolling validation: trajectory-fidelity check vs the cap-100 run after ~15
  turns of game 1 (greedy prefix must reproduce fc/fd exactly), then per-game
  summary checks.
- Game 1 (seed 495097115) fidelity confirmed live: 15/15 prefix turns
  reproduce the cap-100 run exactly; the predicted parse death at t94
  re-occurred byte-identically and the repair tier rescued it
  (`via=repair`, mi=3); the old run died one draw away from its next
  foundation card.
- **Game 1 result: fc=40/52, fd=0 at the 200-turn cap, outcome max_turns,
  2 repair-rescued turns, 0 temp rescues.** Pacing: 50-turn plateau at fc=23
  (TS-pile col1/col5 oscillation), then a breakout cascade of 17 foundation
  cards in the final 18 turns, cut mid-cascade by the cap. Post-hoc exact
  adjudication (new tool `gemma4_finetune/adjudicate_final_position.py`:
  zero-drift engine replay of all 200 decisions, then the sound best-first
  solver on the exact final position): **SOLVED in 15 explored states.** The
  first faithful base-policy win was truncated by the cap, not by play. The
  precise claim: the final position is a near-forced winning cascade the
  model was already executing at 1 card/turn; the end-to-end win artifact
  needs a cap ~250-300 rerun of this seed (deterministic replay, ~70-90 min).
- Adjudication discipline for the rest of the night: every max_turns game
  gets the replay+solve treatment (SOLVED-small-n = banked win truncated by
  cap; UNSOLVABLE = the deal or the play killed it, no cap rerun warranted).
- Games 2-4 adjudicated. #1388178981: fc=18 flat from t43 (157 turns),
  final SOLVED n=4057 = behavioural plateau on a winnable board.
  #3123337720: fc=22 fd=1 flat from t111 (89 turns), final SOLVED n=103 =
  stalled ON a trivially-won position (the crispest close-out-failure
  evidence yet; prime best-of-N test deck). #4250754298: fc=15 flat from
  t51, final UNSOLVABLE n=112; deck winnable at t0, so the model PLAYED INTO
  the dead position. First fatal-move localization on a student trajectory
  (mechanism 4 of section 10.2, working same-night): definitive bracket
  [t1, t42] (t0 SOLVED n=312729; t42 UNSOLVABLE n=565224; t10-t35 UNKNOWN at
  600k nodes), suspect region the t22-26 waste-burying sequence. Exact-turn
  refinement deferred to a CPU-only window with a bigger node cap.
- Scoreboard semantics note: on benchmark decks (all winnable at t0),
  max_turns + UNSOLVABLE final means the policy threw the game, never that
  the deal was dead.

### 11.1 Window closed (2026-06-12): final tally

Operator ended the window after 5 complete games (game 6, seed 350743738,
killed mid-game; no summary written, so resume replays it cleanly). GPU
verified free. All games pure greedy: 9 parse events across the night, all
rescued by the repair tier, 0 temp rescues, 0 artifact deaths.

| seed | cap-100 fc | cap-200 fc/fd | flat since | final position | profile |
|---|---|---|---|---|---|
| 495097115 | 17 (parse-dead) | **40 / 0** | cascading at cap | SOLVED n=15 | cap-truncated WIN |
| 1388178981 | 18 (parse-dead) | 18 / 9 | t43 | SOLVED n=4057 | plateau on winnable |
| 3123337720 | 13 | 22 / 1 | t111 | SOLVED n=103 | stalled ON a won position |
| 4250754298 | 15 | 15 / 8 | t51 | UNSOLVABLE n=112 | threw the game, fatal in [t1, t42] |
| 239901548 | 13 | 13 / 4 | t36 | SOLVED n=1499 | plateau on winnable |

Mean fc on these 5 decks: 15.2 (cap 100) -> 21.6 (cap 200 + parse rescue).
The night's verdict in one line: the parse artifact is fully cured, one
near-certain win was banked, and the remaining ceiling is behavioural
(loop-stall on winnable or even trivially-won positions) plus an
early-game thrown board, neither of which more cap fixes at greedy.

Resume the remaining 8 queue games with the identical launch command
(resume-skip honors the 5 completed summaries):

    .venv/bin/python gemma4_finetune/tournament_A.py --arms base \
      --seeds 495097115,1388178981,3123337720,4250754298,239901548,350743738,3263196305,2703165610,405489085,4221577640,4161700176,3841057237,4197389931 \
      --max-turns 200 --max-parse-failures 10 --parse-retry-temp 0.3 \
      --out-name tourA_v16_rescue --prompt-version v1.6

Next-window candidates ranked by tonight's evidence (operator picks):
1. Cap-300 banking run of seed 495097115 (~2 h, deterministic replay,
   near-certain first end-to-end student win artifact).
2. Best-of-N temp probe on the two stall decks (3123337720 starts from a
   solver-trivial won position; 1388178981 from SOLVED n=4057): the
   designed cure for deterministic loop cycles, and the RFT gate.
3. Finish the rescue queue (8 games, ~11 h) for the complete cap-200 map.
4. The won-only faithful re-grade gate (section 10.3).

### 11.2 Window resumed and completed: full 13-game tally (2026-06-12/13)

Operator resumed the queue; all 13 games completed (resume-skip honored the
first 5; killed game 6 replayed cleanly). Leaderboard: n=13, won=0,
meanFC=14.2, medFC=11.0, maxFC=40, all outcomes max_turns. Paired against
the same 13 decks at cap 100: total fc 137 -> 184 (+47), better on 7,
equal on 6, worse on 0. 18 parse events across the window, every one
rescued by the repair tier, 0 temp rescues: all 13 trajectories are pure
greedy, and parse deaths went 6-of-24 (cap-100 run) to 0-of-13.

| seed | cap-100 fc | cap-200 fc/fd | flat since | final position | profile |
|---|---|---|---|---|---|
| 495097115 | 17 | **40 / 0** | cascading | SOLVED n=15 | cap-truncated WIN |
| 1388178981 | 18 | 18 / 9 | t43 | SOLVED n=4057 | stall on winnable |
| 3123337720 | 13 | 22 / 1 | t111 | SOLVED n=103 | stall ON won position |
| 4250754298 | 15 | 15 / 8 | t51 | UNSOLVABLE n=112 | thrown, fatal [t1, t42] |
| 239901548 | 13 | 13 / 4 | t36 | SOLVED n=1499 | stall on winnable |
| 350743738 | 10 | 10 / 3 | t81 | SOLVED n=159 | stall on near-won position |
| 3263196305 | 10 | 11 / 7 | t108 | SOLVED n=13626 | stall on winnable (doom deck) |
| 2703165610 | 10 | 10 / 12 | t85 | UNSOLVABLE n=72 | thrown |
| 405489085 | 10 | 10 / 9 | t38 | SOLVED n=335 | stall on easy position |
| 4221577640 | 9 | 11 / 5 | t180 | SOLVED n=415 | slow grind, cap-limited |
| 4161700176 | 6 | 8 / 13 | t120 | UNKNOWN at 700k | inconclusive |
| 3841057237 | 4 | 12 / 9 | t146 | UNSOLVABLE n=144 | thrown |
| 4197389931 | 2 | 4 / 13 | t77 | SOLVED n=10790 | stall on winnable |

Window conclusions:
1. The parse artifact is eliminated as a cause of death (18/18 repair
   rescues, zero policy contamination).
2. One proven cap-truncated win (495097115). No other cascade existed:
   every other game ended flat, so further cap raises buy nothing at greedy.
3. The dominant residual failure (8 of 13) is loop-stall on a winnable, in
   three cases trivially-won, position. Median stall onset around t80-110;
   stalls persisted 54-164 turns. This is the precise target for the
   best-of-N temp probe (lever 2): deterministic cycles are exactly what
   sampling breaks.
4. Three thrown games (23%) where the policy converted a winnable deal into
   a structurally dead board; fatal-move bracketing works and the batch
   localization pass (mechanism 4 of 10.2) now has three targets.
5. The night's instruments (tiered parse rescue, exact final-position
   adjudication, fatal bracketing) all validated on first contact; the
   adjudicator should run as standard post-processing on every future
   full-game eval.

### 11.3 Second window (2026-06-12 late): banking run + fatal moves + warm start

- Lever 1 launched: cap-300 banking run of seed 495097115 (deterministic
  greedy replay of the cap-200 trajectory, then the cascade continues).
- Harness extended while the GPU replayed: `--temp` (policy sampling) +
  `--sample-seed` (mx.random.seed; without it every mlx process samples
  identically) + `--warm-start-from/--warm-start-until` (engine-side replay
  of a recorded turns.jsonl, no model calls, with a per-decision drift gate:
  the re-rendered prompt must match the recorded prompt_chars exactly).
  Bookkeeping validated offline: 200/200 decisions of seed 3123337720
  re-render to the exact recorded byte lengths, zero drift.
- Fatal-move localization completed on the remaining two thrown games, at
  SINGLE-TURN precision, and both killing moves are REVEAL moves:
  seed 2703165610 dies at t150 "Move 8H plus 1 more from column 3 to
  column 6 (reveals a hidden card)" (t149 SOLVED n=9144 -> t150 UNSOLVABLE
  n=72, winnable for 150 turns then one move kills it); seed 3841057237
  dies at t135 "Move TC from column 3 to column 7 (reveals a hidden card)"
  (t134 SOLVED n=16079 -> t135 UNSOLVABLE n=21960). The move class the
  prompt guidance prioritizes and the model takes ~90% when offered is what
  destroyed both boards. Honest framing: perfect-info verdicts, so these
  were unknowable risks, not certain blunders; but each position carried a
  solver-proven win-preserving alternative, which is exactly the preference
  signal mechanism 4 mines. Game 4's bracket stays [t1, t42] (UNKNOWN wall).

### 11.4 FIRST END-TO-END STUDENT WIN (2026-06-12, banked and verified)

The cap-300 banking run of seed 495097115 finished `outcome=won`, 52/52
foundations, 211 turns, 2 repair-rescued parses, 0 temp rescues: a pure
greedy trajectory by the untuned Gemma 4 E2B int4 base under the v1.6
prompt on the faithful harness. The final card (KC) was auto-forced at
t210 after the last model decision sent KD home at t209. Independently
verified by full engine replay (210 decisions + trailing auto-forced
moves, zero drift, 52/52). Artifact:
`gemma4_finetune/play_runs/win_banking/seed495097115_cap300/`. The teacher
won the same deck (#109f85), so this deal now carries paired teacher and
student wins. Verification note: the replay verifier initially reported
51/52, a tail-gap in the TOOL (it skipped trailing auto-forced moves after
the last decision); fixed in `adjudicate_final_position.py` same day.

First-win context: every trained checkpoint (v2-v7b) made full-game play
worse; the interventions that produced this win were prompt fidelity,
parse-rescue, and an honest cap. Zero gradient updates. Any future
training must now beat a base policy that wins games.

Warm-start wiring was validated end-to-end before the sampling probe: 150
decisions of seed 3123337720 replayed engine-side in seconds, then ONE
live greedy call at t150 produced a response byte-identical to the
recorded run.

### 11.5 Best-of-4 stall probe: sampling does NOT break the close-out failure

Design: 4 samples on seed 3123337720, each warm-started engine-side to the
t111 stall onset (fc=22 fd=1, greedy had then sat flat for 89 turns on a
SOLVED n=103 position), played live to cap 250 at temp 0.7 with distinct
mx.random seeds. 139 live turns per sample, ~46 min each.

Result: pass@4 = 0. Every sample PROGRESSED but none finished:

| sample | fc | fd | final position |
|---|---|---|---|
| s1 | 28 | 0 | SOLVED n=39 |
| s2 | 23 | 1 | SOLVED n=62 |
| s3 | 23 | 1 | SOLVED n=59 |
| s4 | 29 | 0 | SOLVED n=39 |

All four finals are nearer-forced wins than the position greedy stalled on,
and the policy still would not execute them. Distinct seeds produced
genuinely distinct trajectories (s1/s4 revealed the last card, s2/s3 did
not), so the sampling mechanism worked; the close-out failure survived it.

Reading: the stall is POLICY-DEEP, not decoding-determinism. This is the
student phenotype of the teacher's #3e91a0 endgame loop (124 turns at
fd=0 before stochastically escaping), consistent with the
doom-loop-is-base-model-deep finding. Consequences for the lever ranking:
cheap RFT-via-sampled-stall-escapes is weakly refuted (no win trajectories
to prefer, at least at N=4 / temp 0.7 / 139-turn budgets / this deck);
solver-as-teacher gains weight (clean cascades are exactly what solver
trajectories teach); the engine-wrapper product path gains the most (these
finals are mechanically executable by 1-ply logic). Caveats: one deck, one
temperature, N=4.

Adjudicator extended same-night: warm-start-aware replay (replays the
warm-start source before the run's own records) so warm-started runs verify
end-to-end like everything else.

### 11.6 WON-ONLY SFT GATE: strong signal, but confounded (verdict PENDING a control)

The gate from section 10.3 was trained and graded. Corpus: 36 won teacher
sessions, with the 13 tourA_v16_rescue benchmark decks HELD OUT of training
(`build_wononly_corpus.py --exclude-seeds ...`, since the benchmark is built
from win records and un-held-out eval would be a memorization probe). 2492
rows / 18 won games, split 1881/148/463. Adapter `adapters_gate` (LoRA
identical hypers to v2/v5, `lora_config_gate.yaml`), 1000 iters, val bottomed
at iter 300 (0.212) then drifted to 0.233 (the preregistered overfit
signature; final checkpoint kept as the eval arm). Eval: paired full game vs
the base numbers already in `play_runs/tourA_v16_rescue`, same 13 HELD-OUT
decks, cap 200, faithful harness, greedy base policy.

Headline (verified from summaries): gate better on 12/13, worse on 1 (by one
card), 4 OUTRIGHT WINS vs base's 1, mean paired delta +13.4 fc, median +4,
gate meanFC 27.5 vs base 14.2. Adjudicated gate finals add 2 cap-truncated
wins (#3123337720 fc44 SOLVED n=10; #405489085 fc25 SOLVED n=30), so 6 of 13
gate games are won-or-near-forced-win.

| seed | gate | base | delta | gate outcome / final |
|---|---|---|---|---|
| 4221577640 | 52 | 11 | +41 | won |
| 239901548 | 52 | 13 | +39 | won |
| 1388178981 | 52 | 18 | +34 | won |
| 3123337720 | 44 | 22 | +22 | cap, SOLVED n=10 |
| 405489085 | 25 | 10 | +15 | cap, SOLVED n=30 |
| 495097115 | 52 | 40 | +12 | won |
| 4161700176 | 12 | 8 | +4 | cap, UNSOLVABLE |
| 3263196305 | 14 | 11 | +3 | cap, SOLVED n=3628 |
| 350743738 | 12 | 10 | +2 | cap, SOLVED n=538 |
| 3841057237 | 13 | 12 | +1 | cap, UNSOLVABLE n=16 |
| 4197389931 | 5 | 4 | +1 | cap, UNKNOWN |
| 4250754298 | 16 | 15 | +1 | cap, UNSOLVABLE n=96 |
| 2703165610 | 9 | 10 | -1 | cap, SOLVED n=102 (base threw it: UNSOLVABLE) |

THE CONFOUND (why this is not yet a PROMOTE): won-only training degraded the
adapter's JSON discipline ~6x, so the gate triggers the temp-0.3 parse-retry
path 163 times across the 13 games (base: ~12). The best-of-4 probe (11.5)
established that temperature injection alone breaks deterministic stalls. So
the gate's advantage is entangled: (a) a better learned move policy, vs (b)
self-induced stochasticity from its own brokenness. Two reasons to suspect
real policy over pure noise: the gate's per-deck illegal-grounding failure
(the pre-patch first run died on index 7 x3) was the only NEW regression,
and on #3123337720 the gate gained +22 (to a forced win) where pure temp-0.7
sampling in the probe gained only +6/+7. Suggestive, not conclusive.

CONTROL LAUNCHED (`run_base_temp03_control.sh`): base, no adapter, full
--temp 0.3, on the 5 decks the gate won or near-won
(4221577640/239901548/1388178981/495097115/3123337720), one sample each,
cap 200. Decision rule: if base-at-temp-0.3 also wins ~these, the gate edge
is mostly stochasticity (verdict HOLD, won-only does not teach policy);
if base-at-temp-0.3 still stalls where the gate won, the gate learned real
Solitaire (verdict PROMOTE, green-light the winnable-to-won harvest scaling
of section 10.2). The harness illegal-move path was patched to arm the same
temp-0.3 retry as parse failures (production absorbs invalid responses via
its retry budget; the harness now matches), so both arms are measured under
identical, deployment-faithful retry rules.
