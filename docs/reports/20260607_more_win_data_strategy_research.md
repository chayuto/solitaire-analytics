# More win data from the Gemma teacher: strategy research and log-mining

**Date:** 2026-06-07 | **Author:** analysis pass (6 parallel research strands: codebase capability, corpus numbers, ML methods, Klondike domain, reasoning/behaviour mining, prompt/mechanics mining) | **Scope:** how to grow the supply of WINNING Klondike trajectories that feed the Gemma-4-E2B / Gemma-3n student distillation, plus everything the ~32.8k-interaction game log says we can improve.

All numbers are read off real corpus output or cited literature; confidence is flagged where it matters. The corpus is DRAW-1 Klondike (confirmed). Fidelity rule: the student imitates (visible-state, natural-language reasoning, chosen-move) under IMPERFECT information; any lever that produces data the student cannot reproduce from visible state is a trap and is marked as such.

## TL;DR (one screen)

Where we are: draw-1, win rate about 31% on the 31B teacher (23/75 sessions), which is exactly average-human level. The perfect-information ceiling for draw-1 is about 90.5%; the realistic ceiling for a competent IMPERFECT-information player is about 43 to 52%. So roughly 12 to 21 points of win rate are reachable by better play, and about 40 points are permanently locked behind hidden cards. Dead deals are only about 10% of draw-1 deals, so the dominant opportunity is BEHAVIOURAL (loops on winnable boards), not dead-deal removal.

A central log-mining finding: the loop is frequently outside the 10-move window at the moment it matters. RECENT MOVES is hard-capped at 10, and in 34 of 42 sessions with a dominant oscillation the loop is outside that window at the export snapshot. The anti-undo rule (which references RECENT MOVES) is consequently violated 18 to 38% of the time it is testable, and about twice as often in losses (21% vs 11%). Nuance from a follow-up dig (section 4.1): the dominant loop is TIGHT (median period 2), so while actively looping the model often CAN see the back-and-forth and loops anyway; and WON sessions also reverse a dominant card a median of 8 times (max 23) but productively. So this is partly a visibility gap and partly an actioning gap, the signal must be progress-gated (reversal WITHOUT foundation or reveal progress), and a render alone is necessary but not sufficient.

The five highest-leverage moves, cheapest first:
1. Render loop visibility (a persistent "recently shuffled without progress" summary that survives beyond 10 moves), replacing the v1.5 stall counters that provably do nothing (0 resigns, 9% reasoning uptake while they climb to 292). Prompt/render, state-not-logic. The biggest behavioural lever.
2. Raise the 240s provider timeout or cap the reasoning budget. 47% of turns are non-success, almost entirely provider timeouts, and the timeouts are reasoning-overrun (latency correlates with thinking tokens at r=0.85; v1.5 thinking p90 is about 9.8k tokens against a 240s wall). Harness ask. Biggest lever on usable-data-per-game.
3. Best-of-N replay of winnable decks plus drop solver-confirmed-dead decks. The win check is a free perfect verifier; about 10% of decks are dead. Harvester ask plus offline solver.
4. Remove the prescriptive "prefer exposing a face-down card" heuristic from the prompt body (A/B on 31B). It is the rationalization engine of the loop (justifies 85.9% of empty shuffles in losses) and violates the project's own no-injected-logic principle. Keep the accurate per-move reveal tag.
5. Fix the student objective so wins actually stick: on-policy distillation plus an ORPO loop-penalty. The held v2 result (imitating the teacher's loops reproduces looping) is an imitation-objective problem; pure win-SFT will not fix it.

The rest of this document is the evidence, the full lever menu, the two infrastructure realities that gate the plan, a phased portfolio, and the decisions we need from you and from the harvester team.

## Status (2026-06-07): documented, parked

This is a research and candidate document. Nothing here ships now. v1.6 is in flight (drafted ask, not yet shipped or evaluated), and the prompt track waits on its evaluation. Specifically:
- The prompt-side levers here (loop-visibility render, removing the expose-face-down heuristic) are NOT additions to v1.6 (whose ask is locked to the {NOW} prose fix and the recycle-in-RECENT-MOVES item, with the rest in its internal appendix). They are candidates for the prompt iteration AFTER v1.6 results land, evaluated against what v1.6 actually changed.
- The non-prompt levers (harness timeout / reasoning cap, best-of-N replay, solver line-emit, ORPO and on-policy distillation) are independent of the prompt version but are also not being started yet.
- Revisit this document once v1.6 evaluation is in. Until then it is reference, not a work order.

## 1. The gap (where the points are)

| Quantity | Draw-1 value | Source |
|---|---|---|
| Perfect-information ("thoughtful") solvability ceiling | 90.48% +/- 0.12% | Solvitaire (Blake and Gent, arXiv 1906.12314) |
| Realistic competent imperfect-info ceiling | about 43 to 52% (43% best documented human, 52% best documented AI) | Bjarnason/Fern/Tadepalli; Wikipedia turn-1 52% |
| Population-average human | about 33% | Solitaired player DB |
| Our 31B teacher | about 31% (23/75 sessions) | corpus (data/index/manifest.jsonl, DATASET_NOTES.md) |
| Structurally dead deals (draw-1) | about 10% (= 100% minus 90.5%) | Solvitaire |

Reads:
- We are at average-human level, about 12 to 21 points below the realistic imperfect-info ceiling, and about 40 points below the perfect-info ceiling. About 40 points are unreachable without seeing hidden cards.
- Dead-deal removal is NOT the big lever for win rate. With only about 10% of draw-1 deals dead, screening them out reframes 31% to about 34% on winnable-only (31 / 0.90). Its real value is reclaiming budget and correcting the denominator, not win rate.
- The big win-rate lever is the behavioural bucket: loops on winnable boards. That is where the 12 to 21 reachable points live.

Note on the commonly-quoted "82% of Solitaire is winnable": that is the DRAW-3 number. Ours is draw-1 (90.5%). Several internal planning docs and some DATASET_NOTES seed labels still say draw-3; they are stale (see the corpus strand). Use draw-1 figures.

Win rate by prompt version (31B, session-level; low-N on recent versions):

| version | 31B won/total | rate |
|---|---|---|
| v1.0-hybrid | 1/10 | 10% |
| v1.2 | 2/4 | 50% |
| v1.3 | 16/27 | 59% |
| v1.4 | 0/3 | 0% |
| v1.5 | 1/2 | 50% |

The prior memory figure "v1.3 83% (5/6)" was an early small-N snapshot; the full v1.3 31B set is 16/27 = 59% as more sessions (including dead deals and loops) were ingested. v1.4 and v1.5 are too small to rank.

## 2. Where the wins are lost (the log-mining answer)

This section is the direct answer to "dig deep in the massive game log for any insight." Numbers are corpus-wide, deduped by interaction id (7,020 of 39,804 raw rows are re-export duplicates; always key on the UUIDv7 id, never on turnIndex, which is non-unique at recycle boundaries: 467 collisions, all at recycles).

### 2.1 The win/loss discriminants

- Foundation-move AVAILABILITY is the cleanest separator: a foundation play is available on 33.1% of WON turns vs 5.9% of LOST turns, near-perfectly separable. Foundation PASS-UP is roughly equal (WON 9.9% vs LOST 11.8%), so the model is not refusing foundation plays; losing boards simply never reach positions that offer them. Caveat: this partly reflects deck winnability (winnable decks naturally offer more foundation plays), so it is both a difficulty signal and a process signal.
- Draw rate: WON 40% vs LOST 63%. Excessive drawing is the visible face of "no productive tableau move available."
- Loops are one-card-dominant. De-inflated stitched reversals (exact A to B to A of the same card between the same column pair): WON median 11 per session (max 34) vs LOST median 22 (max 135); dominance (top card's share of reversals) WON 0.53 vs LOST 0.70. A usable kill signal: a single card exceeding about 20 stitched reversals with dominance at or above 0.6 is a near-certain loss. The four worst grinders are all 26B (consistent with the MoE long-context routing-instability hypothesis).
- Resign is dead: 0 resigns in 17,130 success turns, despite stuck/dead/blocked language in 25% of LOST turns and the move_index = -1 option being offered every turn. 16 of 59 losses burn to the roughly 500-turn cap; 38 of 58 end with 10 or fewer foundation cards.
- Reasoning TEXT barely separates wins from losses; BEHAVIOUR separates far better. Speculative-vs-deterministic draw rationale differs only weakly (deterministic 76% WON vs 70% LOST). Implication: do not chase prompt wording for "better reasoning"; filter the student's imitation target by behaviour (foundation-availability growth, bounded reversals), not by phrasing.

### 2.2 Behavioural loops: partly a visibility gap, partly an actioning gap

RECENT MOVES is hard-capped at 10 entries (9,278 turns show exactly 10). Of 42 sessions with a dominant session-wide oscillation, the loop is outside the latest 10-move window at the export snapshot in 34. The anti-undo rule ("do not return a card to a column it occupied in any move shown in RECENT MOVES") therefore references a window that often does not contain the loop. Measured violation rate when an exact-reverse move is offered: v1.3 18.3%, v1.5 37.9%, and about twice as high in non-won sessions (21.0%) as in won (11.1%).

But a follow-up dig (82 sessions with a dominant card reversing 4+ times: 18 WON, 64 LOST) shows the picture is two-sided and bounds what a render can do:
- The dominant loop is TIGHT: median period 2 (an A-to-B-to-A oscillation). 76 of 82 have median period 10 or less, so while actively looping the model usually CAN see the back-and-forth in the last few moves. Only 34 of 82 have the loop ALWAYS inside 10 (max period reaches a median of 14, up to 113), so it interleaves with draws and other shuffles and falls out of the window at the decision/export moment.
- Reversal is NOT itself a loss signal: WON sessions reverse a dominant card a median of 8 times (max 23); they reverse productively (progress interleaves). LOST median is 15 (max 88).

Implications for the fix: (a) it must be PROGRESS-GATED (surface reversal only when it coincides with no foundation growth and no reveal), or it will false-alarm on the 18 winning loopy sessions; (b) it should be CO-LOCATED with the move choice, because RECENT MOVES is the least-cited section (named in only 23 to 50% of reasoning) so a raw history list does not get acted on; (c) it is informational, not a harder predicate (memory v1-3-anti-undo-predicate-design-hole warns a harder predicate is net-negative on 31B). And it is bounded: tight loopers that already see the back-and-forth and loop anyway will not be fixed by a render alone, which is why the harness auto-kill (budget reclaim) and the training-side process-reward (section 6, phase 2) are the necessary complements. Full design in section 4.1.

### 2.3 The injected heuristic is the loop's rationalization engine

The v1.3+ STRATEGY GUIDANCE contains a prescriptive "if any legal move exposes a face-down tableau card, prefer it" rule with a tie-break. Reasoning echoes this rule in 90.6% of LOST turns vs 68.5% of WON turns. Among non-productive tableau shuffles (a non-reveal move that did not drop a face-down next turn), the model justifies the move with this heuristic 85.9% of the time in losses. The rule converts the accurate reveal TAG into a loop attractor: any shuffle between two columns can be narrated as rule-following. This is the corrected reading of today's 26B "fabricated reveal" observation (the reveal tag itself is 100% accurate; the prose is the recited heuristic, and it is corpus-wide, not 26B-specific). Removing the prescriptive heuristic while keeping the per-move tag is a clean A/B and restores the project principle prompt-closes-info-gap-not-logic.

### 2.4 Two loss families, different fixes

Of 59 LOST sessions: 31 behavioural LOOP, 20 DEAD-FLAIL (stuck early, max foundation 12 or fewer with few reversals), 8 MIXED. The documented doom-loop subset (a biased, notably-bad sample of 41) splits about 44% dead / 54% behavioural; the unbiased theoretical dead-from-deal rate is about 10% (section 1). The truth is between, and "dead-deal" in our notes sometimes means "dead at the point reached" (played into) rather than "dead from the deal." Treat the structurally-dead-from-deal fraction as roughly 10 to 20%, uncertain, and Monte-Carlo-over-called on high-unknown boards (memory winnability-montecarlo-false-dead). One LOST session (#0154e1, 31B) reached foundation 51 with a single reversal before hitting one unwinnable card: a near-perfect game and a high-quality data point, not a behavioural failure.

### 2.5 Mechanics findings (validated at scale)

- Reveal tag: 100% accurate both directions (614/614 chosen tagged moves drop a face-down; 0 real missed tags). Do not touch.
- v1.5 stall counters: climb to a maximum of 292, are cited in 9% of reasoning, and produce 0 resigns. Non-functioning and pure token waste. Replace, do not keep.
- {NOW} doc bug: confirmed at scale (18,824 prompts say {NOW}; the literal marker appears 0 times in the data, which wraps the waste top in braces). Geometry is correct (next draw is left of the brace 4,185 times vs 640). Cosmetic but real; already drafted as v1.6 item 1.
- Recycle never logged in RECENT MOVES: 0 of 15,714 blocks, against 482 recycle events. Already drafted as v1.6 item 2.
- Provider errors: 15,653 of 32,784 interactions (47.7%) are non-success, but only 2 are prompt/parse-side; the rest are provider unavailable/timeout. Across v1.3+ the mix shifted to timeout-dominated (v1.5 98.7% of errors are timeouts). Retry recovers 65.6% of retried calls (cap about 4 attempts), so the policy works; the residual hard-failures are the 240s timeouts.
- Output schema: 0 malformed/truncated JSON of 15,790. Clean. The "no markdown fences" instruction is increasingly ignored (v1.5 84% emit a fence), harmless because the parser strips them.
- Style leak: the STRATEGY GUIDANCE block contains an em-dash ("too early ... they are sometimes needed"), which both violates the repo style policy and trains the student to emit em-dashes. Remove it.
- Token waste: a static preamble of about 1,032 tokens is re-sent every call (about 33.8M identical tokens corpus-wide); a candidate for prompt caching if the provider path supports it.

## 3. The lever menu

Grouped by goal. Each lever has a verdict in the deep-dive table (section 4).

A. Make the teacher win more (grow supply):
- A1 Loop-visibility render (persistent shuffle summary). Prompt/render.
- A2 Remove the prescriptive expose-face-down heuristic. Prompt A/B.
- A3 Raise the 240s timeout or cap the reasoning budget. Harness.
- A4 Best-of-N replay of winnable decks (free win verifier). Harvester plus solver.
- A5 Drop solver-dead decks, and add an early dead-deal kill or a working resign. Harvester plus solver plus prompt.
- A6 Temperature bump (0.3 to about 0.6 to 0.8) for replay diversity. Harvester.
- A7 Difficulty curriculum (easy winnable decks first). Harvester plus solver.
- A8 Move-level self-consistency or solver-adjudicated best-of-N. In-house rollouts.
- A9 Imperfect-info search as a win generator (ExIt-OOS style). Build, later phase.

B. Extract more win-quality data from what we already harvest:
- B1 Salvage good prefixes of losing games (already 53% of training; mine good actions, exclude loop tails).
- B2 Oracle relabeling of winnable-but-lost decks (full deck recoverable from the export), with an imperfect-info fairness filter.
- B3 Milestone hindsight relabeling (best foundation reached as an auxiliary sub-goal).
- B4 Pure programmatic solver demonstrations. Trap-leaning (loses reasoning, full-info leakage).

C. Fix the student objective so wins stick:
- C1 On-policy distillation plus loop-penalty (the structural fix for the held v2 problem).
- C2 Step-level ORPO/DPO on hand-exact process rewards (foundation up, face-down down, winnability preserved, vs the loop move). The already-planned pilot.
- C3 Training-data filter: exclude loop tails and dead-flail flailing; segment LOOP vs DEAD-FLAIL before training.

## 4. Per-lever deep dive

Effort: LOW (config/prose), MED (small build), HIGH (substantial build or research). Fidelity: CLEAN (imitable by the imperfect-info student), CONDITIONAL (clean only with a filter), TRAP (not imitable).

| Lever | Effort | Win-data impact | Fidelity | Verdict |
|---|---|---|---|---|
| A1 loop-visibility render | LOW (harness render) | High (targets the dominant behavioural loss) | CLEAN | DO FIRST; replaces dead stall counters |
| A2 remove expose heuristic | LOW (prose) + A/B | Medium-High | CLEAN | A/B on 31B, same-deck |
| A3 raise timeout / cap reasoning | LOW (harness ask) | High (recovers a large share of the 47% error turns) | CLEAN | DO FIRST; biggest yield-per-game lever |
| A4 best-of-N replay of winnable decks | LOW-MED (harvester ask) | Highest raw win supply | CLEAN | Do if harvester can inject a specific deck; else degrades to "play more games" |
| A5 drop dead decks + working resign | MED (solver + prompt/harness) | Budget + denominator (about 31 to 34%), not win rate | CLEAN | Reclaims about 40 to 55 games of budget |
| A6 temperature bump | LOW (harvester ask) | Medium (diversity for A4) | CLEAN | Pair with A4 |
| A7 difficulty curriculum | LOW-MED (solver scoring) | Medium (efficiency) | CLEAN | After A4/A5 |
| A8 move-level self-consistency / solver-adjudicated BoN | MED (in-house rollouts) | Medium | CONDITIONAL (gate to "keeps winnable", not "uniquely optimal") | If we run our own rollouts |
| A9 imperfect-info search win generator | HIGH | High per hard deal | CONDITIONAL | Phase 3, only if cheaper levers plateau |
| B1 salvage loss prefixes | LOW (ingest tweak) | Already capturing 53%; refine | CLEAN | Mine good actions, exclude loop tails |
| B2 oracle relabeling | MED (solver line-emit about 20 LOC + harness + fairness filter) | High (turns winnable losses into win-quality data) | CONDITIONAL (drop hidden-info-dependent plies) | Phase 2-3; apply at mid-game snapshots where the solver is cheap |
| B3 milestone hindsight relabel | MED | Medium | CLEAN (real visible-state prefixes) | Auxiliary sub-goal data |
| B4 programmatic solver demos | MED | Low | TRAP | Avoid; full-info optimal play is un-imitable and has no reasoning text |
| C1 on-policy distillation + loop penalty | HIGH (trainer build) | Structural (stops the student re-learning loops) | CLEAN (student-realizable by construction) | The key student-side fix |
| C2 ORPO loop-penalty pilot | MED-HIGH (trainer; planned) | Structural | CLEAN (visible-state potentials) | Already the next training step; feed it the mined negative pairs |
| C3 training-data filter / segment | LOW | Medium (cleaner SFT) | CLEAN | Cheap, immediate |

Traps to avoid (would produce data the imperfect-info student cannot imitate): behaviour-cloning the full-info solver line move-for-move (T1); full-info tree search distilled (T2); distilling search or voting machinery rather than the single chosen move (T3); process rewards defined on solver-optimality instead of visible potentials (T4); over-iterating the rejection-sampling loop on this small corpus (ReST-EM regresses past 1 to 2 iterations on small data) (T5); ingesting whole doom-loops as positive data (T6).

## 4.1 Deep dive: the loop-visibility render (is it reasonable, what size, is there a better way)

This is the design pressure-test for lever A1, the top behavioural lever. Three questions: is it reasonable, what is the optimal "number" (window or threshold), would it bloat the context, and is there a better way.

Is it reasonable: yes, but PARTIAL and progress-gated. From section 2.2: tight loopers already see the back-and-forth and loop anyway (actioning gap), so a render helps mainly the long-period and decision-moment cases (visibility gap) and adds salience for the rest. Treat it as one of three complementary moves (render + harness auto-kill + training-side process-reward), not a silver bullet. It also has to clear the bar that the v1.5 stall counters failed: a bare scalar in a separate line did nothing (0 resigns, 9% uptake), so the render must be SPECIFIC (name the card and columns), PROGRESS-GATED, and CO-LOCATED with the move choice.

Do NOT just lengthen RECENT MOVES. The loop interleaves, so a raw window that guarantees the loop is always visible would need to be as long as the max period (median 14, up to about 110 moves). That is impractical: it adds hundreds of tokens every turn and grows, it stresses the 26B MoE long-context routing, and it still relies on the model spotting a pattern in a list (the least-cited section). A longer raw list is the worst option at any N. Keep RECENT MOVES at 10 for immediate context; solve the loop with an aggregate, not a longer list.

The optimal "numbers":
- Tracking horizon: WHOLE game, harness-side state (per-card-pair move counts since last progress). Cheap, off-prompt, no context cost.
- Render threshold (when to surface a back-and-forth): start at about 3 unproductive repeats of the same card between the same two columns. Calibration: WON dominant reversals median 8 (max 23) but productive; gating on "no foundation growth and no reveal since" keeps winners from tripping it, and 3 is below the productive-winner range while still early enough to flag before entrenchment.
- Auto-kill threshold (harness, budget reclaim): high and progress-gated, e.g. a dominant card-pair reversing about 12 or more times with zero foundation growth and zero reveal across that span. This sits clear of WON behaviour (whose loops come with progress; WON max dominant reversals is 23 but interleaved with gains), so it will not kill winnable games. Pair with the existing session_oscillation helper in load_export.py.

Context-length budget (answering "too long?"):
- Longer raw RECENT MOVES to N=30 to 50: about +300 to +600 tokens per turn, growing. Rejected.
- Compact summary block, worst case 3 cards ("Repeated without progress: 4S col5/col7 x12; 9D col1/col3 x6; 6H col2/col4 x4"): about 40 to 60 tokens.
- Per-legal-move annotation (only the 0 to 2 legal moves that are reversals get a tag like "(returns 4S to col 5; this back-and-forth has happened 12 times with no progress)"): about +10 to +12 tokens each.
- Net for the recommended design: roughly +50 tokens, and it REPLACES the dead v1.5 stall counters (about -20 tokens), so the swap is close to token-neutral. Negligible against the ~1,032-token static preamble and the multi-thousand-token thinking budget. So context length is a problem only for the naive long-list approach, not for the aggregate.

A better way (ranked):
1. Per-legal-move reversal annotation, progress-gated. Put the count ON the move line where the decision happens. Best, because RECENT MOVES as a separate section is under-read; co-location is the differentiator from the failed stall counters.
2. Compact "repeated without progress" summary block (1 to 3 lines), as the global complement. Replace the v1.5 stall counters with this.
3. Harness-side auto-kill on progress-gated dominant reversals (does not need the model to cooperate; reclaims the budget the tight loopers burn). This is the safety net for the actioning-gap cases a render cannot fix.
4. Training-side process-reward / ORPO pair from the same signal (the durable fix; the student learns to not loop in the states it visits).

Net recommendation: a progress-gated, co-located count (forms 1 and 2), token-neutral by replacing the stall counters, plus the harness auto-kill (form 3) and the training signal (form 4). Validate on 31B against same decks once v1.6 is evaluated, and measure whether the dominant-reversal distribution shifts down and foundation-availability shifts up, not just whether win rate moves (the render is necessary-not-sufficient, so win-rate movement may be small on its own).

## 4.2 Measured evidence the false-reveal failure is real and frequent (2026-06-07)

The hint case rests on the false-reveal failure being common, not anecdotal. Measured corpus-wide over consecutive successful tableau-to-tableau moves (errored turns do not change board state, so consecutive successful moves isolate each move's effect; the change in faceDownTotal is ground truth). Sanity check: of all such moves, 1,055 dropped faceDown by exactly 1 and 3,055 did not, so reveals do occur and are detected.

Of the tableau moves whose reasoning CLAIMED it would reveal a face-down card:

| group | claims reveal? | moves | actually revealed | reveal rate |
|---|---|---|---|---|
| WON | claims reveal | 866 | 389 | 44.9% |
| WON | no claim | 114 | 1 | 0.9% |
| LOST | claims reveal | 2,962 | 664 | 22.4% |
| LOST | no claim | 168 | 1 | 0.6% |

- Pooled: 3,828 moves claimed a reveal; only 1,053 (27.5%) actually revealed. So 2,775 moves (72.5%) claimed a reveal and revealed nothing, across 82 of 131 sessions (63%). The model overclaims reveals about 3.6x beyond reality.
- Win/loss linked: a reveal-claim is right 44.9% of the time in won games vs 22.4% in lost. Losers are wrong about their own reveals roughly twice as often, and do it in far greater volume (2,962 vs 866 claim-moves).
- The claim is not pure noise: claim-moves reveal 27.5% vs no-claim 0.7%, so the model has real but badly-miscalibrated signal.

Caveats: the claim detector is a regex (reveal/expose/uncover near hidden/face-down), validated by the clean no-claim baseline (0.7%). "Claimed but revealed nothing" mixes genuine misreads with multi-step setups that never paid off, so the win/loss split is the robust signal because it roughly controls for the setup rate.

Verbatim receipts (the clean misread end of the distribution):
- #7a4b10 turn 11 (the move that starts the 47-reversal loop): chose "Move 9H from column 5 to column 3", wrote "this move ... will reveal the TS ... it follows the heuristic of prioritizing moves that reveal hidden cards ... This will expose the TS." TS is the face-up ten of spades directly under the 9H; faceDownTotal was 16 before and 16 after. It names the injected heuristic and claims a reveal the board disproves.
- #78c130 turn 15: chose "Move QD from column 2 to column 1", wrote "will reveal a hidden card in column 2 ... the only move that immediately exposes a face-down card." Column 2 was a single QD with nothing beneath it; faceDownTotal 17 before and 17 after. It invented a hidden card.

Loop-visibility receipt: in #523f19 the dominant loop is 4S between col 5 and col 7 (66 stitched reversals). Extracting the actual RECENT MOVES block shown each turn, 4S is absent from that window on 135 of 677 turns, so the model often could not see its dominant loop in the history it was given. (Directional: that count includes pre-loop turns; the cleaner corpus stat is that 34 of 42 looping sessions have the loop outside the 10-move window at the export moment.)

## 4.3 Conviction and sequencing (calibration, not a recommendation)

Recorded 2026-06-07 so a separate next-step discussion starts from a calibrated position rather than this document's advocacy.

Convinced of: the diagnosis above (real, measured, frequent, win/loss-linked), and that the hint is the best-shaped prompt change IF a prompt change is made (state-not-logic, cheap, targets the measured failure, and strictly better than the prescriptive heuristic that backfired).

NOT convinced the hint is the right NEXT move for the project, for four reasons:
1. Causality is unproven and there is a specific inert-risk. All of the above is correlational; the clean test is an A/B we cannot run unilaterally. The v1.5 stall counters are a precedent: an accurate signal the model simply did not act on (0 resigns in 17k turns). The hint could be inert the same way. Tempering point: the model does heed the positive "(reveals a hidden card)" tag (about 90% taken when offered, per reveal-pass-up-kill-signal), so it is tag-asymmetric (never shown the negative case) rather than tag-blind; but heeding "this is good, do it" is easier than heeding "this does nothing, abandon your plan."
2. It is blocked behind v1.6 evaluation (the prompt track is parked).
3. Higher-expected-value, model-independent moves are available now: the harness timeout / reasoning-cap fix (recovers about half of the ~47% of turns currently discarded to reasoning-overrun timeouts, with zero behaviour change) and the training-side ORPO loop-penalty / on-policy distillation (the structural fix for the held v2 problem of the student re-learning the teacher's loops).
4. The hint helps the TEACHER; the STUDENT is the deliverable, and the student's training objective is more central to the project goal than a teacher-prompt tweak.

Cheap de-risk before investing in the hint: confirm whether the model, when it claims a reveal the prompt did not tag, ever had the correct tag available and overrode it. Routine override would be evidence the hint will be inert and effort should redirect to the harness and training levers.

Next-step decision deferred to a separate discussion (operator direction 2026-06-07).

## 4.4 What contributed to the wins: easy deck, chance, or right decisions? (2026-06-07)

Question: across all prompt versions, what drives the 24 wins, deck-forgiveness, variance, or the model playing well? Three measurements, then a calibrated verdict. Hard constraint throughout: losing sessions are ai-log-only with no recorded deck, so won decks cannot be compared head-to-head against lost decks; these tests profile the won decks in absolute terms and against a random-deal baseline.

Test 1, deck difficulty by card accessibility (cheap proxy). Won decks (n=24) vs an 8,000-deal random baseline: ace face-down burial 4.00 vs 4.28 (a wash); aces immediately playable 0.75 vs 0.55 (marginally better); 2s face-down burial 4.92 vs 4.27 (won decks WORSE). By this proxy the won decks are about as hard as random deals. No "easy deck" signal on average. Caveat: ace/low-card burial is a crude proxy and n=24 is small.

Test 2, win-play efficiency (complete win-record move history, authoritative). The median won game uses 237 decisions, 86 draws, and 88 tableau moves to place the 52 foundation cards, with 21 reveals. Of the tableau moves, 47% are exact A-to-B-to-A reversals (1,087 of 2,322; median 36 per game, mean 45, max 169). Wins are achieved by heavy thrashing, not clean lines. Methodology note that matters corpus-wide: the ai-log decision stream captures only about 30% of the true moves (it shows a median of 11 reversals for won games vs the complete record's 36), so the ai-log-derived behavioural counts elsewhere in this document (the false-reveal frequency, the loss reversal counts) are FLOORS, not exact.

Test 3, solving the won decks from turn 0 (most direct difficulty test; engine best-first, 150k-node cap, full information). 9 of 24 SOLVED (node counts 315, 435, 777, 1680, 2363, 3669, 6414, 34482, 57698; median 2363; 4 of 9 trivially easy under 2k nodes); 15 of 24 hit the cap (UNKNOWN, inconclusive). So at least 9/24 (37%) of won decks are demonstrably EASY (the solver finds a win almost immediately, i.e. many forgiving lines). The 15 cap-outs are NOT a hardness verdict: turn-0 full-tree search is a far harsher task than the teacher faces, because the teacher plays one trajectory with cards revealed progressively as it draws and flips, so a cap-out reflects search-tree size, not play difficulty. Read the 9 SOLVED as a firm lower bound on easy decks and the 15 as inconclusive by this instrument.

Verdict (calibrated): no single factor dominates, and "the LLM making clean right decisions" is the WEAKEST of the three.
- Easy deck: a real contributor for a clear minority (at least 37% demonstrably easy), but not a uniform explanation; the won decks are not easier than random on the burial proxy.
- Chance / variance: a major, irreducible factor. The win method is a high-variance grind (47% reversals, 86 draws, 237 decisions), and the same deck wins-or-stalls stochastically at temperature 0.3 (corpus same-seed evidence: seed 2967897202 won once and lost twice; seed 3263196305 won four times but a 26B run on it looped). Given the decks are not uniformly easy and the method is thrash, variance largely decides which winnable deck on which run is won.
- Right decisions: real but limited. The teacher wins about 31% (well above a random or one-step-greedy player, roughly 7 to 13% in the literature for draw-3, and it beats the repo's own lookahead strategy, which won 0/15), it takes offered reveals about 90% of the time, and it plays foundations whenever one is available. But it does not find clean lines (47% of its winning tableau moves are reversals); its edge is local competence plus the structural advantage of progressive reveals plus relentless exploration, not strategic planning.

Blunt version: the model mostly "fails to lose" winnable decks by grinding (draw, recycle, shuffle, reverse) until a winnable deck falls over, with a clear minority being easy decks that fall over readily and variance deciding the individual runs. This is consistent with win rate being statistically flat across prompt versions (overlapping CIs, seed-difficulty confound): if wins were driven by the model's prompted decision-quality, prompt changes would move win rate, and they do not clearly.

What would make this decisive: the harvester emitting the full deck on LOSSES too (so won vs lost decks can be compared head-to-head), and a mid-game solver difficulty pass (fairer than turn-0 search). The full-deck-on-losses request is now written up as a logging ask, add `initialBoardSetup` as a key inside the existing ai-log, no new file (docs/reports/20260607_harvester_ask_ailog_initial_deck.md). The mid-game solver pass remains deferred.

## 5. Two infrastructure realities that gate the plan

1. Seed to full-deck is NOT reproducible in-repo. The repo's dealer (random.Random(seed).shuffle) does not match the web app's JS PRNG (verified: same seed, completely different layout). So we cannot, given a harvester seed, decide its winnability offline. Offline seed pre-filtering (A5/A7 at seed level) is BLOCKED unless we either reverse-engineer the web PRNG or have the harvester emit the pre-game initialBoardSetup so we can screen the board.

2. The full deck IS recoverable per game from the export. Every solitaire-win and solitaire-game file carries a complete 52-card initialBoardSetup with all face-down identities (25/25 verified). So B2 oracle relabeling and any per-game solving are data-feasible without the seed, for games that produced a terminal/snapshot file. The sound solver decides winnability but does not yet emit the winning line (about 20 lines of code to add parent-pointers); it is cheap mid/late-game and expensive or inconclusive at turn 0 (14/25 turn-0 boards solved at a 300k-node cap), so apply oracle work at MID-GAME snapshots, where it is both cheap and aimed exactly at the states the student must learn. Remember to reverse drawPile into the engine stock (wrong orientation multiplied solver cost about 9x).

Note: the only component that actually finds wins is the sound best-first solver; the in-repo lookahead strategy wins 0/15 and the beam solver loses on a known-winnable board. So there is no free programmatic player to mine (B4 is empty without repurposing the solver, which reintroduces the full-info-leakage trap).

## 6. Recommended portfolio (phased)

Phase 0 (after v1.6 is evaluated; no new builds, cheap):
- Prompt (next harvester ask after the drafted v1.6 items 1 and 2): add the loop-visibility render (A1) to REPLACE the v1.5 stall counters; A/B removing the prescriptive expose-face-down heuristic (A2); remove the em-dash from STRATEGY GUIDANCE. Keep the accurate reveal tag.
- Harness asks: raise the 240s timeout to about 360s OR cap the reasoning budget (A3); auto-kill a session after N turns with no face-down drop AND no foundation gain (reclaims budget, A5 without seed control); enable prompt caching for the static preamble.

Phase 1 (small builds):
- Add winning-line emission to the sound solver (about 20 LOC).
- Difficulty/winnability scoring of recoverable decks with a fast sound solver (lonelybot is the recommended tool: Rust, sound, draw-1, seed-range batch, about 3 orders of magnitude faster than Solvitaire). Use it to drop confidently-dead decks and to score difficulty.
- If the harvester can inject a specific deck: best-of-N replay of winnable decks at a raised temperature (A4 + A6), capped at 1 to 2 rejection-sampling iterations.

Phase 2 (training):
- ORPO loop-penalty pilot (C2, already planned) fed by the mined negative pairs: anti-undo-violation turns (759 in non-won sessions) and dominant-card loop turns as "rejected", the offered-but-not-taken progress/reveal move as "chosen". Hand-exact process potentials: foundation up, face-down down, winnability preserved.
- Training-data segmentation (C3): exclude loop tails and dead-flail flailing; keep good loss prefixes (B1).

Phase 3 (if Phase 0 to 2 plateau):
- On-policy distillation as the student objective (C1), paired with the loop penalty so it does not copy teacher loops in the states the student itself visits.
- Oracle relabeling with the imperfect-info fairness filter (B2), applied at mid-game snapshots.
- Imperfect-info search as a win generator (A9) only for the hard winnable decks that replay cannot crack.

## 7. Decisions needed

From you:
- Q1 Fidelity philosophy: are best-of-N-selected trajectories and oracle-hinted (imperfect-info-filtered) trajectories acceptable as training data, or do you want pure unfiltered 31B-only? The clean levers (A4 replay, C1 on-policy distillation) stay in-distribution; oracle-hint (B2) is acceptable only with the fairness filter.
- Q2 Build appetite: green-light the small in-repo builds (solver line-emit, dead-deal screen, ORPO trainer)?

From the harvester team (these gate the biggest levers):
- Q3 Can the 240s timeout be raised or the reasoning budget capped? (A3, biggest yield lever.)
- Q4 Can a specific deck be injected to replay (best-of-N)? (A4.)
- Q5 Can sampling temperature be set per run? (A6.)
- Q6 Can the harvester skip seeds we flag dead, or emit the pre-game initialBoardSetup so we can screen? (A5/A7.)
- Q7 Can an auto-kill stall terminator be added (no face-down drop and no foundation gain for N turns)? (budget reclaim.)

## 8. Corrections to earlier reads this session

- The 26B "fabricated reveal" (#7a4b10, #4aa9f1) is reframed: corpus-wide echo of the injected expose-face-down heuristic, not a 26B-specific spatial hallucination. The reveal tag is 100% accurate. The fix is heuristic removal (A2), not a tag fix. (The behavioural description, looping while narrating a reveal that is not productive, stands; the attributed cause changes.)
- The v1.5 stall-count "first test failed" read is reinforced, not corrected: at corpus scale the counters climb to 292 with 9% uptake and 0 resigns.
- The v1.6 "deferred A: timeline dropped on empty-waste" item is downgraded: the timeline is correctly present whenever the stock is non-empty and correctly absent only when both stock and waste are empty. Not a bug.

## Sources

Klondike domain:
- Blake and Gent, The Winnability of Klondike Solitaire and Many Other Patience Games, arXiv:1906.12314 (draw-1 90.48%, draw-3 81.945%).
- Bjarnason, Tadepalli, Fern, Searching Solitaire in Real Time, ICGA 2007 (feature value functions, dead-end detector).
- Bjarnason, Fern, Tadepalli, Lower Bounding Klondike Solitaire with Monte-Carlo Planning, ICAPS 2009 (UCT/HOP win rates, human 36.6% draw-3).
- lonelybot (Rust solver, sound, fast batch): github.com/vuonghy2442/lonelybot. ShootMe Klondike-Solver. Solvitaire: github.com/thecharlieblake/Solvitaire.
- Solitaired odds: solitaired.com/odds-of-winning-solitaire. Klondike (solitaire) Wikipedia (turn-1 52% AI, 43% human).

ML methods:
- STaR (arXiv:2203.14465); ReST-EM "Beyond Human Data" (arXiv:2312.06585); Exploring Expert Failures (arXiv:2504.13145).
- LATS (arXiv:2310.04406); Expert Iteration / ExIt; ExIt-OOS imperfect-info expert iteration (arXiv:1808.10120).
- DAgger; Distilling Realizable Students from Unrealizable Teachers (arXiv:2505.09546); Student-Informed Teacher Training (ICLR 2025); privileged-information asymmetric actor-critic (arXiv:2509.26000).
- LEMMA (arXiv:2503.17439); SCoRe (arXiv:2409.12917); AgentPRM (arXiv:2511.08325); self-consistency (CISC, ACL 2025).
- On-Policy Distillation (Thinking Machines, 2025); AgentHER hindsight relabeling (arXiv:2603.21357); Data Curation Flywheel (arXiv:2508.03018).
