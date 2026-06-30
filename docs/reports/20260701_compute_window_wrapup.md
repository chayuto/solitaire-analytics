# Compute window wrap-up: the decisive STaR test closes the on-policy SFT track; ship volume (2026-06-30 to 2026-07-01)

Author: research log, synthesized 2026-07-01.
Scope: one overnight compute window running the single decisive experiment (Q1)
queued by the previous wrap-up (`20260629_onpolicy_window_wrapup.md`), plus the
resulting program-state change and the ship decision. Per-experiment detail is in
`20260701_star_diverse_eval.md`; this is the lab-notebook synthesis.

Standing constraints honored: one GPU job, machine on mains, faithful v1.6
harness with exact final-position adjudication available, all eval decks
solver-confirmed winnable and held out of training by seed.

---

## 0. Abstract

The 2026-06-29 window left exactly one decisive question open: with a diverse,
balanced pool of the student's own self-wins (the diversity the oversample
ablation said was missing), does SFT-on-self-wins finally beat the volume
baseline on fresh decks? This window built and ran that test (Q1, STaR diverse
retrain). Result: **star_diverse 4 wins vs volume 5 on heldout12 (delta -1, fc
delta -2)**. The diversity fix worked exactly as predicted in isolation, it
erased the catastrophic x1 regression (fc delta -2 vs x1's -94) and recovered 2
of the 3 decks x1 collapsed, confirming x1 failed from trajectory correlation,
not oversample dose. But removing the poison only bought parity with volume, not
lift. Per the pre-registered gate, this closes the SFT-on-self-wins line: every
configuration tried (x4 correlated, x1 correlated, x1 diverse) lands at or below
volume. With both research lines now closed (static-corpus engineering and
on-policy SFT), the decision taken is to ship volume; reward-weighted RFT/GRPO is
parked as the sole remaining upside lever.

---

## 1. The single experiment (Q1: STaR diverse retrain)

Report: `20260701_star_diverse_eval.md`.

Method note worth surfacing: extracting the diverse pool revealed that a naive
x1 fuse of all 12 trajectories would have been a 28% STaR share (at the x4 poison
dose) with deck 9000237 contributing 41% of the rows (five wins of one board,
re-introducing correlation). The corpus was rebuilt as balanced design B, one
win per deck (9000237 collapsed 5 to 1), 1537 rows at a 21.3% share, so the
result isolates diversity rather than re-running the dose confound. This catch,
made from the measured row counts before training, is the methodological event of
the window.

Result: SFT clean (val 0.220, matching x1). Paired heldout12: volume 5,
star_diverse 4. star_diverse uniquely won 9000102 (volume stalled fc10), lost
9000105 (fc18) and 9000124 (fc42). Against x1 specifically the diverse pool
recovered 2 of 3 collapse decks to wins and cut the sum-fc draw-down from -94 to
-2.

Verdict: does not beat volume; SFT-on-self-wins closed for generalization.

---

## 2. Integrated program state

Two independent lines have now closed on the same conclusion:

1. **Static-corpus engineering does not generalize** (the reweighting/declarative/
   grounded forms all collapse to 3-4 on fresh decks; volume alone holds; closed
   2026-06-23, `solver-grounded-form3-result`).
2. **On-policy SFT-on-self-wins does not beat volume** (this window). Diversity is
   confirmed as the cause of the x1 collapse but is insufficient to exceed the
   baseline; imitation cannot exceed the distribution it imitates.

Volume stands as the best fresh-deck generalizer: ~33% pooled (16/48), roughly 4x
the untuned base (~8%), already published to Hugging Face
(`chayuto/gemma-4-e2b-it-solitaire-advisor-lora`). Its losses are stalls on
winnable boards (the close-out phenotype) and the occasional thrown board; its
greedy resign reflex is well calibrated.

The single open mechanism is reward-weighted RFT/GRPO, which optimizes a reward
rather than imitating trajectories. The best-of-N gate proved a real but sparse
reward signal (~8.6% per sample on hard decks), so RFT is viable but expensive
(multi-week build; must preserve full action-space coverage per the ORPO
draw-starvation lesson).

---

## 3. Decision: ship volume

The product-vs-research fork from the 2026-06-11 strategic review resolves to
product. Both improvement programs are exhausted, and volume is a real, published,
~4x-base generalizer. Ship it as the student.

What "ship volume" entails (no compute, no GPU):

- Volume is already on the Hub; the remaining work is a consolidated
  negative-results write-up (this report plus the per-experiment reports it cites)
  and a model-card refinement noting the pinned ~33% pooled fresh-deck rate and
  that both corpus-engineering and SFT-on-self-wins were tried and did not beat
  it. The card edit is an outward-facing publish step (needs a fresh HF write
  token, user-initiated) and is left for an explicit push.

What is parked:

- RFT/GRPO remains the only route to beating, rather than matching, volume.
  Pre-registered design constraints if resumed: reward-weighted on full
  trajectories, full action-space coverage, gate on a fuller best-of-N reward
  density first.

---

## 4. Artifacts manifest

Reports (new this window):
- `docs/reports/20260701_star_diverse_eval.md`
- `docs/reports/20260701_compute_window_wrapup.md` (this file)

Scaffolding (new):
- `gemma4_finetune/run_star_diverse.sh`
- `gemma4_finetune/lora_config_star_diverse.yaml`

Generated/local (gitignored): `dataset_star_diverse/`, `adapters_star_diverse/`,
`star_corpus_diverse_{gate,harvest}.jsonl`,
`play_runs/star_iter1_eval/star_diverse/`, `play_runs/star_diverse.log`.

Memory updated: `bestofn-rft-gate-result`, `star-iter1-regressed`, and a new
`star-sft-track-closed` capturing the closure and the ship decision; `MEMORY.md`.

## 5. Reproducibility

The result is read from on-disk eval summaries
(`play_runs/star_iter1_eval/{volume,star_diverse}/seed*/summary.json`, fields
`outcome` and `final_foundation_cards`) and the training log
(`play_runs/star_diverse.log`). The chain is a single resumable script
(`run_star_diverse.sh`) with the launch command in its header. All decks are
solver-confirmed winnable and seed-disjoint across train and eval. No result was
written before it was measured.
