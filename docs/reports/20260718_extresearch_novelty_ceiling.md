# External research 2 of 5: novelty check and ceiling sanity (2026-07-18)

Strand 2 of the sequential external research pass requested 2026-07-17.
Method: a single verification agent fetched each lead from the interrupted
2026-07-17 deep-research sweep, confirmed or refuted the unverified claims
against primary sources, and ran a 5-query fresh novelty sweep. 18 fetches
(cap consciously exceeded to chase two direct novelty threats), 5 searches,
no usage-limit errors. Companions: strands 1 and 3 to 5 (same date) and
`20260718_external_strategy_stresstest.md`. Bears on: the public write-up
novelty claim and the 43-52% imperfect-info planning ceiling.

Report body below is the verification agent's output, verbatim.

---

## Verdict summary

**Novelty: CHALLENGED as phrased, SUPPORTED after mandatory narrowing.** As
of 2026-07-18 I found no published work in which an LLM plays full games of
Klondike to win/loss with a reported win rate, so the core differentiator
survives. But the blanket claim "no published LLM-plays-Klondike work
exists" is no longer safe to print: Bateni, Pratt, and Whitehead (AIIDE
2025, AAAI proceedings, peer-reviewed) fine-tune LLMs on Klondike-family
solitaire variants via a custom Solitaire GDL, and explicitly scope to
rule-understanding progression questions, not play ("we only use progression
questions... our goal is to purely measure whether or not the rules are
understood and applied correctly"). Separately, Game-RL/GameQA (arXiv
2505.13886) includes Spider Solitaire (and per a search snippet, an adapted
Pygame Klondike engine, unconfirmed in the fetched text) as VQA
data-generation sources for VLM reasoning, again with no full-game play or
win rates. The write-up must cite and differentiate both, and restate the
claim as: no published work where an LLM (or LLM fine-tune) plays complete
Klondike games and reports a win rate; the checked-date should move from
2026-07-04 to 2026-07-18.

**Ceiling: SUPPORTED and slightly strengthened on one number, unmoved but
weakly sourced on the others.** The 81.9% perfect-information figure got a
citation upgrade: Blake and Gent is now peer-reviewed in JAIR Vol 85 (2026),
DOI 10.1613/jair.1.17167, with 81.945% +/- 0.084% for thoughtful Klondike
("the player knows the rank and suit of all cards"). No new solver results
or imperfect-information win-rate estimates surfaced that move the 43-52%
planning band; however, this pass also found no primary source FOR that
band, and the vendor page previously credited with "expert human draw-1 near
43%" demonstrably does not contain that number, so the 43% human-expert
figure is currently unsourced and should be softened or re-sourced before
publication.

## Source-by-source findings

**1. arXiv 2509.01328, "Can Large Language Models Master Complex Card
Games?" (Wang, Bie, Chen, Zhang, Huang, Kharlamov, Tang). Status:
peer-reviewed, arXiv Comments field verbatim "Accepted by NeurIPS 2025"; v5
dated 2025-10-21 (the conflicting venue reports were all simultaneously
true).**
- (a) CONFIRMED: exactly eight games, all multiplayer: "DouDizhu, Guandan,
  Riichi Mahjong, Uno, Gin Rummy, Leduc Hold'em, Limit Texas Hold'em, and
  No-Limit Texas Hold'em". No Klondike/solitaire/FreeCell/Spider/patience/
  single-player mention anywhere in the fetched full text.
- (b) CONFIRMED: abstract states "LLMs can approach the performance of
  strong game AIs through supervised fine-tuning on high-quality data";
  body shows "the performance of the LLM in Doudizhu and Guandan gradually
  approaches that of the teacher model" as data scales. It SHOWS approach,
  never exceed.
- (c) PARTLY CORRECTED: LoRA confirmed ("LoRA rank and LoRA alpha are set
  to 8 and 16"), but models are Qwen2.5 0.5B-14B PLUS Llama3.1-8B-Instruct
  and GLM4-9B-Chat, not Qwen-only.
- (d) CONFIRMED: "we sample 1,000k instances" (DouDizhu, Guandan, Riichi
  Mahjong) and "400k instances" (the other five).
- (e) CONFIRMED: supervised fine-tuning only, no RL found.
- Bearing on the write-up: the strongest published analogue of your
  distillation finding (SFT converges toward teacher, does not pass it) at
  NeurIPS scale; cite it as convergent evidence and differentiate on:
  single-player planning game, on-device 2B model, and your GRPO step
  attempting to go beyond the teacher.

**2. arXiv 2508.08382, "UrzaGPT" (Magic: The Gathering drafting). Status:
preprint (v1, no acceptance note).**
- (a) CONFIRMED: "In this work, we solely focus on deckbuilding and
  disregard the gameplay." No solitaire/Klondike mention.
- (b) CONFIRMED numbers: "peak accuracy of 66.2% using only 10,000 steps"
  vs "domain-specific drafting models (68%)". The specific attribution of
  the ceiling to "model capacity or SFT objective" was NOT confirmed in the
  fetched text; do not quote that framing.
- (c) CONFIRMED: GPT-4o "achieves a zero-shot performance of 43%"; "Untuned,
  small LLMs like Llama-3-8B are completely unable to draft."
- (d) CONFIRMED: LoRA on annotated draft logs.
- (e) CONFIRMED in substance: "future work should consider reinforcement
  learning from outcomes or incorporate gameplay simulations", which is
  precisely what your project executed.
- (f) PARTLY CONFIRMED: it cites Bateni and Whitehead (2024), but only
  briefly in related work; "nearest single-player card-game LLM work" is an
  interpretation, not the paper's wording.
- Bearing on the write-up: cite as the SFT-imitation-ceiling result whose
  recommended next step (RL from outcomes) your GRPO track actually ran;
  differentiate: single-decision imitation vs full-game sequential play.

**3. Bateni and Whitehead 2024, "Language-Driven Play: Large Language Models
as Game-Playing Agents in Slay the Spire", FDG 2024, ACM, DOI
10.1145/3649921.3650013. Status: peer-reviewed conference paper.** (ACM page
403-blocked; verified via the authors' MiniSTS GitHub, search records, and
the AIIDE 2025 paper's description.) It uses prompting only (OpenAI ChatGPT
API with chain-of-thought variants, no fine-tuning, no RL) on a simplified
battle-only implementation (MiniSTS) of a "single-player deck-building
roguelike card game". Per the AIIDE 2025 summary: LLMs "can play well in
scenarios in which long-term planning is needed, whereas more traditional
game playing methods such as lookahead agents excels where cards have
immediate synergies". No solitaire/patience content.
- Bearing on the write-up: closest published single-player-card-game PLAY
  prior art, but prompting-only and not a patience game; your work differs
  on genre (patience/perfect-shuffle solitaire), training (LoRA SFT + GRPO),
  and deployment (local 2B).

**4. arXiv 2505.15146, lmgame-Bench. Status: preprint on arXiv; the
companion lmgame-org/GamingAgent GitHub repo banner says "[ICLR 2026]"
(search-level evidence of acceptance).**
- CONFIRMED: exactly six games (Super Mario Bros, Tetris, Sokoban, Candy
  Crush, 2048, Ace Attorney), 13 models, and no card game, solitaire, or
  poker mention anywhere in the fetched text.
- CONFIRMED harness stat: "Excluding text-only models, 40% of game runs
  without the harness fail to outperform a random-play baseline" vs "86.7%
  of game runs beat the random baseline" with harness. Frontier models do
  score 0.0 on several games unharnessed (Sokoban, Tetris, 2048, Candy
  Crush, Ace Attorney).
- Bearing on the write-up: supports your scaffolding/harness-matters
  narrative (your eval-fidelity and oracle results echo the harness lift),
  and confirms no card-game coverage in this benchmark family.

**5. BALROG, arXiv 2411.13543. Status: peer-reviewed, Comments field
"Published as a conference paper at ICLR 2025".** Abstract names the NetHack
Learning Environment; neither the abstract nor the fetched README mentions
any card game. Caveat: the fetched README did not enumerate the full env
list, so "current env list" is verified only as "no card game found in
abstract or README" (the canonical list lives at balrogai.com, NOT-FETCHED).
**Game Reasoning Arena, arXiv 2508.03368. Status: preprint (v3, no venue
note).** Built on DeepMind's OpenSpiel; includes Kuhn Poker, so it DOES
contain a card game, but all competitive/multi-agent: "No mention of
solitaire, Klondike, patience, or single-player card games."
**DSGBench, arXiv 2503.06047. Status: preprint (Comments just "43 pages, 5
figures, conference").** Six games confirmed from full text: StarCraft II,
Civilization, Street Fighter III, Diplomacy, Werewolf, Stratego. No card
games.
- Bearing on the write-up: the benchmark-gap claim holds, but phrase it as
  "no solitaire/patience environment"; Game Reasoning Arena and TextArena do
  contain card games.

**6. TextArena (arXiv 2504.11442; env list verified from the repo's
textarena/envs/README.md). Status: framework, preprint paper.** The reviewer
gotcha half-fired: there is NO solitaire, Klondike, patience, FreeCell, or
Spider environment among ~99 envs (28 single-player, 52 two-player, 19
multi-player), BUT single-player CARD games exist: Blackjack-v0 and
ThreeCardMonte-v0, plus multi-player card games (Poker-v0, KuhnPoker-v0,
LeducHoldem-v0, IndianPoker-v0, Briscola-v0, GermanWhist-v0).
- Bearing on the write-up: never write "no benchmark contains a
  single-player card game"; write "no benchmark or framework we found
  contains a solitaire/patience environment".

**7. Blake and Gent, "The Winnability of Klondike Solitaire and Many Other
Patience Games". Status: peer-reviewed journal article, JAIR Vol 85 (2026),
DOI 10.1613/jair.1.17167, published 2026-02-27.** CONFIRMED: Solvitaire, 73
variants of 35 games, and for thoughtful Klondike "81.945% +/- 0.084%",
described as roughly a 30-fold CI tightening over prior work; "thoughtful"
means "the player knows the rank and suit of all cards" (perfect
information).
- Bearing on the write-up: upgrade the citation from arXiv 1906.12314 to the
  JAIR version; it SHOWS perfect-information winnability only and says
  nothing about the imperfect-information ceiling.

**8. solitaire.com "Odds of Winning Solitaire" blog
(https://solitaire.com/blog/the-odds-of-winning-solitaire-backed-with-data/).
Status: vendor, and the page cites no sources for any number.** On the page:
theoretical winnability "about 82%" (stated for both Turn 1 and Turn 3);
"most players win only 11-33% of their games" in the opener versus "most
players win around 40%" for Turn 1 later (internal inconsistency CONFIRMED);
Turn 3 actual "about 11%. That is roughly 1 in 9 games"; FreeCell "about
99%" solvable with skilled players at "75-80%" (beginners ~40%). NOT on the
page: any 43% expert figure, any 30-35% skilled band, any "188k real games /
31%" statistic, and nothing about bots or "iterated-rollout" players; those
search-snippet attributions are REFUTED as claims about this page (they
belong to some other, unidentified source).
- Bearing on the write-up: usable only as a vendor-grade sanity anchor for
  "theoretical ~82% vs typical-player ~11-40%"; the human-expert ~43% number
  currently has NO verified source and must be re-sourced or dropped.

**NEW A. Bateni, Pratt, Whitehead, "LLM Game Rule Understanding through
Out-of-Distribution Fine-Tuning", AIIDE 2025 (Proceedings of the
Twenty-First AAAI Conference on AIIDE). Status: peer-reviewed.** Verified
from the full PDF: uses "Solitaire card games as our testbed", a custom
Solitaire GDL (SGDL) covering "over 30 variants from Solitaire families of
Spider, Klondike, Pairing and Free Cell" (Klondike family: Blind Alley's,
Double Klondike, Klondike, Legion, Thirty-Six, Westcliff, Whitehead);
fine-tunes GPT-4o-mini (~200 samples per config) on action-legality plus
next-state progression questions, compares against untuned DeepSeek v3 and
Llama 3.3 70B; its Table 1 marks itself Game Progression yes, Game Playing
NO; play traces are generated by a non-LLM heuristic DFS agent (its Figure 4
win rates, e.g. Thirteen 83%, Baker's Game 15%, are that bot's, not an
LLM's); RL is only future work ("or using reinforcement learning
strategies"); it cites Blake and Gent. Its "Xiao and Yang 2024" citation
resolves to arXiv 2410.02829 ("LLMs May Not Be Human-Level Players, But They
Can Be Testers"), difficulty measurement in the Slay the Spire context, not
solitaire play.
- Bearing on the write-up: this is now THE closest prior art and an
  unmissable citation; differentiation is clean: rule-understanding QA on
  solitaire variants versus your full-game play with win rates, on-policy
  GRPO, and an on-device model. Code: github.com/iambb5445/SolitaireGDL.

**NEW B. Game-RL / GameQA, arXiv 2505.13886 (v5). Status: preprint per
fetched copy.** CONFIRMED: "GameQA includes 30 games, 158 distinct tasks and
approximately 140K questions", Spider Solitaire is in its game table, games
are converted "into a Visual Question Answering (VQA) format", GRPO is used
with binary answer-correctness rewards, and no model plays complete games to
win/loss. The search snippet's claim that it also adapted an open-source
Pygame Klondike engine was NOT found in the fetched portion: treat as
plausible, unconfirmed.
- Bearing on the write-up: the only found case of solitaire inside an
  LLM/VLM training pipeline; differentiate: state-QA data synthesis for
  general reasoning versus playing the game.

## Mandatory citation list

- Bateni, Pratt, Whitehead, AIIDE 2025 (SolitaireGDL): fine-tunes LLMs on
  Klondike-family solitaire RULES (legality/next-state QA); explicitly does
  not do game playing; no win rates, no RL.
- Bateni and Whitehead, FDG 2024 (Slay the Spire): closest single-player
  card-game PLAY work; prompting-only GPT on a simplified battle sim; not a
  patience game.
- Wang et al., NeurIPS 2025, arXiv 2509.01328: LoRA SFT on 8 multiplayer
  card games approaches but never exceeds the teacher AI; no single-player
  games, no RL; your teacher-parity finding matches, your GRPO attempt goes
  further.
- UrzaGPT, arXiv 2508.08382 (preprint): LoRA SFT for MTG draft picks
  plateaus below domain SOTA and recommends RL from outcomes;
  single-decision, not gameplay.
- Blake and Gent, JAIR 85 (2026), DOI 10.1613/jair.1.17167:
  perfect-information Klondike winnability 81.945% +/- 0.084%; the ceiling
  anchor, now journal-published.
- Game-RL/GameQA, arXiv 2505.13886: uses Spider Solitaire states as VQA
  training data for VLM reasoning with GRPO; no full-game play.
- lmgame-Bench, arXiv 2505.15146 (ICLR 2026 per repo): six-game LLM
  benchmark, no card games; harness lift 40% fail-vs-random unharnessed to
  86.7% beat-random harnessed.
- BALROG, ICLR 2025, arXiv 2411.13543: agentic LLM benchmark on RL
  environments, no card games found.
- TextArena, arXiv 2504.11442: ~99 text game envs including single-player
  card games (Blackjack, ThreeCardMonte) and poker variants, but no
  solitaire/patience env.
- Optional secondary: Game Reasoning Arena (arXiv 2508.03368, OpenSpiel
  incl. Kuhn Poker, competitive only), DSGBench (arXiv 2503.06047, six
  strategy games, no cards), Xiao and Yang (arXiv 2410.02829, LLM agents as
  difficulty testers), Schultz et al. (arXiv 2412.12119, LLM search-based
  play of chess/connect-four/hex, cited by AIIDE 2025 as the
  game-playing-with-training row).

## Fresh sweep results

Five searches total (cap respected): (1) "Bateni Whitehead Slay the Spire
large language model paper" (found FDG 2024 paper); (2) "odds of winning
solitaire backed with data" domain-locked to solitaire.com (found the exact
page); (3) "LLM plays Klondike solitaire agent language model 2026"; (4)
quoted "solitaire" + large language model agent benchmark fine-tuning paper
2025; (5) "reinforcement learning LLM environment card game Klondike
FreeCell Spider solitaire arxiv". Searches 3-5 found NO third-party
published LLM-plays-Klondike work. Search 3's only Klondike-plus-LLM hits
describing "a Klondike engine with an LLM move-decision dataset and
distilling/preference-training small local models" appear to be this
project's own public artifacts (GitHub klondike topic, HF dataset), meaning
the project is already the top discoverable item in its own niche. Search 5
surfaced the two genuine near-misses that were then fetched and verified
(Game-RL 2505.13886; AIIDE 2025 SolitaireGDL paper). NOT-FETCHED leads for
completeness: balrogai.com env page, the ACM full text of the FDG 2024 paper
(403), OpenReview pages for 2509.01328, the git-disl/awesome-LLM-game-agent-
papers survey list, and Xiao and Yang 2410.02829 (characterized only via the
AIIDE paper's citation). Fetch budget note: verification ran to 18 fetches
(the ~10 target was exceeded to chase the two direct novelty threats); no
session or usage errors occurred.

## Ceiling numbers table

| Planning number | Best current source | What the source SHOWS | Evidence grade |
|---|---|---|---|
| ~81.9% perfect-info winnability | Blake and Gent, JAIR Vol 85 (2026), DOI 10.1613/jair.1.17167, published 2026-02-27 | 81.945% +/- 0.084% thoughtful (all ranks/suits known) draw-1 Klondike; ~30x tighter CI than prior work | Peer-reviewed journal; STRONG; cite JAIR, not just arXiv 1906.12314 |
| 43-52% imperfect-info "thoughtful play" ceiling | No primary source verified in this pass; nothing new found that moves it either way | Nothing fetched supports or contradicts the band; JAIR paper is perfect-info only | UNSOURCED in this pass; keep only with its original citation trail or label as project estimate |
| Human expert draw-1 ~43% | Attributed to solitaire.com; REFUTED as an attribution (number absent from the page) | Page gives: typical players 11-33% (opener) vs "around 40%" Turn 1 (later), draw-3 ~11%, theoretical ~82%; no expert figure, no citations | Vendor-grade, internally inconsistent; re-source or drop the 43% expert claim |
| Teacher / student ~31-33% | Internal: docs/reports/20260628_volume_fresh_winrate.md (16/48 pooled fresh-deck) plus teacher corpus rate | Project's own replicated measurement; consistent with vendor "typical player" band, well below the 81.9% perfect-info ceiling | Internal, replicated; strongest number the write-up itself contributes |

One further note for the write-up: the solitaire.com page claims FreeCell
skilled players reach 75-80% against ~99% solvable deals, an uncited vendor
figure, but it is the only found external anchor for "skilled human play
lands well under the perfect-information ceiling", which is directionally
the same story your 33%-versus-81.9% framing tells for Klondike.
