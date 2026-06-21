#!/usr/bin/env python3
"""Quick probe (2026-06-20): does adding declarative Klondike STRATEGY TEXT to
the SFT mix improve full-game play, or does the student only learn to recite it?

Form 1 of the strategy-text idea (the cheap one): a small set of hand-authored
draw-1 imperfect-information strategy Q&A rows, mixed (train-only) into the
volume corpus. valid/test stay byte-identical to volume, so volstrategy-vs-volume
isolates the strategy rows. Principles target the project's MEASURED failure
modes (reveal pass-up, behavioural loops, foundation over-eagerness, the
false-resign / not-resign-on-dead gap), not generic perfect-info advice.

This is a declarative->procedural BET: the model already gets strategy in the
v1.6 PROMPT every turn and under-applies it, so the question is whether strategy
in the WEIGHTS does better. Judge on full-game adjudicated play (recites-well !=
plays-well) and watch the JSON parse-rescue rate (prose rows may erode JSON).

  build_volstrategy_corpus.py [--copies 12] [--src dataset_volume] [--out dataset_volstrategy]
"""
import argparse, json
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC_DEFAULT = THIS / "dataset_volume"
OUT_DEFAULT = THIS / "dataset_volstrategy"

# (question, answer) -- concise, correct draw-1 imperfect-info Klondike strategy.
STRATEGY = [
    # --- reveals / excavation (counters the reveal-pass-up kill signal) ---
    ("What is the single highest-value kind of move in Klondike Solitaire?",
     "A move that flips a face-down tableau card. Every face-down card hides information and a potential play, so a move that reveals one beats a move that only rearranges cards you can already see."),
    ("You can either shuffle a card between two columns (revealing nothing) or play a different card that flips a face-down card. Which do you pick?",
     "Flip the face-down card. Rearranging visible cards without a reveal makes no real progress and invites a loop; revealing adds information and new options."),
    ("Which face-down pile is usually best to dig into first?",
     "The column with the most face-down cards, since unburying it unlocks the most hidden cards and future moves -- but only when a legal sequence of moves can actually expose it."),
    ("Two different moves each flip a face-down card. How do you choose between them?",
     "Prefer the one that also makes other progress (a foundation play, or unburying the larger face-down pile) and that does not bury a card you will need soon."),

    # --- loop avoidance (counters behavioural doom-loops) ---
    ("Last turn you moved the 9H from column 3 to column 4. This turn the only new-looking option is to move it back. Should you?",
     "No. Moving a card back where it came from with no reveal and no foundation gain is a wasted, looping move. If nothing else progresses, draw from the stock, and if the position is truly stuck, resign."),
    ("How do you recognize you are stuck in a behavioural loop rather than making progress?",
     "The same one or two cards shuffle between the same columns over several turns with no drop in the face-down count and no foundation progress. Break it with a reveal, a foundation play, or the stock; if none exist, the board may be dead."),
    ("Why is moving a card back and forth harmful beyond wasting one turn?",
     "It entrenches a loop the player tends to repeat and it burns the limited turn budget without lowering the face-down count, which is the real measure of progress."),

    # --- foundation discipline (counters foundation over-eagerness) ---
    ("Should you always play a card to the foundation as soon as it becomes legal?",
     "No. Aces and twos always go up. For higher cards, keep one in the tableau if it can still receive a card you need to move, because once it is on the foundation it can no longer help the tableau."),
    ("An ace and a two are available to play up. Do you play them?",
     "Yes, immediately and always. They can never be useful in the tableau, so sending them up is pure gain and frees flexibility."),
    ("You can send the 5H to the foundation, but a black 4 in another column needs a red 5 to move onto. What do you do?",
     "Hold the 5H in the tableau and move the black 4 onto it first. Sending the 5H up now would strand the black 4."),
    ("Why can playing a card to the foundation too early hurt you?",
     "Tableau builds need cards of the opposite color one rank lower; a card on the foundation is gone from the tableau, so playing it up early can leave another card with no legal home."),

    # --- not burying needed cards ---
    ("Why avoid covering a low card that a foundation will soon need?",
     "If you bury the 3 of a suit while that foundation sits at 2, under cards you cannot move, you may never retrieve it, blocking that suit for the rest of the game."),
    ("You can move a black 9 onto a red 10, but it would bury the red 4 your foundation needs next. Good move?",
     "No. Do not bury a card the foundation needs soon. Find a move that keeps the needed card retrievable."),

    # --- columns / kings ---
    ("When is it safe to empty a tableau column?",
     "Only when you have a King, or a King-headed sequence, ready to move into the empty space. An empty column with no King to fill it is usually wasted and can lock the board."),
    ("You could empty a column, but you have no King available. Is that wise?",
     "Usually not. Emptying a column you cannot refill with a King removes a working pile for little gain; prefer a move that reveals a face-down card instead."),
    ("Where should Kings go, and which King should you move first?",
     "Kings go into empty columns to anchor a new descending alternating-color build. Prefer moving a King whose move also frees a face-down card or unblocks a needed card."),

    # --- sequencing / planning ---
    ("How should you build sequences in the tableau?",
     "Descending rank in alternating colors (red on black, black on red), aiming to build runs long enough that moving the whole run at once exposes a face-down card."),
    ("Should you move a sequence one card at a time or as a whole unit?",
     "Move the full ordered run as a unit when it exposes a face-down card or frees a column. Piecemeal shuffling that reveals nothing just wastes moves."),

    # --- draw / stock discipline (draw-1, imperfect info) ---
    ("In draw-1 Klondike, what does cycling the stock actually give you?",
     "You see every stock card in order, one per draw, and may recycle the waste when the stock empties. Draw to find a specific card you need or to surface a play, not aimlessly."),
    ("You have already seen every card in the stock this cycle and none helped. Is drawing again useful?",
     "Re-drawing cards you have already seen gives no new information. Draw only if a previously-unplayable seen card is now placeable; otherwise look for a tableau move or accept the position is stuck."),
    ("Should you draw from the stock before exhausting your tableau moves?",
     "Generally make safe, progress-making tableau moves first, especially reveals and aces and twos. Draw when the tableau offers no progress and you need a fresh card."),

    # --- imperfect information ---
    ("You cannot see the face-down cards. How should that shape your play?",
     "Treat reveals as the way to buy information, prefer flexible moves that keep options open, and avoid committing to a plan that depends on a specific hidden card until you have revealed it."),

    # --- dead-board recognition / resign (counters both false-resign and not-resigning-on-dead) ---
    ("When is resigning a Klondike game the correct choice?",
     "When no legal move can make progress: no face-down card can be revealed, no card can reach a foundation now or after drawing, and the stock is exhausted or only recycles cards that do not help. Resigning a truly dead board is correct; looping on it is not."),
    ("The stock is empty, no reveal is possible, and the next foundation cards are buried under sequences you cannot move. What is this position?",
     "Structurally dead. There is no winning line, so resign rather than shuffle cards indefinitely."),
    ("Is it ever right to keep playing a board that looks stuck?",
     "Yes, when a reveal or a foundation play still exists, or the stock still holds an unplayed card that helps. Only resign once you have confirmed none of those remain."),
    ("You reached a near-complete board with no face-down cards left and a legal foundation play is available. Should you resign?",
     "No. A legal foundation play is progress, so play it. Resigning a winnable, fully-revealed endgame throws the game away; resign only when no progressing move exists."),

    # --- opening ---
    ("On the first move you can play an ace up or make a move that reveals a face-down card. What order?",
     "Play the ace up first, since it is free gain, then prioritize a move that flips a face-down card."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=12,
                    help="copies of each strategy row in train (exposure vs memorisation tradeoff)")
    ap.add_argument("--src", default=str(SRC_DEFAULT), help="volume split to mix into")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # valid/test verbatim from volume so the comparison isolates the train mix
    for name in ("valid", "test"):
        (out / f"{name}.jsonl").write_text((src / f"{name}.jsonl").read_text())

    train = [json.loads(l) for l in (src / "train.jsonl").read_text().splitlines() if l.strip()]
    base_n = len(train)
    strat_rows = [{"prompt": q, "completion": a} for q, a in STRATEGY]
    for _ in range(args.copies):
        train.extend(dict(r) for r in strat_rows)
    added = len(strat_rows) * args.copies
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))

    print(f"unique strategy principles: {len(STRATEGY)}")
    print(f"base volume train rows:     {base_n}")
    print(f"strategy rows added (x{args.copies}):  {added}  ({100*added/len(train):.1f}% of train)")
    print(f"-> {out} train rows: {len(train)}  (valid/test unchanged vs volume)")


if __name__ == "__main__":
    main()
