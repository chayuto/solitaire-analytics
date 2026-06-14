# The training I gave up on started working.

This is part two. In part one, I found out my tests were broken, fixed them, and watched my do-nothing baseline win a full game of Solitaire for the first time. If you missed it, start there.

Quick recap. I'm a senior dev teaching a small model to play Klondike by copying a bigger one. Every training run I tried had lost to doing nothing. Then I learned the test was lying, not the model.

So now I had a fair test. Time to go back and retry the training I'd written off.

## With honest tests, training works

The experiment was the same one I'd given up on. Train the small model on the teacher's winning games. But this time, grade it on the fixed test, and keep the test games out of training so it can't just memorize them.

It beat the plain base model on 12 of 13 games. It won 4 outright where the base won 1. First time in the project that training beat not-training, on a test I trusted.

After a month of nothing working, that felt great. Then I made myself stop and poke at it.

## Was it just luck?

Here's the catch. The trained model got sloppier with its output format. That made it hit the retry path more often. Retries add a little randomness. And randomness can shake a model out of a loop by accident.

So maybe I wasn't measuring better play. Maybe I was measuring lucky dice.

There's one clean way to settle that: a control. I gave the plain base model the exact same dose of randomness, on the exact games the trained model had won.

It won zero of them. Often it did worse. The dice weren't the reason. The training had taught the model something real.

I almost skipped that control because I liked the result. Running it anyway is the only reason I trust the result now.

## The clever idea that wasn't

I also had a theory about why it worked. I figured the trick was being picky with the data: keep only the games the teacher won, throw out the losses.

So I tested it. I trained another model on a plain mix of wins and losses, same amount of data. It did just as well.

My clever filter wasn't the reason. The reason was boring: enough good data. That's it. I keep relearning that the unglamorous answer is usually the right one.

## What I don't know yet

Here's the honest gap. I don't know if the model learned Solitaire, or just learned my decks.

Every game I've tested came from the same pile as its training games. Different games, but the same kind. That's not proof it can play. It might just be good at the decks it grew up around.

The real test is a fresh, random deck it has never seen, with no link to anything it trained on. That test is running on my laptop as I write this.

If it holds up, I have a small model that actually plays. If it falls apart, then it memorized instead of learned, which is its own useful lesson. Either way, that's the next post.

## The takeaway

Two things I'm keeping from this one.

Run the control even when you love the result. The control is what turns "it worked" into "it worked for the reason I think." Without it, I'd have shipped luck and called it skill.

And your clever idea is probably not the reason. Test it against the boring version. Mine lost.
