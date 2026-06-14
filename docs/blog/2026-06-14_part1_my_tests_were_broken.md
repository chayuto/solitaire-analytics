# My model wasn't broken. My tests were.

This is the first post in a series. I'm building a small AI that plays Solitaire on my laptop. I'm a senior dev, not an ML expert, and I'm figuring this out as I go.

For about a month, I was sure the model just couldn't learn. The real problem was my tests. They were green, and they were lying to me.

If you've ever lost days to a bug and found out the test was wrong, not the code, you know this feeling.

## The setup

I want a tiny model, small enough to run on my laptop, to play Klondike Solitaire.

A bigger model already plays it okay, so the plan was to copy it. You let the big model play thousands of games and record every move. Then you train the small model to make the same moves. It's basically learning from a recording.

I built all of it: the data, the trainer, a solver that tells me if a game is winnable, and a test that deals a winnable game and lets the small model play to the end.

That test is what this post is about. I spent all my time on the model and the data, and almost none on the test. That was the mistake.

## Everything lost to nothing

The result that haunted me was simple. I'd train the model, test it, and it scored worse than the same model with no training at all.

Every time. Seven different setups. I cleaned the data, kept only the winning games, tried a smarter training method. They all came out worse than doing nothing.

When every road hits the same wall, you start to believe in the wall. So I wrote it down: copying doesn't work, the small model can't beat the teacher, move on.

I was wrong. The experiments were fine. The test was broken, in three different ways.

## Three broken tests

The first was training loss, the number the trainer prints while it learns. It dropped nicely and meant nothing about real play. It's your build going green while the app crashes on launch.

The second was a quick check: show the model 20 hard positions, ask for one move each, score them. Fast and useless. It once rated the tiny model above the big teacher, which is nonsense. One move says almost nothing about a 200-move game. It's unit testing a single function and shipping.

The third was the one I trusted most, the full game. And my test setup didn't match production. It used an old prompt. It was missing a rule, so the game quietly became unwinnable halfway through, every time. It handed the model forced moves the real system plays for you. And on a small formatting slip, my test gave up after three tries. The real system retries seventeen.

None of these were loud bugs. They were small gaps between my test and the real thing. Together, they faked failure.

## I fixed the tests, and the model won

So I stopped touching the model and spent a week making the test honest. Same prompt as production. The missing rule added back. No more forced moves. Retries on the formatting slips, like the real system does. And a second check: hand the final board to the solver and ask if the model was one move from winning, or had killed the game fifty moves ago.

Then I tested the plain base model. No training. The do-nothing baseline.

It won. A full game, all 52 cards home. The first complete win in the whole project. I replayed all 211 moves to be sure it was real. It was.

That broke the spell. If the baseline can win, then "trained models lose to the baseline" was never about training. It was a broken test making everything look like a loss.

## The takeaway

If you train models, keep this on a sticky note: when results are bad, check the test before you blame the model.

Broken code fails loud. A broken test fails quiet, in the exact shape of a real result. That's what makes it dangerous. For a month I had a clean, steady, completely false finding, and the steadiness was the trap. Seven setups losing wasn't seven bad setups. It was one bad test, read seven times.

The model was fine the whole time. I just couldn't see it.

Next post: I go back to the training I'd written off, and find out if it actually works once the test is honest.
