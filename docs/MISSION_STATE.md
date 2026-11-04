# Mission state

Read this first when resuming. Update it at each milestone and commit it on the mission branch.

## Goal

Find techniques from any domain worth integrating into world models and physical AI. Hypothesize,
implement rigorously, test in the cloud, and commit to `meta-lucky-builds/intr` on branch
`mission/world-model-techniques`.

## Rules

- Models, all at medium effort: Opus for implementation and thinking; Sonnet for executing finished work,
  simple tasks, and web search; Haiku for simple retrieval.
- Tell agents to use the how-to-think skill when needed. Use it for hypotheses, planning, analysis, design.
- Never run tests locally, even CPU ones. Colab first, for small tests.
- Follow the protocol in `docs/HYPOTHESES.md`. Changing a prediction or threshold after results exist needs
  its own commit explaining why.

## Usage habits

- Check plan usage before launching a batch of agents, and stagger batches when usage is high.
- Give agents narrow briefs pointing at specific sections, not whole files.
- Use the cheapest tier that fits. Keep quality-critical work on Opus.
- Commit often so nothing is lost if a limit stops work mid-step.
- Do not shorten seeds, protocol, or verification to save usage. Delay instead.

## Context habits

- Have agents write findings to files and return short summaries. Read only the sections needed.
- Keep this file current, so a resume does not need old conversation or re-reading.

## Avoiding local optima

- After each thread's result, decide explicitly: continue, refine, or drop. Record the reason here.
- Drop or reframe a thread when its kill criterion fires, or when two rounds of fixes produce no effect above
  the protocol threshold. Do not keep tuning to rescue a hypothesis.
- A null result is a result. Write it up, then ask what it implies before moving on.
- New hypotheses and approaches are welcome at any time. Add them to `docs/HYPOTHESES.md` with a kill criterion
  before testing them, and mark any that came from seeing results as post hoc.
- Check for new work and unread leads between threads, starting with 2610.42146, 2611.02782, and the 22
  robotics-tagged papers in `docs/lit/hot_feed_triage.md`.

## Status

- Done: literature notes, prior-art check, pre-registered hypotheses, benchmark and Thread A code.
- Verified on Colab at commit 28df3b3: 92 tests pass, smoke run completes.
- Oracle planner gate at full settings: pendulum interp 1.0, extrap 0.38-0.48, oof 0.86-0.95; puck 1.0 on all tiers.
  A stronger planner does not raise pendulum extrap, so read closed-loop results there against the oracle.
- Thread C code committed, never run. Sync to Colab and test it only after Thread A finishes, so two jobs on 2 cores
  do not distort Thread A's wall-clock numbers.
- H4 and H10 code committed in `experiments/thread_b.py`, never run in full.
- Threads B and C tests and smokes run at low priority from `/content/next`, a separate copy, so `/content/intr` stays
  unchanged during the Thread A run.
- Thread A timing: about 1 hour per pendulum seed and slower for puck, so the 300-minute guard stops it at about 2-3 of
  the 5 required seeds. It needs a way to resume the remaining seeds, and results must leave Colab before any restart.
  A partial run does not satisfy the protocol and gets no verdicts beyond "preliminary".
- Running: Thread A full run on Colab in the background, launched from a notebook cell. Log at
  `/content/intr/full.log`, results at `/content/intr/results/thread_a/`. Check with a short cell. Each finished
  job is saved, and it stops starting new jobs after 300 minutes. Colab files are lost if the runtime drops, so
  copy results into the repo as soon as it finishes.

- Threads B and C tests are rerunning at nice 5 in `/content/next` (log `pytest_next.log`). nice 19 starved them: 11 CPU
  seconds in 30 minutes. Hashes verified. 21 tests had passed before the restart.
- This contention makes Thread A wall-clock numbers unreliable for that window. Compare compute using pass and write
  counts, and say so in the write-up.
- Do not inspect Thread A method comparisons before the run ends. Pipeline sanity on seed 0 pendulum passed.

## Next steps

0. Threads B and C: 100 of 100 tests pass on Colab and both smokes finish cleanly. Run them in full after Thread A finishes.
   `--resume-from` was added to `thread_a.py`. Its test is running in `/content/next` (`resume_status.log`, results in
   `r1/thread_a_smoke.json`). Pass condition: the resumed 2-seed smoke's `jobs` match the uninterrupted
   `/content/intr/results/thread_a/thread_a_smoke.json` up to wall-clock fields. Do not use resume until it passes.
   If the Thread A process has stopped, go to step 1, adding a way to resume the remaining seeds if fewer than 5 finished.
1. When the full run ends, copy the summary into `results/`, write up verdicts per the protocol, and commit.
2. Start Thread C in parallel, then Thread B if H4 survives, then Threads D and E. H4 needs its own analysis:
   the full run does not report calibration.
3. Read leads between threads.

## Runtime notes

- Colab is CPU only: 2 cores, 12 GB RAM, torch 2.11. Its session depends on the browser tab staying connected.
- The repo is private, and no token goes into Colab. Code reaches Colab by writing files from notebook cells.
