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

## Status

- Done: literature notes in `docs/lit/`, pre-registered hypotheses.
- In progress: benchmark and Thread A code in `src/intr/`, prior art in `docs/lit/prior_art.md`, hot-feed list
  in `docs/lit/hot_feed_triage.md`. If these files are missing or incomplete after an interruption, redo that
  piece.

## Next steps

1. Review and commit the implementation, then run `tests/` on Colab, then the Thread A smoke run.
2. Revise `docs/HYPOTHESES.md` for anything prior art already answered.
3. Run Thread A in full and write up results with verdicts per the protocol.
4. Start Thread C in parallel, then Thread B if H4 survives, then Threads D and E.

## Runtime notes

- Colab is CPU only: 2 cores, 12 GB RAM, torch 2.11. Its session depends on the browser tab staying connected.
- The repo is private, and no token goes into Colab. Code reaches Colab by writing files from notebook cells.
