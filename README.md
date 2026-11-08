# forge — Training Ladder, Stage 1

A public fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (preserved as [UPSTREAM.md](UPSTREAM.md)), used for **stage 1 of my training ladder**: reproduce the current NanoGPT speedrun record and run one measured bet.

- **Track:** train to ≤3.28 val loss (FineWeb) on 8×H100, fast as possible.
- **Current record:** 0.665 min (record #92, 2026-08-30, upstream `master` @ `4ea6b93`).
- **My goals for stage 1** (full design: `~/Projects/forge/docs/superpowers/specs/2026-11-09-training-ladder-design.md`):
  1. Reproduce the record with disciplined timing (untimed kernel-warmup is not counted).
  2. Understand every technique in the record stack — see [NOTES.md](NOTES.md) (mechanism map).
  3. Run exactly one bet: hypothesis fixed before the run, measured delta, negative results published with numbers — see [BENCH.md](BENCH.md).
  4. Write up honestly in [WRITEUP.md](WRITEUP.md).
- **Rules I follow:** upstream speedrun rules (data pipelines untouched; p<0.01 logs for ML-touching submissions; no extra inductor/compile flags). Money discipline in [runs/SESSIONS.md](runs/SESSIONS.md) — rented GPU time is logged per session.
- A merged record is only possible by **beating** 0.665 min on same-class hardware; anything else ships as a writeup, not a claim.

Logs of all runs live in `runs/`. This repo is public from the first commit.
