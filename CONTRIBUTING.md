# Contributing

Thanks for your interest in **VLA-Collapse-Recover**. This is a research-engineering portfolio
project — a reproducible diagnostic *study*, not a framework — but issues and focused PRs are
welcome.

## Issues welcome

- Bug reports, reproduction problems, and questions about the method or results are all welcome.
  Use the issue templates and include enough to reproduce: OS, Python version, `uv --version`, and
  the exact command.
- **A number that doesn't reproduce is the most valuable issue.** `make stats` should regenerate
  [`analysis/runs/phase5_summary.md`](analysis/runs/phase5_summary.md) byte-for-byte — if it
  doesn't on your machine, please report it.

## Ground rules (the project's non-negotiables)

- **No fabricated results.** Every metric comes from a real run; it stays `TBD` until then.
- **Delta-only headlines.** We do not claim absolute SOTA — only ΔSR / Recovery / Δ_method.
- **In-dist vs held-out is always labeled.** In-distribution recovery is never presented as
  generalization.
- **Negative findings stay.** `viewpoint` 0%, `noise` degradation after LoRA, and C-`lighting` < B
  are real conclusions, not bugs to be smoothed over.
- **The reproduced statistics are frozen.** Do not change the logic in `eval/stats/` or
  `eval/runners/` in a way that alters `phase5_summary` — the `make stats` diff must stay empty.

## PRs need a passing test

1. Fork and branch.
2. `make setup`, then make your change.
3. **Add or update a test** that covers it. PRs that change behavior without a test won't be merged.
4. `make lint` (ruff, zero errors) and `make test` (pytest, all green) must pass locally — CI runs
   both on Python 3.12.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`,
   `refactor:`, `chore:`, `ci:`); keep PRs small and reviewable.

## Scope note

GPU-dependent code (training, rollouts) runs only on the rented RTX 5090 stack; its logic is
unit-tested off-GPU behind seams, so the suite stays green without `torch`/`lerobot`. New GPU code
should keep that pattern (pure logic testable without the heavy stack). See
[docs/REPRODUCING.md](docs/REPRODUCING.md).
