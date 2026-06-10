# Changelog

All notable changes to VLA-Collapse-Recover are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-06-10

**First complete single-seed run (Phases 2–5)** with real, per-episode paired statistics.

### Added
- **Collapse curve** (Phase 2, condition A) — base SmolVLA under LIBERO-Plus
  {viewpoint, lighting, texture, noise} × {L2, L4}; base clean SR 64.0%.
- **Recovery experiment** (Phase 4) — condition B (LoRA + standard aug) and C (LoRA +
  perturbation-targeted aug) on the in-distribution families plus the held-out `layout` family.
- **Paired-statistics harness** (Phase 5; `eval/stats/` + `eval/runners/phase5_stats.py`) —
  bootstrap CIs, McNemar, Holm–Bonferroni; regenerates `analysis/runs/phase5_summary.{md,json}`.
- **Diagnostic-probe battery** (`docs/PROBES.md`) — held-out cross-family generalization (live) +
  language-conditioning probe (scaffolded).
- **Headline results** — H1 ΔSR +7.4 pp [+2.8, +11.9], McNemar p ≈ 0.0018 (significant);
  H2 no family-matching advantage (all Holm-p > 0.05); H3 held-out +15.0 pp [+5.0, +25.0],
  p ≈ 0.0072 (significant).
- **Honest negative findings** recorded as conclusions — `viewpoint` 0% (2-D proxy), `noise`
  degradation after LoRA, C-`lighting` < B.
- **Tooling** — `Makefile`, off-GPU test suite (115 tests), `deploy.py` (AutoDL), `run.sh`, budget
  gating, GitHub Actions CI; LICENSE (MIT), CITATION.cff.
- **Docs** — README, `docs/RESULTS.md`, `docs/REPRODUCING.md`, `docs/PROBES.md`,
  `docs/EVALUATION.md`, `docs/LIBERO_PLUS_NOTES.md`.

### Changed
- Reorganized `eval/` into `eval/stats/` (statistics primitives) and `eval/runners/` (phase
  drivers); pinned dependencies and targeted Python 3.12.

### Known limitations
- Single seed — within-seed paired statistics only; a 3-seed run on the key B/C comparison is the
  natural extension.
- `viewpoint` augmentation is a 2-D image-space proxy (no true 3-D camera shift); Probe 3 (visual
  feature shift) and the language-conditioning probe are scaffolded but not run in this budget.

[Unreleased]: https://github.com/IntheFesh/project2/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IntheFesh/project2/releases/tag/v0.1.0
