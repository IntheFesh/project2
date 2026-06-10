"""Phase drivers: end-to-end rollout/eval/stats entry points (``python -m eval.runners.*``).

These orchestrate the experiment phases (collapse, recovery, paired-statistics report) and depend
on the heavier rollout stack at *run* time; the reusable statistics primitives they call live in
``eval.stats``. Pure-logic helpers remain importable off-GPU.
"""
