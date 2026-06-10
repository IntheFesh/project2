"""Run the noise family per-task with timeout protection.

noise perturbations (fog/blur via plasma_fractal + per-pixel shuffle) are ~5x slower than
other families (~60s/episode) and can hang under EGL. Running all 12 tasks in one eval call
risks a 2h hang. Instead we run each task as its own eval call with a timeout; a hung task
is skipped (recorded as missing) and the batch continues. Each completed task is flushed.
"""
from __future__ import annotations
import csv, subprocess, sys
from pathlib import Path
sys.path.insert(0, ".")
from eval.runners.lerobot_runner import run_eval
from eval.runners.phase2_collapse import select_cell_task_ids, SUITE, POLICY_PATH

OUT = Path("analysis/runs/phase2_collapse.csv")
N_EP = 5
SEED = 0
PER_TASK_TIMEOUT = 600  # 10 min/task (5 ep x ~60s + margin)

def main():
    rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
    rows = [r for r in rows if r["family"] != "noise"]  # clear prior noise
    print(f"[noise] starting from {len(rows)} non-noise rows")
    fields = ["condition","task_id","family","level","seed","success"]

    def flush():
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    for level in [2, 4]:
        task_ids = select_cell_task_ids(SUITE, "noise", level, 12)
        print(f"[noise] L{level}: {len(task_ids)} tasks, ids={task_ids}")
        for i, tid in enumerate(task_ids):
            try:
                res = run_eval(policy_path=POLICY_PATH, suite=SUITE, task_ids=[tid],
                               n_episodes=N_EP, seed=SEED, is_libero_plus=True,
                               timeout_s=PER_TASK_TIMEOUT)
                for r in res:
                    rows.append({"condition":"A","task_id":r["task_id"],"family":"noise",
                                 "level":level,"seed":SEED,"success":int(r["success"])})
                ns = sum(r["success"] for r in res)
                print(f"[noise] L{level} task {tid} ({i+1}/{len(task_ids)}): {ns}/{len(res)}")
                flush()
            except subprocess.TimeoutExpired:
                print(f"[noise] L{level} task {tid} ({i+1}/{len(task_ids)}): TIMEOUT, skipped")
            except subprocess.CalledProcessError as e:
                print(f"[noise] L{level} task {tid} ({i+1}/{len(task_ids)}): ERROR {e.returncode}, skipped")

    # summary
    n2 = [r for r in rows if r["family"]=="noise" and r["level"]=="2"]
    n4 = [r for r in rows if r["family"]=="noise" and r["level"]=="4"]
    def sr(rs): return f"{sum(int(r['success']) for r in rs)}/{len(rs)} = {100*sum(int(r['success']) for r in rs)/len(rs):.1f}%" if rs else "0/0"
    print(f"\n[noise] DONE. L2: {sr(n2)}  L4: {sr(n4)}  total rows: {len(rows)}")

if __name__ == "__main__":
    main()
