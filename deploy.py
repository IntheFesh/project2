#!/usr/bin/env python3
"""VLA-Collapse-Recover -- integrated one-click deploy (AutoDL-aware).

A single, stdlib-only file that prepares the project to run on a rented GPU. It:
  1. detects the environment (AutoDL? GPU? a suitable pre-installed PyTorch?),
  2. routes model/dataset caches to the big data disk (/root/autodl-tmp on AutoDL),
  3. turns on AutoDL academic network acceleration + an HF mirror (China network),
  4. installs the dependency stack -- REUSING a suitable pre-installed cu128 PyTorch
     instead of re-downloading multiple GB,
  5. pulls LeRobot (SmolVLA / pi0.5) and LIBERO-Plus from source,
  6. verifies the RTX 5090 exposes sm_120, runs the off-GPU tests, and gates the
     GPU-day budget.

Honest scope: the rollout / training / demo entry points (Phases 1-4, 6) are scaffolded
with NotImplementedError seams. This script makes the project *ready to run* and exercises
everything that is implemented (stats, budget, tests, env verification). It fabricates
nothing -- all result tables stay TBD until a real run fills them.

Usage (on an AutoDL RTX 5090 instance, from the repo root):
    python deploy.py                 # full deploy with AutoDL-friendly defaults
    python deploy.py --check         # DRY RUN: detect + print the plan, install nothing
    python deploy.py --stage core    # off-GPU deps + tests only (no torch/lerobot)
    python deploy.py --budget-only   # just project the GPU-day budget
    python deploy.py --help

Runs with a bare `python3` (stdlib only); third-party imports happen only in subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

CORE_DEPS = [
    "numpy>=1.26", "scipy>=1.11", "pandas>=2.0", "matplotlib>=3.7",
    "pyyaml>=6.0", "tqdm>=4.66", "pytest>=8.0", "ruff>=0.6",
]
GPU_ADJACENT = [
    "transformers>=4.44", "huggingface-hub>=0.24", "peft>=0.11",
    "imageio>=2.34", "imageio-ffmpeg>=0.5",
]
TORCH_PKGS = ["torch>=2.7.0", "torchvision>=0.22.0"]
CU128_INDEX = "https://download.pytorch.org/whl/cu128"
LEROBOT_SPEC = "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot"
LEROBOT_FALLBACK = "lerobot @ git+https://github.com/huggingface/lerobot"
LIBERO_PLUS_URL = "https://github.com/sylvestf/LIBERO-plus"  # official repo (arXiv 2510.13626)
HF_MIRROR = "https://hf-mirror.com"

# ---------------------------------------------------------------- logging --
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def log(s: str) -> None:
    print(_c("1;36", "==> " + s))


def ok(s: str) -> None:
    print(_c("32", "OK ") + s)


def warn(s: str) -> None:
    print(_c("33", "!! " + s))


def err(s: str) -> None:
    print(_c("31", "xx " + s), file=sys.stderr)


def step(i: int, n: int, s: str) -> None:
    print(_c("1", f"\n[{i}/{n}] {s}"))


def run(cmd: list[str], env: dict | None = None, check: bool = True, cwd: Path | None = None) -> int:
    print(_c("2", "$ " + " ".join(cmd)))
    res = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res.returncode


# -------------------------------------------------------------- detection --
def probe_torch() -> dict | None:
    """Probe torch in a subprocess (so deploy.py itself stays stdlib-only)."""
    code = (
        "import json,torch\n"
        "d={'version':torch.__version__,'cuda':getattr(torch.version,'cuda',None),"
        "'avail':torch.cuda.is_available()}\n"
        "d['cap']=list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None\n"
        "print(json.dumps(d))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.strip())
    except (ValueError, json.JSONDecodeError):
        return None


def detect() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "autodl_tmp": Path("/root/autodl-tmp").is_dir(),
        "network_turbo": Path("/etc/network_turbo").exists(),
    }
    info["is_autodl"] = info["autodl_tmp"] or info["network_turbo"]
    nvsmi = shutil.which("nvidia-smi")
    info["has_gpu"] = False
    info["gpu_name"] = None
    if nvsmi:
        r = subprocess.run([nvsmi, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
        names = [x for x in r.stdout.strip().splitlines() if x]
        info["has_gpu"] = r.returncode == 0 and bool(names)
        info["gpu_name"] = names[0] if names else None
    info["torch"] = probe_torch()
    return info


def torch_build_ok(t: dict | None) -> bool:
    """True if a pre-installed torch is >= 2.7 built against CUDA 12.8 (cu128)."""
    if not t:
        return False
    try:
        mm = tuple(int(x) for x in t["version"].split("+")[0].split(".")[:2])
    except (ValueError, KeyError):
        return False
    return mm >= (2, 7) and bool(t.get("cuda")) and str(t["cuda"]).startswith("12.8")


def print_detection(info: dict) -> None:
    log("Environment detection")
    print(f"  python      : {info['python']}  ({info['executable']})")
    print(f"  platform    : {info['platform']}")
    print(f"  AutoDL      : {info['is_autodl']}  (autodl-tmp={info['autodl_tmp']}, "
          f"network_turbo={info['network_turbo']})")
    print(f"  GPU         : {info['gpu_name'] or 'none detected'}")
    t = info["torch"]
    if t:
        usable = "yes (cu128, >=2.7)" if torch_build_ok(t) else "NO -- needs cu128 torch>=2.7"
        print(f"  torch       : {t['version']}  cuda={t['cuda']}  "
              f"avail={t['avail']}  cap={t['cap']}")
        print(f"  torch usable: {usable}")
    else:
        print("  torch       : not importable in this interpreter")


# -------------------------------------------------------------- env setup --
def setup_storage(data_dir: Path, apply: bool) -> None:
    """Point HF / datasets / torch caches at the big data disk; export into os.environ."""
    mapping = {
        "HF_HOME": "hf",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "TRANSFORMERS_CACHE": "hf/transformers",
        "HF_DATASETS_CACHE": "datasets",
        "TORCH_HOME": "torch",
    }
    log(f"Cache/storage -> {data_dir}")
    for key, rel in mapping.items():
        p = data_dir / rel
        if apply:
            p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
        print(f"  {key}={p}")
    if apply:
        env_sh = data_dir / "vcr_env.sh"
        lines = [f'export {k}="{data_dir / rel}"' for k, rel in mapping.items()]
        env_sh.write_text("\n".join(lines) + "\n")
        ok(f"wrote {env_sh}  (source it in new shells)")


def setup_network(use_accel: bool, use_hf_mirror: bool, apply: bool) -> None:
    """Apply AutoDL academic acceleration proxies + HF mirror to os.environ."""
    if use_hf_mirror:
        if apply:
            os.environ["HF_ENDPOINT"] = HF_MIRROR
        log(f"HF mirror -> {HF_MIRROR}")
    if use_accel:
        turbo = Path("/etc/network_turbo")
        if not turbo.exists():
            warn("AutoDL acceleration (/etc/network_turbo) not found -- skipping.")
            return
        r = subprocess.run(["bash", "-lc", "source /etc/network_turbo >/dev/null 2>&1 && env"],
                           capture_output=True, text=True)
        keys = ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
        n = 0
        for line in r.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k in keys:
                    if apply:
                        os.environ[k] = v
                    n += 1
        ok(f"AutoDL academic acceleration applied ({n} proxy vars)") if n else \
            warn("no proxy vars captured from /etc/network_turbo")


# --------------------------------------------------------------- installs --
def pip_install(pkgs: list[str], index_url: str | None = None, editable: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-input"]
    if index_url:
        cmd += ["--index-url", index_url]
    if editable:
        cmd += ["-e"]
    cmd += pkgs
    run(cmd, env=os.environ.copy())


def ensure_torch(info: dict, install: bool) -> None:
    if torch_build_ok(info["torch"]) and not install:
        ok(f"reusing pre-installed torch {info['torch']['version']} (cu128) -- no re-download")
        return
    if not install:
        warn("no suitable cu128 torch>=2.7 found. On AutoDL, prefer an image with "
             "'CUDA 12.8 / PyTorch 2.7+'. To install it anyway, re-run with --install-torch.")
        return
    warn("installing cu128 torch + torchvision from the PyTorch index (large download).")
    pip_install(TORCH_PKGS, index_url=CU128_INDEX)


def install_lerobot(spec: str) -> None:
    try:
        pip_install([spec])
    except RuntimeError:
        warn(f"`{spec}` failed; retrying without the [smolvla] extra.")
        pip_install([LEROBOT_FALLBACK])


def install_libero_plus(url: str, path: Path | None) -> None:
    target = path or (REPO / "third_party" / "LIBERO-plus")
    if not target.exists():
        if not url:
            warn(f"LIBERO-Plus not present at {target} and no URL given -- skipping. "
                 f"Clone {LIBERO_PLUS_URL} there, then: pip install -e {target}")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", url, str(target)], env=os.environ.copy())
    pip_install([str(target)], editable=True)
    ok("LIBERO-Plus installed (drop-in replacement for `libero`).")


# ----------------------------------------------------------- run helpers --
def verify_env() -> bool:
    rc = run([sys.executable, "scripts/verify_env.py"], env=os.environ.copy(),
             check=False, cwd=REPO)
    return rc == 0


def run_tests() -> bool:
    rc = run([sys.executable, "-m", "pytest", "-q"], env=os.environ.copy(),
             check=False, cwd=REPO)
    return rc == 0


def run_budget(sec_per_episode: float, train_hours: float) -> None:
    run([sys.executable, "-m", "scripts.estimate_budget",
         "--sec-per-episode", str(sec_per_episode),
         "--train-hours-per-run", str(train_hours)], env=os.environ.copy(),
        check=False, cwd=REPO)


# ---------------------------------------------------------------- summary --
def print_summary(info: dict, gpu_ready: bool, tests_ok: bool) -> None:
    log("Summary")
    print(f"  tests (off-GPU)   : {'PASS' if tests_ok else 'see output above'}")
    print(f"  GPU env (sm_120)  : {'READY' if gpu_ready else 'NOT ready (see verify_env output)'}")
    print()
    warn("Phases 1-4,6 (rollouts / LoRA training / demos) are scaffolded with "
         "NotImplementedError seams -- implement them before scaling. The off-GPU "
         "stats / budget / tests run today; all result tables remain TBD.")
    print()
    log("Next steps")
    print("  1) Smoke-test per-episode wall-clock, then gate the matrix:")
    print("       python -m scripts.smoke_timing --smoke-tasks 2 --smoke-episodes 5 "
          "--train-hours-per-run 4")
    print("  2) Keep heavy work inside tmux (AutoDL SSH can drop):  tmux new -s vcr")
    if info["is_autodl"]:
        print("  3) AutoDL tip: do downloads/setup in 无卡模式 (no-GPU mode) to save GPU-hours, "
              "then switch the instance to GPU mode for rollouts/training.")


# ------------------------------------------------------------------- main --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Integrated one-click deploy for VLA-Collapse-Recover (AutoDL-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["core", "gpu", "all"], default="all",
                   help="core = off-GPU deps + tests; gpu = + torch/lerobot/libero; all = everything.")
    p.add_argument("--check", action="store_true", help="Dry run: detect + plan, install nothing.")
    p.add_argument("--budget-only", action="store_true", help="Only project the GPU-day budget.")
    p.add_argument("--data-dir", default=None,
                   help="Cache/storage dir (default: /root/autodl-tmp/vcr on AutoDL, else ./.vcr-cache).")
    p.add_argument("--no-accel", action="store_true", help="Do NOT apply AutoDL academic acceleration.")
    p.add_argument("--no-hf-mirror", action="store_true", help="Do NOT use the hf-mirror.com endpoint.")
    p.add_argument("--install-torch", action="store_true",
                   help="Force-install cu128 torch even if a suitable one is present.")
    p.add_argument("--skip-lerobot", action="store_true")
    p.add_argument("--skip-libero-plus", action="store_true")
    p.add_argument("--libero-plus-url", default=LIBERO_PLUS_URL)
    p.add_argument("--libero-plus-path", default=None, help="Use an existing local LIBERO-Plus checkout.")
    p.add_argument("--sec-per-episode", type=float, default=30.0, help="Budget step: per-episode seconds.")
    p.add_argument("--train-hours-per-run", type=float, default=4.0, help="Budget step: hours/train run.")
    return p


def default_data_dir(info: dict) -> Path:
    return Path("/root/autodl-tmp/vcr") if info["autodl_tmp"] else (REPO / ".vcr-cache")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    info = detect()
    print_detection(info)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(info)

    if args.budget_only:
        run_budget(args.sec_per_episode, args.train_hours_per_run)
        return 0

    if args.check:
        log("DRY RUN (--check): the following WOULD run; nothing is installed.")
        setup_storage(data_dir, apply=False)
        setup_network(not args.no_accel, not args.no_hf_mirror, apply=False)
        plan = ["core deps + pytest/ruff", "run tests"]
        if args.stage != "core":
            plan = (["storage + network", *plan[:1]]
                    + ["ensure cu128 torch (reuse if suitable)", "gpu-adjacent deps"]
                    + ([] if args.skip_lerobot else ["LeRobot from source"])
                    + ([] if args.skip_libero_plus else ["LIBERO-Plus from source"])
                    + ["verify_env (sm_120)", "run tests", "budget gate"])
        for i, s in enumerate(plan, 1):
            print(f"  plan[{i}] {s}")
        ok("dry run complete.")
        return 0

    # ---- real deploy ----
    setup_storage(data_dir, apply=True)
    setup_network(not args.no_accel, not args.no_hf_mirror, apply=True)

    total = 3 if args.stage == "core" else 8
    step(1, total, "Core dependencies (off-GPU)")
    pip_install(CORE_DEPS)

    tests_ok = False
    gpu_ready = False

    if args.stage == "core":
        step(2, total, "Run off-GPU tests")
        tests_ok = run_tests()
        step(3, total, "Budget gate")
        run_budget(args.sec_per_episode, args.train_hours_per_run)
        print_summary(info, gpu_ready, tests_ok)
        return 0 if tests_ok else 1

    step(2, total, "Ensure cu128 PyTorch (>=2.7, sm_120)")
    ensure_torch(info, args.install_torch)
    step(3, total, "GPU-adjacent dependencies")
    pip_install(GPU_ADJACENT)
    step(4, total, "LeRobot (SmolVLA / pi0.5) from source")
    if args.skip_lerobot:
        warn("skipped (--skip-lerobot)")
    else:
        install_lerobot(LEROBOT_SPEC)
    step(5, total, "LIBERO-Plus (drop-in for libero) from source")
    if args.skip_libero_plus:
        warn("skipped (--skip-libero-plus)")
    else:
        install_libero_plus(args.libero_plus_url,
                            Path(args.libero_plus_path) if args.libero_plus_path else None)
    step(6, total, "Verify GPU env (sm_120 / cu128 / torch>=2.7 / LeRobot)")
    gpu_ready = verify_env()
    step(7, total, "Run tests")
    tests_ok = run_tests()
    step(8, total, "Budget gate")
    run_budget(args.sec_per_episode, args.train_hours_per_run)

    print_summary(info, gpu_ready, tests_ok)
    return 0 if (tests_ok and gpu_ready) else 1


if __name__ == "__main__":
    sys.exit(main())
