"""
run_all.py
==========
Master pipeline script for the Risk-Averse Predictive Maintenance paper.

Runs the full pipeline in order:
  Step 1: python src/feature_extractor.py   (precompute features + HI)
  Step 2: python src/train_rul.py           (train RUL model)
  Step 3: python src/baselines.py           (verify baselines)
  Step 4: python src/train.py               (train RL agent)
  Step 5: python src/evaluate.py            (generate all figures + tables)

Usage
-----
    python run_all.py                     # full pipeline
    python run_all.py --dry-run           # fast smoke-test (2 eps, 2 epochs)
    python run_all.py --skip-rul          # skip Step 2 if checkpoint exists
    python run_all.py --from-step 3       # restart from Step 3
    python run_all.py --steps 1 2         # run only Steps 1 and 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_PROJ = Path(__file__).resolve().parent
_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], step: int, name: str) -> int:
    """Run a subprocess command; return its exit code."""
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJ))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n{'!'*60}")
        print(f"  PIPELINE FAILED AT STEP {step}: {name}")
        print(f"  Exit code: {result.returncode}")
        print(f"{'!'*60}")
    else:
        print(f"\n  Step {step} OK  ({elapsed:.1f}s)")
    return result.returncode


def _rul_checkpoint_ok(results_dir: Path) -> bool:
    """Check if RUL model checkpoint passes the discriminability threshold."""
    ckpt = results_dir / "rul_model_best.pth"
    if not ckpt.exists():
        return False
    try:
        import torch
        state = torch.load(ckpt, map_location="cpu")
        gap = state.get("discriminability_gap", 0.0)
        print(f"  RUL checkpoint found: discriminability_gap={gap:.1f} (threshold=75)")
        return float(gap) > 75.0
    except Exception as exc:
        print(f"  RUL checkpoint check failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full PdM paper pipeline end-to-end."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fast smoke-test: 2 episodes per policy, 2 training epochs, "
            "no LSTM/CNN baseline training.  Verifies the pipeline runs without errors."
        ),
    )
    parser.add_argument(
        "--skip-rul",
        action="store_true",
        help=(
            "Skip Step 2 (train_rul.py) if rul_model_best.pth already exists "
            "and passes the discriminability check (gap > 75)."
        ),
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        metavar="N",
        help="Start from Step N (1-5).  Default: 1.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        metavar="N",
        help="Run only these specific steps (e.g. --steps 1 3 5).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory.  Default: results/",
    )
    args = parser.parse_args(argv)

    results_dir = (_PROJ / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Decide which steps to run
    if args.steps:
        steps_to_run = set(args.steps)
    else:
        steps_to_run = set(range(args.from_step, 6))

    dry    = args.dry_run
    rul_ok = _rul_checkpoint_ok(results_dir)

    print(f"\n{'='*60}")
    print(f"  PdM PIPELINE  ({'DRY RUN' if dry else 'FULL'})")
    print(f"  Steps to run: {sorted(steps_to_run)}")
    if dry:
        print("  ⚠  Dry-run mode: minimal epochs/episodes for smoke-testing")
    print(f"{'='*60}")

    pipeline_start = time.time()
    step_results: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Feature extraction
    # ------------------------------------------------------------------
    if 1 in steps_to_run:
        cmd = [_PYTHON, "src/feature_extractor.py"]
        if dry:
            # feature_extractor doesn't have a --dry-run flag; just run it normally.
            # It will exit fast if data/raw is empty (no raw files).
            pass
        rc = _run(cmd, step=1, name="Feature extraction & HI precompute")
        step_results[1] = rc
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Step 2: RUL model training
    # ------------------------------------------------------------------
    if 2 in steps_to_run:
        if args.skip_rul and rul_ok:
            print(f"\n  STEP 2: SKIPPED (--skip-rul: checkpoint OK)")
            step_results[2] = 0
        else:
            cmd = [_PYTHON, "src/train_rul.py"]
            if dry:
                cmd += ["--epochs", "2", "--patience", "999"]
            rc = _run(cmd, step=2, name="RUL model training (Conv-SA)")
            step_results[2] = rc
            if rc != 0:
                return rc

    # ------------------------------------------------------------------
    # Step 3: Baseline verification
    # ------------------------------------------------------------------
    if 3 in steps_to_run:
        rc = _run(
            [_PYTHON, "src/baselines.py"],
            step=3,
            name="Baseline policies verification",
        )
        step_results[3] = rc
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Step 4: RL agent training
    # ------------------------------------------------------------------
    if 4 in steps_to_run:
        cmd = [_PYTHON, "src/train.py"]
        if dry:
            cmd += ["--no-train"]   # train.py supports --no-train to skip training
        rc = _run(cmd, step=4, name="QR-DQN RL agent training")
        step_results[4] = rc
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Step 5: Evaluation & figure generation
    # ------------------------------------------------------------------
    if 5 in steps_to_run:
        cmd = [_PYTHON, "src/evaluate.py", f"--results-dir={args.results_dir}"]
        if dry:
            cmd += ["--dry-run"]
        rc = _run(cmd, step=5, name="Figure & table generation")
        step_results[5] = rc
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start

    print(f"\n{'='*60}")
    print("  === PIPELINE COMPLETE ===")
    print(f"  Total time: {total_time/60:.1f} min")

    # RUL metrics from checkpoint
    ckpt = results_dir / "rul_model_best.pth"
    if ckpt.exists():
        try:
            import torch
            state = torch.load(ckpt, map_location="cpu")
            full_r = state.get("full_rmse", float("nan"))
            late_r = state.get("late_rmse", float("nan"))
            print(f"  RUL model:   full-RMSE={full_r:.2f}  late-RMSE={late_r:.2f}")
        except Exception:
            pass

    # Best policy from table2_policy.csv
    t2 = results_dir / "table2_policy.csv"
    if t2.exists():
        import csv
        import math
        try:
            with open(t2, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best = min(rows, key=lambda r: float(r.get("catastrophe_rate", "1")))
                print(f"  Best policy: {best['policy']} "
                      f"(cost={best.get('mean_total_cost', '?')}  "
                      f"catastrophe={float(best.get('catastrophe_rate', 1))*100:.1f}%)")
        except Exception:
            pass

    # List generated files
    figs   = sorted(results_dir.glob("fig*.png"))
    tables = sorted(results_dir.glob("table*.tex")) + sorted(results_dir.glob("table*.csv"))
    if figs:
        print(f"  Figures:  {' '.join(p.name for p in figs)}")
    if tables:
        print(f"  Tables:   {' '.join(p.name for p in tables)}")

    print("  Ready for paper writing.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
