"""
explainability.py
==================
Parts D-E of the experimental audit: CFP importance quantification and RL
decision explainability.

No new RL models are implemented here -- Part D re-analyses the existing
state-ablation results (src/state_ablation.py), and Part E rolls out the
already-trained State-C ablation agent (ablation_stateC.pth). Its actual
3D decision input is [RUL_norm, sigma_norm, CFP], so visualizing those
components next to the chosen action is a literal, faithful explanation
of the policy -- not an approximation built from a different agent's state.

Usage
-----
    python -m src.explainability                  # Parts D+E
    python -m src.explainability --part d
    python -m src.explainability --part e
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from src.qrdqn_agent import QRDQNAgent
    from src.state_ablation import (
        _ALL_BEARINGS, _CFP_TAU, _precompute_mc_cache, make_ablation_env,
    )
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.qrdqn_agent import QRDQNAgent
    from src.state_ablation import (
        _ALL_BEARINGS, _CFP_TAU, _precompute_mc_cache, make_ablation_env,
    )

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

_ACTION_NAMES  = {0: "Do-Nothing", 1: "Repair", 2: "Replace"}
_ACTION_COLORS = {0: "#CCCCCC", 1: "#5fa8d3", 2: "#0072B2"}


# ===========================================================================
# Part D -- CFP importance study
# ===========================================================================

def run_part_d(results_dir: Path) -> Dict[str, Any]:
    """Quantify the marginal contribution of CFP using existing ablation results.

    State B = [RUL_norm, sigma_norm], State C = [RUL_norm, sigma_norm, CFP].
    The B -> C delta isolates CFP's marginal effect, holding RUL+variance fixed.
    """
    csv_path = results_dir.parent / "05_state_ablation" / "table_state_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run `python -m src.state_ablation` first."
        )

    df = pd.read_csv(csv_path).set_index("State_Config")
    b, c = df.loc["B"], df.loc["C"]

    marginal = {
        "cost_delta":        float(c["Cost_mu"] - b["Cost_mu"]),
        "cost_delta_pct":    float((c["Cost_mu"] - b["Cost_mu"]) / b["Cost_mu"] * 100),
        "catastrophe_delta": float(c["Catastrophe_pct"] - b["Catastrophe_pct"]),
        "reward_delta":      float(c["Avg_Reward"] - b["Avg_Reward"]),
    }
    return {"table": df, "marginal_cfp": marginal}


def generate_part_d_outputs(part_d: Dict[str, Any], results_dir: Path) -> None:
    df       = part_d["table"]
    marginal = part_d["marginal_cfp"]

    csv_path = results_dir / "table_cfp_marginal.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in marginal.items():
            w.writerow([k, f"{v:.4f}"])
    logger.info("Saved %s", csv_path)

    configs = list(df.index)
    costs   = df["Cost_mu"].values
    catast  = df["Catastrophe_pct"].values
    blues   = ["#a8c8e8", "#5fa8d3", "#2980b9", "#1a5276"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    axes[0].bar(configs, costs, color=blues[: len(configs)], edgecolor="white")
    axes[0].set_ylabel("Mean Cost")
    axes[0].set_title("(a) Cost by State Configuration")
    b_idx, c_idx = configs.index("B"), configs.index("C")
    axes[0].annotate(
        f"CFP: {marginal['cost_delta']:+.1f} ({marginal['cost_delta_pct']:+.1f}%)",
        xy=(c_idx, costs[c_idx]),
        xytext=(c_idx, costs[b_idx] + max(costs) * 0.08),
        ha="center", fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )

    axes[1].bar(configs, catast, color=blues[: len(configs)], edgecolor="white")
    axes[1].set_ylabel("Catastrophe Rate (%)")
    axes[1].set_title("(b) Catastrophe Rate by State Configuration")

    fig.tight_layout()
    out = results_dir / "fig_cfp_marginal.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)

    print("\n=== PART D: CFP MARGINAL CONTRIBUTION (State B -> State C) ===")
    print(f"  Cost:             {marginal['cost_delta']:+.2f} ({marginal['cost_delta_pct']:+.1f}%)")
    print(f"  Catastrophe rate: {marginal['catastrophe_delta']:+.2f} pp")
    print(f"  Avg reward:       {marginal['reward_delta']:+.2f}")
    print("=== END PART D ===\n")


# ===========================================================================
# Part E -- RL decision explainability
# ===========================================================================

def run_part_e(
    processed_dir: Path,
    results_dir:   Path,
    device_str:    str,
    n_candidate_episodes: int = 30,
) -> Dict[str, List[float]]:
    """Roll out the State-C ablation agent and select a representative episode.

    State C obs = [RUL_norm, sigma_norm (variance), CFP] -- this *is* the
    agent's actual decision input, so plotting these 3 components alongside
    the chosen action is a literal explanation of the policy, not a proxy.
    """
    ckpt_path = results_dir.parent / "05_state_ablation" / "ablation_stateC.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found -- run `python -m src.state_ablation` first."
        )

    rul_ckpt = results_dir.parent / "01_rul_predictor" / "rul_model_best.pth"
    mc_cache, sigma2_max = _precompute_mc_cache(
        processed_dir, rul_ckpt, bearing_ids=_ALL_BEARINGS,
        n_mc=10, device_str=device_str,
    )
    env = make_ablation_env(processed_dir, "C", mc_cache, sigma2_max, seed=42)

    agent = QRDQNAgent(state_dim=3, n_actions=3, device=device_str)
    agent.load_checkpoint(ckpt_path)

    best_episode: Dict[str, List[float]] = {}
    best_score = -1.0

    for i in range(n_candidate_episodes):
        obs, _ = env.reset(seed=1_000_000 + i, force_degraded=False)
        rec: Dict[str, List[float]] = {
            "t": [0], "rul": [float(obs[0])], "var": [float(obs[1])],
            "cfp": [float(obs[2])], "action": [],
        }
        done = False
        t = 0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t += 1
            rec["t"].append(t)
            rec["rul"].append(float(obs[0]))
            rec["var"].append(float(obs[1]))
            rec["cfp"].append(float(obs[2]))
            rec["action"].append(action)

        n_switches = sum(
            1 for j in range(1, len(rec["action"])) if rec["action"][j] != rec["action"][j - 1]
        )
        n_nonzero = sum(1 for a in rec["action"] if a != 0)
        score = n_switches * 2 + n_nonzero
        if score > best_score:
            best_score   = score
            best_episode = rec

    return best_episode


def generate_part_e_outputs(episode: Dict[str, List[float]], results_dir: Path) -> None:
    t        = episode["t"]
    rul      = episode["rul"]
    var      = episode["var"]
    cfp      = episode["cfp"]
    action   = episode["action"]
    action_t = t[1:]   # action[j] is the action taken between t[j] and t[j+1]

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.0), sharex=True)

    axes[0].plot(t, rul, color="#1a5276", linewidth=1.2)
    axes[0].set_ylabel("Mean RUL\n(normalized)")
    axes[0].set_title("State-C Agent Decision Trace (representative episode)")

    axes[1].plot(t, var, color="#d7191c", linewidth=1.2)
    axes[1].set_ylabel("Variance\n(normalized)")

    axes[2].plot(t, cfp, color="#1a9641", linewidth=1.2)
    axes[2].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel(f"CFP\nP(RUL<={_CFP_TAU:.0f})")

    for j, a in enumerate(action):
        axes[3].scatter(action_t[j], a, color=_ACTION_COLORS[a], s=12)
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(["Do-Nothing", "Repair", "Replace"])
    axes[3].set_ylabel("Action")
    axes[3].set_xlabel("Timestep")

    switch_points = [j for j in range(1, len(action)) if action[j] != action[j - 1]]
    for j in switch_points:
        for ax in axes:
            ax.axvline(action_t[j], color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    out = results_dir / "fig_explainability.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)

    print("\n=== PART E: DECISION EXPLAINABILITY ===")
    print(f"  Representative episode: {len(action)} steps, {len(switch_points)} action changes")
    for j in switch_points[:15]:
        print(
            f"  t={action_t[j]:4d}: action -> {_ACTION_NAMES[action[j]]:<10} | "
            f"RUL={rul[j + 1]:.3f}  Var={var[j + 1]:.3f}  CFP={cfp[j + 1]:.3f}"
        )
    print(
        f"  Final action counts: do_nothing={action.count(0)}, "
        f"repair={action.count(1)}, replace={action.count(2)}"
    )
    print("=== END PART E ===\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Part D (CFP importance) and Part E (decision explainability)."
    )
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--results-dir",   default="results/09_explainability", type=Path)
    p.add_argument("--device",        default=None)
    p.add_argument("--part",          choices=["d", "e", "all"], default="all")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.part in ("d", "all"):
        part_d = run_part_d(results_dir)
        generate_part_d_outputs(part_d, results_dir)

    if args.part in ("e", "all"):
        episode = run_part_e(args.processed_dir, results_dir, device_str)
        generate_part_e_outputs(episode, results_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
