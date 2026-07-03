"""
experimental_audit.py
======================
Parts A-C of the experimental audit: current performance audit, multi-seed
robustness sweep, and statistical significance testing.

No new RL models are implemented here -- only existing agents (DDQNAgent,
DuelingDQNAgent, PPOAgent, QRDQNAgent, ThresholdPolicy) are trained/evaluated
under different seeds and risk-mode decodings.

Naming note: no plain vanilla DQN (single network, no double-target) exists
in this codebase -- only DDQN (double-target) is implemented. Per agreement,
"DQN" in Parts B/C refers to Risk-Neutral QR-DQN: the same QRDQNAgent network
as CVaR QR-DQN, decoded with risk_mode="mean" instead of "cvar". This follows
the existing convention already used in train.py / baselines.RiskNeutralDQN.

Usage
-----
    python -m src.experimental_audit                  # Parts A+B+C
    python -m src.experimental_audit --part a          # Part A only
    python -m src.experimental_audit --part b          # Part B only
    python -m src.experimental_audit --part c           # Part C only (needs Part B output)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

try:
    from src.baselines import ThresholdPolicy
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer, train_qrdqn
    from src.rl_benchmarks import (
        DDQNAgent, DuelingDQNAgent, PPOAgent,
        evaluate_all_agents, _train_dqn_agent,
    )
    from src.rl_environment import make_env_from_processed
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.baselines import ThresholdPolicy
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer, train_qrdqn
    from src.rl_benchmarks import (
        DDQNAgent, DuelingDQNAgent, PPOAgent,
        evaluate_all_agents, _train_dqn_agent,
    )
    from src.rl_environment import make_env_from_processed

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

SEEDS:               List[int] = [42, 123]
SEED_SWEEP_EPISODES: int       = 2000
EVAL_EPISODES:        int       = 300
CVAR_ALPHA:            float     = 0.40

_METRIC_LABELS = {
    "mean_cost":           "Total Cost",
    "catastrophe_rate":    "Catastrophe Rate",
    "mean_n_repairs":      "Repairs",
    "mean_n_replacements": "Replacements",
    "mean_reward":         "Avg Reward",
}
_METRIC_KEYS = list(_METRIC_LABELS.keys())

_PART_B_MODELS = ["Risk-Neutral QR-DQN", "Dueling DQN", "CVaR QR-DQN (ours)"]

_COLORS = {
    "ThresholdPolicy":      "#999999",
    "DDQN":                 "#E69F00",
    "Dueling DQN":          "#56B4E9",
    "PPO":                  "#009E73",
    "Risk-Neutral QR-DQN": "#CC79A7",
    "CVaR QR-DQN (ours)":  "#0072B2",
}


# ===========================================================================
# Part A -- current performance audit
# ===========================================================================

def run_part_a(env, results_dir: Path, device_str: str) -> Dict[str, Dict[str, Any]]:
    """Evaluate every existing trained RL agent over EVAL_EPISODES greedy rollouts."""
    agents: Dict[str, Any] = {"ThresholdPolicy": ThresholdPolicy()}

    benchmarks_dir = results_dir.parent / "04_rl_benchmarks_ddqn_dueling_ppo"
    primary_dir    = results_dir.parent / "00_primary_cvar_qrdqn"

    ddqn_ckpt = benchmarks_dir / "ddqn_best.pth"
    if ddqn_ckpt.exists():
        a = DDQNAgent(device=device_str)
        a.load_checkpoint(ddqn_ckpt)
        agents["DDQN"] = a

    dueling_ckpt = benchmarks_dir / "dueling_dqn_best.pth"
    if dueling_ckpt.exists():
        a = DuelingDQNAgent(device=device_str)
        a.load_checkpoint(dueling_ckpt)
        agents["Dueling DQN"] = a

    ppo_ckpt = benchmarks_dir / "ppo_best.pth"
    if ppo_ckpt.exists():
        a = PPOAgent(device=device_str)
        a.load_checkpoint(ppo_ckpt)
        agents["PPO"] = a

    qrdqn_ckpt = primary_dir / "qrdqn_best.pth"
    if qrdqn_ckpt.exists():
        rn = QRDQNAgent(risk_mode="cvar", cvar_alpha=CVAR_ALPHA, device=device_str)
        rn.load_checkpoint(qrdqn_ckpt)
        rn.risk_mode = "mean"          # decode the same weights risk-neutrally
        agents["Risk-Neutral QR-DQN"] = rn

        cvar = QRDQNAgent(risk_mode="cvar", cvar_alpha=CVAR_ALPHA, device=device_str)
        cvar.load_checkpoint(qrdqn_ckpt)
        agents["CVaR QR-DQN (ours)"] = cvar
    else:
        logger.warning("qrdqn_best.pth not found; Risk-Neutral/CVaR rows skipped.")

    logger.info("Part A: evaluating %d agents over %d episodes...", len(agents), EVAL_EPISODES)
    return evaluate_all_agents(agents, env, n_episodes=EVAL_EPISODES, seed=42)


def generate_part_a_outputs(results: Dict[str, Dict[str, Any]], results_dir: Path) -> None:
    """Write table_audit_partA.csv/.tex and fig_audit_partA.png."""
    order = [n for n in ["ThresholdPolicy", "DDQN", "Dueling DQN", "PPO",
                          "Risk-Neutral QR-DQN", "CVaR QR-DQN (ours)"] if n in results]

    rows = []
    for name in order:
        m = results[name]
        rows.append({
            "policy":          name,
            "total_cost":      m["mean_cost"],
            "catastrophe_pct": m["catastrophe_rate"] * 100,
            "repairs_per_ep":  m["mean_n_repairs"],
            "replaces_per_ep": m["mean_n_replacements"],
            "avg_reward":      m["mean_reward"],
        })

    csv_path = results_dir / "table_audit_partA.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %s", csv_path)

    best_cost   = min(r["total_cost"]      for r in rows)
    best_catast = min(r["catastrophe_pct"] for r in rows)
    best_reward = max(r["avg_reward"]      for r in rows)

    def _b(v: float, best: float, fmt: str, tol: float = 0.01) -> str:
        s = format(v, fmt)
        return f"\\textbf{{{s}}}" if abs(v - best) <= tol else s

    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Part A -- current performance audit across all existing RL agents.}",
        r"\label{tab:audit_a}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Policy & Total Cost & Catast.\% & Repairs & Replacements & Avg Reward \\",
        r"\midrule",
    ]
    for r in rows:
        is_ours = "(ours)" in r["policy"]
        name = "\\textbf{" + r["policy"] + "}" if is_ours else r["policy"]
        tex.append(
            f"{name} & {_b(r['total_cost'], best_cost, '.2f')} & "
            f"{_b(r['catastrophe_pct'], best_catast, '.2f', 0.005)} & "
            f"{r['repairs_per_ep']:.2f} & {r['replaces_per_ep']:.2f} & "
            f"{_b(r['avg_reward'], best_reward, '.2f')} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = results_dir / "table_audit_partA.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    logger.info("Saved %s", tex_path)

    # Figure: 1x3 panels (cost, catastrophe, action counts)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    names  = [r["policy"] for r in rows]
    colors = [_COLORS.get(n, "#AAAAAA") for n in names]

    axes[0].bar(names, [r["total_cost"] for r in rows], color=colors, edgecolor="white")
    axes[0].set_ylabel("Total Cost")
    axes[0].set_title("(a) Total Cost")
    axes[0].tick_params(axis="x", rotation=30)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")

    axes[1].bar(names, [r["catastrophe_pct"] for r in rows], color=colors, edgecolor="white")
    axes[1].set_ylabel("Catastrophe Rate (%)")
    axes[1].set_title("(b) Catastrophe Rate")
    axes[1].tick_params(axis="x", rotation=30)
    for lbl in axes[1].get_xticklabels():
        lbl.set_ha("right")

    w = 0.35
    xs = np.arange(len(names))
    axes[2].bar(xs - w / 2, [r["repairs_per_ep"] for r in rows], w,
                label="Repairs/ep", color="#5fa8d3", edgecolor="white")
    axes[2].bar(xs + w / 2, [r["replaces_per_ep"] for r in rows], w,
                label="Replacements/ep", color="#1a9641", edgecolor="white")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(names, rotation=30, ha="right")
    axes[2].set_ylabel("Count per Episode")
    axes[2].set_title("(c) Maintenance Actions")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    out = results_dir / "fig_audit_partA.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===========================================================================
# Part B -- multiple random seeds
# ===========================================================================

def run_part_b(
    env,
    results_dir:   Path,
    device_str:    str,
    seeds:         List[int] = SEEDS,
    n_episodes:    int       = SEED_SWEEP_EPISODES,
    force_retrain: bool      = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Train (or load) QRDQNAgent + DuelingDQNAgent per seed; evaluate 3 model views."""
    raw: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in _PART_B_MODELS}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("Part B: seed=%d", seed)
        logger.info("=" * 60)

        # --- QR-DQN: trained once under CVaR, decoded two ways -------------
        qrdqn_ckpt = results_dir / f"audit_seed{seed}_qrdqn.pth"
        if qrdqn_ckpt.exists() and not force_retrain:
            logger.info("Loading existing checkpoint %s", qrdqn_ckpt)
        else:
            agent  = QRDQNAgent(risk_mode="cvar", cvar_alpha=CVAR_ALPHA, device=device_str)
            buffer = ReplayBuffer(maxlen=50_000)
            train_qrdqn(
                agent, env, buffer,
                n_episodes=n_episodes, warmup_steps=500,
                eval_every=200, n_eval_episodes=20,
                save_every=n_episodes + 1,     # disable periodic dump, save-on-best only
                save_path=qrdqn_ckpt, seed=seed,
            )
            if not qrdqn_ckpt.exists():
                agent.save_checkpoint(qrdqn_ckpt)

        rn_agent = QRDQNAgent(risk_mode="cvar", cvar_alpha=CVAR_ALPHA, device=device_str)
        rn_agent.load_checkpoint(qrdqn_ckpt)
        rn_agent.risk_mode = "mean"

        cvar_agent = QRDQNAgent(risk_mode="cvar", cvar_alpha=CVAR_ALPHA, device=device_str)
        cvar_agent.load_checkpoint(qrdqn_ckpt)

        # --- Dueling DQN: independently trained -----------------------------
        dueling_ckpt = results_dir / f"audit_seed{seed}_dueling.pth"
        dueling = DuelingDQNAgent(device=device_str)
        if dueling_ckpt.exists() and not force_retrain:
            logger.info("Loading existing checkpoint %s", dueling_ckpt)
            dueling.load_checkpoint(dueling_ckpt)
        else:
            _train_dqn_agent(
                dueling, env,
                n_episodes=n_episodes, warmup_episodes=100,
                save_path=dueling_ckpt, eval_every=200, n_eval_episodes=20,
                seed=seed, label=f"DuelingDQN-seed{seed}",
            )
            if dueling_ckpt.exists():
                dueling.load_checkpoint(dueling_ckpt)
            else:
                dueling.save_checkpoint(dueling_ckpt)

        seed_agents = {
            "Risk-Neutral QR-DQN": rn_agent,
            "Dueling DQN":          dueling,
            "CVaR QR-DQN (ours)":  cvar_agent,
        }
        seed_eval = evaluate_all_agents(seed_agents, env, n_episodes=EVAL_EPISODES, seed=seed)
        for name in _PART_B_MODELS:
            raw[name][seed] = seed_eval[name]

    return raw


def generate_part_b_outputs(
    raw: Dict[str, Dict[int, Dict[str, Any]]],
    results_dir: Path,
) -> None:
    """Mean +/- std table across seeds, and boxplots per metric."""
    rows = []
    for model in _PART_B_MODELS:
        seed_results = raw[model]
        row: Dict[str, Any] = {"model": model, "n_seeds": len(seed_results)}
        for mk in _METRIC_KEYS:
            vals  = np.array([seed_results[s][mk] for s in seed_results])
            scale = 100.0 if mk == "catastrophe_rate" else 1.0
            row[f"{mk}_mean"] = float(np.mean(vals)) * scale
            row[f"{mk}_std"]  = float(np.std(vals))  * scale
        rows.append(row)

    csv_path = results_dir / "table_seed_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %s", csv_path)

    fig, axes = plt.subplots(1, len(_METRIC_KEYS), figsize=(3.2 * len(_METRIC_KEYS), 3.4))
    for ax, mk in zip(axes, _METRIC_KEYS):
        data = []
        for model in _PART_B_MODELS:
            vals = [raw[model][s][mk] for s in raw[model]]
            if mk == "catastrophe_rate":
                vals = [v * 100 for v in vals]
            data.append(vals)
        labels = [m.replace(" (ours)", "*").replace("Risk-Neutral QR-DQN", "RN-QRDQN")
                   .replace("Dueling DQN", "Dueling") for m in _PART_B_MODELS]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, model in zip(bp["boxes"], _PART_B_MODELS):
            patch.set_facecolor(_COLORS.get(model, "#AAAAAA"))
            patch.set_alpha(0.7)
        ax.set_title(_METRIC_LABELS[mk], fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    n_seeds = len(next(iter(raw.values())))
    fig.suptitle(
        f"Part B: {n_seeds}-seed sweep ({SEED_SWEEP_EPISODES} eps/run, seeds={SEEDS})",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = results_dir / "fig_seed_boxplots.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _save_part_b_raw(raw: Dict[str, Dict[int, Dict[str, Any]]], results_dir: Path) -> None:
    serializable = {m: {str(s): v for s, v in seeds.items()} for m, seeds in raw.items()}
    path = results_dir / "_part_b_raw.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    logger.info("Saved %s", path)


def _load_part_b_raw(results_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    path = results_dir / "_part_b_raw.json"
    with open(path, encoding="utf-8") as fh:
        serializable = json.load(fh)
    return {m: {int(s): v for s, v in seeds.items()} for m, seeds in serializable.items()}


# ===========================================================================
# Part C -- statistical significance
# ===========================================================================

def run_part_c(
    raw: Dict[str, Dict[int, Dict[str, Any]]],
    results_dir: Path,
) -> None:
    """Independent t-test + Mann-Whitney U: (Risk-Neutral, Dueling) vs CVaR QR-DQN."""
    comparisons = [
        ("Risk-Neutral QR-DQN", "CVaR QR-DQN (ours)"),
        ("Dueling DQN",          "CVaR QR-DQN (ours)"),
    ]

    n_seeds = len(next(iter(raw.values())))
    print("\n" + "=" * 60)
    print("=== PART C: STATISTICAL SIGNIFICANCE ===")
    print(f"CAVEAT: n={n_seeds} seeds per group. With n={n_seeds}, Mann-Whitney U's")
    print("smallest achievable two-sided p-value is ~0.33 -- no comparison can")
    print("reach p<0.05 regardless of effect size at this sample size. Treat")
    print("these as illustrative effect-size indicators only, NOT significance.")
    print("=" * 60)

    rows = []
    for model_a, model_b in comparisons:
        for mk in _METRIC_KEYS:
            a_vals = np.array([raw[model_a][s][mk] for s in raw[model_a]])
            b_vals = np.array([raw[model_b][s][mk] for s in raw[model_b]])

            t_stat, t_p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
            try:
                u_stat, u_p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            except ValueError:
                u_stat, u_p = float("nan"), float("nan")

            rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "metric":     _METRIC_LABELS[mk],
                "mean_a":     float(np.mean(a_vals)),
                "mean_b":     float(np.mean(b_vals)),
                "t_stat":     float(t_stat),
                "t_pvalue":   float(t_p),
                "u_stat":     float(u_stat),
                "u_pvalue":   float(u_p),
            })
            print(
                f"  {model_a} vs {model_b} | {_METRIC_LABELS[mk]:<18} | "
                f"t-test p={t_p:.3f} | Mann-Whitney p={u_p:.3f}"
            )

    csv_path = results_dir / "table_significance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved %s", csv_path)
    print("=" * 60 + "\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Experimental audit: Part A (current performance), "
                     "Part B (seed sweep), Part C (significance tests)."
    )
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--results-dir",   default="results/08_experimental_audit_seed_sweep", type=Path)
    p.add_argument("--device",        default=None)
    p.add_argument("--part",          choices=["a", "b", "c", "all"], default="all")
    p.add_argument("--force-retrain", action="store_true")
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
    env = make_env_from_processed(args.processed_dir, seed=42)

    if args.part in ("a", "all"):
        part_a_results = run_part_a(env, results_dir, device_str)
        generate_part_a_outputs(part_a_results, results_dir)

    part_b_raw = None
    if args.part in ("b", "all"):
        part_b_raw = run_part_b(
            env, results_dir, device_str, force_retrain=args.force_retrain
        )
        generate_part_b_outputs(part_b_raw, results_dir)
        _save_part_b_raw(part_b_raw, results_dir)

    if args.part in ("c", "all"):
        if part_b_raw is None:
            part_b_raw = _load_part_b_raw(results_dir)
        run_part_c(part_b_raw, results_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
