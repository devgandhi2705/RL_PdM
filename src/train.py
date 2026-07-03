"""
train.py
========
QR-DQN training loop for risk-averse bearing predictive maintenance.

Usage
-----
    python -m src.train                            # from project root
    python -m src.train --config path/to/cfg.yaml
    python -m src.train --no-train                 # skip training, run final eval only

Prerequisites
-------------
    data/processed/  with {bid}_hi.npy and {bid}_rul.npy files
    (produced by src/precompute_hi.py or equivalent)

Outputs (written to results/, or results_dir from config)
----------------------------------------------------------
    qrdqn_best.pth           best agent checkpoint (lowest catastrophe_rate on eval)
    training_log.csv         per-episode metrics
    training_curve.png       smoothed reward + eval catastrophe rate
    final_comparison.csv     4-policy comparison
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    from src.baselines import (
        CorrectiveMaintenance,
        PeriodicPreventiveMaintenance,
        RiskNeutralDQN,
        evaluate_policy,
    )
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import PdMBearingEnv, make_env_from_processed, ACTION_DO_NOTHING
except ImportError:
    _proj_root = Path(__file__).resolve().parent.parent
    if str(_proj_root) not in sys.path:
        sys.path.insert(0, str(_proj_root))
    from src.baselines import (
        CorrectiveMaintenance,
        PeriodicPreventiveMaintenance,
        RiskNeutralDQN,
        evaluate_policy,
    )
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import PdMBearingEnv, make_env_from_processed, ACTION_DO_NOTHING

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

_RL_DEFAULTS: Dict[str, Any] = {
    "total_episodes":         5000,
    "warmup_episodes":        100,
    "eval_interval":          200,
    "max_steps":              400,
    "batch_size":             128,
    "learning_rate":          5e-4,
    "gamma":                  0.99,
    "epsilon_start":          1.0,
    "epsilon_end":            0.10,
    "epsilon_decay_episodes": 3000,
    "target_update_freq":     200,
    "replay_buffer_size":     50000,
    "N_quantiles":            51,
    "cvar_alpha":             0.25,
    "risk_mode":              "cvar",
    "n_mc_samples_train":     10,
    "n_mc_samples_eval":      50,
    "infer_interval":         1,
    "n_eval_episodes":        50,
    "n_final_episodes":       300,
    "seed":                   42,
    "processed_dir":          "data/processed",
    "results_dir":            "results/00_primary_cvar_qrdqn",
}


def load_train_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = dict(_RL_DEFAULTS)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg.update(raw.get("rl", {}))
    if "results_dir" not in raw.get("rl", {}):
        cfg["results_dir"] = raw.get("results_dir", cfg["results_dir"])
    if "processed_dir" not in raw.get("rl", {}):
        cfg["processed_dir"] = raw.get("dataset", {}).get("processed_dir", cfg["processed_dir"])
    return cfg


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(results_dir / "train_rl.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Pre-training sanity check
# ===========================================================================

def _run_sanity_check(env: PdMBearingEnv, seed: int = 42) -> None:
    """Run 20 random episodes and verify the env is healthy.

    Exits via sys.exit(1) if any check fails.
    Checks:
      - action distribution roughly 33% each (+/-15%)
      - at least 1 failure occurred
      - reward standard deviation > 5.0
      - mean HI at episode end < 0.4
    """
    print("\n--- PRE-TRAINING SANITY CHECK ---")
    action_counts = [0, 0, 0]
    rewards_all: List[float] = []
    n_failures = 0
    hi_ends: List[float] = []

    # 20 random episodes: check action distribution
    for ep in range(20):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_all.append(reward)
            done = terminated or truncated

    # 5 do_nothing episodes with uncapped max_steps to check degradation + failure
    _saved_max = env._max_steps
    env._max_steps = 3000           # any bearing fails within 3000 do_nothing steps
    for ep in range(5):
        obs, info = env.reset(seed=seed + 200 + ep)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(ACTION_DO_NOTHING)
            rewards_all.append(reward)
            done = terminated or truncated
        if info.get("is_failure", False):
            n_failures += 1
        hi_ends.append(float(info.get("hi_t", obs[0])))
    env._max_steps = _saved_max

    total_ac = sum(action_counts)
    fracs    = [c / total_ac for c in action_counts]
    rew_std  = float(np.std(rewards_all)) if len(rewards_all) > 1 else 0.0
    mean_hi  = float(np.mean(hi_ends))

    passed = True
    for label, frac in zip(["do_nothing", "repair", "replace"], fracs):
        ok = 0.18 <= frac <= 0.48
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag}: action {label} fraction={frac:.2%} (expected 33%+-15%)")
        if not ok:
            passed = False

    ok = n_failures >= 1
    print(f"  {'OK  ' if ok else 'FAIL'}: n_failures={n_failures} (expected >= 1)")
    if not ok:
        passed = False

    ok = rew_std > 5.0
    print(f"  {'OK  ' if ok else 'FAIL'}: reward_std={rew_std:.3f} (expected > 5.0)")
    if not ok:
        passed = False

    ok = mean_hi < 0.4
    print(f"  {'OK  ' if ok else 'FAIL'}: mean_hi_at_end={mean_hi:.3f} (expected < 0.4)")
    if not ok:
        passed = False

    if not passed:
        print("PRE-TRAINING SANITY: FAILED -- fix environment before training.\n")
        sys.exit(1)

    print("PRE-TRAINING SANITY: PASSED\n")


# ===========================================================================
# Warmup
# ===========================================================================

def _warmup(
    env: PdMBearingEnv,
    buffer: ReplayBuffer,
    n_episodes: int,
    seed: int,
) -> None:
    """Fill replay buffer with diverse transitions using cyclic [0,0,0,1,1,2] actions.

    No network updates performed during warmup.
    """
    print(f"--- WARMUP: {n_episodes} episodes with cyclic [0,0,0,1,1,2] actions ---")
    action_cycle  = cycle([0, 0, 0, 1, 1, 2])
    action_counts = [0, 0, 0]

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = next(action_cycle)
            action_counts[action] += 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs

    total = sum(action_counts)
    print(f"  Buffer size after warmup: {len(buffer):,}")
    print(f"  Action distribution:  "
          f"dn={action_counts[0]/total:.0%}  "
          f"rp={action_counts[1]/total:.0%}  "
          f"rx={action_counts[2]/total:.0%}")
    print(f"  Warmup complete.\n")


# ===========================================================================
# Diagnostic helper
# ===========================================================================

def _print_diagnostic(
    ep: int,
    window_rewards: List[float],
    window_failures: List[int],
    window_actions: List[List[int]],
    agent: QRDQNAgent,
    obs_sample: np.ndarray,
) -> float:
    """Print 50-episode diagnostic line; return dn_pct for collapse detection."""
    n_eps  = len(window_rewards)
    mean_r = float(np.mean(window_rewards))
    catast = sum(window_failures) / max(n_eps, 1)

    dn  = sum(a[0] for a in window_actions)
    rp  = sum(a[1] for a in window_actions)
    rx  = sum(a[2] for a in window_actions)
    tot = max(dn + rp + rx, 1)
    dn_pct = dn / tot
    rp_pct = rp / tot
    rx_pct = rx / tot

    qs     = agent.get_q_stats(obs_sample)
    q_dn, q_rp, q_rx = qs["cvar_q"]

    print(
        f"Ep {ep:5d}: reward={mean_r:6.1f} | catast={catast:.0%} | "
        f"act=[dn={dn_pct:.0%} rp={rp_pct:.0%} rx={rx_pct:.0%}] | "
        f"eps={agent.epsilon:.3f} | "
        f"Q=[{q_dn:.2f},{q_rp:.2f},{q_rx:.2f}]"
    )
    return dn_pct


# ===========================================================================
# Greedy evaluation (training-internal, also tracks mean_HI)
# ===========================================================================

def _greedy_eval(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    n_episodes: int,
    seed: int,
) -> Dict[str, Any]:
    """Run n_episodes fully greedy (epsilon=0); return metrics including mean_hi_end."""
    saved_eps  = agent.epsilon
    agent.epsilon = 0.0

    costs: List[float]   = []
    catasts: List[int]   = []
    hi_ends: List[float] = []
    hi_starts: List[float] = []
    action_counts = [0, 0, 0]

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i, force_degraded=False)
        hi_starts.append(float(obs[0]))
        done = False
        while not done:
            action = agent.select_action(obs)
            action_counts[action] += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(float(info["total_cost"]))
        catasts.append(1 if info.get("is_failure", False) else 0)
        hi_ends.append(float(info.get("hi_t", obs[0])))

    agent.epsilon = saved_eps

    total_ac = max(sum(action_counts), 1)
    return {
        "mean_total_cost":  float(np.mean(costs)),
        "catastrophe_rate": float(np.mean(catasts)),
        "mean_hi_end":      float(np.mean(hi_ends)),
        "mean_hi_start":    float(np.mean(hi_starts)),
        "action_distribution": {
            "do_nothing": action_counts[0] / total_ac,
            "repair":     action_counts[1] / total_ac,
            "replace":    action_counts[2] / total_ac,
        },
    }


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    buffer: ReplayBuffer,
    cfg: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, List[float]]:
    """Main training loop (post-warmup).

    Returns history dict with train_rewards, eval_catastrophe_rates,
    eval_episodes.
    """
    total_eps  = int(cfg["total_episodes"])
    eval_intv  = int(cfg["eval_interval"])
    n_eval_eps = int(cfg["n_eval_episodes"])
    seed       = int(cfg["seed"])

    best_path        = results_dir / "qrdqn_best.pth"
    best_catast_rate = float("inf")
    learning_confirmed = False

    # 50-episode sliding windows for diagnostics
    window_rewards:  List[float]     = []
    window_failures: List[int]       = []
    window_actions:  List[List[int]] = []
    last_obs = np.zeros(5, dtype=np.float32)

    # collapse detection: dn_pct from each 50-ep window
    collapse_dn_hist: List[float] = []

    history: Dict[str, List[float]] = {
        "train_rewards":          [],
        "eval_catastrophe_rates": [],
        "eval_episodes":          [],
    }

    # CSV log
    csv_path = results_dir / "training_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w  = csv.writer(csv_fh)
    csv_w.writerow([
        "episode", "phase", "ep_reward", "ep_length",
        "failure", "n_repairs", "n_replaces",
        "epsilon", "eval_catast", "eval_mean_cost",
    ])

    for ep in range(1, total_eps + 1):
        obs, info    = env.reset(seed=seed + ep, force_degraded=(random.random() < 0.4))
        ep_reward    = 0.0
        ep_step      = 0
        ep_act_cnts  = [0, 0, 0]
        done         = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, float(done))
            if buffer.is_ready(agent.batch_size):
                agent.update(buffer.sample(agent.batch_size))

            ep_act_cnts[action] += 1
            ep_reward += reward
            ep_step   += 1
            last_obs   = obs
            obs        = next_obs

        # episode-level epsilon decay
        agent.step_episode()

        failure = 1 if info.get("is_failure", False) else 0
        window_rewards.append(ep_reward)
        window_failures.append(failure)
        window_actions.append(ep_act_cnts)
        history["train_rewards"].append(ep_reward)

        csv_w.writerow([
            ep, "train", round(ep_reward, 3), ep_step,
            failure,
            info.get("n_repairs", 0),
            info.get("n_replacements", 0),
            round(agent.epsilon, 5),
            "", "",
        ])

        # ------------------------------------------------------------------
        # Diagnostic every 50 episodes
        # ------------------------------------------------------------------
        if ep % 50 == 0:
            dn_pct = _print_diagnostic(
                ep, window_rewards, window_failures, window_actions, agent, last_obs
            )
            collapse_dn_hist.append(dn_pct)

            if len(collapse_dn_hist) >= 3 and all(x > 0.90 for x in collapse_dn_hist[-3:]):
                print(f"  WARNING: do-nothing collapse detected (dn > 90% for 3 consecutive windows)")
                qs = agent.get_q_stats(last_obs)
                print(f"  Q-values (CVaR): dn={qs['cvar_q'][0]:.3f}  "
                      f"rp={qs['cvar_q'][1]:.3f}  rx={qs['cvar_q'][2]:.3f}")
                print(f"  Recommend: verify reward shaping and failure transitions in buffer")

            window_rewards  = []
            window_failures = []
            window_actions  = []

        # ------------------------------------------------------------------
        # Evaluation every 200 episodes (greedy, epsilon=0)
        # ------------------------------------------------------------------
        if ep % eval_intv == 0:
            eval_m    = _greedy_eval(agent, env, n_episodes=n_eval_eps,
                                     seed=seed + 1_000_000 + ep)
            catast    = eval_m["catastrophe_rate"]
            mean_cost = eval_m["mean_total_cost"]
            mean_hi   = eval_m["mean_hi_end"]
            act       = eval_m["action_distribution"]
            dn_e      = act.get("do_nothing", 0.0)
            rp_e      = act.get("repair",     0.0)
            rx_e      = act.get("replace",    0.0)
            all_dn    = dn_e > 0.995

            history["eval_catastrophe_rates"].append(catast)
            history["eval_episodes"].append(ep)

            mean_hi_start = eval_m.get("mean_hi_start", float("nan"))
            print(
                f"  [EVAL ep {ep:5d}] catast={catast:.0%}  cost={mean_cost:.2f}  "
                f"HI_start={mean_hi_start:.3f}  HI_end={mean_hi:.3f}  "
                f"act=[dn={dn_e:.0%} rp={rp_e:.0%} rx={rx_e:.0%}]"
            )

            csv_w.writerow([
                ep, "eval", "", "", "", "", "",
                round(agent.epsilon, 5),
                round(catast, 5),
                round(mean_cost, 3),
            ])
            csv_fh.flush()

            # Save checkpoint only if policy is not a do-nothing collapse
            not_collapsed = dn_e < 0.95
            if catast < 0.15 and not_collapsed:
                if catast < best_catast_rate:
                    best_catast_rate = catast
                    agent.save_checkpoint(best_path)
                    print(f"  -> New best: catast={catast:.1%}, dn={dn_e:.0%}; checkpoint saved.")
            elif ep >= 500 and not_collapsed and catast < best_catast_rate:
                # After ep 500: save best non-collapsed policy even if catast >= 0.15
                best_catast_rate = catast
                agent.save_checkpoint(best_path)
                print(f"  -> New best (non-collapsed): catast={catast:.1%}, dn={dn_e:.0%}; checkpoint saved.")

            if ep <= 1000 and catast < 0.10 and not all_dn and not learning_confirmed:
                learning_confirmed = True
                print(
                    f"\n  LEARNING CONFIRMED at ep {ep}: "
                    f"catastrophe_rate={catast:.1%}, policy is not all do-nothing.\n"
                )

    csv_fh.close()
    return history


# ===========================================================================
# Training curve
# ===========================================================================

def _smooth(arr: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(arr) < w:
        return np.arange(len(arr)), arr.copy()
    smoothed = np.convolve(arr, np.ones(w) / w, mode="valid")
    return np.arange(w - 1, len(arr)), smoothed


def plot_training_curve(history: Dict[str, List[float]], results_dir: Path) -> None:
    rewards  = np.array(history["train_rewards"])
    eval_eps = np.array(history.get("eval_episodes", []))
    eval_cr  = np.array(history.get("eval_catastrophe_rates", []))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(rewards, alpha=0.2, color="steelblue", linewidth=0.7)
    xr, yr = _smooth(rewards, 50)
    ax.plot(xr, yr, color="steelblue", linewidth=1.6, label="reward (smoothed w=50)")
    ax.set_ylabel("Episode reward")
    ax.set_title("QR-DQN Training -- reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if eval_eps.size > 0:
        ax.plot(eval_eps, eval_cr * 100, marker="D", ms=5, color="coral",
                label="eval catastrophe %")
    ax.axhline(10, ls="--", color="gray", alpha=0.5, label="10% target")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Catastrophe rate (%)")
    ax.set_title("QR-DQN Training -- evaluation catastrophe rate (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = results_dir / "training_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve saved to {path}")


# ===========================================================================
# Final 4-policy comparison
# ===========================================================================

def run_final_comparison(
    agent: QRDQNAgent,
    env: PdMBearingEnv,
    cfg: Dict[str, Any],
    results_dir: Path,
) -> None:
    """Evaluate CM, periodic-PM, risk-neutral DQN, and CVaR QR-DQN."""
    n_eps = int(cfg.get("n_final_episodes", 300))
    seed  = int(cfg.get("seed", 42))

    best_path = results_dir / "qrdqn_best.pth"
    if best_path.exists():
        agent.load_checkpoint(best_path)
        print("Loaded best checkpoint for final comparison.")
    else:
        print("WARNING: no best checkpoint found -- using current weights.")

    # risk-neutral variant: load same weights, override to mean aggregation
    rn_agent = QRDQNAgent(
        state_dim=agent.state_dim,
        n_actions=agent.n_actions,
        N_quantiles=agent.N_quantiles,
        risk_mode="mean",
        device=str(agent.device),
    )
    if best_path.exists():
        rn_agent.load_checkpoint(best_path)
        rn_agent.risk_mode = "mean"

    policies: Dict[str, Any] = {
        "CorrectiveMaintenance":   CorrectiveMaintenance(),
        "PeriodicPM(interval=50)": PeriodicPreventiveMaintenance(fixed_interval=50),
        "RiskNeutral-DQN":         rn_agent,
        "RiskAverse-QR-DQN":       agent,
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, policy in policies.items():
        print(f"  Evaluating '{name}' ({n_eps} episodes) ...")
        results[name] = evaluate_policy(policy, env, n_episodes=n_eps, seed=seed)

    _print_comparison_table(results)
    _save_comparison_csv(results, results_dir / "final_comparison.csv")
    _print_interpretation(results)


def _print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    headers = ["Policy",          "Cost_mean", "Cost_std", "Catastrophe%",
               "TTR_mean",        "Repairs/ep", "Replaces/ep", "DoNothing%"]
    widths  = [30,                10,           10,         13,
               10,                12,           13,          11]
    sep     = "  "
    header  = sep.join(f"{h:<{w}}" for h, w in zip(headers, widths))
    div     = "-" * len(header)

    lines = ["", "=== FINAL POLICY COMPARISON ===", div, header, div]
    for name, m in results.items():
        act   = m.get("action_distribution", {})
        ttr   = m.get("mean_TTR", float("nan"))
        ttr_s = f"{ttr:.1f}" if not math.isnan(ttr) else "nan"
        vals  = [
            name,
            f"{m.get('mean_total_cost', 0):.3f}",
            f"{m.get('std_total_cost', 0):.3f}",
            f"{m.get('catastrophe_rate', 0)*100:.1f}%",
            ttr_s,
            f"{m.get('mean_n_repairs', 0):.2f}",
            f"{m.get('mean_n_replacements', 0):.2f}",
            f"{act.get('do_nothing', 0)*100:.1f}%",
        ]
        lines.append(sep.join(f"{v:<{w}}" for v, w in zip(vals, widths)))
    lines.append(div)
    print("\n".join(lines))


def _print_interpretation(results: Dict[str, Dict[str, Any]]) -> None:
    names   = list(results.keys())
    costs   = {n: results[n].get("mean_total_cost", float("inf")) for n in names}
    catasts = {n: results[n].get("catastrophe_rate", 1.0) for n in names}

    best_cost = min(names, key=lambda n: costs[n])
    safest    = min(names, key=lambda n: catasts[n])

    median_c      = float(np.median(list(catasts.values())))
    candidates    = [n for n in names if catasts[n] <= median_c]
    best_tradeoff = min(candidates, key=lambda n: costs[n]) if candidates else best_cost

    print("\n--- INTERPRETATION ---")
    print(f"Best cost policy:  {best_cost}  (cost={costs[best_cost]:.3f})")
    print(f"Safest policy:     {safest}  (catastrophe={catasts[safest]:.1%})")
    print(f"Best tradeoff:     {best_tradeoff}  "
          f"(cost={costs[best_tradeoff]:.3f}, catastrophe={catasts[best_tradeoff]:.1%})")


def _save_comparison_csv(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    fields = ["policy", "mean_total_cost", "std_total_cost", "catastrophe_rate",
              "mean_TTR", "mean_n_repairs", "mean_n_replacements",
              "dn_pct", "rp_pct", "rx_pct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in results.items():
            act = m.get("action_distribution", {})
            ttr = m.get("mean_TTR", float("nan"))
            w.writerow({
                "policy":              name,
                "mean_total_cost":     round(m.get("mean_total_cost", 0), 5),
                "std_total_cost":      round(m.get("std_total_cost", 0), 5),
                "catastrophe_rate":    round(m.get("catastrophe_rate", 0), 5),
                "mean_TTR":            "" if math.isnan(ttr) else round(ttr, 3),
                "mean_n_repairs":      round(m.get("mean_n_repairs", 0), 4),
                "mean_n_replacements": round(m.get("mean_n_replacements", 0), 4),
                "dn_pct":              round(act.get("do_nothing", 0), 5),
                "rp_pct":              round(act.get("repair", 0), 5),
                "rx_pct":              round(act.get("replace", 0), 5),
            })
    print(f"Final comparison saved to {path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train QR-DQN for risk-averse bearing predictive maintenance."
    )
    p.add_argument("--config", default="config.yaml", type=Path,
                   help="Path to config.yaml")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training; load checkpoint and run final comparison only.")
    p.add_argument("--device", default=None,
                   help="Override compute device: cuda | mps | cpu")
    return p.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    args = parse_args()
    cfg  = load_train_cfg(args.config)

    results_dir = Path(str(cfg["results_dir"]))
    setup_logging(results_dir)

    seed = int(cfg["seed"])
    set_seed(seed)

    if args.device:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    logger.info("=" * 60)
    logger.info("QR-DQN PdM Trainer | risk_mode=%s | seed=%d | device=%s",
                cfg["risk_mode"], seed, device_str)
    logger.info("=" * 60)

    # ---- environment --------------------------------------------------------
    processed_dir = Path(str(cfg["processed_dir"]))
    env = make_env_from_processed(processed_dir)
    logger.info("Environment loaded: %d bearing(s) from %s",
                len(env._bearing_ids), processed_dir)

    state_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    # ---- agent + buffer -----------------------------------------------------
    agent = QRDQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        N_quantiles=int(cfg["N_quantiles"]),
        lr=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        epsilon_start=float(cfg["epsilon_start"]),
        epsilon_end=float(cfg["epsilon_end"]),
        epsilon_decay_episodes=int(cfg["epsilon_decay_episodes"]),
        target_update_freq=int(cfg["target_update_freq"]),
        batch_size=int(cfg["batch_size"]),
        risk_mode=str(cfg["risk_mode"]),
        cvar_alpha=float(cfg["cvar_alpha"]),
        device=device_str,
    )
    buffer = ReplayBuffer(maxlen=int(cfg["replay_buffer_size"]))

    logger.info(
        "Agent: %d quantiles | risk_mode=%s | cvar_alpha=%.2f | lr=%.1e | eps_decay=%d eps",
        agent.N_quantiles, agent.risk_mode, agent.cvar_alpha,
        float(cfg["learning_rate"]), int(cfg["epsilon_decay_episodes"]),
    )

    if not args.no_train:
        # sanity check
        _run_sanity_check(env, seed=seed)

        # warmup
        _warmup(env, buffer, n_episodes=int(cfg["warmup_episodes"]), seed=seed)

        # training
        t0      = time.time()
        history = train(agent, env, buffer, cfg, results_dir)
        elapsed = time.time() - t0
        logger.info(
            "Training complete: %d episodes in %.1f min (%.1f ep/min).",
            int(cfg["total_episodes"]),
            elapsed / 60,
            int(cfg["total_episodes"]) / max(elapsed / 60, 1e-9),
        )
        plot_training_curve(history, results_dir)
    else:
        logger.info("--no-train: skipping training.")

    # ---- final comparison ---------------------------------------------------
    print(f"\nRunning final 4-policy comparison "
          f"({cfg.get('n_final_episodes', 300)} episodes each) ...")
    run_final_comparison(agent, env, cfg, results_dir)


if __name__ == "__main__":
    main()
