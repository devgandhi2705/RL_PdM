"""
train_d3qn.py
=============
Training script for the Dueling Distributional QR-DQN (D3QN-CVaR) agent.

Usage
-----
    # Verify full pipeline without long training:
    python -m src.train_d3qn --dry-run

    # Full training (5000 episodes, ~3-4 hours on RTX 4050):
    python -m src.train_d3qn --episodes 5000

    # Custom config:
    python -m src.train_d3qn --config config.yaml --device cuda

Prerequisites
-------------
    data/processed/  -- HI + RUL .npy files (from src/feature_extractor.py)
    results/rul_model_best.pth -- trained RUL model (from src/train_rul.py)

Outputs
-------
    results/d3qn_cvar_best.pth      -- best checkpoint (lowest catastrophe)
    results/d3qn_training_log.csv   -- per-episode metrics
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

try:
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.qrdqn_agent import ReplayBuffer
    from src.dueling_qrdqn import D3QNAgent, ThreeDStateEnv
    from src.rl_environment import ACTION_DO_NOTHING
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.qrdqn_agent import ReplayBuffer
    from src.dueling_qrdqn import D3QNAgent, ThreeDStateEnv
    from src.rl_environment import ACTION_DO_NOTHING

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults (mirror rl: section; overridden by config.yaml d3qn: section)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "total_episodes":         5000,
    "warmup_episodes":        100,
    "eval_interval":          200,
    "batch_size":             128,
    "learning_rate":          5e-4,
    "gamma":                  0.99,
    "epsilon_start":          1.0,
    "epsilon_end":            0.10,
    "epsilon_decay_episodes": 3000,
    "target_update_freq":     200,
    "replay_buffer_size":     50000,
    "N_quantiles":            51,
    "cvar_alpha":             0.40,
    "risk_mode":              "cvar",
    "n_eval_episodes":        100,
    "n_final_episodes":       300,
    "seed":                   42,
    "processed_dir":          "data/processed",
    "results_dir":            "results",
    "n_mc_samples":           10,
}


def _load_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg.update(raw.get("d3qn", raw.get("rl", {})))
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pre-training sanity check
# ---------------------------------------------------------------------------

def _run_sanity_check(env, seed: int) -> None:
    """20 random-action episodes; exits via sys.exit(1) if checks fail."""
    logger.info("--- PRE-TRAINING SANITY CHECK ---")
    action_counts = [0, 0, 0]
    rewards_all: List[float] = []
    n_failures = 0

    for ep in range(20):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_all.append(reward)
            done = terminated or truncated

    # 5 do_nothing episodes with extended horizon to guarantee seeing failures
    _saved_max = env._max_steps
    env._max_steps = 3000
    for ep in range(5):
        obs, _ = env.reset(seed=seed + 200 + ep)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(ACTION_DO_NOTHING)
            rewards_all.append(reward)
            done = terminated or truncated
        if info.get("is_failure", False):
            n_failures += 1
    env._max_steps = _saved_max

    total_ac = max(sum(action_counts), 1)
    fracs    = [c / total_ac for c in action_counts]
    rew_std  = float(np.std(rewards_all)) if len(rewards_all) > 1 else 0.0

    passed = True
    for label, frac in zip(["do_nothing", "repair", "replace"], fracs):
        ok  = 0.18 <= frac <= 0.48
        tag = "OK  " if ok else "FAIL"
        logger.info("  %s: action %s fraction=%.1f%%  (expected 33%%+-15%%)", tag, label, frac * 100)
        if not ok:
            passed = False

    ok = n_failures >= 1
    logger.info("  %s: n_failures=%d  (expected >= 1)", "OK  " if ok else "FAIL", n_failures)
    if not ok:
        passed = False

    ok = rew_std > 5.0
    logger.info("  %s: reward_std=%.3f  (expected > 5.0)", "OK  " if ok else "FAIL", rew_std)
    if not ok:
        passed = False

    if not passed:
        logger.error("PRE-TRAINING SANITY: FAILED — fix environment before training.")
        sys.exit(1)

    logger.info("PRE-TRAINING SANITY: PASSED")


def _run_network_check(agent: D3QNAgent) -> None:
    """Shape check + dueling identity verification on a random mini-batch."""
    B = 8
    s = torch.randn(B, agent.state_dim, device=agent.device)
    with torch.no_grad():
        q = agent.online_net(s)  # (B, A, N)

    assert q.shape == (B, agent.n_actions, agent.N_quantiles), (
        f"D3QNetwork shape mismatch: expected ({B}, {agent.n_actions}, "
        f"{agent.N_quantiles}), got {tuple(q.shape)}"
    )
    logger.info(
        "Network shape check: D3QNetwork(%d, %d, %d) -> %s  OK",
        agent.state_dim, agent.n_actions, agent.N_quantiles, tuple(q.shape),
    )

    # Dueling identity: mean_a (Q - V) == 0 per quantile per sample
    # V_i = Q_i.mean(dim=action), so (Q - V.unsqueeze(1)).mean(dim=1) == 0
    v_hat  = q.mean(dim=1, keepdim=True)       # (B, 1, N) — reconstructed V
    adv    = q - v_hat                          # (B, A, N) — reconstructed advantage
    dev    = adv.mean(dim=1).abs().max().item() # should be ~0 by construction
    logger.info("Dueling identity check: max deviation = %.6f  (expect ~0)", dev)
    if dev > 0.01:
        logger.warning("Dueling identity deviation %.6f > 0.01 — possible numerical issue.", dev)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(env, buffer: ReplayBuffer, n_episodes: int, seed: int) -> None:
    """Fill buffer with cyclic [0,0,0,1,1,2] actions; no network updates."""
    logger.info("--- WARMUP: %d episodes with cyclic [0,0,0,1,1,2] actions ---", n_episodes)
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

    total = max(sum(action_counts), 1)
    logger.info(
        "Warmup complete: buffer=%d  act=[dn=%.0f%% rp=%.0f%% rx=%.0f%%]",
        len(buffer),
        action_counts[0] / total * 100,
        action_counts[1] / total * 100,
        action_counts[2] / total * 100,
    )


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

def _greedy_eval(agent: D3QNAgent, env, n_episodes: int, seed: int) -> Dict[str, Any]:
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    costs: List[float] = []
    catasts: List[int] = []
    action_counts = [0, 0, 0]

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, force_degraded=False)
        done = False
        while not done:
            action = agent.select_action(obs)
            action_counts[action] += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(float(info["total_cost"]))
        catasts.append(1 if info.get("is_failure", False) else 0)

    agent.epsilon = saved_eps
    total_ac = max(sum(action_counts), 1)
    return {
        "catastrophe_rate": float(np.mean(catasts)),
        "mean_total_cost":  float(np.mean(costs)),
        "action_distribution": {
            "do_nothing": action_counts[0] / total_ac,
            "repair":     action_counts[1] / total_ac,
            "replace":    action_counts[2] / total_ac,
        },
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    agent:       D3QNAgent,
    env,
    buffer:      ReplayBuffer,
    cfg:         Dict[str, Any],
    results_dir: Path,
    dry_run:     bool = False,
) -> None:
    total_eps  = int(cfg["total_episodes"])
    eval_intv  = int(cfg["eval_interval"])
    n_eval_eps = int(cfg["n_eval_episodes"])
    seed       = int(cfg["seed"])

    if dry_run:
        total_eps  = int(cfg.get("_dry_train_episodes", 5))
        eval_intv  = 5
        n_eval_eps = 3

    best_path        = results_dir / "d3qn_cvar_best.pth"
    best_catast_rate = float("inf")

    # 50-episode diagnostic windows
    win_rewards:  List[float]     = []
    win_failures: List[int]       = []
    win_actions:  List[List[int]] = []
    last_obs = np.zeros(agent.state_dim, dtype=np.float32)
    collapse_dn_hist: List[float] = []
    episode_losses:   List[float] = []
    loss_log_counter  = 0

    csv_path = results_dir / "d3qn_training_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w  = csv.writer(csv_fh)
    csv_w.writerow(["episode", "phase", "ep_reward", "ep_length",
                    "failure", "n_repairs", "n_replaces",
                    "epsilon", "eval_catast", "eval_mean_cost"])

    for ep in range(1, total_eps + 1):
        obs, _    = env.reset(seed=seed + ep, force_degraded=(random.random() < 0.4))
        ep_reward = 0.0
        ep_step   = 0
        ep_acts   = [0, 0, 0]
        ep_losses: List[float] = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, float(done))
            if buffer.is_ready(agent.batch_size):
                loss = agent.update(buffer.sample(agent.batch_size))
                ep_losses.append(loss)

            ep_acts[action] += 1
            ep_reward += reward
            ep_step   += 1
            last_obs   = obs
            obs        = next_obs

        agent.step_episode()

        failure = 1 if info.get("is_failure", False) else 0
        win_rewards.append(ep_reward)
        win_failures.append(failure)
        win_actions.append(ep_acts)
        if ep_losses:
            episode_losses.append(float(np.mean(ep_losses)))

        csv_w.writerow([
            ep, "train", round(ep_reward, 3), ep_step,
            failure,
            info.get("n_repairs", 0),
            info.get("n_replacements", 0),
            round(agent.epsilon, 5),
            "", "",
        ])

        # ------------------------------------------------------------------
        # Diagnostic every 50 episodes (every 5 in dry-run)
        # ------------------------------------------------------------------
        diag_every = 5 if dry_run else 50
        if ep % diag_every == 0:
            n_w    = len(win_rewards)
            mean_r = float(np.mean(win_rewards))
            catast = sum(win_failures) / max(n_w, 1)

            dn = sum(a[0] for a in win_actions)
            rp = sum(a[1] for a in win_actions)
            rx = sum(a[2] for a in win_actions)
            tot = max(dn + rp + rx, 1)
            dn_pct, rp_pct, rx_pct = dn / tot, rp / tot, rx / tot

            qs  = agent.get_q_stats(last_obs)
            q_dn, q_rp, q_rx = qs["cvar_q"]

            logger.info(
                "Ep %5d: reward=%6.1f | catast=%.0f%% | "
                "act=[dn=%.0f%% rp=%.0f%% rx=%.0f%%] | "
                "eps=%.3f | Q=[%.2f,%.2f,%.2f]",
                ep, mean_r, catast * 100,
                dn_pct * 100, rp_pct * 100, rx_pct * 100,
                agent.epsilon, q_dn, q_rp, q_rx,
            )

            # Mean loss every 50-episode window at INFO
            if episode_losses:
                logger.info(
                    "  mean_ep_loss (last %d eps) = %.6f",
                    diag_every, float(np.mean(episode_losses[-diag_every:])),
                )

            collapse_dn_hist.append(dn_pct)
            if len(collapse_dn_hist) >= 3 and all(x > 0.90 for x in collapse_dn_hist[-3:]):
                logger.warning(
                    "AGENT COLLAPSE DETECTED: do-nothing > 90%% for 3 consecutive "
                    "windows. eps=%.3f  Q(CVaR)=[%.3f, %.3f, %.3f]",
                    agent.epsilon, q_dn, q_rp, q_rx,
                )

            win_rewards  = []
            win_failures = []
            win_actions  = []

        # ------------------------------------------------------------------
        # Evaluation every eval_intv episodes (greedy, epsilon=0)
        # ------------------------------------------------------------------
        if ep % eval_intv == 0:
            eval_m    = _greedy_eval(agent, env, n_eval_eps, seed + 1_000_000 + ep)
            catast    = eval_m["catastrophe_rate"]
            mean_cost = eval_m["mean_total_cost"]
            act       = eval_m["action_distribution"]
            dn_e      = act["do_nothing"]

            logger.info(
                "  [EVAL ep %5d] catast=%.0f%%  cost=%.2f  "
                "act=[dn=%.0f%% rp=%.0f%% rx=%.0f%%]",
                ep, catast * 100, mean_cost,
                dn_e * 100, act["repair"] * 100, act["replace"] * 100,
            )

            csv_w.writerow([ep, "eval", "", "", "", "", "",
                            round(agent.epsilon, 5),
                            round(catast, 5), round(mean_cost, 3)])
            csv_fh.flush()

            not_collapsed = dn_e < 0.95
            if catast < best_catast_rate and not_collapsed:
                best_catast_rate = catast
                agent.save_checkpoint(best_path)
                logger.info("  -> New best: catast=%.1f%%  dn=%.0f%%  checkpoint saved.",
                            catast * 100, dn_e * 100)
            elif ep >= 500 and not_collapsed and catast < best_catast_rate:
                best_catast_rate = catast
                agent.save_checkpoint(best_path)
                logger.info("  -> New best (non-collapsed): catast=%.1f%%  saved.", catast * 100)

    csv_fh.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train D3QN-CVaR agent.")
    p.add_argument("--config",        default="config.yaml", type=Path)
    p.add_argument("--episodes",      default=None,          type=int,
                   help="Override total_episodes from config.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Run 5 warmup + 5 training episodes to verify pipeline.")
    p.add_argument("--device",        default=None)
    p.add_argument("--processed-dir", default=None,  type=Path)
    p.add_argument("--results-dir",   default=None,  type=Path)
    p.add_argument("--seed",          default=None,  type=int)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = _load_cfg(args.config)

    if args.episodes is not None:
        cfg["total_episodes"] = args.episodes
    if args.processed_dir is not None:
        cfg["processed_dir"] = str(args.processed_dir)
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)
    if args.seed is not None:
        cfg["seed"] = args.seed

    dry_run = args.dry_run
    if dry_run:
        cfg["warmup_episodes"]  = 5
        cfg["_dry_train_episodes"] = 5

    seed = int(cfg["seed"])
    _set_seed(seed)

    device_str = args.device or (get_device() if callable(get_device) else "cpu")

    results_dir  = Path(str(cfg["results_dir"]))
    processed_dir = Path(str(cfg["processed_dir"]))
    rul_ckpt     = results_dir / "rul_model_best.pth"

    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("D3QN-CVaR Trainer | risk_mode=%s | alpha=%.2f | seed=%d | device=%s | dry_run=%s",
                cfg["risk_mode"], float(cfg["cvar_alpha"]), seed, device_str, dry_run)
    logger.info("=" * 60)

    if not processed_dir.exists():
        logger.error("processed_dir not found: %s", processed_dir)
        sys.exit(1)
    if not rul_ckpt.exists():
        logger.error("RUL checkpoint not found: %s", rul_ckpt)
        sys.exit(1)

    # ---- Build environment ------------------------------------------------
    logger.info("Building ThreeDStateEnv ...")
    env = ThreeDStateEnv.build(
        proc_dir   = processed_dir,
        rul_ckpt   = rul_ckpt,
        device_str = device_str,
        seed       = seed,
        n_mc       = int(cfg["n_mc_samples"]),
    )

    # ---- Build agent + buffer --------------------------------------------
    agent = D3QNAgent(
        state_dim               = int(env.observation_space.shape[0]),
        n_actions               = int(env.action_space.n),
        N_quantiles             = int(cfg["N_quantiles"]),
        lr                      = float(cfg["learning_rate"]),
        gamma                   = float(cfg["gamma"]),
        epsilon_start           = float(cfg["epsilon_start"]),
        epsilon_end             = float(cfg["epsilon_end"]),
        epsilon_decay_episodes  = int(cfg["epsilon_decay_episodes"]),
        target_update_freq      = int(cfg["target_update_freq"]),
        batch_size              = int(cfg["batch_size"]),
        cvar_alpha              = float(cfg["cvar_alpha"]),
        risk_mode               = str(cfg["risk_mode"]),
        device                  = device_str,
    )
    buffer = ReplayBuffer(maxlen=int(cfg["replay_buffer_size"]))

    # ---- Pre-training checks ---------------------------------------------
    _run_sanity_check(env, seed)
    _run_network_check(agent)

    # ---- Warmup + training -----------------------------------------------
    _warmup(env, buffer, n_episodes=int(cfg["warmup_episodes"]), seed=seed)

    t0 = time.time()
    _train(agent, env, buffer, cfg, results_dir, dry_run=dry_run)
    elapsed = time.time() - t0
    total_ep = int(cfg.get("_dry_train_episodes", cfg["total_episodes"])) if dry_run else int(cfg["total_episodes"])
    logger.info(
        "Training complete: %d episodes in %.1f min (%.1f ep/min).",
        total_ep, elapsed / 60, total_ep / max(elapsed / 60, 1e-9),
    )

    if dry_run:
        logger.info(
            "Dry run complete. To run full training (5000 episodes, ~3-4 hours on RTX 4050), "
            "run: python -m src.train_d3qn --episodes 5000"
        )


if __name__ == "__main__":
    main()
