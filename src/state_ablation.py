"""
state_ablation.py
=================
Ablation study: what happens when we replace the full 5D state with
progressively richer RUL-derived signals?

State configurations
--------------------
A — 1D  [RUL_norm]                  pure point estimate
B — 2D  [RUL_norm, sigma_norm]      adds MC uncertainty
C — 3D  [RUL_norm, sigma_norm, CFP] Xu & Zhang (2025) design
D — 5D  [HI, slope, RUL, repairs, steps]  full state (ours, pre-trained)

Usage
-----
    python -m src.state_ablation               # train A/B/C + evaluate all
    python -m src.state_ablation --eval-only   # skip training
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

import sys
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from src.device import get_device
    from src.qrdqn_agent import QRDQNAgent, ReplayBuffer
    from src.rl_environment import PdMBearingEnv, make_env_from_processed
    from src.rul_predictor import (
        ConvSARULPredictor, create_sliding_windows,
        mc_dropout_inference, MAX_RUL,
    )
except ImportError:
    from device import get_device                     # type: ignore[no-redef]
    from qrdqn_agent import QRDQNAgent, ReplayBuffer  # type: ignore[no-redef]
    from rl_environment import PdMBearingEnv, make_env_from_processed  # type: ignore[no-redef]
    from rul_predictor import (                        # type: ignore[no-redef]
        ConvSARULPredictor, create_sliding_windows,
        mc_dropout_inference, MAX_RUL,
    )

logger = logging.getLogger(__name__)

_WINDOW_SIZE  = 32
_CFP_TAU      = 30.0          # RUL threshold (in original scale) for CFP
_MC_SAMPLES   = 10            # MC passes for precomputation (speed/quality trade-off)
_ALL_BEARINGS = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2"]

_STATE_DIMS: Dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 5}
_STATE_LABELS = {
    "A": "RUL only",
    "B": "RUL+sigma",
    "C": "RUL+sigma+CFP",
    "D": "Full (ours)",
}
_STATE_COMPONENTS = {
    "A": "RUL_norm",
    "B": "RUL_norm, sigma_norm",
    "C": "RUL_norm, sigma_norm, CFP",
    "D": "HI, slope, RUL, repairs, steps",
}

# Blue gradient from light to dark (for 4 bars A→D)
_BAR_COLORS = ["#a8c8e8", "#5fa8d3", "#2980b9", "#1a5276"]


# ===========================================================================
# 1.  MC Dropout precomputation
# ===========================================================================

def _precompute_mc_cache(
    proc_dir: Path,
    ckpt_path: Path,
    bearing_ids: List[str] = _ALL_BEARINGS,
    n_mc: int              = _MC_SAMPLES,
    device_str: str        = "cpu",
) -> Tuple[Dict[str, Dict[str, np.ndarray]], float]:
    """Run MC Dropout inference for every bearing; return (cache, sigma2_max).

    Cache structure per bearing
    ---------------------------
    mc_cache[bid]["mean_rul"]  : (M,) float32 — MC mean in [0, 1] (divided by MAX_RUL)
    mc_cache[bid]["sigma2"]    : (M,) float32 — MC variance in [0, MAX_RUL]^2 units
    mc_cache[bid]["samples"]   : (n_mc, M) float32 — individual samples in [0, MAX_RUL]

    M = N_bearing_timesteps - _WINDOW_SIZE + 1.
    Index i corresponds to the prediction for bearing timestep i + (_WINDOW_SIZE - 1).
    For earlier timesteps (< _WINDOW_SIZE - 1) the caller uses index 0 as padding.
    """
    dev = torch.device(device_str)

    # Load model
    model = ConvSARULPredictor()
    ckpt  = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd    = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    model.to(dev)
    print(f"  [MC cache] Loaded RUL model from {ckpt_path.name}")

    mc_cache:  Dict[str, Dict[str, np.ndarray]] = {}
    sigma2_max = 0.0

    for bid in bearing_ids:
        feat_path = proc_dir / f"{bid}_features.npy"
        rul_path  = proc_dir / f"{bid}_rul.npy"
        if not feat_path.exists() or not rul_path.exists():
            logger.warning("Missing processed files for bearing %s; skipped.", bid)
            continue

        features = np.load(feat_path).astype(np.float32)   # (N, 32)
        rul_raw  = np.load(rul_path).astype(np.float32)    # (N,) in [0, 125]

        # Create sliding windows (M, window_size, n_features)
        windows, _ = create_sliding_windows(
            features, rul_raw, window_size=_WINDOW_SIZE, stride=1
        )

        # MC inference — outputs in original RUL scale [0, 125]
        mean_rul, sigma2, samples = mc_dropout_inference(
            model, windows, n_samples=n_mc, device=device_str
        )
        # mean_rul: (M,), sigma2: (M,), samples: (n_mc, M)

        mc_cache[bid] = {
            "mean_rul": np.clip(mean_rul / MAX_RUL, 0.0, 1.0).astype(np.float32),
            "sigma2":   sigma2.astype(np.float32),
            "samples":  samples.astype(np.float32),
        }
        sigma2_max = max(sigma2_max, float(np.max(sigma2)) if len(sigma2) else 0.0)
        print(
            f"  [MC cache] {bid}: N={len(features)}  M={len(mean_rul)}"
            f"  rul_range=[{mean_rul.min():.1f},{mean_rul.max():.1f}]"
            f"  sigma2_max={float(np.max(sigma2)):.2f}"
        )

    sigma2_max = max(sigma2_max, 1e-8)
    print(f"  [MC cache] Global sigma2_max = {sigma2_max:.4f}")
    return mc_cache, sigma2_max


# ===========================================================================
# 2.  Wrapper environment
# ===========================================================================

class StateAblationEnv(PdMBearingEnv):
    """PdMBearingEnv wrapper that returns a reduced state vector.

    For modes A/B/C the observation is derived from precomputed MC Dropout
    predictions (cached at __init__ time).  The environment dynamics
    (reward, termination) are identical to PdMBearingEnv — only the
    observation visible to the agent changes.

    Parameters
    ----------
    hi_sequences, rul_sequences : as in PdMBearingEnv
    state_mode : "A", "B", "C", or "D"
    mc_cache   : dict returned by _precompute_mc_cache (required for A/B/C)
    sigma2_max : global normalisation constant for sigma2 (required for B/C)
    """

    def __init__(
        self,
        hi_sequences:  Dict[str, np.ndarray],
        rul_sequences: Dict[str, np.ndarray],
        state_mode:    str  = "D",
        mc_cache:      Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        sigma2_max:    float = 1.0,
        seed:          Optional[int] = None,
        render_mode:   Optional[str] = None,
        max_steps:     int  = 400,
    ) -> None:
        super().__init__(
            hi_sequences  = hi_sequences,
            rul_sequences = rul_sequences,
            seed          = seed,
            render_mode   = render_mode,
            max_steps     = max_steps,
        )
        self.state_mode  = state_mode.upper()
        self._mc_cache   = mc_cache or {}
        self._sigma2_max = max(float(sigma2_max), 1e-8)

        # Only override observation space for reduced-dimension modes.
        # Mode D keeps the parent's full 5D space (slope lives in [-1, 1]).
        if self.state_mode != "D":
            obs_dim = _STATE_DIMS[self.state_mode]
            self.observation_space = spaces.Box(
                low  = np.zeros(obs_dim, dtype=np.float32),
                high = np.ones(obs_dim,  dtype=np.float32),
                dtype = np.float32,
            )

    # ------------------------------------------------------------------
    # Gymnasium overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:           Optional[int]  = None,
        options:        Optional[dict] = None,
        force_degraded: bool           = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options, force_degraded=force_degraded)
        return self._build_ablation_obs(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        _, reward, terminated, truncated, info = super().step(action)
        return self._build_ablation_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_ablation_obs(self) -> np.ndarray:
        if self.state_mode == "D":
            return self._build_obs()   # parent's 5D observation

        t   = min(self._t, len(self._rul_seq) - 1)
        bid = self._current_bid
        mc  = self._mc_cache.get(bid)

        if mc is None:
            # Fallback to normalised RUL from the environment
            rul_t = float(np.clip(self._rul_seq[t], 0.0, 1.0))
            out   = np.zeros(_STATE_DIMS[self.state_mode], dtype=np.float32)
            out[0] = rul_t
            return out

        # Offset into the full (unsliced) bearing sequence.
        # _rul_seq is always a slice/view of _rul_sequences[bid], so:
        #   len(full) - len(slice) == start index of the slice.
        full_len   = len(self._rul_sequences[bid])
        seq_offset = full_len - len(self._rul_seq)
        orig_t     = seq_offset + t

        # Map bearing timestep → window index (causal, padded at the start).
        # Window i was built from features[i : i+WINDOW_SIZE], labelled at i+31.
        # Inverse: for timestep orig_t, use window max(0, orig_t - 31).
        n_windows  = len(mc["mean_rul"])
        window_idx = max(0, min(orig_t - (_WINDOW_SIZE - 1), n_windows - 1))

        rul_norm = float(np.clip(mc["mean_rul"][window_idx], 0.0, 1.0))

        if self.state_mode == "A":
            return np.array([rul_norm], dtype=np.float32)

        sigma2_norm = float(
            np.clip(mc["sigma2"][window_idx] / self._sigma2_max, 0.0, 1.0)
        )

        if self.state_mode == "B":
            return np.array([rul_norm, sigma2_norm], dtype=np.float32)

        # CFP: fraction of MC samples predicting RUL <= tau (imminent failure)
        samples_t = mc["samples"][:, window_idx]         # (n_mc,)
        cfp = float(np.mean(samples_t <= _CFP_TAU))

        return np.array([rul_norm, sigma2_norm, cfp], dtype=np.float32)


# ===========================================================================
# 3.  Factory helper
# ===========================================================================

def make_ablation_env(
    proc_dir:   Path,
    state_mode: str,
    mc_cache:   Optional[Dict] = None,
    sigma2_max: float          = 1.0,
    seed:       Optional[int]  = None,
) -> StateAblationEnv:
    """Load HI/RUL arrays from proc_dir and return a StateAblationEnv."""
    hi_seqs:  Dict[str, np.ndarray] = {}
    rul_seqs: Dict[str, np.ndarray] = {}

    for bid in _ALL_BEARINGS:
        hi_p  = proc_dir / f"{bid}_hi.npy"
        rul_p = proc_dir / f"{bid}_rul.npy"
        if not hi_p.exists() or not rul_p.exists():
            continue
        hi_seqs[bid]  = np.load(hi_p).astype(np.float32)
        rul_seqs[bid] = (np.load(rul_p) / MAX_RUL).astype(np.float32)

    if not hi_seqs:
        raise FileNotFoundError(f"No bearing files found in {proc_dir}")

    return StateAblationEnv(
        hi_sequences  = hi_seqs,
        rul_sequences = rul_seqs,
        state_mode    = state_mode,
        mc_cache      = mc_cache,
        sigma2_max    = sigma2_max,
        seed          = seed,
    )


# ===========================================================================
# 4.  Training loop (mirrors train.py: CVaR, force_degraded, same HPs)
# ===========================================================================

def _train_ablation_agent(
    agent:        QRDQNAgent,
    env:          StateAblationEnv,
    n_episodes:   int           = 5_000,
    warmup_steps: int           = 500,
    save_path:    Optional[Path] = None,
    eval_every:   int           = 250,
    n_eval_ep:    int           = 10,
    seed:         int           = 42,
    label:        str           = "",
    fast_diag:    bool          = False,
) -> Dict[str, List[float]]:
    """QR-DQN training with force_degraded (40%) and warmup random policy.

    fast_diag=True overrides: 1500 episodes, eval every 400, 60 eval episodes.
    Also switches mid-training eval to _greedy_eval_diag (catastrophe + action dist)
    and prints a DIAGNOSTIC VERDICT at the end.
    """
    if fast_diag:
        n_episodes = 1500
        eval_every = 400
        n_eval_ep  = 60

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    buffer    = ReplayBuffer(maxlen=100_000)
    history: Dict[str, List[float]] = {"train_returns": [], "losses": []}
    best_eval = -math.inf
    total_steps = 0
    tag = f"[{label}]" if label else ""
    collapse_dn_hist: List[float] = []
    last_dn_pct = 1.0

    for ep in range(1, n_episodes + 1):
        force_deg = (random.random() < 0.40)
        obs, _    = env.reset(seed=seed + ep, force_degraded=force_deg)
        ep_return = 0.0
        ep_losses: List[float] = []
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = random.randrange(3)
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))
            obs         = next_obs
            ep_return  += reward
            total_steps += 1

            if buffer.is_ready(agent.batch_size) and total_steps >= warmup_steps:
                loss = agent.update(buffer.sample(agent.batch_size))
                ep_losses.append(loss)

        agent.step_episode()
        history["train_returns"].append(ep_return)
        if ep_losses:
            history["losses"].append(float(np.mean(ep_losses)))

        if ep % eval_every == 0:
            if fast_diag:
                m = _greedy_eval_diag(agent, env, n_eval_ep, seed)
                last_dn_pct = m["dn_pct"]
                collapse_dn_hist.append(last_dn_pct)
                collapsed = (
                    len(collapse_dn_hist) >= 3
                    and all(x > 0.90 for x in collapse_dn_hist[-3:])
                )
                print(
                    f"  {tag} ep={ep:5d}/{n_episodes}  eps={agent.epsilon:.3f}"
                    f"  eval_ret={m['mean_return']:.1f}"
                    f"  catast={m['catastrophe_rate']:.1%}"
                    f"  act=[dn={m['dn_pct']:.0%} rp={m['rp_pct']:.0%}"
                    f" rx={m['rx_pct']:.0%}]"
                    + ("  *** COLLAPSE ***" if collapsed else "")
                )
                if m["mean_return"] > best_eval and save_path is not None:
                    best_eval = m["mean_return"]
                    agent.save_checkpoint(save_path)
                    print(f"    -> new best eval={best_eval:.2f}, saved.")
            else:
                eval_ret = _greedy_eval(agent, env, n_eval_ep, seed)
                print(
                    f"  {tag} ep={ep:5d}/{n_episodes}  eps={agent.epsilon:.3f}"
                    f"  train={ep_return:.1f}  eval={eval_ret:.1f}"
                )
                if eval_ret > best_eval and save_path is not None:
                    best_eval = eval_ret
                    agent.save_checkpoint(save_path)
                    print(f"    -> new best eval={best_eval:.2f}, saved.")

    if save_path is not None and best_eval == -math.inf:
        agent.save_checkpoint(save_path)

    if fast_diag:
        verdict = (
            "COLLAPSED -- do-nothing dominates"
            if last_dn_pct >= 0.95
            else "NOT COLLAPSED -- QR-DQN stable under fast-diag settings"
        )
        msg = (
            f"DIAGNOSTIC VERDICT [{label} | risk_mode={agent.risk_mode}]:"
            f" dn%={last_dn_pct * 100:.0f}% at final eval. {verdict}"
        )
        print(f"\n{msg}\n")

    return history


def _greedy_eval(agent: QRDQNAgent, env, n_episodes: int, seed: int) -> float:
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        total, done = 0.0, False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        returns.append(total)
    return float(np.mean(returns))


def _greedy_eval_diag(
    agent: QRDQNAgent, env, n_episodes: int, seed: int
) -> Dict[str, Any]:
    """Like _greedy_eval but also returns catastrophe_rate and action distribution."""
    returns: List[float] = []
    catastrophes: List[int] = []
    action_counts = [0, 0, 0]
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        total, done = 0.0, False
        info: Dict[str, Any] = {}
        while not done:
            action = agent.select_action(obs, greedy=True)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
        catastrophes.append(1 if info.get("is_failure", False) else 0)
    total_ac = max(1, sum(action_counts))
    return {
        "mean_return":      float(np.mean(returns)),
        "catastrophe_rate": float(np.mean(catastrophes)),
        "dn_pct":           action_counts[0] / total_ac,
        "rp_pct":           action_counts[1] / total_ac,
        "rx_pct":           action_counts[2] / total_ac,
    }


# ===========================================================================
# 5.  Evaluation
# ===========================================================================

def evaluate_ablation(
    agents: Dict[str, QRDQNAgent],
    envs:   Dict[str, Any],
    n_episodes: int = 300,
    seed:       int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate agents A–D on their respective environments.

    Returns
    -------
    {mode: {mean_cost, std_cost, catastrophe_rate, mean_reward,
             mean_n_repairs, mean_n_replacements, action_dist}}
    """
    rng     = np.random.default_rng(seed)
    results: Dict[str, Dict[str, float]] = {}

    for mode in ("A", "B", "C", "D"):
        agent = agents.get(mode)
        env   = envs.get(mode)
        if agent is None or env is None:
            continue

        total_costs:  List[float] = []
        catastrophes: List[int]   = []
        ep_rewards:   List[float] = []
        n_repairs_ep: List[int]   = []
        n_replace_ep: List[int]   = []
        action_counts = np.zeros(3, dtype=np.int64)

        for ep in range(n_episodes):
            ep_seed  = int(rng.integers(0, 2 ** 31))
            obs, _   = env.reset(seed=ep_seed)
            done     = False
            ep_rew   = 0.0

            while not done:
                action = agent.select_action(obs, greedy=True)
                action_counts[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done   = terminated or truncated
                ep_rew += reward

            total_costs.append(float(info["total_cost"]))
            catastrophes.append(1 if info.get("is_failure", False) else 0)
            ep_rewards.append(ep_rew)
            n_repairs_ep.append(int(info.get("n_repairs", 0)))
            n_replace_ep.append(int(info.get("n_replacements", 0)))

        total = max(1, int(action_counts.sum()))
        results[mode] = {
            "mean_cost":           float(np.mean(total_costs)),
            "std_cost":            float(np.std(total_costs)),
            "catastrophe_rate":    float(np.mean(catastrophes)),
            "mean_reward":         float(np.mean(ep_rewards)),
            "mean_n_repairs":      float(np.mean(n_repairs_ep)),
            "mean_n_replacements": float(np.mean(n_replace_ep)),
            "action_dist": {
                "do_nothing": float(action_counts[0] / total),
                "repair":     float(action_counts[1] / total),
                "replace":    float(action_counts[2] / total),
            },
        }
        print(
            f"  State {mode} ({_STATE_LABELS[mode]:20s})"
            f"  cost={results[mode]['mean_cost']:.2f}"
            f"  catast={results[mode]['catastrophe_rate']:.1%}"
            f"  rew={results[mode]['mean_reward']:.2f}"
        )

    return results


# ===========================================================================
# 6.  Table (CSV + LaTeX)
# ===========================================================================

def generate_ablation_table(
    results:     Dict[str, Dict[str, float]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    modes = [m for m in ("A", "B", "C", "D") if m in results]

    # CSV ------------------------------------------------------------------
    csv_path = results_dir / "table_state_ablation.csv"
    fields   = ["State_Config", "Dims", "Components",
                 "Cost_mu", "Catastrophe_pct", "Avg_Reward"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for m in modes:
            r = results[m]
            w.writerow({
                "State_Config":    m,
                "Dims":            _STATE_DIMS[m],
                "Components":      _STATE_COMPONENTS[m],
                "Cost_mu":         round(r["mean_cost"],        3),
                "Catastrophe_pct": round(r["catastrophe_rate"] * 100, 2),
                "Avg_Reward":      round(r["mean_reward"],      3),
            })
    print(f"  Saved -> {csv_path}")

    # LaTeX ----------------------------------------------------------------
    # Best per column
    cost_vals  = [results[m]["mean_cost"]        for m in modes]
    catas_vals = [results[m]["catastrophe_rate"]  for m in modes]
    rew_vals   = [results[m]["mean_reward"]       for m in modes]

    best_cost  = min(cost_vals)
    best_catas = min(catas_vals)
    best_rew   = max(rew_vals)

    def _b(val: float, best: float, fmt: str) -> str:
        s = format(val, fmt)
        return f"\\textbf{{{s}}}" if abs(val - best) < 1e-9 else s

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{State representation ablation (300 evaluation episodes, seed=42)}",
        "\\label{tab:state_ablation}",
        "\\begin{tabular}{clcrrr}",
        "\\toprule",
        "\\textbf{State} & \\textbf{Components} & \\textbf{Dims}"
        " & \\textbf{Cost $\\mu$} & \\textbf{Catastrophe \\%}"
        " & \\textbf{Avg Reward} \\\\",
        "\\midrule",
    ]
    for m in modes:
        r = results[m]
        row = (
            f"{m} & {_STATE_COMPONENTS[m]} & {_STATE_DIMS[m]}"
            f" & {_b(r['mean_cost'],       best_cost,  '.2f')}"
            f" & {_b(r['catastrophe_rate']*100, best_catas*100, '.1f')}\\%"
            f" & {_b(r['mean_reward'],      best_rew,   '.2f')}"
            " \\\\"
        )
        if m == "D":
            row = row.replace("\\\\", "  % ours \\\\")
        tex_lines.append(row)
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    tex_path = results_dir / "table_state_ablation.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines) + "\n")
    print(f"  Saved -> {tex_path}")


# ===========================================================================
# 7.  Plots
# ===========================================================================

def generate_ablation_plot(
    results:     Dict[str, Dict[str, float]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "axes.spines.right":  False,
        "axes.spines.top":    False,
    })

    modes   = [m for m in ("A", "B", "C", "D") if m in results]
    xlabels = [_STATE_LABELS[m] for m in modes]
    xpos    = np.arange(len(modes))
    colors  = [_BAR_COLORS[i] for i in range(len(modes))]

    catast_pct  = [results[m]["catastrophe_rate"] * 100.0 for m in modes]
    mean_rew    = [results[m]["mean_reward"]               for m in modes]

    fig, (ax_cat, ax_rew) = plt.subplots(1, 2, figsize=(7, 3.5))

    # --- (a) Catastrophe rate --------------------------------------------
    ax = ax_cat
    bars = ax.bar(xpos, catast_pct, color=colors, edgecolor="white", width=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylabel("Catastrophe rate (%)")
    ax.set_title("(a) Catastrophe rate by state config")
    ax.grid(axis="y", alpha=0.25)

    # Annotate each bar
    for bar, v in zip(bars, catast_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}%",
            ha="center", va="bottom", fontsize=7,
        )

    # "Best" arrow on state D bar
    if "D" in modes:
        d_idx  = modes.index("D")
        d_val  = catast_pct[d_idx]
        best_v = min(catast_pct)
        ax.annotate(
            "Best" if d_val <= best_v + 1e-6 else "Full",
            xy     = (d_idx, d_val),
            xytext = (d_idx + 0.35, d_val + max(catast_pct) * 0.12),
            fontsize = 7,
            color    = "#c0392b",
            arrowprops = dict(arrowstyle="->", color="#c0392b", lw=1.0),
        )

    # --- (b) Mean reward -------------------------------------------------
    ax = ax_rew
    bars2 = ax.bar(xpos, mean_rew, color=colors, edgecolor="white", width=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylabel("Mean episode reward (higher = better)")
    ax.set_title("(b) Mean reward by state config")
    ax.grid(axis="y", alpha=0.25)

    # Horizontal dashed reference line at State A reward
    if "A" in modes:
        a_rew = results["A"]["mean_reward"]
        ax.axhline(a_rew, color="#888888", linestyle="--", linewidth=1.0,
                   label=f"State A baseline ({a_rew:.1f})")
        ax.legend(loc="lower right", framealpha=0.7)

    for bar, v in zip(bars2, mean_rew):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (max(mean_rew) - min(mean_rew)) * 0.01,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout(pad=0.9)
    out = results_dir / "fig_state_ablation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ===========================================================================
# 8.  Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State representation ablation study.")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; load existing checkpoints.")
    parser.add_argument("--n-episodes", type=int, default=5_000)
    parser.add_argument("--n-eval",     type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--fast-diag",  action="store_true",
                        help="Screening mode: 1500 eps, 60 eval eps, eval every 400. "
                             "Trains State C only by default. Matches train_d3qn --fast-diag.")
    parser.add_argument("--risk-mode",  default="cvar", choices=["cvar", "mean"],
                        help="Risk aggregation mode: 'cvar' (default) or 'mean' (risk-neutral).")
    parser.add_argument("--states",     nargs="+", default=None,
                        choices=["A", "B", "C"],
                        help="Which state configs to train. Default: C in fast-diag, A B C otherwise.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    proc_dir    = _PROJ / "data" / "processed"
    results_dir = _PROJ / "results" / "05_state_ablation"
    rul_ckpt    = _PROJ / "results" / "01_rul_predictor" / "rul_model_best.pth"
    qrdqn_ckpt  = _PROJ / "results" / "00_primary_cvar_qrdqn" / "qrdqn_best.pth"

    for p in (proc_dir, rul_ckpt):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found.")

    # Determine which states to process
    if args.fast_diag:
        train_states = args.states or ["C"]
        print(
            f"\n=== FAST DIAGNOSTIC MODE: 1500 episodes, 60 eval episodes,"
            f" eval every 400 | risk_mode={args.risk_mode}"
            f" | states={train_states} ==="
        )
    else:
        train_states = args.states or ["A", "B", "C"]

    # Checkpoint suffix: separate files per (fast_diag, risk_mode) to avoid
    # overwriting the original 5000-episode cvar checkpoints.
    if args.fast_diag:
        _CKPT_SUFFIX = f"_diag_rm{args.risk_mode}"
    elif args.risk_mode != "cvar":
        _CKPT_SUFFIX = f"_rm{args.risk_mode}"
    else:
        _CKPT_SUFFIX = ""

    # --- Precompute MC Dropout cache for all bearings --------------------
    print("\nPrecomputing MC Dropout predictions ...")
    mc_cache, sigma2_max = _precompute_mc_cache(
        proc_dir    = proc_dir,
        ckpt_path   = rul_ckpt,
        bearing_ids = _ALL_BEARINGS,
        n_mc        = _MC_SAMPLES,
        device_str  = get_device(verbose=False),
    )

    # --- Build environments -----------------------------------------------
    print("\nBuilding environments ...")
    envs: Dict[str, Any] = {}
    for mode in train_states:
        envs[mode] = make_ablation_env(
            proc_dir   = proc_dir,
            state_mode = mode,
            mc_cache   = mc_cache,
            sigma2_max = sigma2_max,
            seed       = args.seed,
        )
        print(f"  State {mode}: obs_dim={envs[mode].observation_space.shape[0]}")

    # State D: full 5D env — only in standard (non-fast-diag) full-ablation mode
    if not args.fast_diag:
        envs["D"] = make_env_from_processed(proc_dir, seed=args.seed)
        print(f"  State D: obs_dim={envs['D'].observation_space.shape[0]}")

    # --- Train / load agents A, B, C -------------------------------------
    print("\nTraining / loading agents ...")
    agents: Dict[str, QRDQNAgent] = {}

    _CKPT = {
        m: results_dir / f"ablation_state{m}{_CKPT_SUFFIX}.pth"
        for m in ("A", "B", "C")
    }

    for mode in train_states:
        state_dim = _STATE_DIMS[mode]
        ckpt_path = _CKPT[mode]

        agent = QRDQNAgent(
            state_dim              = state_dim,
            n_actions              = 3,
            risk_mode              = args.risk_mode,
            cvar_alpha             = 0.25,
            lr                     = 1e-3,
            gamma                  = 0.99,
            epsilon_start          = 1.0,
            epsilon_end            = 0.05,
            epsilon_decay_episodes = 3_000,
            target_update_freq     = 100,
            batch_size             = 64,
            N_quantiles            = 51,
        )

        if args.eval_only or ckpt_path.exists():
            if ckpt_path.exists():
                agent.load_checkpoint(ckpt_path)
                print(f"  State {mode}: loaded from {ckpt_path.name}")
            else:
                print(f"  State {mode}: checkpoint not found, using random weights.")
        else:
            n_eps_display = 1500 if args.fast_diag else args.n_episodes
            print(
                f"  State {mode}: training for {n_eps_display} episodes"
                f" (risk_mode={args.risk_mode}) ..."
            )
            _train_ablation_agent(
                agent      = agent,
                env        = envs[mode],
                n_episodes = args.n_episodes,  # overridden inside when fast_diag=True
                save_path  = ckpt_path,
                seed       = args.seed,
                label      = f"State {mode}",
                fast_diag  = args.fast_diag,
            )

        agents[mode] = agent

    # --- Load existing QR-DQN (state D, CVaR) — full ablation only -------
    if not args.fast_diag:
        d_agent = QRDQNAgent(state_dim=5, risk_mode="cvar")
        if qrdqn_ckpt.exists():
            d_agent.load_checkpoint(qrdqn_ckpt)
            print(f"  State D: loaded from {qrdqn_ckpt.name}")
        else:
            print(f"  WARNING: {qrdqn_ckpt} not found — State D uses random weights.")
        agents["D"] = d_agent

    # --- Evaluate all trained modes --------------------------------------
    n_eval = 60 if args.fast_diag else args.n_eval
    print(f"\nEvaluating states {list(agents.keys())} ({n_eval} episodes each) ...")
    results = evaluate_ablation(agents, envs, n_episodes=n_eval, seed=args.seed)

    # --- Summary table to stdout -----------------------------------------
    print("\n" + "=" * 80)
    print(f"  {'State':>5}  {'Dims':>4}  {'Components':30s}"
          f"  {'Cost_mu':>8}  {'Catast%':>8}  {'AvgRew':>8}  {'dn%':>5}")
    print("-" * 80)
    for mode in ("A", "B", "C", "D"):
        if mode not in results:
            continue
        r  = results[mode]
        ad = r.get("action_dist", {})
        print(
            f"  {mode:>5}  {_STATE_DIMS[mode]:>4}  {_STATE_COMPONENTS[mode]:30s}"
            f"  {r['mean_cost']:>8.2f}"
            f"  {r['catastrophe_rate']:>7.1%}"
            f"  {r['mean_reward']:>8.2f}"
            f"  {ad.get('do_nothing', 0):>4.0%}"
        )
    print("=" * 80)

    # --- Fast-diag collapse verdict --------------------------------------
    if args.fast_diag:
        print()
        for mode in train_states:
            if mode not in results:
                continue
            r  = results[mode]
            dn = r.get("action_dist", {}).get("do_nothing", 1.0)
            verdict = "COLLAPSED" if dn >= 0.95 else "NOT COLLAPSED"
            print(
                f"FAST DIAG VERDICT [State {mode} | risk_mode={args.risk_mode}]:"
                f" dn={dn:.0%}  catast={r['catastrophe_rate']:.1%}. {verdict}"
            )
    else:
        # Generate full outputs only in standard mode
        print("\nGenerating table and figure ...")
        generate_ablation_table(results, results_dir)
        generate_ablation_plot(results, results_dir)

    print("\nDone. Outputs written to", results_dir)
