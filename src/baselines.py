"""
baselines.py
============
Deterministic maintenance baselines compatible with the 5D state environment.

Policies
--------
1. CorrectiveMaintenance  — replace only after failure (reactive lower bound)
2. PeriodicPM             — replace every fixed interval (simple scheduled PM)
3. ThresholdPolicy        — domain-expert HI-based rule (non-trivial baseline)
4. RiskNeutralDQN         — QR-DQN with risk_mode='mean' (isolates CVaR benefit)

All policies expose:
    policy.select_action(obs, step, info) -> int

``evaluate_policy`` accepts any of these or a bare QRDQNAgent.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from src.device import get_device
except ImportError:
    from device import get_device  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Action constants (mirror rl_environment.py)
ACTION_DO_NOTHING = 0
ACTION_REPAIR     = 1
ACTION_REPLACE    = 2


# ---------------------------------------------------------------------------
# 1. Corrective Maintenance
# ---------------------------------------------------------------------------

class CorrectiveMaintenance:
    """Replace bearing only when it has just failed (hi_t < 0.05); otherwise do nothing.

    This is the reactive lower-bound baseline: it accrues maximum degradation
    cost but never wastes money on healthy bearings.
    """

    def reset(self) -> None:
        pass

    def select_action(
        self,
        obs:  np.ndarray,
        step: int,
        info: Dict[str, Any],
    ) -> int:
        if float(info.get("hi_t", 1.0)) < 0.05:
            return ACTION_REPLACE
        return ACTION_DO_NOTHING


# ---------------------------------------------------------------------------
# 2. Periodic Preventive Maintenance
# ---------------------------------------------------------------------------

class PeriodicPM:
    """Replace bearing every *interval* steps; do nothing otherwise.

    Parameters
    ----------
    interval : int
        Steps between scheduled replacements.  Default 50.
    """

    def __init__(self, interval: int = 50) -> None:
        self.interval = int(interval)

    def reset(self) -> None:
        pass

    def select_action(
        self,
        obs:  np.ndarray,
        step: int,
        info: Dict[str, Any],
    ) -> int:
        if step > 0 and step % self.interval == 0:
            return ACTION_REPLACE
        return ACTION_DO_NOTHING


class PeriodicPreventiveMaintenance(PeriodicPM):
    """Alias for PeriodicPM; accepts ``fixed_interval`` kwarg for backward compat."""

    def __init__(
        self,
        interval:       int           = 50,
        fixed_interval: Optional[int] = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(interval=fixed_interval if fixed_interval is not None else interval)


# ---------------------------------------------------------------------------
# 3. Threshold Policy (domain-expert rule-based)
# ---------------------------------------------------------------------------

class ThresholdPolicy:
    """HI-based rule policy — a non-trivial domain-expert baseline.

    Decision rules (obs[0] = hi, obs[3]*5 = n_repairs):
      hi < 0.15             → replace  (bearing critically degraded)
      hi < 0.35, n_rep < 2 → repair   (degrading, repair still effective)
      otherwise             → do nothing
    """

    def reset(self) -> None:
        pass

    def select_action(
        self,
        obs:  np.ndarray,
        step: int,
        info: Dict[str, Any],
    ) -> int:
        hi       = float(obs[0])
        n_repairs = float(obs[3]) * 5.0
        if hi < 0.15:
            return ACTION_REPLACE
        if hi < 0.35 and n_repairs < 2:
            return ACTION_REPAIR
        return ACTION_DO_NOTHING


# ---------------------------------------------------------------------------
# 4. Risk-neutral QR-DQN wrapper
# ---------------------------------------------------------------------------

class RiskNeutralDQN:
    """QR-DQN with risk_mode='mean' — isolates the marginal benefit of CVaR.

    Parameters
    ----------
    checkpoint_path : str, Path, or None
        Optional .pth checkpoint to load weights from.
    **agent_kwargs :
        Forwarded to QRDQNAgent (risk_mode is always overridden to 'mean').
    """

    def __init__(
        self,
        checkpoint_path: Optional[Any] = None,
        **agent_kwargs: Any,
    ) -> None:
        try:
            from src.qrdqn_agent import QRDQNAgent as _Agent
        except ImportError:
            from qrdqn_agent import QRDQNAgent as _Agent  # type: ignore[no-redef]

        agent_kwargs.pop("risk_mode", None)
        self.agent = _Agent(risk_mode="mean", **agent_kwargs)
        if checkpoint_path is not None:
            self.agent.load_checkpoint(checkpoint_path)

    def reset(self) -> None:
        pass

    def select_action(
        self,
        obs:  np.ndarray,
        step: int,
        info: Dict[str, Any],
    ) -> int:
        return self.agent.select_action(obs, greedy=True)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def _call_policy(
    policy: Any,
    obs:    np.ndarray,
    step:   int,
    info:   Dict[str, Any],
) -> int:
    """Route to the correct select_action signature."""
    try:
        from src.qrdqn_agent import QRDQNAgent as _Agent
    except ImportError:
        from qrdqn_agent import QRDQNAgent as _Agent  # type: ignore[no-redef]

    if isinstance(policy, _Agent):
        return policy.select_action(obs, greedy=True)
    return policy.select_action(obs, step, info)


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy:     Any,
    env:        Any,
    n_episodes: int = 300,
    seed:       int = 42,
) -> Dict[str, Any]:
    """Evaluate any maintenance policy over *n_episodes* rollouts.

    Compatible with CorrectiveMaintenance, PeriodicPM, ThresholdPolicy,
    RiskNeutralDQN, and bare QRDQNAgent instances.

    Returns
    -------
    dict with keys:
        mean_total_cost, std_total_cost, catastrophe_rate,
        mean_TTR, std_TTR, mean_n_repairs, mean_n_replacements,
        action_distribution, mean_episode_length
    """
    rng = np.random.default_rng(seed)

    total_costs:    List[float] = []
    catastrophes:   List[int]   = []
    ttrs:           List[int]   = []
    n_repairs_ep:   List[int]   = []
    n_replace_ep:   List[int]   = []
    ep_lengths:     List[int]   = []
    action_counts = np.zeros(3, dtype=np.int64)

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2 ** 31))
        obs, info = env.reset(seed=ep_seed)

        if hasattr(policy, "reset"):
            policy.reset()

        done                          = False
        step                          = 0
        first_replace_step: int | None = None

        while not done:
            action = _call_policy(policy, obs, step, info)
            action_counts[action] += 1
            if action == ACTION_REPLACE and first_replace_step is None:
                first_replace_step = step
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        total_costs.append(float(info["total_cost"]))
        catastrophes.append(1 if info.get("is_failure", False) else 0)
        if first_replace_step is not None:
            ttrs.append(first_replace_step)
        n_repairs_ep.append(int(info.get("n_repairs", 0)))
        n_replace_ep.append(int(info.get("n_replacements", 0)))
        ep_lengths.append(step)

        if (ep + 1) % 100 == 0:
            logger.debug(
                "evaluate_policy: %d/%d  mean_cost=%.2f  catast=%.1%%",
                ep + 1, n_episodes,
                float(np.mean(total_costs)),
                float(np.mean(catastrophes)) * 100,
            )

    total = int(action_counts.sum())
    ttr_arr = np.array(ttrs, dtype=np.float64) if ttrs else np.array([], dtype=np.float64)
    return {
        "mean_total_cost":     float(np.mean(total_costs)),
        "std_total_cost":      float(np.std(total_costs)),
        "catastrophe_rate":    float(np.mean(catastrophes)),
        "mean_TTR":            float(np.mean(ttr_arr))  if ttr_arr.size else float("nan"),
        "std_TTR":             float(np.std(ttr_arr))   if ttr_arr.size else float("nan"),
        "mean_n_repairs":      float(np.mean(n_repairs_ep)),
        "mean_n_replacements": float(np.mean(n_replace_ep)),
        "action_distribution": {
            "do_nothing": float(action_counts[0] / total) if total else 0.0,
            "repair":     float(action_counts[1] / total) if total else 0.0,
            "replace":    float(action_counts[2] / total) if total else 0.0,
        },
        "mean_episode_length": float(np.mean(ep_lengths)),
    }


# ---------------------------------------------------------------------------
# Legacy classes kept for backward compat (used by old scripts / smoke tests)
# ---------------------------------------------------------------------------

class FixedIntervalPolicy:
    """Replace every *interval* steps (legacy name, use PeriodicPM)."""
    def __init__(self, interval: int = 50) -> None:
        self.interval = interval; self._step = 0
    def reset(self) -> None: self._step = 0
    def select_action(self, obs: np.ndarray, step: int = 0, info: Dict[str, Any] = None) -> int:
        self._step += 1
        return ACTION_REPLACE if self._step % self.interval == 0 else ACTION_DO_NOTHING


class RandomPolicy:
    """Random action baseline."""
    def __init__(self, n_actions: int = 3, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed); self.n_actions = n_actions
    def select_action(self, obs: np.ndarray, step: int = 0, info: Dict[str, Any] = None) -> int:
        return int(self._rng.integers(self.n_actions))


# ---------------------------------------------------------------------------
# Entry point — evaluate all 3 baselines and save comparison CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    try:
        from src.rl_environment import make_env_from_processed
    except ImportError:
        from rl_environment import make_env_from_processed  # type: ignore[no-redef]

    _PROC_DIR    = _proj / "data" / "processed"
    _RESULTS_DIR = _proj / "results"
    _N_EPISODES  = 100
    _SEED        = 42

    if not _PROC_DIR.exists():
        sys.exit(f"ERROR: {_PROC_DIR} not found. Run src/feature_extractor.py first.")

    print(f"Loading environment from {_PROC_DIR} …")
    _env = make_env_from_processed(
        processed_dir=_PROC_DIR,
        seed=_SEED,
    )

    _policies: Dict[str, Any] = {
        "CorrectiveMaintenance": CorrectiveMaintenance(),
        "PeriodicPM(50)":        PeriodicPM(interval=50),
        "ThresholdPolicy":       ThresholdPolicy(),
    }

    _results: Dict[str, Dict[str, Any]] = {}
    for _name, _policy in _policies.items():
        print(f"Evaluating {_name} over {_N_EPISODES} episodes …", flush=True)
        _results[_name] = evaluate_policy(_policy, _env, n_episodes=_N_EPISODES, seed=_SEED)

    # ---- print comparison table ----
    _COLS = [
        ("mean_total_cost",     "Cost_mean", 10),
        ("std_total_cost",      "Cost_std",   9),
        ("catastrophe_rate",    "Catast%",    8),
        ("mean_TTR",            "TTR_mean",   9),
        ("mean_n_repairs",      "Repairs/ep", 10),
        ("mean_n_replacements", "Replaces/ep",11),
        ("mean_episode_length", "Ep_len",     7),
    ]
    _W = 25
    _hdr = f"{'Policy':<{_W}}" + "".join(f"{lbl:>{w}}" for _, lbl, w in _COLS)
    print(f"\n{'='*len(_hdr)}")
    print("Baseline comparison — all bearings")
    print(f"{'='*len(_hdr)}")
    print(_hdr)
    print(f"{'-'*len(_hdr)}")
    for _name, _res in _results.items():
        _row = f"{_name:<{_W}}"
        for _key, _, _w in _COLS:
            _v = _res[_key]
            if _key == "catastrophe_rate":
                _cell = f"{_v*100:.1f}%"
            elif math.isnan(_v):
                _cell = "nan"
            else:
                _cell = f"{_v:.3f}"
            _row += f"{_cell:>{_w}}"
        print(_row)
    print(f"{'='*len(_hdr)}")

    # ---- save CSV ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _csv_path = _RESULTS_DIR / "baseline_comparison.csv"
    _fields = ["policy", "mean_total_cost", "std_total_cost", "catastrophe_rate",
               "mean_TTR", "std_TTR", "mean_n_repairs", "mean_n_replacements",
               "dn_pct", "rp_pct", "rx_pct", "mean_episode_length"]
    with open(_csv_path, "w", newline="", encoding="utf-8") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_fields)
        _w.writeheader()
        for _name, _res in _results.items():
            _act = _res.get("action_distribution", {})
            _w.writerow({
                "policy":              _name,
                "mean_total_cost":     round(_res["mean_total_cost"], 5),
                "std_total_cost":      round(_res["std_total_cost"], 5),
                "catastrophe_rate":    round(_res["catastrophe_rate"], 5),
                "mean_TTR":            "" if math.isnan(_res["mean_TTR"]) else round(_res["mean_TTR"], 3),
                "std_TTR":             "" if math.isnan(_res["std_TTR"])  else round(_res["std_TTR"],  3),
                "mean_n_repairs":      round(_res["mean_n_repairs"], 4),
                "mean_n_replacements": round(_res["mean_n_replacements"], 4),
                "dn_pct":              round(_act.get("do_nothing", 0), 5),
                "rp_pct":              round(_act.get("repair",     0), 5),
                "rx_pct":              round(_act.get("replace",    0), 5),
                "mean_episode_length": round(_res["mean_episode_length"], 2),
            })
    print(f"\nSaved → {_csv_path}")
