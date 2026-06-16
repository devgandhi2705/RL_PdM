"""
rl_environment.py
=================
Gymnasium environment for bearing predictive maintenance.

No neural-network calls during rollout.  HI and RUL sequences are loaded as
plain numpy arrays at __init__ time and indexed directly during each step.

State (5D, float32):
  obs[0] = hi(t)                      Health Index, [0, 1]
  obs[1] = hi_slope(t)                10-step linear slope, clipped [-1, 1]
  obs[2] = rul_norm(t)                MC mean RUL / 125, [0, 1]
  obs[3] = n_repairs / 5.0            Repair count (capped at 5), [0, 1]
  obs[4] = steps_since_replace / 100  Steps since last replacement (capped 100), [0, 1]

Actions: Discrete(3)
  0 = do_nothing
  1 = repair
  2 = replace
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

logger = logging.getLogger(__name__)

ACTION_DO_NOTHING = 0
ACTION_REPAIR     = 1
ACTION_REPLACE    = 2
N_ACTIONS         = 3
_ACTION_NAMES     = ["do_nothing", "repair", "replace"]


# ===========================================================================
# Environment
# ===========================================================================

class PdMBearingEnv(gym.Env):
    """Bearing maintenance scheduling environment (Gymnasium API).

    Parameters
    ----------
    hi_sequences : dict
        {bearing_id: np.ndarray (N,)} -- precomputed Health Index (1=healthy,
        0=failed).  Arrays are COPIED at reset / replace and may be mutated by
        repair actions.
    rul_sequences : dict
        {bearing_id: np.ndarray (N,)} -- normalised RUL (MC mean / 125).
        Read-only; must have the same length N as the corresponding HI array.
    seed : int or None
        Global RNG seed.
    render_mode : str or None
        Only ``"ansi"`` is supported.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        hi_sequences:  Dict[str, np.ndarray],
        rul_sequences: Dict[str, np.ndarray],
        seed:          Optional[int] = None,
        render_mode:   Optional[str] = None,
        max_steps:     int           = 400,
    ) -> None:
        super().__init__()

        if not hi_sequences:
            raise ValueError("hi_sequences must not be empty.")
        missing = [b for b in hi_sequences if b not in rul_sequences]
        if missing:
            raise ValueError(f"rul_sequences missing bearings: {missing}")

        self._hi_sequences  = {b: np.asarray(v, dtype=np.float32)
                               for b, v in hi_sequences.items()}
        self._rul_sequences = {b: np.asarray(v, dtype=np.float32)
                               for b, v in rul_sequences.items()}
        self._bearing_ids   = list(self._hi_sequences.keys())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng       = np.random.default_rng(seed)
        self._max_steps = max(1, int(max_steps))

        # Episode state -- filled by reset()
        first = self._bearing_ids[0]
        self._current_bid:          str        = first
        self._hi_copy:              np.ndarray = self._hi_sequences[first].copy()
        self._rul_seq:              np.ndarray = self._rul_sequences[first]
        self._t:                    int        = 0
        self._episode_steps:        int        = 0   # absolute counter — survives replacements
        self._n_repairs:            int        = 0
        self._n_replacements:       int        = 0
        self._steps_since_replace:  int        = 0
        self._total_cost:           float      = 0.0
        self._n_failures:           int        = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_hi(self, t: int) -> float:
        return float(self._hi_copy[min(t, len(self._hi_copy) - 1)])

    def _safe_rul(self, t: int) -> float:
        return float(self._rul_seq[min(t, len(self._rul_seq) - 1)])

    def _build_obs(self) -> np.ndarray:
        t       = min(self._t, len(self._hi_copy) - 1)
        hi_t    = float(np.clip(self._hi_copy[t], 0.0, 1.0))
        rul_t   = float(np.clip(self._rul_seq[min(t, len(self._rul_seq) - 1)], 0.0, 1.0))

        start = max(0, t - 9)
        win   = self._hi_copy[start: t + 1].astype(np.float64)
        if len(win) >= 2:
            xs    = np.arange(len(win), dtype=np.float64)
            slope = float(np.polyfit(xs, win, 1)[0])
        else:
            slope = 0.0
        hi_slope = float(np.clip(slope, -1.0, 1.0))

        return np.array([
            hi_t,
            hi_slope,
            rul_t,
            float(np.clip(self._n_repairs          / 5.0,   0.0, 1.0)),
            float(np.clip(self._steps_since_replace / 100.0, 0.0, 1.0)),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:           Optional[int]  = None,
        options:        Optional[dict] = None,
        force_degraded: bool           = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Parameters
        ----------
        force_degraded : bool
            If True, start from the last 40% of the bearing's lifetime so the
            agent encounters degraded states immediately.  Pass False (default)
            for evaluation to ensure unbiased start.  The training loop applies
            this flag with 40% probability to bias the replay buffer toward
            low-HI transitions.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        bid = (
            options["bearing_id"]
            if (options and "bearing_id" in options)
            else str(self._rng.choice(self._bearing_ids))
        )

        hi_arr  = self._hi_sequences[bid]
        rul_arr = self._rul_sequences[bid]
        hi_len  = len(hi_arr)

        if force_degraded:
            # Start from a random position in the last 40% of the bearing's life
            start_min = max(0, int(hi_len * 0.6))
            restart_t = int(self._rng.integers(start_min, hi_len))
            self._hi_copy = hi_arr[restart_t:].copy()
            self._rul_seq = rul_arr[restart_t:]
        else:
            self._hi_copy = hi_arr.copy()
            self._rul_seq = rul_arr

        self._current_bid         = bid
        self._t                   = 0
        self._episode_steps       = 0
        self._n_repairs           = 0
        self._n_replacements      = 0
        self._steps_since_replace = 0
        self._total_cost          = 0.0
        self._n_failures          = 0

        obs  = self._build_obs()
        info: Dict[str, Any] = {
            "bearing_id":          bid,
            "hi_t":                float(self._hi_copy[0]),
            "rul_t":               float(self._rul_seq[0]),
            "n_repairs":           0,
            "n_replacements":      0,
            "steps_since_replace": 0,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}.")

        t     = self._t
        hi_t  = self._safe_hi(t)
        rul_t = self._safe_rul(t)

        # ---- apply action ------------------------------------------------
        if action == ACTION_DO_NOTHING:
            if hi_t >= 0.5:
                step_cost = 0.1
            elif hi_t >= 0.3:
                step_cost = 2.0
            else:
                step_cost = 0.1 + max(0.0, (0.5 - hi_t) * 20.0)
            reward = -step_cost

        elif action == ACTION_REPAIR:
            efficacy = max(0.1, 0.35 * math.exp(-0.4 * self._n_repairs))
            hi_gain  = efficacy * (1.0 - hi_t)
            # Shift remaining HI up by hi_gain; cap each value at hi_t + hi_gain
            self._hi_copy[t:] = np.minimum(
                self._hi_copy[t:] + hi_gain,
                hi_t + hi_gain,
            )
            self._n_repairs += 1
            step_cost = 4.0 + 1.5 * self._n_repairs
            reward    = -step_cost

        else:  # ACTION_REPLACE
            new_bid    = str(self._rng.choice(self._bearing_ids))
            hi_len     = len(self._hi_sequences[new_bid])
            restart_t  = int(self._rng.integers(0, max(1, hi_len // 5)))
            self._hi_copy             = self._hi_sequences[new_bid][restart_t:].copy()
            self._rul_seq             = self._rul_sequences[new_bid][restart_t:]
            self._current_bid         = new_bid
            self._t                   = 0
            self._n_repairs           = 0
            self._steps_since_replace = 0
            self._n_replacements     += 1
            step_cost                 = 8.0
            reward                    = -8.0

        # ---- advance timestep (all actions) ------------------------------
        self._t                   += 1
        self._episode_steps       += 1
        self._steps_since_replace += 1

        # ---- termination check -------------------------------------------
        tn     = self._t
        hi_nxt = float(self._hi_copy[tn]) if tn < len(self._hi_copy) else 0.0
        terminated = hi_nxt < 0.055  # float32(0.05) ≈ 0.050000007 > 0.05, so use 0.055
        truncated  = (tn >= min(len(self._hi_copy) - 1, self._max_steps)
                      or self._episode_steps >= self._max_steps)

        is_failure = False
        if terminated:
            reward     -= 100.0
            step_cost  += 100.0
            is_failure  = True
            self._n_failures += 1

        self._total_cost += step_cost

        obs  = self._build_obs()
        info: Dict[str, Any] = {
            "cost":                step_cost,
            "action_name":         _ACTION_NAMES[action],
            "hi_t":                hi_t,
            "rul_t":               rul_t,
            "mean_rul":            rul_t * 125.0,   # for CM/PM baselines (original RUL scale)
            "n_repairs":           self._n_repairs,
            "n_replacements":      self._n_replacements,
            "steps_since_replace": self._steps_since_replace,
            "is_failure":          is_failure,
            "failure":             is_failure,      # alias for backward compatibility
            "total_cost":          self._total_cost,
            "bearing_id":          self._current_bid,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        t    = min(self._t, len(self._hi_copy) - 1)
        lines = [
            "=" * 54,
            f"  bearing={self._current_bid}  t={self._t}  "
            f"HI={self._hi_copy[t]:.3f}  RUL={self._rul_seq[min(t, len(self._rul_seq)-1)]:.3f}",
            f"  repairs={self._n_repairs}  replacements={self._n_replacements}  "
            f"failures={self._n_failures}",
            f"  total_cost={self._total_cost:.2f}  "
            f"steps_since_replace={self._steps_since_replace}",
            "=" * 54,
        ]
        return "\n".join(lines)

    def close(self) -> None:
        pass


# ===========================================================================
# Factory helper
# ===========================================================================

def make_env_from_processed(
    processed_dir: str | Path = "data/processed",
    bearing_ids:   Optional[List[str]] = None,
    seed:          Optional[int] = None,
    render_mode:   Optional[str] = None,
) -> "PdMBearingEnv":
    """Load precomputed HI/RUL arrays and return a ready-to-use environment.

    Expects files named ``{bid}_hi.npy`` and ``{bid}_rul.npy`` inside
    *processed_dir*.  RUL values are divided by 125 to normalise to [0, 1].
    """
    processed_dir = Path(processed_dir)
    hi_seqs:  Dict[str, np.ndarray] = {}
    rul_seqs: Dict[str, np.ndarray] = {}

    candidates = sorted(processed_dir.glob("*_hi.npy"))
    if not candidates:
        raise FileNotFoundError(f"No *_hi.npy files found in {processed_dir}")

    for hi_path in candidates:
        bid = hi_path.stem.replace("_hi", "")
        if bearing_ids and bid not in bearing_ids:
            continue
        rul_path = processed_dir / f"{bid}_rul.npy"
        if not rul_path.exists():
            logger.warning("No RUL file for bearing %s; skipped.", bid)
            continue
        hi_seqs[bid]  = np.load(hi_path).astype(np.float32)
        rul_seqs[bid] = (np.load(rul_path) / 125.0).astype(np.float32)
        logger.info("Loaded bearing %s: HI%s RUL%s", bid,
                    hi_seqs[bid].shape, rul_seqs[bid].shape)

    if not hi_seqs:
        raise ValueError("No bearings loaded -- check bearing_ids and processed_dir.")

    return PdMBearingEnv(hi_seqs, rul_seqs, seed=seed, render_mode=render_mode)


# ===========================================================================
# Mandatory self-test -- runs at import time
# ===========================================================================

def _run_self_test() -> None:
    """Verify environment correctness using synthetic sequences.

    Checks reward values, termination behaviour, and the critical constraint
    that do_nothing is costlier than repair when the bearing is critical.
    Raises AssertionError if any check fails.
    """
    _G = "\033[32m"   # green
    _R = "\033[31m"   # red
    _E = "\033[0m"    # reset
    errors: List[str] = []

    def _ok(name: str, cond: bool, detail: str = "") -> None:
        tag = f"{_G}PASS{_E}" if cond else f"{_R}FAIL{_E}"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            errors.append(f"{name}: {detail}")

    print("=== PdMBearingEnv self-test ===")

    # --- synthetic data ---------------------------------------------------
    N = 300

    # Near-failure: hi starts at 0.15 and falls to 0
    hi_nf  = np.clip(np.linspace(0.15, -0.05, N), 0.0, 1.0).astype(np.float32)
    # Dying: 20 healthy steps then quick descent to 0 over 130 steps (fails ~step 140)
    hi_dy  = np.concatenate([
        np.full(20, 0.90, dtype=np.float32),
        np.linspace(0.90, 0.00, 130, dtype=np.float32),
    ])
    rul_mock = np.linspace(1.0, 0.0, N, dtype=np.float32)

    seqs_nf = {"nf": hi_nf, "healthy": np.ones(N, dtype=np.float32)}
    rul_nf  = {"nf": rul_mock, "healthy": rul_mock}
    rul_dy_arr = np.linspace(1.0, 0.0, len(hi_dy), dtype=np.float32)
    seqs_dy = {"dying": hi_dy}
    rul_dy  = {"dying": rul_dy_arr}

    # ------------------------------------------------------------------
    # Test 1 -- do_nothing from near-failure (hi~0.15): reward < -5 fires
    # ------------------------------------------------------------------
    env1 = PdMBearingEnv(seqs_nf, rul_nf, seed=0)
    env1.reset(options={"bearing_id": "nf"})
    rewards1: List[float] = []
    for _ in range(20):
        _, r, done, trunc, _ = env1.step(ACTION_DO_NOTHING)
        rewards1.append(r)
        if done or trunc:
            break
    _ok("T1: degradation penalty fires (r < -5) from hi~0.15",
        any(r < -5.0 for r in rewards1),
        f"first 5 rewards: {[round(v, 2) for v in rewards1[:5]]}")

    # ------------------------------------------------------------------
    # Test 2 -- pure do_nothing terminates via failure within 200 steps
    # ------------------------------------------------------------------
    env2 = PdMBearingEnv(seqs_dy, rul_dy, seed=0)
    obs2, _ = env2.reset()
    failed2 = False
    for _ in range(200):
        obs2, _, done2, trunc2, info2 = env2.step(ACTION_DO_NOTHING)
        if done2 and info2["is_failure"]:
            failed2 = True
            break
        if done2 or trunc2:
            failed2 = float(obs2[0]) < 0.05
            break
    _ok("T2: pure do_nothing terminates via failure <= 200 steps",
        failed2, f"hi={obs2[0]:.3f}")

    # ------------------------------------------------------------------
    # Test 3 -- replace every 50 steps: zero failures, cost > 0
    # ------------------------------------------------------------------
    env3 = PdMBearingEnv(seqs_dy, rul_dy, seed=1)
    env3.reset()
    n_fail3 = 0
    cost3   = 0.0
    for si in range(300):
        a = ACTION_REPLACE if (si % 50 == 49) else ACTION_DO_NOTHING
        _, _, done3, trunc3, info3 = env3.step(a)
        if info3["is_failure"]:
            n_fail3 += 1
        cost3 = info3["total_cost"]
        if done3 or trunc3:
            break
    _ok("T3: replace every 50 steps -> zero failures",
        n_fail3 == 0, f"n_failures={n_fail3}")
    _ok("T3: total_cost > 0",
        cost3 > 0.0, f"total_cost={cost3:.2f}")

    # ------------------------------------------------------------------
    # Test 4 -- reward values at three specific HI levels
    # ------------------------------------------------------------------
    def _env_at(hi_start: float) -> PdMBearingEnv:
        hi_arr = np.array(
            [hi_start, hi_start * 0.9, hi_start * 0.7,
             hi_start * 0.5, hi_start * 0.3, hi_start * 0.1],
            dtype=np.float32,
        )
        rl_arr = hi_arr.copy()
        return PdMBearingEnv({"t": hi_arr}, {"t": rl_arr}, seed=0)

    def _reward(e: PdMBearingEnv, action: int) -> float:
        e.reset(options={"bearing_id": "t"})
        _, r, _, _, _ = e.step(action)
        return r

    env09 = _env_at(0.9)
    env05 = _env_at(0.5)
    env01 = _env_at(0.1)

    r_dn_09  = _reward(env09, ACTION_DO_NOTHING)
    r_dn_05  = _reward(env05, ACTION_DO_NOTHING)
    r_dn_01  = _reward(env01, ACTION_DO_NOTHING)
    r_rep_09 = _reward(env09, ACTION_REPAIR)
    r_rep_01 = _reward(env01, ACTION_REPAIR)
    r_repl   = _reward(env09, ACTION_REPLACE)

    env04 = _env_at(0.4)
    r_dn_04 = _reward(env04, ACTION_DO_NOTHING)

    # Expected values (three-zone do_nothing reward):
    #   hi >= 0.5: step_cost = 0.1
    #   0.3 <= hi < 0.5: step_cost = 2.0
    #   hi < 0.3: step_cost = 0.1 + (0.5-hi)*20
    #   repair (1st)  -> -(4.0 + 1.5*1) = -5.5
    #   replace       -> -8.0

    _ok("T4a: do_nothing hi=0.9 -> -0.1",
        abs(r_dn_09 - (-0.1)) < 0.01, f"got {r_dn_09:.3f}")
    _ok("T4b: do_nothing hi=0.5 -> -0.1",
        abs(r_dn_05 - (-0.1)) < 0.01, f"got {r_dn_05:.3f}")
    _ok("T4b2: do_nothing hi=0.4 -> -2.0",
        abs(r_dn_04 - (-2.0)) < 0.01, f"got {r_dn_04:.3f}")
    _ok("T4c: do_nothing hi=0.1 -> -8.1",
        abs(r_dn_01 - (-8.1)) < 0.01, f"got {r_dn_01:.3f}")
    _ok("T4d: repair (1st) -> -5.5",
        abs(r_rep_09 - (-5.5)) < 0.01, f"got {r_rep_09:.3f}")
    _ok("T4e: replace -> -8.0",
        abs(r_repl - (-8.0)) < 0.01, f"got {r_repl:.3f}")

    # CRITICAL: at hi=0.1, do_nothing must be MORE negative than repair
    _ok("T4f CRITICAL: do_nothing(hi=0.1) < repair(hi=0.1)",
        r_dn_01 < r_rep_01,
        f"do_nothing={r_dn_01:.2f}  repair={r_rep_01:.2f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if errors:
        print(f"\n  {len(errors)} check(s) FAILED:")
        for e in errors:
            print(f"    - {e}")
        raise AssertionError(
            f"PdMBearingEnv self-test failed ({len(errors)} error(s)): {errors}"
        )
    print("  All checks passed.\n")


_run_self_test()
