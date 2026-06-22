"""
dueling_qrdqn.py
================
Dueling Distributional QR-DQN (D3QN-CVaR) — combines:
  - Value-advantage decomposition (Dueling DQN, Wang et al. 2016)
  - Quantile regression distributional RL (QR-DQN, Dabney et al. 2018)
  - CVaR risk aggregation for risk-averse action selection

Architecture
------------
  Input:  state_dim=3  [RUL_norm, sigma_norm, CFP]
  Shared: Linear(3,128)->LayerNorm->ReLU->Linear(128,128)->LayerNorm->ReLU
  Value:  Linear(128,64)->ReLU->Linear(64,N_quantiles)           -> (B, N)
  Adv:    Linear(128,64)->ReLU->Linear(64,A*N_quantiles)->reshape -> (B, A, N)
  Q(s,a)_i = V_i + A_i - mean_a(A_i)    [per-quantile dueling combination]
  Output: (B, n_actions, N_quantiles)

Public interface mirrors QRDQNAgent exactly so D3QNAgent drops into any
existing evaluate_policy() / evaluate_all_agents() harness without changes.
"""

from __future__ import annotations

import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.qrdqn_agent import ReplayBuffer
    from src.state_ablation import _ALL_BEARINGS, _precompute_mc_cache, make_ablation_env
    from src.rul_predictor import MAX_RUL
except ImportError:
    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from src.logging_config import setup_logger
    from src.device import get_device
    from src.qrdqn_agent import ReplayBuffer
    from src.state_ablation import _ALL_BEARINGS, _precompute_mc_cache, make_ablation_env
    from src.rul_predictor import MAX_RUL

logger = setup_logger(__name__)

_KAPPA: float = 1.0  # Huber threshold — matches QR-DQN convention


# ===========================================================================
# 1.  Environment wrapper
# ===========================================================================

class ThreeDStateEnv:
    """Factory + cache wrapper for StateAblationEnv(state_mode='C').

    Checks for a pre-built MC Dropout cache at
    ``<proc_dir>/mc_cache.pkl`` to avoid expensive recomputation (10-sample
    MC Dropout over all 6 bearings, already done by state_ablation.py).
    Saves the cache on first run so subsequent calls are instant.
    """

    _CACHE_FILE = "mc_cache.pkl"

    def __new__(
        cls,
        proc_dir: Path,
        rul_ckpt: Path,
        device_str: str = "cpu",
        seed: Optional[int] = None,
        n_mc: int = 10,
    ):
        """Return a StateAblationEnv configured for 3D state (mode C).

        Returns a StateAblationEnv instance — this class is a factory, not
        a subclass, so isinstance(env, ThreeDStateEnv) is False by design.
        Call ThreeDStateEnv.build(...) for the same effect with named args.
        """
        return cls.build(proc_dir, rul_ckpt, device_str=device_str, seed=seed, n_mc=n_mc)

    @staticmethod
    def build(
        proc_dir:   Path,
        rul_ckpt:   Path,
        device_str: str           = "cpu",
        seed:       Optional[int] = None,
        n_mc:       int           = 10,
    ):
        """Load or recompute MC Dropout cache; return StateAblationEnv(C)."""
        proc_dir   = Path(proc_dir)
        cache_path = proc_dir / ThreeDStateEnv._CACHE_FILE

        if cache_path.exists():
            logger.info(
                "MC cache found at %s — reusing (skipping MC Dropout recomputation).",
                cache_path,
            )
            with open(cache_path, "rb") as fh:
                mc_cache, sigma2_max = pickle.load(fh)
        else:
            logger.info(
                "No MC cache at %s — running MC Dropout inference (n_mc=%d) ...",
                cache_path, n_mc,
            )
            mc_cache, sigma2_max = _precompute_mc_cache(
                proc_dir,
                rul_ckpt,
                bearing_ids=_ALL_BEARINGS,
                n_mc=n_mc,
                device_str=device_str,
            )
            with open(cache_path, "wb") as fh:
                pickle.dump((mc_cache, sigma2_max), fh)
            logger.info("MC cache saved to %s.", cache_path)

        env = make_ablation_env(
            proc_dir, "C", mc_cache, sigma2_max, seed=seed
        )
        logger.info(
            "ThreeDStateEnv ready: obs_dim=%d  n_bearings=%d",
            env.observation_space.shape[0],
            len(mc_cache),
        )
        return env


# ===========================================================================
# 2.  Network
# ===========================================================================

class D3QNetwork(nn.Module):
    """Dueling distributional network — value + advantage streams over quantiles.

    Parameters
    ----------
    state_dim   : observation length (default 3 for [RUL_norm, sigma_norm, CFP]).
    n_actions   : discrete action count (default 3).
    N_quantiles : quantile atoms (default 51, matches QR-DQN convention).
    """

    def __init__(
        self,
        state_dim:   int = 3,
        n_actions:   int = 3,
        N_quantiles: int = 51,
    ) -> None:
        super().__init__()

        self.n_actions   = n_actions
        self.N_quantiles = N_quantiles

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_quantiles),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions * N_quantiles),
        )

        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for p in self.encoder.parameters())
        val   = sum(p.numel() for p in self.value_stream.parameters())
        adv   = sum(p.numel() for p in self.advantage_stream.parameters())
        logger.info(
            "D3QNetwork: total_params=%d  (encoder=%d  value=%d  advantage=%d)",
            total, enc, val, adv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, state_dim)

        Returns
        -------
        (batch, n_actions, N_quantiles)
        """
        h = self.encoder(x)                                  # (B, 128)

        v = self.value_stream(h)                             # (B, N)
        a = self.advantage_stream(h)                         # (B, A*N)
        a = a.view(-1, self.n_actions, self.N_quantiles)     # (B, A, N)

        # Per-quantile dueling combination
        # Q_i(s,a) = V_i(s) + A_i(s,a) - mean_a A_i(s,a)
        q = v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True) # (B, A, N)
        return q


# ===========================================================================
# 3.  Agent
# ===========================================================================

class D3QNAgent:
    """Dueling Distributional QR-DQN agent with CVaR risk aggregation.

    Public interface is identical to QRDQNAgent so this agent drops into
    any existing evaluate_policy() / evaluate_all_agents() call.
    """

    def __init__(
        self,
        state_dim:               int   = 3,
        n_actions:               int   = 3,
        N_quantiles:             int   = 51,
        lr:                      float = 5e-4,
        gamma:                   float = 0.99,
        epsilon_start:           float = 1.0,
        epsilon_end:             float = 0.10,
        epsilon_decay_episodes:  int   = 3000,
        target_update_freq:      int   = 200,
        batch_size:              int   = 128,
        cvar_alpha:              float = 0.40,
        risk_mode:               str   = "cvar",
        device:                  Optional[str] = None,
    ) -> None:

        if risk_mode not in ("mean", "cvar"):
            raise ValueError(f"risk_mode must be 'mean' or 'cvar', got {risk_mode!r}.")

        self.state_dim              = state_dim
        self.n_actions              = n_actions
        self.N_quantiles            = N_quantiles
        self.gamma                  = gamma
        self.epsilon_start          = epsilon_start
        self.epsilon_end            = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.current_episode        = 0
        self.epsilon                = epsilon_start
        self.target_update_freq     = target_update_freq
        self.batch_size             = batch_size
        self.risk_mode              = risk_mode
        self.cvar_alpha             = float(cvar_alpha)
        self._cvar_k = max(1, int(math.floor(self.cvar_alpha * N_quantiles)))

        self.device = torch.device(device if device is not None else get_device())

        self.online_net = D3QNetwork(state_dim, n_actions, N_quantiles).to(self.device)
        self.target_net = D3QNetwork(state_dim, n_actions, N_quantiles).to(self.device)
        self.hard_update_target()
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self._taus = torch.tensor(
            [(2 * i - 1) / (2 * N_quantiles) for i in range(1, N_quantiles + 1)],
            dtype=torch.float32,
            device=self.device,
        )

        self._env_steps: int = 0
        self._opt_steps: int = 0

    # ------------------------------------------------------------------
    # Epsilon decay (episode-based, identical to QRDQNAgent)
    # ------------------------------------------------------------------

    def step_episode(self) -> None:
        """Exponential epsilon decay — call exactly once per training episode."""
        self.current_episode += 1
        self.epsilon = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end)
            * math.exp(-self.current_episode / self.epsilon_decay_episodes)
        )

    # ------------------------------------------------------------------
    # Risk aggregation
    # ------------------------------------------------------------------

    def _aggregate_q(self, quantiles: torch.Tensor) -> torch.Tensor:
        """(B, A, N) -> (B, A) scalar Q-values under chosen risk mode."""
        if self.risk_mode == "mean":
            return quantiles.mean(dim=2)
        sorted_q = quantiles.sort(dim=2).values
        return sorted_q[:, :, : self._cvar_k].mean(dim=2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_q_stats(self, state: np.ndarray) -> Dict[str, Any]:
        """Per-action mean-Q and CVaR-Q for a single state (diagnostic)."""
        self.online_net.eval()
        with torch.no_grad():
            s      = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_dist = self.online_net(s)
            q_mean = q_dist.mean(dim=2).squeeze(0)
            sorted_q = q_dist.sort(dim=2).values
            q_cvar = sorted_q[:, :, : self._cvar_k].mean(dim=2).squeeze(0)
        return {
            "mean_q": [float(q_mean[i]) for i in range(self.n_actions)],
            "cvar_q": [float(q_cvar[i]) for i in range(self.n_actions)],
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: "np.ndarray | torch.Tensor",
        greedy: bool = False,
    ) -> int:
        """Epsilon-greedy with CVaR or mean aggregation over quantile distribution."""
        self._env_steps += 1

        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        self.online_net.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            s      = state.unsqueeze(0).to(self.device)
            q_dist = self.online_net(s)
            q_vals = self._aggregate_q(q_dist)
            action = int(q_vals.argmax(dim=1).item())

        if self._env_steps % 200 == 0:
            logger.debug(
                "select_action sample: step=%d  state=%s  q=%s  action=%d",
                self._env_steps,
                [round(float(x), 3) for x in state],
                [round(float(v), 3) for v in q_vals.squeeze(0).tolist()],
                action,
            )

        self.online_net.train()
        return action

    # ------------------------------------------------------------------
    # Gradient update (Quantile Huber loss — Dabney et al. 2018)
    # ------------------------------------------------------------------

    def update(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """One gradient step; returns scalar loss."""
        states, actions, rewards, next_states, dones = batch

        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,     dtype=torch.long).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        s2 = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)
        B  = s.shape[0]

        # Predicted quantiles for taken actions
        pred_all = self.online_net(s)                       # (B, A, N)
        pred_q   = pred_all[torch.arange(B), a]             # (B, N)

        with torch.no_grad():
            next_q_dist   = self.target_net(s2)             # (B, A, N)
            next_q_online = self._aggregate_q(self.online_net(s2))  # (B, A)
            best_a        = next_q_online.argmax(dim=1)     # (B,)
            best_q        = next_q_dist[torch.arange(B), best_a]    # (B, N)
            targets       = r.unsqueeze(1) + self.gamma * (1.0 - d.unsqueeze(1)) * best_q

        # Quantile Huber loss  rho_tau_i(u_ij)
        u      = targets.unsqueeze(1) - pred_q.unsqueeze(2)  # (B, N, N)
        huber  = torch.where(
            u.abs() < _KAPPA,
            0.5 * u ** 2,
            _KAPPA * (u.abs() - 0.5 * _KAPPA),
        )
        tau    = self._taus.view(1, -1, 1)                   # (1, N, 1)
        weights = (tau - (u.detach() < 0).float()).abs()     # (B, N, N)
        loss   = (weights * huber).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_freq == 0:
            self.hard_update_target()
            logger.debug("Target network updated at opt_step=%d.", self._opt_steps)

        loss_val = float(loss.item())
        logger.debug("update opt_step=%d loss=%.6f", self._opt_steps, loss_val)
        return loss_val

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: "str | Path") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "online_net":             self.online_net.state_dict(),
            "target_net":             self.target_net.state_dict(),
            "optimiser":              self.optimiser.state_dict(),
            "env_steps":              self._env_steps,
            "opt_steps":              self._opt_steps,
            "risk_mode":              self.risk_mode,
            "N_quantiles":            self.N_quantiles,
            "state_dim":              self.state_dim,
            "n_actions":              self.n_actions,
            "epsilon":                self.epsilon,
            "current_episode":        self.current_episode,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
            "cvar_alpha":             self.cvar_alpha,
            "architecture":           "dueling_distributional",
        }
        torch.save(meta, path)
        logger.info(
            "Saved D3QN checkpoint to %s  "
            "(env_steps=%d  opt_steps=%d  risk_mode=%s  architecture=%s)",
            path, self._env_steps, self._opt_steps,
            self.risk_mode, meta["architecture"],
        )

    def load_checkpoint(self, path: "str | Path") -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self._env_steps              = ckpt.get("env_steps",              0)
        self._opt_steps              = ckpt.get("opt_steps",              0)
        self.risk_mode               = ckpt.get("risk_mode",              self.risk_mode)
        self.epsilon                 = ckpt.get("epsilon",                self.epsilon_start)
        self.current_episode         = ckpt.get("current_episode",        0)
        self.epsilon_decay_episodes  = ckpt.get("epsilon_decay_episodes", self.epsilon_decay_episodes)
        if "cvar_alpha" in ckpt:
            self.cvar_alpha = float(ckpt["cvar_alpha"])
            self._cvar_k    = max(1, int(math.floor(self.cvar_alpha * self.N_quantiles)))
        arch = ckpt.get("architecture", "unknown")
        logger.info(
            "Loaded D3QN checkpoint from %s  "
            "(env_steps=%d  opt_steps=%d  risk_mode=%s  architecture=%s)",
            path, self._env_steps, self._opt_steps, self.risk_mode, arch,
        )
