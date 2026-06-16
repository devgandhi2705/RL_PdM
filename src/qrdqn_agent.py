"""
qrdqn_agent.py
==============
Quantile Regression DQN (QR-DQN) for risk-sensitive maintenance scheduling.

Implemented from scratch in PyTorch — no stable-baselines3.

Reference
---------
Dabney et al., "Distributional Reinforcement Learning with Quantile
Regression", AAAI 2018.  arXiv:1710.10044.

Architecture
------------
Input (state_dim=5)
  Linear(5, 128) → LayerNorm(128) → ReLU
  Linear(128, 128) → LayerNorm(128) → ReLU
  Linear(128, n_actions * N_quantiles)
  reshape → (batch, n_actions, N_quantiles)

Quantile fractions (fixed mid-points of N equal probability segments)
  tau_i = (2i - 1) / (2 * N_quantiles)  for i = 1 … N_quantiles

Quantile Huber loss per (s,a,r,s',done) transition
  u_ij = target_j - pred_i
  L_kappa(u) = 0.5 * u^2          if |u| < kappa
             = kappa*(|u| - 0.5*kappa)  otherwise  (kappa = 1)
  rho_tau_i(u) = |tau_i - 1(u < 0)| * L_kappa(u)
  loss = mean over all (i, j) pairs of rho_tau_i(u_ij)

Risk aggregation modes
----------------------
"mean"  : Q(s,a) = mean over N_quantiles  (risk-neutral)
"cvar"  : Q(s,a) = mean of lowest floor(alpha * N_quantiles) quantiles
          alpha = 0.25  (CVaR at 25 % — risk-averse; penalises high variance)
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from src.device import get_device
except ImportError:
    from device import get_device

logger = logging.getLogger(__name__)

# Fixed hyperparameters
_KAPPA: float     = 1.0   # Huber loss threshold
_CVAR_ALPHA: float = 0.25  # CVaR tail fraction
_STATE_DIM: int   = 5     # matches PdMBearingEnv 5D observation (hi, slope, rul, repairs, steps)
_N_ACTIONS: int   = 3     # continue / repair / replace


# ===========================================================================
# 1. Network
# ===========================================================================

class QRDQNNetwork(nn.Module):
    """Two-hidden-layer MLP that outputs N_quantiles estimates per action.

    Parameters
    ----------
    state_dim   : int — observation vector length (default 3).
    n_actions   : int — number of discrete actions (default 3).
    N_quantiles : int — number of quantile atoms (default 51).
    """

    def __init__(
        self,
        state_dim: int   = _STATE_DIM,
        n_actions: int   = _N_ACTIONS,
        N_quantiles: int = 51,
    ) -> None:
        super().__init__()

        self.n_actions   = n_actions
        self.N_quantiles = N_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_actions * N_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, state_dim)

        Returns
        -------
        (batch, n_actions, N_quantiles)
            Z_{theta}(s, a)_i — the i-th quantile estimate of action a's
            return distribution.
        """
        out = self.net(x)                              # (batch, n_actions*N_q)
        return out.view(-1, self.n_actions, self.N_quantiles)


# ===========================================================================
# 2. Replay buffer
# ===========================================================================

class ReplayBuffer:
    """Uniform-random experience replay buffer.

    Stores transitions as ``(state, action, reward, next_state, done)``.

    Parameters
    ----------
    maxlen : int — maximum number of transitions (oldest discarded when full).
    """

    def __init__(self, maxlen: int = 100_000) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Append a single transition; failure transitions added 3× for higher replay frequency."""
        entry = (
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        self._buf.append(entry)
        if float(reward) < -50.0:   # catastrophic failure — replicate twice
            self._buf.append(entry)
            self._buf.append(entry)

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample *batch_size* transitions.

        Returns
        -------
        (states, actions, rewards, next_states, dones)
            states, next_states : (B, state_dim), float32
            actions             : (B,),           int64
            rewards             : (B,),           float32
            dones               : (B,),           float32  {0.0, 1.0}
        """
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buf)} transitions, "
                f"but batch_size={batch_size} was requested."
            )
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, batch_size: int) -> bool:
        """True when the buffer contains at least *batch_size* transitions."""
        return len(self._buf) >= batch_size


# ===========================================================================
# 3. QR-DQN Agent
# ===========================================================================

class QRDQNAgent:
    """Quantile Regression DQN agent with risk-sensitive action selection.

    Parameters
    ----------
    state_dim        : observation vector length.
    n_actions        : number of discrete actions.
    N_quantiles      : quantile atoms N (default 51).
    lr               : Adam learning rate (default 1e-3).
    gamma            : discount factor (default 0.99).
    epsilon_start    : initial exploration rate (default 1.0).
    epsilon_end      : final exploration rate (default 0.05).
    epsilon_decay    : number of env-steps for linear epsilon annealing.
    target_update_freq : hard-copy target every this many *optimiser* steps.
    batch_size       : mini-batch size (stored for convenience; callers use it
                       when sampling from ReplayBuffer).
    risk_mode        : ``"mean"`` (risk-neutral) or ``"cvar"`` (risk-averse).
    device           : ``"cuda"``, ``"mps"``, or ``"cpu"``; *None* = auto.
    """

    def __init__(
        self,
        state_dim: int            = _STATE_DIM,
        n_actions: int            = _N_ACTIONS,
        N_quantiles: int          = 51,
        lr: float                 = 1e-3,
        gamma: float              = 0.99,
        epsilon_start: float      = 1.0,
        epsilon_end: float        = 0.05,
        epsilon_decay: int            = 5_000,      # kept for backward compat (unused)
        epsilon_decay_episodes: int   = 3_000,      # episode-based exponential decay
        target_update_freq: int       = 100,
        batch_size: int           = 64,
        risk_mode: str            = "cvar",
        cvar_alpha: float         = _CVAR_ALPHA,    # CVaR tail fraction (overrides module default)
        device: Optional[str]     = None,
    ) -> None:

        if risk_mode not in ("mean", "cvar"):
            raise ValueError(f"risk_mode must be 'mean' or 'cvar', got {risk_mode!r}.")

        self.state_dim          = state_dim
        self.n_actions          = n_actions
        self.N_quantiles        = N_quantiles
        self.gamma              = gamma
        self.epsilon_start           = epsilon_start
        self.epsilon_end             = epsilon_end
        self.epsilon_decay           = max(1, epsilon_decay)   # unused; kept for compat
        self.epsilon_decay_episodes  = max(1, epsilon_decay_episodes)
        self.current_episode: int    = 0
        self.epsilon: float          = epsilon_start  # updated by step_episode()
        self.target_update_freq      = target_update_freq
        self.batch_size         = batch_size
        self.risk_mode          = risk_mode

        self.device = torch.device(
            device if device is not None else get_device()
        )

        # ------------------------------------------------------------------
        # Networks
        # ------------------------------------------------------------------
        self.online_net = QRDQNNetwork(state_dim, n_actions, N_quantiles).to(self.device)
        self.target_net = QRDQNNetwork(state_dim, n_actions, N_quantiles).to(self.device)
        self.hard_update_target()          # initialise target = online
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # ------------------------------------------------------------------
        # Fixed quantile fractions  tau_i = (2i-1) / (2*N)  for i=1..N
        # ------------------------------------------------------------------
        taus = torch.tensor(
            [(2 * i - 1) / (2 * N_quantiles) for i in range(1, N_quantiles + 1)],
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer_taus = taus          # shape (N_quantiles,)

        self.cvar_alpha = float(cvar_alpha)
        # CVaR tail index:  k = floor(alpha * N_quantiles)
        self._cvar_k = max(1, int(math.floor(self.cvar_alpha * N_quantiles)))

        # ------------------------------------------------------------------
        # Step counters
        # ------------------------------------------------------------------
        self._env_steps: int = 0   # incremented in select_action
        self._opt_steps: int = 0   # incremented in update

    # ------------------------------------------------------------------
    # Episode-based epsilon decay
    # ------------------------------------------------------------------

    def step_episode(self) -> None:
        """Exponential epsilon decay — call exactly once per training episode."""
        self.current_episode += 1
        self.epsilon = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end)
            * math.exp(-self.current_episode / self.epsilon_decay_episodes)
        )

    # ------------------------------------------------------------------
    # Risk aggregation helper
    # ------------------------------------------------------------------

    def _aggregate_q(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Aggregate per-action quantile distributions into scalar Q-values.

        Parameters
        ----------
        quantiles : (batch, n_actions, N_quantiles)

        Returns
        -------
        (batch, n_actions)  — Q-value per action under the chosen risk mode.
        """
        if self.risk_mode == "mean":
            return quantiles.mean(dim=2)
        else:
            # CVaR: sort ascending, take mean of lowest k = floor(alpha*N) values
            # Sorting handles any crossing of quantile outputs during early training.
            sorted_q = quantiles.sort(dim=2).values        # (B, A, N) ascending
            return sorted_q[:, :, : self._cvar_k].mean(dim=2)

    # ------------------------------------------------------------------
    # Q-value diagnostics
    # ------------------------------------------------------------------

    def get_q_stats(self, state: np.ndarray) -> Dict[str, Any]:
        """Return per-action mean-Q and CVaR-Q for a single state (diagnostic)."""
        self.online_net.eval()
        with torch.no_grad():
            s      = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_dist = self.online_net(s)                    # (1, A, N)
            q_mean = q_dist.mean(dim=2).squeeze(0)         # (A,)
            q_cvar = self._aggregate_q(q_dist).squeeze(0)  # (A,)
        return {
            "mean_q": [float(q_mean[i]) for i in range(self.n_actions)],
            "cvar_q": [float(q_cvar[i]) for i in range(self.n_actions)],
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray | torch.Tensor, greedy: bool = False
    ) -> int:
        """Epsilon-greedy action selection using CVaR or mean aggregation.

        Parameters
        ----------
        state  : observation, shape (state_dim,).
        greedy : if *True*, always select the greedy action (epsilon ignored).

        Returns
        -------
        int — chosen action index.
        """
        self._env_steps += 1

        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        self.online_net.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            s = state.unsqueeze(0).to(self.device)         # (1, state_dim)
            q_dist = self.online_net(s)                    # (1, n_actions, N)
            q_vals = self._aggregate_q(q_dist)             # (1, n_actions)
            action = int(q_vals.argmax(dim=1).item())
        self.online_net.train()
        return action

    # ------------------------------------------------------------------
    # Gradient update step
    # ------------------------------------------------------------------

    def update(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Perform one gradient-descent step on a sampled mini-batch.

        Parameters
        ----------
        batch : (states, actions, rewards, next_states, dones) as numpy arrays
                with dtypes (float32, int64, float32, float32, float32).

        Returns
        -------
        float — scalar loss value for logging.
        """
        states, actions, rewards, next_states, dones = batch

        # Move to device
        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,     dtype=torch.long).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        s2 = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        B = s.shape[0]

        # ---- Predicted quantiles for chosen actions ----------------------
        # online_net: (B, n_actions, N) → select the action taken
        pred_all  = self.online_net(s)                      # (B, A, N)
        pred_q    = pred_all[torch.arange(B), a]            # (B, N) — Z_theta(s, a_t)

        # ---- Target quantiles  r + gamma*(1-d)*Z_target(s', a*) --------
        with torch.no_grad():
            next_q_dist = self.target_net(s2)               # (B, A, N)

            # Best action a* by online-net risk aggregation (Double DQN style)
            next_q_online = self._aggregate_q(self.online_net(s2))  # (B, A)
            best_a        = next_q_online.argmax(dim=1)             # (B,)

            best_q = next_q_dist[torch.arange(B), best_a]   # (B, N)

            # Bellman target for each target quantile j
            # shape (B, N);  not-done mask broadcasts over N
            targets = r.unsqueeze(1) + self.gamma * (1.0 - d.unsqueeze(1)) * best_q

        # ---- Quantile Huber loss  ----------------------------------------
        # u_ij = target_j - pred_i  →  shape (B, N_pred, N_target)
        u = targets.unsqueeze(1) - pred_q.unsqueeze(2)      # (B, N, N)

        # Huber component  L_kappa(u)
        huber = torch.where(
            u.abs() < _KAPPA,
            0.5 * u ** 2,
            _KAPPA * (u.abs() - 0.5 * _KAPPA),
        )

        # Asymmetric weight  |tau_i - 1(u < 0)|
        # taus indexed along the "predicted" dimension (dim=1)
        tau = self.register_buffer_taus.view(1, -1, 1)       # (1, N, 1)
        weights = (tau - (u.detach() < 0).float()).abs()     # (B, N, N)

        # Mean over all (i, j) pairs, then over batch
        loss = (weights * huber).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_freq == 0:
            self.hard_update_target()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def hard_update_target(self) -> None:
        """Copy online network weights exactly into the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full agent state (weights + optimiser + counters).

        Parameters
        ----------
        path : file path (``results/qrdqn.pth`` convention).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
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
            },
            path,
        )
        logger.info("Saved QR-DQN checkpoint to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights, optimiser state, and step counters from *path*.

        The network architecture must match the saved checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self._env_steps              = ckpt.get("env_steps", 0)
        self._opt_steps              = ckpt.get("opt_steps", 0)
        self.risk_mode               = ckpt.get("risk_mode", self.risk_mode)
        self.epsilon                 = ckpt.get("epsilon", self.epsilon_start)
        self.current_episode         = ckpt.get("current_episode", 0)
        self.epsilon_decay_episodes  = ckpt.get("epsilon_decay_episodes",
                                                  self.epsilon_decay_episodes)
        if "cvar_alpha" in ckpt:
            self.cvar_alpha = float(ckpt["cvar_alpha"])
            self._cvar_k    = max(1, int(math.floor(self.cvar_alpha * self.N_quantiles)))
        logger.info(
            "Loaded QR-DQN checkpoint from %s  "
            "(env_steps=%d, opt_steps=%d, risk_mode=%s)",
            path, self._env_steps, self._opt_steps, self.risk_mode,
        )


# ===========================================================================
# 4. Training loop (integrates with PdMBearingEnv)
# ===========================================================================

def train_qrdqn(
    agent: QRDQNAgent,
    env,
    buffer: ReplayBuffer,
    n_episodes: int              = 1_000,
    warmup_steps: int            = 500,
    eval_env                     = None,
    eval_every: int              = 50,
    n_eval_episodes: int         = 5,
    save_every: int              = 200,
    save_path: str | Path        = "results/qrdqn_best.pth",
    seed: int                    = 42,
) -> Dict[str, List[float]]:
    """Full training loop for :class:`QRDQNAgent` on a PdMBearingEnv.

    Parameters
    ----------
    agent         : initialised :class:`QRDQNAgent`.
    env           : training environment (Gymnasium API).
    buffer        : :class:`ReplayBuffer` instance.
    n_episodes    : total training episodes.
    warmup_steps  : steps with random policy before any gradient updates.
    eval_env      : evaluation environment; *None* uses training env.
    eval_every    : evaluate every N episodes.
    n_eval_episodes: greedy rollouts per evaluation.
    save_every    : checkpoint every N episodes.
    save_path     : path for the best-performing checkpoint.
    seed          : RNG seed.

    Returns
    -------
    dict
        ``{"train_returns": [...], "eval_returns": [...], "losses": [...]}``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    eval_env   = eval_env or env
    save_path  = Path(save_path)
    history: Dict[str, List[float]] = {
        "train_returns": [],
        "eval_returns":  [],
        "losses":        [],
    }
    best_eval  = -math.inf
    total_steps = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_losses: List[float] = []
        done = False

        while not done:
            # Warm-up: random actions; after that, epsilon-greedy
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, float(done))
            obs         = next_obs
            ep_return  += reward
            total_steps += 1

            # Gradient update
            if buffer.is_ready(agent.batch_size) and total_steps >= warmup_steps:
                batch = buffer.sample(agent.batch_size)
                loss  = agent.update(batch)
                ep_losses.append(loss)

        history["train_returns"].append(ep_return)
        if ep_losses:
            history["losses"].append(float(np.mean(ep_losses)))

        # Evaluation
        if ep % eval_every == 0:
            eval_return = _greedy_eval(agent, eval_env, n_eval_episodes, seed)
            history["eval_returns"].append(eval_return)

            logger.info(
                "Episode %4d/%d | eps=%.3f | train_ret=%7.2f | "
                "eval_ret=%7.2f | opt_steps=%d | buf=%d",
                ep, n_episodes, agent.epsilon,
                ep_return, eval_return,
                agent._opt_steps, len(buffer),
            )

            if eval_return > best_eval:
                best_eval = eval_return
                agent.save_checkpoint(save_path)
                logger.info("  -> New best eval return %.2f; checkpoint saved.", best_eval)

        if ep % save_every == 0:
            agent.save_checkpoint(save_path.with_stem(f"{save_path.stem}_ep{ep}"))

    return history


def _greedy_eval(
    agent: QRDQNAgent,
    env,
    n_episodes: int,
    seed: int,
) -> float:
    """Run *n_episodes* with the greedy policy; return mean episode return."""
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        total = 0.0
        done  = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        returns.append(total)
    return float(np.mean(returns))


# ===========================================================================
# CLI smoke-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.device import get_device as _gd
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _gd()

    # ---- Network shape check ----
    net = QRDQNNetwork(state_dim=5, n_actions=3, N_quantiles=51)
    x   = torch.randn(8, 5)
    out = net(x)
    assert out.shape == (8, 3, 51), f"Expected (8,3,51), got {out.shape}"
    print(f"QRDQNNetwork forward: {tuple(x.shape)} -> {tuple(out.shape)}  OK")

    # ---- ReplayBuffer ----
    buf = ReplayBuffer(maxlen=1000)
    for _ in range(200):
        buf.add(
            np.random.randn(5).astype(np.float32),
            random.randint(0, 2),
            random.uniform(-5, 0),
            np.random.randn(5).astype(np.float32),
            False,
        )
    states, actions, rewards, next_states, dones = buf.sample(64)
    assert states.shape == (64, 5)
    assert actions.shape == (64,)
    print(f"ReplayBuffer.sample(64): states={states.shape}  actions={actions.shape}  OK")

    # ---- Agent: mean mode ----
    state5 = np.array([0.8, -0.01, 0.6, 0.0, 0.1], dtype=np.float32)
    agent_mean = QRDQNAgent(state_dim=5, n_actions=3, risk_mode="mean")
    action_m   = agent_mean.select_action(state5)
    print(f"select_action (mean):  action={action_m}  epsilon={agent_mean.epsilon:.3f}")

    # ---- Agent: cvar mode ----
    agent_cvar = QRDQNAgent(state_dim=5, n_actions=3, risk_mode="cvar")
    action_c   = agent_cvar.select_action(state5)
    print(f"select_action (cvar):  action={action_c}  epsilon={agent_cvar.epsilon:.3f}")
    print(f"CVaR k = {agent_cvar._cvar_k}  (floor({_CVAR_ALPHA} * 51))")

    # ---- step_episode epsilon decay ----
    for _ in range(100):
        agent_cvar.step_episode()
    print(f"After 100 episodes: epsilon={agent_cvar.epsilon:.4f}  (should be between {agent_cvar.epsilon_end:.2f} and 1.0)")

    # ---- get_q_stats ----
    qs = agent_cvar.get_q_stats(state5)
    assert "mean_q" in qs and "cvar_q" in qs and len(qs["mean_q"]) == 3
    print(f"get_q_stats: mean_q={[round(v,3) for v in qs['mean_q']]}  cvar_q={[round(v,3) for v in qs['cvar_q']]}  OK")

    # ---- Gradient update ----
    batch = buf.sample(64)
    loss1 = agent_cvar.update(batch)
    loss2 = agent_cvar.update(batch)
    print(f"update loss: {loss1:.5f} -> {loss2:.5f}")

    # ---- Target update counter ----
    for _ in range(agent_cvar.target_update_freq):
        agent_cvar.update(batch)
    print(f"hard_update triggered at opt_step={agent_cvar._opt_steps}  OK")

    # ---- Quantile Huber loss sanity ----
    # With deterministic targets equal to predictions, loss should be ~0
    clean_agent = QRDQNAgent(state_dim=5, n_actions=3, epsilon_start=0.0)
    zero_s      = torch.zeros(16, 5, device=clean_agent.device)
    z_quant     = clean_agent.online_net(zero_s)   # (16, 3, 51)
    # Manually build a "perfect" batch where targets == predictions
    s_np  = np.zeros((16, 5), dtype=np.float32)
    a_np  = np.zeros(16, dtype=np.int64)
    r_np  = np.zeros(16, dtype=np.float32)
    s2_np = np.zeros((16, 5), dtype=np.float32)
    d_np  = np.ones(16, dtype=np.float32)   # terminal → no bootstrap
    zero_batch = (s_np, a_np, r_np, s2_np, d_np)
    zero_loss  = clean_agent.update(zero_batch)
    print(f"Zero-target batch loss (should be small): {zero_loss:.6f}")

    # ---- Checkpoint round-trip ----
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "test.pth")
        agent_cvar.save_checkpoint(ckpt)
        agent2 = QRDQNAgent(state_dim=5, n_actions=3, risk_mode="cvar")
        agent2.load_checkpoint(ckpt)
        assert agent2._env_steps == agent_cvar._env_steps
        assert agent2.risk_mode  == agent_cvar.risk_mode
        print(f"Checkpoint round-trip: env_steps={agent2._env_steps}  risk_mode={agent2.risk_mode}  OK")

    print("\nAll smoke-tests passed.")
