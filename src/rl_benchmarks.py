"""
rl_benchmarks.py
================
Three additional RL agents (Double DQN, Dueling DQN, PPO) with a unified
comparison evaluation against the existing QR-DQN and rule-based policies.

Comparison rows
---------------
ThresholdPolicy | DDQN | Dueling DQN | PPO | Risk-Neutral DQN | CVaR QR-DQN (ours)

Usage
-----
    python -m src.rl_benchmarks          # train new agents + evaluate all + generate outputs
    python -m src.rl_benchmarks --eval-only   # skip training (load existing checkpoints)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from src.device import get_device
    from src.qrdqn_agent import ReplayBuffer, QRDQNAgent
    from src.baselines import ThresholdPolicy, evaluate_policy
    from src.rl_environment import make_env_from_processed
except ImportError:
    from device import get_device          # type: ignore[no-redef]
    from qrdqn_agent import ReplayBuffer, QRDQNAgent   # type: ignore[no-redef]
    from baselines import ThresholdPolicy, evaluate_policy  # type: ignore[no-redef]
    from rl_environment import make_env_from_processed  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_STATE_DIM = 5
_N_ACTIONS = 3
_GAMMA     = 0.99

# Colour / linestyle palette (colour-blind-safe, IEEE style)
_AGENT_COLORS = {
    "DDQN":                "#E69F00",
    "Dueling DQN":         "#56B4E9",
    "PPO":                 "#009E73",
    "Risk-Neutral DQN":    "#CC79A7",
    "CVaR QR-DQN (ours)": "#0072B2",
    "ThresholdPolicy":     "#999999",
}
_AGENT_LS = {
    "DDQN":                "dashed",
    "Dueling DQN":         "dotted",
    "PPO":                 "dashdot",
    "Risk-Neutral DQN":    "dashed",
    "CVaR QR-DQN (ours)": "solid",
}
_ROW_ORDER = [
    "ThresholdPolicy",
    "DDQN",
    "Dueling DQN",
    "PPO",
    "Risk-Neutral DQN",
    "CVaR QR-DQN (ours)",
]


# ===========================================================================
# 0. BaseAgent ABC
# ===========================================================================

class BaseAgent(ABC):
    """Minimal interface all benchmark agents must satisfy."""

    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Return a discrete action. epsilon=0.0 -> greedy."""
        ...

    @abstractmethod
    def update(self, batch: Any) -> float:
        """Gradient update step. Returns scalar loss."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None: ...


# ===========================================================================
# 1. Double DQN (DDQN)
# ===========================================================================

class _DQNNetwork(nn.Module):
    """Two-hidden-layer MLP outputting Q(s,a) for all actions."""

    def __init__(self, state_dim: int = _STATE_DIM, n_actions: int = _N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, n_actions)


class DDQNAgent(BaseAgent):
    """Double DQN: online net selects action, target net evaluates it.

    Hyperparameters mirror QRDQNAgent (same epsilon schedule, same
    target-update frequency, same batch size) so comparison is apples-to-apples.
    """

    def __init__(
        self,
        state_dim: int               = _STATE_DIM,
        n_actions: int               = _N_ACTIONS,
        lr: float                    = 1e-3,
        gamma: float                 = _GAMMA,
        epsilon_start: float         = 1.0,
        epsilon_end: float           = 0.05,
        epsilon_decay_episodes: int  = 3_000,
        target_update_freq: int      = 100,
        batch_size: int              = 64,
        device: Optional[str]        = None,
    ) -> None:
        self.state_dim              = state_dim
        self.n_actions              = n_actions
        self.gamma                  = gamma
        self.epsilon_start          = epsilon_start
        self.epsilon_end            = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.current_episode: int   = 0
        self.epsilon: float         = epsilon_start
        self.target_update_freq     = target_update_freq
        self.batch_size             = batch_size

        self.device = torch.device(device if device is not None else get_device())

        self.online_net = _DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = _DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser  = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self._opt_steps = 0

    def step_episode(self) -> None:
        """Exponential epsilon decay — call once per episode."""
        self.current_episode += 1
        self.epsilon = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end)
            * math.exp(-self.current_episode / self.epsilon_decay_episodes)
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        self.online_net.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = int(self.online_net(s).argmax(dim=1).item())
        self.online_net.train()
        return action

    def update(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        states, actions, rewards, next_states, dones = batch
        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,     dtype=torch.long).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        s2 = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        q_pred = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects best next action, target evaluates it
            a_star = self.online_net(s2).argmax(dim=1)
            q_next = self.target_net(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * (1.0 - d) * q_next

        loss = F.huber_loss(q_pred, target)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net":             self.online_net.state_dict(),
            "target_net":             self.target_net.state_dict(),
            "optimiser":              self.optimiser.state_dict(),
            "opt_steps":              self._opt_steps,
            "epsilon":                self.epsilon,
            "current_episode":        self.current_episode,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
        }, path)
        logger.info("Saved DDQN checkpoint -> %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self._opt_steps              = ckpt.get("opt_steps", 0)
        self.epsilon                 = ckpt.get("epsilon", self.epsilon_start)
        self.current_episode         = ckpt.get("current_episode", 0)
        self.epsilon_decay_episodes  = ckpt.get("epsilon_decay_episodes",
                                                  self.epsilon_decay_episodes)
        logger.info("Loaded DDQN checkpoint <- %s", path)


# ===========================================================================
# 2. Dueling DQN
# ===========================================================================

class _DuelingNetwork(nn.Module):
    """Shared trunk + separate Value and Advantage streams."""

    def __init__(self, state_dim: int = _STATE_DIM, n_actions: int = _N_ACTIONS) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
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
            nn.Linear(64, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.trunk(x)
        v   = self.value_stream(h)                       # (B, 1)
        adv = self.adv_stream(h)                         # (B, n_actions)
        return v + adv - adv.mean(dim=1, keepdim=True)  # Q(s,a) = V(s) + A(s,a) - mean_a[A]


class DuelingDQNAgent(BaseAgent):
    """Dueling DQN with Double DQN targets and episode-based epsilon decay."""

    def __init__(
        self,
        state_dim: int               = _STATE_DIM,
        n_actions: int               = _N_ACTIONS,
        lr: float                    = 1e-3,
        gamma: float                 = _GAMMA,
        epsilon_start: float         = 1.0,
        epsilon_end: float           = 0.05,
        epsilon_decay_episodes: int  = 3_000,
        target_update_freq: int      = 100,
        batch_size: int              = 64,
        device: Optional[str]        = None,
    ) -> None:
        self.state_dim              = state_dim
        self.n_actions              = n_actions
        self.gamma                  = gamma
        self.epsilon_start          = epsilon_start
        self.epsilon_end            = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.current_episode: int   = 0
        self.epsilon: float         = epsilon_start
        self.target_update_freq     = target_update_freq
        self.batch_size             = batch_size

        self.device = torch.device(device if device is not None else get_device())

        self.online_net = _DuelingNetwork(state_dim, n_actions).to(self.device)
        self.target_net = _DuelingNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser  = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self._opt_steps = 0

    def step_episode(self) -> None:
        self.current_episode += 1
        self.epsilon = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end)
            * math.exp(-self.current_episode / self.epsilon_decay_episodes)
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        self.online_net.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = int(self.online_net(s).argmax(dim=1).item())
        self.online_net.train()
        return action

    def update(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        states, actions, rewards, next_states, dones = batch
        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,     dtype=torch.long).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        s2 = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        q_pred = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            a_star = self.online_net(s2).argmax(dim=1)
            q_next = self.target_net(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * (1.0 - d) * q_next

        loss = F.huber_loss(q_pred, target)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net":             self.online_net.state_dict(),
            "target_net":             self.target_net.state_dict(),
            "optimiser":              self.optimiser.state_dict(),
            "opt_steps":              self._opt_steps,
            "epsilon":                self.epsilon,
            "current_episode":        self.current_episode,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
        }, path)
        logger.info("Saved DuelingDQN checkpoint -> %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self._opt_steps              = ckpt.get("opt_steps", 0)
        self.epsilon                 = ckpt.get("epsilon", self.epsilon_start)
        self.current_episode         = ckpt.get("current_episode", 0)
        self.epsilon_decay_episodes  = ckpt.get("epsilon_decay_episodes",
                                                  self.epsilon_decay_episodes)
        logger.info("Loaded DuelingDQN checkpoint <- %s", path)


# ===========================================================================
# 3. PPO
# ===========================================================================

class _PPOActor(nn.Module):
    def __init__(self, state_dim: int = _STATE_DIM, n_actions: int = _N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class _PPOCritic(nn.Module):
    def __init__(self, state_dim: int = _STATE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _PPORolloutBuffer:
    """Fixed-length rollout buffer for one PPO update cycle."""

    def __init__(self, rollout_steps: int, state_dim: int) -> None:
        self.rollout_steps = rollout_steps
        self._ptr          = 0
        self.states    = np.zeros((rollout_steps, state_dim), dtype=np.float32)
        self.actions   = np.zeros(rollout_steps, dtype=np.int64)
        self.rewards   = np.zeros(rollout_steps, dtype=np.float32)
        self.values    = np.zeros(rollout_steps, dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.dones     = np.zeros(rollout_steps, dtype=np.float32)
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns    = np.zeros(rollout_steps, dtype=np.float32)

    def add(
        self, state: np.ndarray, action: int,
        reward: float, value: float, log_prob: float, done: float,
    ) -> None:
        i = self._ptr
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = done
        self._ptr         = i + 1

    def is_ready(self) -> bool:
        return self._ptr >= self.rollout_steps

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95,
    ) -> None:
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            mask       = 1.0 - self.dones[t]
            next_val   = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta      = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae        = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_tensors(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s   = torch.tensor(self.states,     dtype=torch.float32).to(device)
        a   = torch.tensor(self.actions,    dtype=torch.long).to(device)
        lp  = torch.tensor(self.log_probs,  dtype=torch.float32).to(device)
        adv = torch.tensor(self.advantages, dtype=torch.float32).to(device)
        ret = torch.tensor(self.returns,    dtype=torch.float32).to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return s, a, lp, adv, ret

    def reset(self) -> None:
        self._ptr = 0


class PPOAgent(BaseAgent):
    """Proximal Policy Optimisation with GAE and clipped surrogate objective.

    select_action() returns the argmax (most-probable) action and is safe to
    call during evaluation.  Training uses get_action_and_value() to sample
    stochastically and collect log-probabilities.
    """

    def __init__(
        self,
        state_dim: int       = _STATE_DIM,
        n_actions: int       = _N_ACTIONS,
        actor_lr: float      = 3e-4,
        critic_lr: float     = 1e-3,
        gamma: float         = _GAMMA,
        gae_lambda: float    = 0.95,
        clip_ratio: float    = 0.2,
        ent_coef: float      = 0.01,
        ppo_epochs: int      = 4,
        mini_batch_size: int = 64,
        rollout_steps: int   = 2048,
        device: Optional[str] = None,
    ) -> None:
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_ratio     = clip_ratio
        self.ent_coef       = ent_coef
        self.ppo_epochs     = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_steps  = rollout_steps
        self.n_actions      = n_actions

        self.device = torch.device(device if device is not None else get_device())

        self.actor  = _PPOActor(state_dim, n_actions).to(self.device)
        self.critic = _PPOCritic(state_dim).to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = _PPORolloutBuffer(rollout_steps, state_dim)

    # --- action helpers ---------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Return greedy (argmax) action — used during evaluation."""
        self.actor.eval()
        with torch.no_grad():
            s      = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs  = self.actor(s)
            action = int(probs.argmax(dim=1).item())
        return action

    def get_action_and_value(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        """Sample action from actor; return (action, log_prob, value)."""
        self.actor.train()
        self.critic.train()
        with torch.no_grad():
            s     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs = self.actor(s)
            val   = self.critic(s).detach()
            dist  = Categorical(probs)
            act   = dist.sample()
            lp    = dist.log_prob(act)
        return int(act.item()), float(lp.item()), float(val.item())

    def get_value(self, state: np.ndarray) -> float:
        self.critic.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            v = self.critic(s)
        return float(v.item())

    # --- update -----------------------------------------------------------

    def update(self, batch: Any = None) -> float:
        """Run ppo_epochs of mini-batch updates on the full rollout buffer."""
        if not self.buffer.is_ready():
            return 0.0

        s, a, old_lp, adv, ret = self.buffer.get_tensors(self.device)
        n_steps    = self.buffer.rollout_steps   # actual buffer size (may differ from self.rollout_steps in tests)
        total_loss = 0.0
        n_updates  = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_steps, device=self.device)
            for start in range(0, n_steps, self.mini_batch_size):
                idx = perm[start: start + self.mini_batch_size]
                sb, ab, old_lpb, advb, retb = (
                    s[idx], a[idx], old_lp[idx], adv[idx], ret[idx]
                )

                probs   = self.actor(sb)
                dist    = Categorical(probs)
                new_lp  = dist.log_prob(ab)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lpb)
                surr1 = ratio * advb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                           1.0 + self.clip_ratio) * advb
                actor_loss  = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy
                critic_loss = F.mse_loss(self.critic(sb), retb)

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_opt.step()

                total_loss += float(actor_loss.item()) + float(critic_loss.item())
                n_updates  += 1

        self.buffer.reset()
        return total_loss / max(1, n_updates)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)
        logger.info("Saved PPO checkpoint -> %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        logger.info("Loaded PPO checkpoint <- %s", path)


# ===========================================================================
# 4. Training helpers
# ===========================================================================

def _smooth(x: List[float], window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean. Returns (x_coords, smoothed_values)."""
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) < window:
        return np.arange(len(arr)), arr
    kernel = np.ones(window, dtype=np.float64) / window
    sm     = np.convolve(arr, kernel, mode="valid")
    xs     = np.arange(window - 1, len(arr))
    return xs, sm


def _greedy_eval_dqn(
    agent: DDQNAgent | DuelingDQNAgent,
    env,
    n_episodes: int,
    seed: int,
) -> float:
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        total, done = 0.0, False
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        returns.append(total)
    return float(np.mean(returns))


def _greedy_eval_ppo(agent: PPOAgent, env, n_episodes: int, seed: int) -> float:
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        total, done = 0.0, False
        while not done:
            action = agent.select_action(obs)   # uses argmax
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        returns.append(total)
    return float(np.mean(returns))


# ===========================================================================
# 5. DQN training loop (shared by DDQN and Dueling DQN)
# ===========================================================================

def _train_dqn_agent(
    agent: DDQNAgent | DuelingDQNAgent,
    env,
    n_episodes: int         = 5_000,
    warmup_episodes: int    = 100,
    buffer_maxlen: int      = 100_000,
    save_path: str | Path   = "results/agent_best.pth",
    eval_every: int         = 100,
    n_eval_episodes: int    = 10,
    seed: int               = 42,
    label: str              = "DQN",
) -> Dict[str, List[float]]:
    """Replay-buffer training with 100-episode warmup using action cycling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    buffer    = ReplayBuffer(maxlen=buffer_maxlen)
    save_path = Path(save_path)
    history: Dict[str, List[float]] = {
        "train_returns": [], "eval_returns": [], "losses": [],
    }
    best_eval = -math.inf

    # --- Warmup: forced 0-1-2-0-1-2... cycling to pre-fill buffer --------
    print(f"  [{label}] Warmup: {warmup_episodes} episodes with action cycling ...")
    cycle_step = 0
    for ep in range(warmup_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        while not done:
            action           = cycle_step % _N_ACTIONS
            cycle_step      += 1
            next_obs, r, terminated, truncated, _ = env.step(action)
            done             = terminated or truncated
            buffer.add(obs, action, r, next_obs, float(done))
            obs              = next_obs
    print(f"  [{label}] Buffer filled with {len(buffer)} transitions. Training ...")

    # --- Main training loop -----------------------------------------------
    for ep in range(1, n_episodes + 1):
        force_deg = (random.random() < 0.40)
        obs, _    = env.reset(seed=seed + ep, force_degraded=force_deg)
        ep_return = 0.0
        ep_losses: List[float] = []
        done      = False

        while not done:
            action = agent.select_action(obs, epsilon=agent.epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))
            obs        = next_obs
            ep_return += reward

            if buffer.is_ready(agent.batch_size):
                batch = buffer.sample(agent.batch_size)
                loss  = agent.update(batch)
                ep_losses.append(loss)

        agent.step_episode()
        history["train_returns"].append(ep_return)
        if ep_losses:
            history["losses"].append(float(np.mean(ep_losses)))

        if ep % eval_every == 0:
            eval_ret = _greedy_eval_dqn(agent, env, n_eval_episodes, seed)
            history["eval_returns"].append(eval_ret)
            print(
                f"  [{label}] ep={ep:5d}/{n_episodes}"
                f"  eps={agent.epsilon:.3f}"
                f"  train={ep_return:.1f}  eval={eval_ret:.1f}"
            )
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save_checkpoint(save_path)
                print(f"    -> new best eval={best_eval:.2f}, saved.")

    return history


# ===========================================================================
# 6. PPO training loop
# ===========================================================================

def _train_ppo_agent(
    agent: PPOAgent,
    env,
    n_episodes: int        = 5_000,
    save_path: str | Path  = "results/ppo_best.pth",
    eval_every: int        = 100,
    n_eval_episodes: int   = 10,
    seed: int              = 42,
) -> Dict[str, List[float]]:
    """Collect rollout_steps steps, compute GAE, then run ppo_epochs updates."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(save_path)
    history: Dict[str, List[float]] = {
        "train_returns": [], "eval_returns": [], "losses": [],
    }
    best_eval     = -math.inf
    last_eval_ep  = 0

    ep            = 0
    ep_return     = 0.0
    last_done     = False
    obs, _        = env.reset(seed=seed)

    print(f"  [PPO] Training: collecting {agent.rollout_steps}-step rollouts ...")

    while ep < n_episodes:
        agent.buffer.reset()

        for _ in range(agent.rollout_steps):
            action, log_prob, value = agent.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done      = terminated or truncated
            last_done = done

            agent.buffer.add(obs, action, reward, value, log_prob, float(done))
            obs        = next_obs
            ep_return += reward

            if done:
                ep += 1
                history["train_returns"].append(ep_return)
                ep_return  = 0.0
                force_deg  = (random.random() < 0.40)
                obs, _     = env.reset(seed=seed + ep, force_degraded=force_deg)
                if ep >= n_episodes:
                    break

        # Bootstrap last state value if episode still in progress
        last_val = 0.0 if last_done else agent.get_value(obs)
        agent.buffer.compute_gae(last_val, gamma=agent.gamma,
                                  gae_lambda=agent.gae_lambda)
        loss = agent.update()
        if history["train_returns"]:
            history["losses"].append(loss)

        # Evaluate roughly every eval_every episodes
        if ep - last_eval_ep >= eval_every or ep >= n_episodes:
            eval_ret     = _greedy_eval_ppo(agent, env, n_eval_episodes, seed)
            last_eval_ep = ep
            history["eval_returns"].append(eval_ret)
            recent = history["train_returns"][-eval_every:] if history["train_returns"] else [0.0]
            print(
                f"  [PPO]  ep~{ep:5d}/{n_episodes}"
                f"  loss={loss:.4f}"
                f"  train_mean={float(np.mean(recent)):.1f}"
                f"  eval={eval_ret:.1f}"
            )
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save_checkpoint(save_path)
                print(f"    -> new best eval={best_eval:.2f}, saved.")

    return history


# ===========================================================================
# 7. Master training function
# ===========================================================================

def train_all_agents(
    env,
    results_dir: Path,
    n_episodes: int     = 5_000,
    seed: int           = 42,
    eval_every: int     = 100,
    n_eval_episodes: int = 10,
) -> Tuple[
    Dict[str, BaseAgent],
    Dict[str, Dict[str, List[float]]],
]:
    """Train DDQN, Dueling DQN, PPO. Skip agents whose checkpoints already exist.

    Returns
    -------
    agents     : dict of trained agent instances
    histories  : dict of training histories (train_returns, eval_returns, losses)
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    agents:    Dict[str, BaseAgent] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}

    # --- DDQN -------------------------------------------------------------
    ddqn_path = results_dir / "ddqn_best.pth"
    ddqn      = DDQNAgent()
    if ddqn_path.exists():
        print(f"  [DDQN] Loading existing checkpoint: {ddqn_path}")
        ddqn.load_checkpoint(ddqn_path)
        histories["DDQN"] = {"train_returns": [], "eval_returns": [], "losses": []}
    else:
        print(f"  [DDQN] Training for {n_episodes} episodes ...")
        histories["DDQN"] = _train_dqn_agent(
            ddqn, env,
            n_episodes=n_episodes,
            save_path=ddqn_path,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            label="DDQN",
        )
    agents["DDQN"] = ddqn

    # --- Dueling DQN ------------------------------------------------------
    dueling_path = results_dir / "dueling_dqn_best.pth"
    dueling      = DuelingDQNAgent()
    if dueling_path.exists():
        print(f"  [Dueling] Loading existing checkpoint: {dueling_path}")
        dueling.load_checkpoint(dueling_path)
        histories["Dueling DQN"] = {"train_returns": [], "eval_returns": [], "losses": []}
    else:
        print(f"  [Dueling] Training for {n_episodes} episodes ...")
        histories["Dueling DQN"] = _train_dqn_agent(
            dueling, env,
            n_episodes=n_episodes,
            save_path=dueling_path,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            label="Dueling",
        )
    agents["Dueling DQN"] = dueling

    # --- PPO --------------------------------------------------------------
    ppo_path = results_dir / "ppo_best.pth"
    ppo      = PPOAgent()
    if ppo_path.exists():
        print(f"  [PPO] Loading existing checkpoint: {ppo_path}")
        ppo.load_checkpoint(ppo_path)
        histories["PPO"] = {"train_returns": [], "eval_returns": [], "losses": []}
    else:
        print(f"  [PPO] Training for {n_episodes} episodes ...")
        histories["PPO"] = _train_ppo_agent(
            ppo, env,
            n_episodes=n_episodes,
            save_path=ppo_path,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
        )
    agents["PPO"] = ppo

    return agents, histories


# ===========================================================================
# 8. Unified evaluation
# ===========================================================================

def _agent_action_eval(agent: Any, obs: np.ndarray) -> int:
    """Get greedy action from any agent type (duck-typed)."""
    if isinstance(agent, QRDQNAgent):
        return agent.select_action(obs, greedy=True)
    if isinstance(agent, (DDQNAgent, DuelingDQNAgent)):
        return agent.select_action(obs, epsilon=0.0)
    if isinstance(agent, PPOAgent):
        return agent.select_action(obs)
    # Rule-based baseline: select_action(obs, step, info) — obs is enough
    return agent.select_action(obs, 0, {"hi_t": float(obs[0])})


def evaluate_all_agents(
    agents_dict: Dict[str, Any],
    env,
    n_episodes: int = 300,
    seed: int       = 42,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all agents/policies over n_episodes greedy rollouts.

    Compatible with DDQNAgent, DuelingDQNAgent, PPOAgent, QRDQNAgent,
    and rule-based policies (ThresholdPolicy, CorrectiveMaintenance, etc.).

    Returns
    -------
    dict  {agent_name: {mean_cost, std_cost, catastrophe_rate, mean_reward,
                         mean_n_repairs, mean_n_replacements, action_dist}}
    """
    rng     = np.random.default_rng(seed)
    results: Dict[str, Dict[str, Any]] = {}

    for name, agent in agents_dict.items():
        total_costs:  List[float] = []
        catastrophes: List[int]   = []
        ep_rewards:   List[float] = []
        n_repairs_ep: List[int]   = []
        n_replace_ep: List[int]   = []
        action_counts = np.zeros(3, dtype=np.int64)

        for ep in range(n_episodes):
            ep_seed  = int(rng.integers(0, 2 ** 31))
            obs, _   = env.reset(seed=ep_seed)
            if hasattr(agent, "reset"):
                agent.reset()
            done     = False
            ep_rew   = 0.0

            while not done:
                action = _agent_action_eval(agent, obs)
                action_counts[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done   = terminated or truncated
                ep_rew += reward

            total_costs.append(float(info["total_cost"]))
            catastrophes.append(1 if info.get("is_failure", False) else 0)
            ep_rewards.append(ep_rew)
            n_repairs_ep.append(int(info.get("n_repairs", 0)))
            n_replace_ep.append(int(info.get("n_replacements", 0)))

            if (ep + 1) % 100 == 0:
                logger.info(
                    "  [eval %s] %d/%d  catast=%.1f%%",
                    name, ep + 1, n_episodes,
                    float(np.mean(catastrophes)) * 100,
                )

        total = int(action_counts.sum())
        results[name] = {
            "mean_cost":           float(np.mean(total_costs)),
            "std_cost":            float(np.std(total_costs)),
            "catastrophe_rate":    float(np.mean(catastrophes)),
            "mean_reward":         float(np.mean(ep_rewards)),
            "mean_n_repairs":      float(np.mean(n_repairs_ep)),
            "mean_n_replacements": float(np.mean(n_replace_ep)),
            "action_dist": {
                "do_nothing": float(action_counts[0] / total) if total else 0.0,
                "repair":     float(action_counts[1] / total) if total else 0.0,
                "replace":    float(action_counts[2] / total) if total else 0.0,
            },
        }
        print(
            f"  {name:25s}  cost={results[name]['mean_cost']:.2f}"
            f"  catast={results[name]['catastrophe_rate']:.1%}"
            f"  rew={results[name]['mean_reward']:.2f}"
        )

    return results


# ===========================================================================
# 9. Comparison table (CSV + LaTeX)
# ===========================================================================

def generate_comparison_table(
    eval_results: Dict[str, Dict[str, Any]],
    results_dir: Path,
) -> None:
    """Write results/table_rl_benchmarks.csv and .tex."""
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [n for n in _ROW_ORDER if n in eval_results]

    # CSV ------------------------------------------------------------------
    csv_path = results_dir / "table_rl_benchmarks.csv"
    fields   = [
        "Policy", "Cost_mu", "Cost_sigma", "Catastrophe_pct",
        "Avg_Reward", "Repairs_per_ep", "Replaces_per_ep", "DoNothing_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name in rows:
            m   = eval_results[name]
            act = m["action_dist"]
            w.writerow({
                "Policy":         name,
                "Cost_mu":        round(m["mean_cost"],           3),
                "Cost_sigma":     round(m["std_cost"],            3),
                "Catastrophe_pct":round(m["catastrophe_rate"] * 100, 2),
                "Avg_Reward":     round(m["mean_reward"],         3),
                "Repairs_per_ep": round(m["mean_n_repairs"],      3),
                "Replaces_per_ep":round(m["mean_n_replacements"], 3),
                "DoNothing_pct":  round(act["do_nothing"] * 100,  2),
            })
    print(f"  Saved -> {csv_path}")

    # LaTeX ----------------------------------------------------------------
    # Collect per-column values to identify best
    cols = [
        ("Cost_mu",        "mean_cost",           "lower"),
        ("Cost_sigma",     "std_cost",            "lower"),
        ("Catastrophe_%",  "catastrophe_rate",    "lower"),
        ("Avg_Reward",     "mean_reward",         "higher"),
        ("Repairs/ep",     "mean_n_repairs",      "lower"),
        ("Replaces/ep",    "mean_n_replacements", "lower"),
        ("DoNothing_%",    "action_dist.dn",      "none"),
    ]

    def _get_val(m: Dict, key: str) -> float:
        if key == "action_dist.dn":
            return m["action_dist"]["do_nothing"] * 100.0
        if key == "catastrophe_rate":
            return m["catastrophe_rate"] * 100.0
        return float(m[key])

    # Find best per column
    best: Dict[str, float] = {}
    for _, mkey, direction in cols:
        vals = [_get_val(eval_results[n], mkey) for n in rows]
        if direction == "lower":
            best[mkey] = min(vals)
        elif direction == "higher":
            best[mkey] = max(vals)

    def _fmt(val: float, mkey: str, direction: str) -> str:
        if mkey in ("catastrophe_rate", "action_dist.dn"):
            s = f"{val:.1f}\\%"
        elif mkey in ("mean_cost", "std_cost", "mean_reward"):
            s = f"{val:.2f}"
        else:
            s = f"{val:.3f}"
        if direction != "none" and abs(val - best.get(mkey, float("nan"))) < 1e-9:
            return f"\\textbf{{{s}}}"
        return s

    col_headers = " & ".join(
        ["\\textbf{Policy}"] + [f"\\textbf{{{h}}}" for h, _, _ in cols]
    )
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{RL benchmark comparison (300 evaluation episodes, seed=42)}",
        "\\label{tab:rl_benchmarks}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        col_headers + " \\\\",
        "\\midrule",
    ]
    for name in rows:
        m    = eval_results[name]
        disp = name.replace("CVaR QR-DQN (ours)", "CVaR QR-DQN$^{\\dagger}$")
        cells = [f"\\textit{{{disp}}}"]
        for _, mkey, direction in cols:
            cells.append(_fmt(_get_val(m, mkey), mkey, direction))
        tex_lines.append(" & ".join(cells) + " \\\\")
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]

    tex_path = results_dir / "table_rl_benchmarks.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines) + "\n")
    print(f"  Saved -> {tex_path}")


# ===========================================================================
# 10. Figures
# ===========================================================================

def _load_qrdqn_training_history(results_dir: Path) -> List[float]:
    """Try to read ep_reward from results/training_log.csv (train.py output)."""
    log = results_dir / "training_log.csv"
    if not log.exists():
        return []
    try:
        rewards = []
        with open(log, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "ep_reward" in row and row["ep_reward"] != "":
                    rewards.append(float(row["ep_reward"]))
        return rewards
    except Exception:
        return []


def generate_plots(
    training_histories: Dict[str, Dict[str, List[float]]],
    eval_results: Dict[str, Dict[str, Any]],
    results_dir: Path,
) -> None:
    """IEEE-style 2x2 figure comparing all agents.

    Panel (a) — Smoothed training reward curves (window=100)
    Panel (b) — Catastrophe rate horizontal bar chart
    Panel (c) — Mean cost vertical bar chart with +/-1 std error bars
    Panel (d) — Action-composition stacked bar with zoomed inset
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.spines.right": False,
        "axes.spines.top":   False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    ax_curve, ax_catast = axes[0]
    ax_cost,  ax_action = axes[1]

    # --- (a) Training curves -----------------------------------------------
    ax = ax_curve
    _TRAIN_AGENTS = ["DDQN", "Dueling DQN", "PPO", "Risk-Neutral DQN", "CVaR QR-DQN (ours)"]

    # Try to inject QR-DQN history from training_log.csv
    qrdqn_hist = _load_qrdqn_training_history(results_dir)
    for name in ("Risk-Neutral DQN", "CVaR QR-DQN (ours)"):
        if name not in training_histories or not training_histories[name]["train_returns"]:
            training_histories[name] = {"train_returns": qrdqn_hist,
                                        "eval_returns": [], "losses": []}

    any_plotted = False
    for name in _TRAIN_AGENTS:
        hist = training_histories.get(name, {})
        returns = hist.get("train_returns", [])
        if len(returns) < 2:
            continue
        xs, sm = _smooth(returns, window=100)
        ax.plot(
            xs, sm,
            color=_AGENT_COLORS.get(name, "#333333"),
            linestyle=_AGENT_LS.get(name, "solid"),
            linewidth=1.2,
            label=name,
            alpha=0.85,
        )
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "Training histories not available\n(run without --eval-only)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed reward (w=100)")
    ax.set_title("(a) Training curves")
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.25)

    # --- (b) Catastrophe rate horizontal bar chart -------------------------
    ax = ax_catast
    names_b = [n for n in _ROW_ORDER if n in eval_results]
    catast  = [eval_results[n]["catastrophe_rate"] * 100.0 for n in names_b]
    colors_b = [_AGENT_COLORS.get(n, "#333333") for n in names_b]
    short_b  = [n.replace("CVaR QR-DQN (ours)", "CVaR QR-DQN").replace("Risk-Neutral DQN", "RN-DQN")
                for n in names_b]

    ypos = np.arange(len(names_b))
    ax.barh(ypos, catast, color=colors_b, edgecolor="white", height=0.6)
    ax.axvline(10.0, color="red", linestyle="--", linewidth=1.0, label="10% target")
    ax.set_yticks(ypos)
    ax.set_yticklabels(short_b)
    ax.set_xlabel("Catastrophe rate (%)")
    ax.set_title("(b) Catastrophe rate")
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(catast):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)

    # --- (c) Mean cost vertical bar chart with error bars ------------------
    ax = ax_cost
    names_c = [n for n in _ROW_ORDER if n in eval_results]
    means_c  = [eval_results[n]["mean_cost"] for n in names_c]
    stds_c   = [eval_results[n]["std_cost"]  for n in names_c]
    colors_c = [_AGENT_COLORS.get(n, "#333333") for n in names_c]
    short_c  = [n.replace("CVaR QR-DQN (ours)", "CVaR").replace("Risk-Neutral DQN", "RN-DQN")
                .replace("ThresholdPolicy", "Threshold").replace("Dueling DQN", "Dueling")
                for n in names_c]

    xpos = np.arange(len(names_c))
    bars = ax.bar(xpos, means_c, yerr=stds_c, color=colors_c,
                  edgecolor="white", capsize=4, error_kw={"linewidth": 1.0})
    ax.set_xticks(xpos)
    ax.set_xticklabels(short_c, rotation=30, ha="right")
    ax.set_ylabel("Mean episode cost")
    ax.set_title("(c) Mean cost (+/- 1 std)")
    ax.grid(axis="y", alpha=0.25)

    # --- (d) Action composition stacked bar + inset ------------------------
    ax = ax_action
    names_d  = [n for n in _ROW_ORDER if n in eval_results]
    dn = np.array([eval_results[n]["action_dist"]["do_nothing"] for n in names_d])
    rp = np.array([eval_results[n]["action_dist"]["repair"]     for n in names_d])
    rx = np.array([eval_results[n]["action_dist"]["replace"]    for n in names_d])
    short_d  = [n.replace("CVaR QR-DQN (ours)", "CVaR").replace("Risk-Neutral DQN", "RN-DQN")
                .replace("ThresholdPolicy", "Threshold").replace("Dueling DQN", "Dueling")
                for n in names_d]
    yd = np.arange(len(names_d))

    ax.barh(yd, dn, color="#aaaaaa", label="Do nothing")
    ax.barh(yd, rp, left=dn,      color="#4c72b0", label="Repair")
    ax.barh(yd, rx, left=dn + rp, color="#c44e52", label="Replace")
    ax.set_yticks(yd)
    ax.set_yticklabels(short_d)
    ax.set_xlabel("Action fraction")
    ax.set_title("(d) Action composition")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(axis="x", alpha=0.25)

    # Zoomed inset: repair + replace only (0-15%)
    axins = ax.inset_axes([0.38, 0.50, 0.60, 0.46])
    axins.barh(yd, rp, color="#4c72b0")
    axins.barh(yd, rx, left=rp, color="#c44e52")
    axins.set_xlim(0, 0.15)
    axins.set_yticks(yd)
    axins.set_yticklabels(short_d, fontsize=6)
    axins.set_xticks([0.0, 0.05, 0.10, 0.15])
    axins.set_xticklabels(["0%", "5%", "10%", "15%"], fontsize=6)
    axins.set_title("Repair & Replace (0-15% zoom)", fontsize=7)
    axins.grid(axis="x", alpha=0.25)
    for i, (r_, x_) in enumerate(zip(rp, rx)):
        if r_ > 5e-4:
            axins.text(r_ / 2, i, f"{r_:.2%}", ha="center", va="center",
                       fontsize=5.5, color="white", fontweight="bold")
        if x_ > 5e-4:
            axins.text(r_ + x_ / 2, i, f"{x_:.2%}", ha="center", va="center",
                       fontsize=5.5, color="white", fontweight="bold")

    fig.tight_layout(pad=0.8)
    out = results_dir / "fig_rl_benchmarks.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ===========================================================================
# 11. Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL benchmark agents and compare.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load existing checkpoints only.")
    parser.add_argument("--n-episodes", type=int, default=5_000)
    parser.add_argument("--n-eval",     type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    proc_dir    = _PROJ / "data" / "processed"
    results_dir = _PROJ / "results"

    if not proc_dir.exists():
        sys.exit(f"ERROR: {proc_dir} not found. Run feature_extractor.py first.")

    print(f"\nLoading environment from {proc_dir} ...")
    env = make_env_from_processed(proc_dir, seed=args.seed)

    # --- Train / load new agents ------------------------------------------
    if args.eval_only:
        print("\n[--eval-only] Loading checkpoints (skipping training).")
        ddqn = DDQNAgent()
        ddqn_path = results_dir / "ddqn_best.pth"
        if ddqn_path.exists():
            ddqn.load_checkpoint(ddqn_path)
        else:
            print(f"  WARNING: {ddqn_path} not found — DDQN uses random weights.")

        dueling = DuelingDQNAgent()
        dup_path = results_dir / "dueling_dqn_best.pth"
        if dup_path.exists():
            dueling.load_checkpoint(dup_path)
        else:
            print(f"  WARNING: {dup_path} not found — Dueling DQN uses random weights.")

        ppo = PPOAgent()
        ppo_path = results_dir / "ppo_best.pth"
        if ppo_path.exists():
            ppo.load_checkpoint(ppo_path)
        else:
            print(f"  WARNING: {ppo_path} not found — PPO uses random weights.")

        agents = {"DDQN": ddqn, "Dueling DQN": dueling, "PPO": ppo}
        histories: Dict[str, Dict[str, List[float]]] = {
            k: {"train_returns": [], "eval_returns": [], "losses": []} for k in agents
        }
    else:
        print(f"\nTraining benchmark agents ({args.n_episodes} episodes each) ...")
        agents, histories = train_all_agents(
            env,
            results_dir=results_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )

    # --- Load QR-DQN agents -----------------------------------------------
    qrdqn_path = results_dir / "qrdqn_best.pth"

    rn_agent = QRDQNAgent(risk_mode="mean")
    if qrdqn_path.exists():
        # Load weights without overriding risk_mode
        ckpt = torch.load(qrdqn_path, map_location="cpu", weights_only=False)
        rn_agent.online_net.load_state_dict(ckpt["online_net"])
        rn_agent.target_net.load_state_dict(ckpt["target_net"])
        print(f"  Loaded Risk-Neutral DQN from {qrdqn_path} (risk_mode=mean)")
    else:
        print(f"  WARNING: {qrdqn_path} not found — Risk-Neutral DQN uses random weights.")

    cvar_agent = QRDQNAgent(risk_mode="cvar")
    if qrdqn_path.exists():
        cvar_agent.load_checkpoint(qrdqn_path)
        print(f"  Loaded CVaR QR-DQN from {qrdqn_path} (risk_mode=cvar)")

    agents["Risk-Neutral DQN"]    = rn_agent
    agents["CVaR QR-DQN (ours)"] = cvar_agent

    # --- Evaluate all agents + ThresholdPolicy ----------------------------
    print(f"\nEvaluating all agents ({args.n_eval} episodes each) ...")
    eval_results = evaluate_all_agents(agents, env, n_episodes=args.n_eval, seed=args.seed)

    # Evaluate ThresholdPolicy via existing evaluate_policy utility
    print("  Evaluating ThresholdPolicy ...")
    threshold     = ThresholdPolicy()
    tp_metrics    = evaluate_policy(threshold, env, n_episodes=args.n_eval, seed=args.seed)
    eval_results["ThresholdPolicy"] = {
        "mean_cost":           tp_metrics["mean_total_cost"],
        "std_cost":            tp_metrics["std_total_cost"],
        "catastrophe_rate":    tp_metrics["catastrophe_rate"],
        "mean_reward":         -tp_metrics["mean_total_cost"],  # reward = -cost (no action costs differ)
        "mean_n_repairs":      tp_metrics["mean_n_repairs"],
        "mean_n_replacements": tp_metrics["mean_n_replacements"],
        "action_dist":         tp_metrics["action_distribution"],
    }

    # --- Print summary table ----------------------------------------------
    print("\n" + "=" * 90)
    print(f"  {'Policy':28s}  {'Cost_mu':>8}  {'Cost_sd':>8}  {'Catast%':>8}"
          f"  {'AvgRew':>8}  {'Rp/ep':>6}  {'Rx/ep':>6}  {'DN%':>7}")
    print("-" * 90)
    for name in _ROW_ORDER:
        if name not in eval_results:
            continue
        m   = eval_results[name]
        act = m["action_dist"]
        print(
            f"  {name:28s}"
            f"  {m['mean_cost']:>8.2f}"
            f"  {m['std_cost']:>8.2f}"
            f"  {m['catastrophe_rate']:>7.1%}"
            f"  {m['mean_reward']:>8.2f}"
            f"  {m['mean_n_repairs']:>6.3f}"
            f"  {m['mean_n_replacements']:>6.3f}"
            f"  {act['do_nothing']:>6.2%}"
        )
    print("=" * 90)

    # --- Generate outputs -------------------------------------------------
    print("\nGenerating table and figures ...")
    generate_comparison_table(eval_results, results_dir)
    generate_plots(histories, eval_results, results_dir)

    print("\nDone. Outputs written to", results_dir)
