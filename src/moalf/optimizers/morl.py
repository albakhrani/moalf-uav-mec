"""morl — Multiobjective RL offloading via a DQN function approximator (spec §7).

Architecture (spec §7, RESOLVED 2026-06-23): a **DQN** — MLP Q-network + target
network + experience replay — not a tabular Q-table (the state is continuous /
high-dimensional). Matches the config replay/batch/ε-decay hyperparameters.

D1 contract (spec §6, same as APSO/MPC): the agent holds **NO objective weights**.
Its scalar reward is `R_MORL = −(value of the 'morl' projection)` (terms
{task, energy, completion, util} = m∈{1,2,3,5}), computed by :class:`OffloadEnv`
through ``projection.value(...)``. The agent is weight-agnostic, so the §5.4
weight-sweep (A2) is just a re-run with a different :class:`~moalf.objective.Objective`.
Offload-only action space (A4: no migration).

Reproducibility: NumPy and PyTorch RNGs are seeded from the frozen seed (B27=42)
and training runs on CPU. **Exact bit-reproducibility additionally requires the
same PyTorch build** (pinned torch==2.0.1 in requirements.txt); different builds
may differ in the last digits.

Scope note: :class:`OffloadEnv` is a synthetic single-decision environment for
training/verifying the agent in isolation. It will be replaced by the real
Algorithm-1 environment when the simulation loop is wired up; the agent and the
projection contract are unchanged by that swap.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from moalf.objective import Objective, Projection, Term


def seed_everything(seed: int) -> None:
    """Seed python/NumPy/PyTorch RNGs (CPU) for as-reproducible-as-possible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# Q-network
# ----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """MLP Q-network: state -> Q-value per action. Width/depth are standard,
    non-paper hyperparameters (frozen before any 94.5% comparison; §7.1)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# DQN agent — generic; holds NO objective weights and NO projection
# ----------------------------------------------------------------------------
@dataclass
class MORLAgent:
    """DQN agent (spec §7). Construct via :meth:`from_config`.

    Holds only RL hyperparameters and network/buffer state — deliberately no
    objective weights and no projection (those live on the Objective; reward is
    supplied by the environment).
    """

    state_dim: int
    action_dim: int
    learning_rate: float
    discount: float
    epsilon: float
    epsilon_decay: float
    replay_size: int
    batch_size: int
    # standard, non-paper hyperparameters (frozen; §7.1)
    hidden: int = 64
    target_update_every: int = 50
    epsilon_min: float = 0.05
    device: str = "cpu"

    def __post_init__(self):
        self.q = QNetwork(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.q_target = QNetwork(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.buffer: deque = deque(maxlen=self.replay_size)
        self._learn_steps = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any], state_dim: int, action_dim: int,
                    *, seed: Optional[int] = None, device: str = "cpu") -> "MORLAgent":
        if seed is None:
            seed = int(config["run"]["seed"])
        seed_everything(seed)
        m = config["morl"]
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=float(m["learning_rate"]),
            discount=float(m["discount_factor"]),
            epsilon=float(m["epsilon_init"]),
            epsilon_decay=float(m["epsilon_decay"]),
            replay_size=int(m["replay_buffer_size"]),
            batch_size=int(m["batch_size"]),
            device=device,
        )

    # ---- action selection ---------------------------------------------------
    def act(self, state, epsilon: Optional[float] = None) -> int:
        """ε-greedy action. ``epsilon=0`` (or omitted at eval) -> greedy."""
        eps = self.epsilon if epsilon is None else epsilon
        if np.random.random() < eps:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
            return int(torch.argmax(self.q(s)).item())

    def greedy(self, state) -> int:
        return self.act(state, epsilon=0.0)

    # ---- replay / learning --------------------------------------------------
    def remember(self, s, a, r, s2, done) -> None:
        self.buffer.append((np.asarray(s, np.float32), int(a), float(r),
                            np.asarray(s2, np.float32), bool(done)))

    def learn(self) -> Optional[float]:
        """One DQN gradient step. Returns the loss, or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        s = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        a = torch.as_tensor([b[1] for b in batch], device=self.device).long().unsqueeze(1)
        r = torch.as_tensor([b[2] for b in batch], device=self.device).float().unsqueeze(1)
        s2 = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        done = torch.as_tensor([b[4] for b in batch], device=self.device).float().unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.q_target(s2).max(1, keepdim=True).values
            target = r + self.discount * q_next * (1.0 - done)
        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ----------------------------------------------------------------------------
# Synthetic offload environment — consumes the 'morl' projection
# ----------------------------------------------------------------------------
class OffloadEnv:
    """Single-decision offload MDP for training/verifying MORL (spec §7).

    Each step: sample M UAV states (compute, energy, queue) and a task; the agent
    picks a UAV; the reward is `−projection.value(raw_terms)` for that choice. The
    'morl' projection is REQUIRED — this is where the agent's reward is tied to the
    one objective's weights (D1). The environment, not the agent, consumes it.
    """

    def __init__(self, config: Mapping[str, Any], projection: Projection,
                 *, seed: int = 42, reward_scale: Optional[float] = None):
        if not isinstance(projection, Projection):
            raise TypeError("OffloadEnv requires an objective Projection (no inline weights)")
        if projection.name != "morl":
            raise ValueError(f"MORL must consume the 'morl' projection, got '{projection.name}'")
        self.projection = projection
        self.rng = np.random.default_rng(seed)
        self.M = int(config["network"]["num_uavs"])

        uav = config["uav"]
        net = config["network"]
        self.c_lo, self.c_hi = uav["compute_capacity_ghz"]["low"], uav["compute_capacity_ghz"]["high"]
        self.e_c_j = float(uav["compute_energy_rate_wh_per_cycle"]) * 3600.0
        self.e_max_j = float(uav["energy_capacity_wh"]) * 3600.0
        self.reserve_frac = float(config["energy"]["min_reserve_wh"]) / float(uav["energy_capacity_wh"])
        task = config["task"]
        self.w_lo, self.w_hi = task["compute_req_megacycles"]["low"], task["compute_req_megacycles"]["high"]
        self.d_lo, self.d_hi = task["deadline_s"]["low"], task["deadline_s"]["high"]
        self.q_max_cyc = self.c_hi * 1e9 * 2.0  # up to ~2 s of backlog at top speed
        # uniform positive scale for numerical conditioning (policy-invariant)
        self.reward_scale = float(reward_scale) if reward_scale is not None \
            else float(net["num_iot_devices"]) * 0.2 * net["num_time_steps"]  # ~N_task_hat

        self.state_dim = 3 * self.M + 2
        self.action_dim = self.M

    # ---- sampling -----------------------------------------------------------
    def reset(self):
        c = self.rng.uniform(self.c_lo, self.c_hi, self.M)        # GHz
        e = self.rng.uniform(0.0, 1.0, self.M)                    # energy fraction
        q = self.rng.uniform(0.0, self.q_max_cyc, self.M)         # cycles backlog
        w = self.rng.uniform(self.w_lo, self.w_hi)               # Mcyc
        d = self.rng.uniform(self.d_lo, self.d_hi)               # s
        info = {"c": c, "e": e, "q": q, "w": w, "d": d}
        return self.encode(info), info

    def encode(self, info) -> np.ndarray:
        c = (info["c"] - self.c_lo) / (self.c_hi - self.c_lo)
        e = info["e"]
        q = info["q"] / self.q_max_cyc
        w = (info["w"] - self.w_lo) / (self.w_hi - self.w_lo)
        d = (info["d"] - self.d_lo) / (self.d_hi - self.d_lo)
        return np.concatenate([c, e, q, [w, d]]).astype(np.float32)

    # ---- raw objective terms for choosing UAV `a` ---------------------------
    def raw_terms(self, info, a: int):
        c_a = info["c"][a] * 1e9             # cycles/s
        e_a = info["e"][a]
        q_a = info["q"][a]                   # cycles
        w_cyc = info["w"] * 1e6             # cycles
        d = info["d"]

        latency = (q_a + w_cyc) / c_a                       # s
        exec_e = w_cyc * self.e_c_j                          # J
        available = e_a * self.e_max_j                       # J
        depleted = (available < exec_e) or (e_a < self.reserve_frac)
        completed = (latency <= d) and (not depleted)
        depletion_penalty = 10.0 * exec_e if depleted else 0.0
        util = min(1.0, c_a * d / (q_a + w_cyc))             # fraction clearable by deadline

        return {
            Term.TASK: latency,                              # cost
            Term.ENERGY: exec_e + depletion_penalty,         # cost
            Term.COMPLETION: -1.0 if completed else 0.0,     # reward
            Term.UTIL: -util,                                # reward
        }, {"completed": bool(completed), "depleted": bool(depleted)}

    def reward(self, info, a: int):
        raw, outcome = self.raw_terms(info, a)
        r = -self.projection.value(raw) * self.reward_scale
        return r, outcome

    def step(self, info, a: int):
        r, outcome = self.reward(info, a)
        return r, True, outcome  # single-decision: always done


# ----------------------------------------------------------------------------
# Baseline policies (for the pre-registered comparison, §7.1) — no learning
# ----------------------------------------------------------------------------
def random_policy(env: OffloadEnv, rng: np.random.Generator) -> Callable:
    return lambda state, info: int(rng.integers(env.action_dim))


def jsq_policy(config: Mapping[str, Any]) -> Callable:
    """Join-shortest-queue softmax baseline (eq 64 / §13, A3).

    score_j = −Q_j / T_sm + λ_b·θ_rel ; pick argmax. With no per-link reliability
    in this env, θ_rel is constant, so this reduces to shortest-compute-queue.
    """
    T_sm = 1.0
    lam_b = 1.0
    theta_rel = float(config["failures"]["link_reliability_threshold"])

    def policy(state, info):
        score = -info["q"] / T_sm + lam_b * theta_rel
        return int(np.argmax(score))

    return policy


# ----------------------------------------------------------------------------
# Training / evaluation helpers
# ----------------------------------------------------------------------------
def train(agent: MORLAgent, env: OffloadEnv, steps: int, *, window: int = 100) -> list:
    """Train the agent for ``steps`` single-decision episodes. Returns the
    learning curve: mean reward per ``window`` steps."""
    curve, bucket = [], []
    for _ in range(steps):
        state, info = env.reset()
        a = agent.act(state)
        r, done, _ = env.step(info, a)
        agent.remember(state, a, r, state, done)
        agent.learn()
        agent.decay_epsilon()
        bucket.append(r)
        if len(bucket) == window:
            curve.append(float(np.mean(bucket)))
            bucket = []
    if bucket:
        curve.append(float(np.mean(bucket)))
    return curve


def evaluate(policy: Callable, env: OffloadEnv, episodes: int):
    """Mean reward and completion rate of a policy(state, info) -> action."""
    rewards, completes = [], []
    for _ in range(episodes):
        state, info = env.reset()
        a = policy(state, info)
        r, _, outcome = env.step(info, a)
        rewards.append(r)
        completes.append(outcome["completed"])
    return float(np.mean(rewards)), float(np.mean(completes))
