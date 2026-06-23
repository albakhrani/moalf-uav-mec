"""Tests for MORL DQN (src/moalf/optimizers/morl.py), spec §7 / §7.1.

Smoke versions of the pre-registered success criteria (§7.1): learning curve
trends up, learned policy beats random AND the JSQ baseline, and a hand-checkable
corner case (near-depleted UAV offloaded-to less). Plus the D1 no-own-weights
guard, the projection contract, reward-via-projection, and determinism.

Full-scale training lives in experiments/train_morl.py.
"""

from __future__ import annotations

import io
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from moalf.objective import Objective, Term
from moalf.optimizers.morl import (
    MORLAgent, OffloadEnv, evaluate, jsq_policy, random_policy, seed_everything, train,
)
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
SMOKE_STEPS = 1500


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def objective(config) -> Objective:
    return Objective.from_config(config)


@pytest.fixture(scope="module")
def trained(config, objective):
    """Train one agent (smoke length) and reuse across criteria tests."""
    proj = objective.projection("morl")
    env = OffloadEnv(config, proj, seed=42)
    agent = MORLAgent.from_config(config, env.state_dim, env.action_dim, seed=42)
    curve = train(agent, env, steps=SMOKE_STEPS, window=150)
    return {"agent": agent, "env": env, "curve": curve, "config": config, "proj": proj}


# --- D1: no objective weights, projection contract ---------------------------
def test_agent_has_no_objective_weights(config):
    env_proj = Objective.from_config(config).projection("morl")
    env = OffloadEnv(config, env_proj, seed=0)
    agent = MORLAgent.from_config(config, env.state_dim, env.action_dim, seed=0)
    field_names = {f.name for f in fields(MORLAgent)}
    for forbidden in ("w1_task", "w2_energy", "w3_completion", "w5_util",
                      "weights", "_weights", "projection", "objective"):
        assert forbidden not in field_names
        assert not hasattr(agent, forbidden)


def test_env_requires_morl_projection(config, objective):
    with pytest.raises(ValueError, match="morl"):
        OffloadEnv(config, objective.projection("apso"), seed=0)
    with pytest.raises(TypeError):
        OffloadEnv(config, lambda r: 0.0, seed=0)  # raw callable, not a Projection


def test_reward_is_negative_projection_value(config, objective):
    proj = objective.projection("morl")
    env = OffloadEnv(config, proj, seed=3)
    _, info = env.reset()
    for a in range(env.action_dim):
        raw, _ = env.raw_terms(info, a)
        expected = -proj.value(raw) * env.reward_scale
        r, _, _ = env.step(info, a)
        assert r == pytest.approx(expected)


# --- determinism -------------------------------------------------------------
def test_same_seed_same_initialization(config, objective):
    proj = objective.projection("morl")
    env = OffloadEnv(config, proj, seed=1)
    a1 = MORLAgent.from_config(config, env.state_dim, env.action_dim, seed=123)
    a2 = MORLAgent.from_config(config, env.state_dim, env.action_dim, seed=123)
    for p1, p2 in zip(a1.q.parameters(), a2.q.parameters()):
        assert torch.equal(p1, p2)
    state, _ = env.reset()
    assert a1.greedy(state) == a2.greedy(state)


# --- §7.1 criterion 1: learning curve trends up ------------------------------
def test_learning_curve_trends_up(trained):
    curve = trained["curve"]
    assert len(curve) >= 4
    first = np.mean(curve[: max(1, len(curve) // 4)])
    last = np.mean(curve[-max(1, len(curve) // 4):])
    assert last > first  # improved over training


# --- §7.1 criterion 2: beats random AND JSQ ----------------------------------
def test_beats_random_and_jsq(trained):
    config, proj, agent = trained["config"], trained["proj"], trained["agent"]
    eval_env = OffloadEnv(config, proj, seed=999)
    dqn_r, dqn_c = evaluate(lambda s, i: agent.greedy(s), eval_env, 1000)

    eval_env = OffloadEnv(config, proj, seed=999)  # same eval stream for fairness
    rng = np.random.default_rng(7)
    rand_r, rand_c = evaluate(random_policy(eval_env, rng), eval_env, 1000)

    eval_env = OffloadEnv(config, proj, seed=999)
    jsq_r, jsq_c = evaluate(jsq_policy(config), eval_env, 1000)

    assert dqn_r > rand_r, f"DQN {dqn_r:.3f} did not beat random {rand_r:.3f}"
    assert dqn_r > jsq_r, f"DQN {dqn_r:.3f} did not beat JSQ {jsq_r:.3f} (would be a FINDING)"
    assert dqn_c >= rand_c and dqn_c >= jsq_c  # completion no worse


# --- §7.1 criterion 3: corner case (near-depleted offloaded to less) ---------
def test_near_depleted_uav_is_avoided(trained):
    agent, env = trained["agent"], trained["env"]
    rng = np.random.default_rng(5)
    picked_depleted = 0
    trials = 200
    for _ in range(trials):
        # two-relevant-UAV setup: index 0 near-depleted, index 1 charged, else equal
        c = np.full(env.M, 3.5)
        q = np.full(env.M, env.q_max_cyc * 0.5)
        e = np.full(env.M, 0.02)         # everyone depleted...
        e[1] = 0.95                       # ...except a charged alternative at idx 1
        info = {"c": c, "e": e, "q": q,
                "w": rng.uniform(env.w_lo, env.w_hi), "d": rng.uniform(env.d_lo, env.d_hi)}
        if agent.greedy(env.encode(info)) == 0:   # picked the near-depleted UAV
            picked_depleted += 1
    # the agent should rarely choose the depleted UAV when a charged one exists
    assert picked_depleted < 0.2 * trials, f"chose depleted {picked_depleted}/{trials} times"


# --- A2 hybrid is REAL: the weight-sweep is actually available ----------------
def _weights_from(config, **overrides):
    """Build a config copy with overridden objective weights (everything else equal)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["objective"] = dict(config["objective"], **overrides)
    return cfg


def test_weight_sweep_changes_optimized_reward_without_touching_agent(config):
    """Two Objectives with DIFFERENT weights, same agent code: the reward the env
    optimizes reflects the new weights, and training runs unchanged. Structural
    proof the §5.4 weight-sweep (A2 multiobjective/Pareto evidence) is real."""
    # Objective A: default weights. Objective B: energy weight x100 (heavily
    # penalize energy) -> the SAME (state, action) must score differently.
    obj_a = Objective.from_config(config)
    cfg_b = _weights_from(config, w2_energy=100.0)
    obj_b = Objective.from_config(cfg_b)

    env_a = OffloadEnv(config, obj_a.projection("morl"), seed=11)
    env_b = OffloadEnv(cfg_b, obj_b.projection("morl"), seed=11)  # same seed -> same states

    _, info = env_a.reset()
    env_b.reset()  # advance B's identical stream to the same state
    # an action that consumes energy should be penalized MORE under B's weights
    a = 0
    r_a, _, _ = env_a.step(info, a)
    r_b, _, _ = env_b.step(info, a)
    assert r_b < r_a  # heavier energy weight -> lower (worse) reward for the same choice

    # and training runs with the *same agent code* on the reweighted objective
    agent_b = MORLAgent.from_config(cfg_b, env_b.state_dim, env_b.action_dim, seed=11)
    curve_b = train(agent_b, env_b, steps=300, window=150)
    assert len(curve_b) >= 2  # it ran end-to-end, agent untouched


def test_agent_is_weight_agnostic_same_init_under_different_weights(config):
    """The agent's initialization does not depend on objective weights at all."""
    cfg_b = _weights_from(config, w2_energy=100.0, w3_completion=7.0)
    a1 = MORLAgent.from_config(config, 17, 5, seed=321)
    a2 = MORLAgent.from_config(cfg_b, 17, 5, seed=321)
    for p1, p2 in zip(a1.q.parameters(), a2.q.parameters()):
        assert torch.equal(p1, p2)  # identical -> agent carries no weight info
