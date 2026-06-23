"""Increment 2 tests: in-loop MORL training against the full coupled system.

Fast/downsized smoke versions (N=20, M=3, high load): confirm that in-loop
training materially reduces compute backlog vs the untrained policy, that
conservation stays exact during training, and that the trained policy keeps the
system more stable. Full-scale training lives in experiments/train_morl_inloop.py.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.simulation import Simulation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def small_cfg() -> dict:
    cfg = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    return {
        **cfg,
        "network": {**cfg["network"], "num_iot_devices": 20, "num_uavs": 3},
        "task": {**cfg["task"],
                 "generation_rate_tps": {"dist": "uniform", "low": 0.9, "high": 0.9}},
    }


@pytest.fixture(scope="module")
def trained(small_cfg):
    sim = Simulation(small_cfg, seed=7, offload_policy="morl", lyapunov_biasing=False)
    eps, max_ident = 1.0, 0.0
    for _ in range(60):
        rec = sim.train_step(eps)
        eps = max(0.05, eps * 0.95)
        max_ident = max(max_ident, abs(rec.handoff_cycles - rec.handoff_cycles_check))
    return {"agent": sim.morl, "max_ident": max_ident}


def _eval_backlog(small_cfg, agent=None, slots=20, seed=123):
    sim = Simulation(small_cfg, seed=seed, offload_policy="morl", lyapunov_biasing=False)
    if agent is not None:
        sim.morl = agent
        sim.morl.epsilon = 0.0
    qc, max_ident = [], 0.0
    for _ in range(slots):
        rec = sim.step()
        qc.append(sim.state.compute_q_cycles.sum())
        max_ident = max(max_ident, abs(rec.handoff_cycles - rec.handoff_cycles_check))
    return np.array(qc), max_ident


def test_conservation_holds_during_training(trained):
    # the in-loop training step must not break the bits->cycles hand-off accounting
    assert trained["max_ident"] < 1e-2


def test_training_reduces_backlog_vs_untrained(small_cfg, trained):
    trained_qc, _ = _eval_backlog(small_cfg, agent=trained["agent"])
    untrained_qc, _ = _eval_backlog(small_cfg, agent=None)  # fresh untrained agent
    assert trained_qc.mean() < 0.5 * untrained_qc.mean(), (
        f"trained mean {trained_qc.mean():.2e} not << untrained {untrained_qc.mean():.2e}"
    )
    assert trained_qc[-1] < untrained_qc[-1]  # lower final backlog too


def test_conservation_holds_under_trained_policy(small_cfg, trained):
    _, max_ident = _eval_backlog(small_cfg, agent=trained["agent"])
    assert max_ident < 1e-2  # hand-off identity still exact under the trained policy
