"""Increment 3 tests: per-task APSO compute allocation (spec §9).

APSO now allocates each UAV's full capacity C_j across the tasks in its compute
queue (consuming the 'apso' projection, D1). Checks: the capacity constraint
(Σ f ≤ C_j), exact end-to-end conservation under contention, sensible per-task
differentiation, and full service when there is no contention.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.simulation import Simulation, MB_TO_BITS, MCYCLES_TO_CYCLES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def config() -> dict:
    return yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))


def _served(sim, j):
    before = dict(sim.state.compute_items[j])
    ex = sim._serve_compute(j)
    after = sim.state.compute_items[j]
    served = {k: before[k] - after.get(k, 0.0) for k in before}
    return served, ex


# --- capacity constraint (Σ f ≤ C_j·Δt) --------------------------------------
def test_allocation_respects_capacity(config):
    sim = Simulation(config, seed=3)
    j = 0
    sim.state.uav_capacity_cps[j] = 1.0e9
    sim.state.compute_items[j] = {0: 2.0e9, 1: 1.5e9, 2: 8.0e8}  # total 4.3e9 >> cap
    served, ex = _served(sim, j)
    cap = sim.state.uav_capacity_cps[j] * sim.dt
    assert ex <= cap + 1e-3                       # Σ served ≤ C_j·Δt
    assert all(v >= -1e-6 for v in served.values())


def test_each_task_served_at_most_its_remaining(config):
    sim = Simulation(config, seed=4)
    j = 0
    sim.state.uav_capacity_cps[j] = 5.0e9
    rem = {0: 3.0e8, 1: 2.0e8}                    # total 5e8 < cap -> no contention
    sim.state.compute_items[j] = dict(rem)
    served, ex = _served(sim, j)
    for k in rem:
        assert served[k] <= rem[k] + 1e-6
    assert ex == pytest.approx(sum(rem.values()), rel=1e-6)  # all served (full CPU, no contention)
    assert not sim.state.compute_items[j]                     # queue emptied


# --- sensible per-task differentiation under contention ----------------------
def test_larger_task_gets_more_capacity_under_contention(config):
    sim = Simulation(config, seed=3)
    j = 0
    sim.state.uav_capacity_cps[j] = 1.0e9
    sim.state.compute_items[j] = {0: 2.0e9, 1: 5.0e8}  # large + small, contention
    served, ex = _served(sim, j)
    assert served[0] > served[1]                  # larger task gets more capacity
    assert ex == pytest.approx(1.0e9, rel=1e-3)   # full capacity used under contention


# --- exact conservation under per-task allocation + contention ---------------
def _high_load(config):
    return {**config, "task": {**config["task"],
            "generation_rate_tps": {"dist": "uniform", "low": 0.6, "high": 0.6}}}


def test_conservation_holds_with_per_task_allocation(config):
    sim = Simulation(_high_load(config), seed=7, offload_policy="nearest", lyapunov_biasing=True)
    for _ in range(8):
        rec = sim.step()
        # hand-off identity (radio tier) still exact
        assert rec.handoff_cycles == pytest.approx(rec.handoff_cycles_check, rel=1e-9, abs=1e-3)
        # compute cycle balance with per-task drain
        assert rec.qc_after == pytest.approx(
            rec.qc_before + rec.handoff_cycles - rec.executed_cycles, rel=1e-9, abs=1e-1)
        # aggregate equals the sum of per-task items (kept in sync)
        agg = sum(sum(items.values()) for items in sim.state.compute_items)
        assert rec.qc_after == pytest.approx(agg, rel=1e-9, abs=1e-1)


def test_full_task_executes_exactly_W_over_time(config):
    # inject one task, deliver it, and confirm exactly W cycles get executed
    sim = Simulation(config, seed=3)
    sim.lam[:] = 0.0
    sim.state.device_pos[0] = np.array([200.0, 200.0])
    sim.state.uav_pos[:] = np.array([200.0, 200.0])
    W = 300 * MCYCLES_TO_CYCLES
    sim.inject_task(device=0, bits=0.2 * MB_TO_BITS, cycles=W, deadline_s=15.0)
    total_exec = 0.0
    for _ in range(5):
        rec = sim.step()
        total_exec += rec.executed_cycles
    # the task's W cycles are fully executed and the compute queue drains to empty
    assert total_exec == pytest.approx(W, rel=1e-6)
    assert sum(sum(it.values()) for it in sim.state.compute_items) == pytest.approx(0.0, abs=1e-3)
