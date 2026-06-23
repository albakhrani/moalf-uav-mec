"""Tests for the Algorithm-1 skeleton (src/moalf/simulation.py), spec §11.

Verifies the loop runs one slot with the four optimizers wired in order, and that
it CONSERVES what it should — especially the two-tier bits->cycles hand-off
(spec §4.7): cycles handed off equal Σ b_{i,k}·r_{i,k}, and a fully transmitted
task contributes exactly its W cycles (the r·L=W identity end-to-end).

No §18.4 paper comparison is produced (by request) — these are structural and
conservation checks only.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.simulation import MB_TO_BITS, MCYCLES_TO_CYCLES, Simulation, Task

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _silence_arrivals(sim) -> None:
    """Stop Poisson arrivals for controlled scenarios, WITHOUT zeroing the
    objective's λ̄ (which would make S_task=0). Suppresses at the sim level only."""
    sim.lam[:] = 0.0


# --- one slot runs end to end ------------------------------------------------
def test_single_slot_runs(config):
    sim = Simulation(config, seed=42)
    rec = sim.step()
    assert rec.t == 0
    assert sim.state.t == 1                      # advanced one slot
    assert sim.state.uav_pos.shape == (sim.M, 2)  # 2-D positions (A5)
    assert np.all(np.isfinite(sim.state.uav_energy_j))
    assert "drift" in rec.lyapunov                # Lyapunov was called


def test_four_optimizers_are_wired(config):
    # smoke: a slot with pending work exercises MORL (assignment), MPC (motion),
    # APSO (allocation), Lyapunov (diagnostics) without error.
    sim = Simulation(config, seed=1)
    _silence_arrivals(sim)
    sim.inject_task(device=0, bits=0.5 * MB_TO_BITS, cycles=500 * MCYCLES_TO_CYCLES,
                    deadline_s=10.0, priority=3.0)
    pos_before = sim.state.uav_pos.copy()
    rec = sim.step()
    assert rec.assignment  # MORL produced an assignment for device 0
    assert not np.allclose(sim.state.uav_pos, pos_before)  # MPC moved UAVs


# --- conservation: the hand-off identity (the core check) --------------------
def test_handoff_cycles_equal_sum_b_times_intensity(config):
    sim = Simulation(config, seed=7)
    for _ in range(5):
        rec = sim.step()
        # cycles handed off this slot == Σ b_{i,k}·r_{i,k} (independent recomputation)
        assert rec.handoff_cycles == pytest.approx(rec.handoff_cycles_check, rel=1e-9, abs=1e-6)


def test_full_task_hands_off_exactly_W_cycles(config):
    # one device, one task, a reachable UAV, no arrivals: if the whole task
    # transmits in the slot, exactly W cycles must enter the compute tier (r·L=W).
    sim = Simulation(config, seed=3)
    _silence_arrivals(sim)
    # place a UAV right over the device so the link is fast enough to send it all
    sim.state.device_pos[0] = np.array([200.0, 200.0])
    sim.state.uav_pos[:] = np.array([200.0, 200.0])
    L = 0.2 * MB_TO_BITS
    W = 300 * MCYCLES_TO_CYCLES
    sim.inject_task(device=0, bits=L, cycles=W, deadline_s=15.0, priority=2.0)

    qc_before = sim.state.compute_q_cycles.sum()
    rec = sim.step()
    # the task fully transmitted -> radio queue empty, exactly W cycles handed off
    assert sum(len(q) for q in sim.state.radio_q) == 0
    assert rec.transmitted_bits == pytest.approx(L)
    assert rec.handoff_cycles == pytest.approx(W, rel=1e-9)
    # compute queue gained W minus whatever was executed this slot
    assert (sim.state.compute_q_cycles.sum() + rec.executed_cycles) == pytest.approx(qc_before + W, rel=1e-9)


# --- conservation: bit and cycle balances ------------------------------------
def test_radio_bit_balance(config):
    sim = Simulation(config, seed=11)
    for _ in range(5):
        rec = sim.step()
        # Q^r_after = Q^r_before + arrivals - transmitted
        assert rec.qr_after == pytest.approx(rec.qr_before + rec.arrivals_bits - rec.transmitted_bits, rel=1e-9, abs=1e-6)


def test_compute_cycle_balance(config):
    sim = Simulation(config, seed=13)
    for _ in range(5):
        rec = sim.step()
        # Q^c_after = Q^c_before + handoff - executed
        assert rec.qc_after == pytest.approx(rec.qc_before + rec.handoff_cycles - rec.executed_cycles, rel=1e-9, abs=1e-3)


# --- energy / coverage updates -----------------------------------------------
def test_energy_capped_and_updated(config):
    sim = Simulation(config, seed=2)
    e_before = sim.state.uav_energy_j.copy()
    sim.step()
    assert np.all(sim.state.uav_energy_j <= sim.energy.capacity_j + 1e-6)  # E_max cap
    assert not np.array_equal(sim.state.uav_energy_j, e_before)            # changed


def test_coverage_in_range(config):
    sim = Simulation(config, seed=4)
    rec = sim.step()
    # D(p) = Σ_i Σ_j exp(...) ∈ [0, N*M]
    assert 0.0 <= rec.coverage <= sim.N * sim.M + 1e-6


# --- Increment 1: Lyapunov drift-plus-penalty biasing ------------------------
def _high_load(config) -> dict:
    """Test-local high-load config (λ≈0.6: demand ~94% of total compute capacity)
    so an imbalanced base policy overloads UAVs and load-balancing matters.
    Stresses the mechanism — NOT the frozen real config, NOT tuning a result."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["task"] = dict(config["task"],
                       generation_rate_tps={"dist": "uniform", "low": 0.6, "high": 0.6})
    return cfg


def _run_compute_backlog(cfg, biasing, slots=25, seed=7):
    sim = Simulation(cfg, seed=seed, offload_policy="nearest", lyapunov_biasing=biasing)
    qc, max_ident = [], 0.0
    for _ in range(slots):
        rec = sim.step()
        qc.append(float(sim.state.compute_q_cycles.sum()))
        max_ident = max(max_ident, abs(rec.handoff_cycles - rec.handoff_cycles_check))
    return np.array(qc), max_ident


def test_biasing_materially_reduces_queue_growth(config):
    cfg = _high_load(config)
    off, _ = _run_compute_backlog(cfg, biasing=False)
    on, _ = _run_compute_backlog(cfg, biasing=True)
    # same seed, same fixed 'nearest' base policy: biasing must bend drift down
    assert on.mean() < 0.5 * off.mean(), f"ON mean {on.mean():.3e} not << OFF {off.mean():.3e}"
    assert on[-1] < off[-1]  # lower final backlog too


def test_biasing_preserves_conservation_exactly(config):
    # the biasing changes DECISIONS, not the bit/cycle accounting
    cfg = _high_load(config)
    sim = Simulation(cfg, seed=7, offload_policy="nearest", lyapunov_biasing=True)
    for _ in range(10):
        rec = sim.step()
        assert rec.handoff_cycles == pytest.approx(rec.handoff_cycles_check, rel=1e-9, abs=1e-3)
        assert rec.qr_after == pytest.approx(rec.qr_before + rec.arrivals_bits - rec.transmitted_bits, rel=1e-9, abs=1e-3)
        assert rec.qc_after == pytest.approx(rec.qc_before + rec.handoff_cycles - rec.executed_cycles, rel=1e-9, abs=1e-1)


def test_biasing_changes_assignment(config):
    cfg = _high_load(config)
    sim_off = Simulation(cfg, seed=7, offload_policy="nearest", lyapunov_biasing=False)
    sim_on = Simulation(cfg, seed=7, offload_policy="nearest", lyapunov_biasing=True)
    differed = False
    for _ in range(6):
        r_off = sim_off.step()
        r_on = sim_on.step()
        if r_off.assignment != r_on.assignment:
            differed = True
    assert differed, "biasing never changed the assignment vs the base policy"


# --- Task hand-off intensity identity (unit) ---------------------------------
def test_task_intensity_identity():
    L = 0.5 * MB_TO_BITS
    W = 400 * MCYCLES_TO_CYCLES
    tk = Task(0, L, W, 10.0, 1.0, 0, L)
    assert tk.intensity * tk.bits == pytest.approx(tk.cycles)  # r·L = W
