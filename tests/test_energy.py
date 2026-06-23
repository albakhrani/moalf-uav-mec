"""Tests for the energy model (src/moalf/system_model/energy.py).

Covers spec §4.6 (eq 10 harvesting/update; constraint 23): Wh->J conversions,
component unit checks, the E_max cap, a limiting case (harvest-only saturates at
E_max), depletion detection, and the compute-energy coupling to cycles executed
(the §4.7 hand-off feed). All parameters come from config/default.yaml.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.system_model.energy import EnergyModel
from moalf.system_model.computation import WH_TO_J

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def em(config) -> EnergyModel:
    return EnergyModel.from_config(config)


# --- construction / unit conversion ------------------------------------------
def test_capacity_and_reserve_converted_wh_to_j(em):
    assert em.capacity_j == pytest.approx(1000 * WH_TO_J)     # 1000 Wh -> 3.6e6 J
    assert em.min_reserve_j == pytest.approx(100 * WH_TO_J)   # 100 Wh  -> 3.6e5 J
    assert em.harvest_power_w == pytest.approx(5.0)
    assert em.flight_power_w == pytest.approx(100.0)


def test_rejects_unknown_harvest_model(config):
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    bad["energy"] = dict(config["energy"], harvest_model="p_harv")
    with pytest.raises(ValueError, match="linear_eta_dt"):
        EnergyModel.from_config(bad)


# --- component unit checks ---------------------------------------------------
def test_harvested_per_slot(em):
    # eta_j * dt = 5 W * 1 s = 5 J
    assert em.harvested_j() == pytest.approx(5.0)


def test_consumed_flight_only_when_moving(em):
    # not moving, no compute -> 0 J
    assert float(em.consumed_j(0.0, moving=0)) == pytest.approx(0.0)
    # moving, no compute -> P_flight * dt = 100 J
    assert float(em.consumed_j(0.0, moving=1)) == pytest.approx(100.0)


def test_consumed_includes_compute_energy(em):
    # compute energy = e_c * cycles; 1e9 cycles -> 3600 J (matches computation.py)
    moving_cost = em.flight_power_w * em.dt_s
    assert float(em.consumed_j(1e9, moving=1)) == pytest.approx(moving_cost + 3600.0)
    assert float(em.consumed_j(1e9, moving=0)) == pytest.approx(3600.0)


def test_negative_cycles_rejected(em):
    with pytest.raises(ValueError):
        em.consumed_j(-1.0, moving=0)


# --- eq (10): one-slot update ------------------------------------------------
def test_step_net_change(em):
    E0 = em.capacity_j - 1e6  # below cap so no clamping
    # moving, 1e9 cycles: consumed = 100 + 3600 = 3700 J; harvested = 5 J
    E1 = float(em.step(E0, cycles_executed=1e9, moving=1))
    assert E1 == pytest.approx(E0 - 3700.0 + 5.0)


def test_step_caps_at_capacity(em):
    # at full charge, idle, harvesting -> stays at E_max (cannot exceed cap)
    E1 = float(em.step(em.capacity_j, cycles_executed=0.0, moving=0))
    assert E1 == pytest.approx(em.capacity_j)
    # even starting slightly below, if harvest > consumption it cannot pass cap
    E2 = float(em.step(em.capacity_j - 1.0, cycles_executed=0.0, moving=0))
    assert E2 <= em.capacity_j


def test_step_never_exceeds_capacity_invariant(em):
    rng = np.random.default_rng(0)
    Es = rng.uniform(0, em.capacity_j, size=1000)
    cyc = rng.uniform(0, 1e9, size=1000)
    mov = rng.integers(0, 2, size=1000)
    out = em.step(Es, cycles_executed=cyc, moving=mov)
    assert np.all(out <= em.capacity_j + 1e-9)


# --- limiting case: harvest-only saturates at E_max --------------------------
def test_harvest_only_saturates_at_capacity(em):
    # Start within a few harvest-steps of the cap; harvest-only must climb to
    # E_max and then stay there (monotone non-decreasing, bounded by cap).
    step_gain = em.harvested_j()                  # 5 J/slot
    E = em.capacity_j - 2.5 * step_gain           # ~3 steps from full
    prev = -np.inf
    for _ in range(5):
        E = float(em.step(E, cycles_executed=0.0, moving=0))
        assert E >= prev                          # non-decreasing
        assert E <= em.capacity_j                 # never exceeds cap
        prev = E
    assert E == pytest.approx(em.capacity_j)       # saturated
    assert float(em.step(E, 0.0, 0)) == pytest.approx(em.capacity_j)  # stays capped


def test_consumption_only_drains_monotonically(em):
    E = em.capacity_j
    prev = E
    for _ in range(5):
        E = float(em.step(E, cycles_executed=5e8, moving=1))  # 100 + 1800 = 1900 J; net -1895 J
        assert E < prev
        prev = E


# --- constraint (23): depletion ----------------------------------------------
def test_is_depleted(em):
    assert bool(em.is_depleted(em.min_reserve_j - 1.0)) is True
    assert bool(em.is_depleted(em.min_reserve_j + 1.0)) is False


def test_step_may_fall_below_reserve_no_clamp(em):
    # spec eq (10) has no lower clamp; large consumption drives E below E_min.
    E = em.min_reserve_j + 1000.0
    E1 = float(em.step(E, cycles_executed=1e9, moving=1))  # consumes ~3700 J
    assert E1 < em.min_reserve_j
    assert bool(em.is_depleted(E1)) is True
