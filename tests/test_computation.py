"""Tests for the computation model (src/moalf/system_model/computation.py).

Covers spec §4.5 (eq 8 execution time; linear execution energy per B6):
unit checks, a limiting case (t_exec -> 0 as capacity -> inf), the e_c Wh->J
conversion, linearity/proportionality, vectorization, and guard errors. All
parameters come from config/default.yaml.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.system_model.computation import ComputationModel, WH_TO_J

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def comp(config) -> ComputationModel:
    return ComputationModel.from_config(config)


# --- construction / unit conversion ------------------------------------------
def test_energy_per_cycle_converted_wh_to_j(comp, config):
    e_c_wh = config["uav"]["compute_energy_rate_wh_per_cycle"]
    assert comp.energy_per_cycle_j == pytest.approx(e_c_wh * WH_TO_J)
    assert comp.energy_per_cycle_j == pytest.approx(3.6e-6)  # 1e-9 Wh/cycle * 3600


def test_rejects_non_linear_energy_model(config):
    bad = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}
    bad["computation"] = {"energy_model": "quadratic"}
    with pytest.raises(ValueError, match="linear"):
        ComputationModel.from_config(bad)


# --- eq (8): execution time --------------------------------------------------
def test_exec_time_definition(comp):
    # W = 1000 Megacycles = 1e9 cycles, C = 2 GHz = 2e9 cycles/s -> 0.5 s
    assert float(comp.exec_time_s(1e9, 2e9)) == pytest.approx(0.5)


def test_exec_time_units_seconds(comp):
    # W [cycles] / C [cycles/s] = [s]; faster UAV -> shorter time
    W = 5.5e8  # mean task (550 Mcyc)
    t_slow = float(comp.exec_time_s(W, 2e9))
    t_fast = float(comp.exec_time_s(W, 5e9))
    assert t_slow > t_fast > 0.0


def test_exec_time_limiting_capacity_to_infinity(comp):
    W = 1e9
    times = [float(comp.exec_time_s(W, C)) for C in (1e9, 1e12, 1e15, 1e18)]
    assert all(t1 > t2 for t1, t2 in zip(times, times[1:]))  # strictly decreasing
    assert times[-1] == pytest.approx(0.0, abs=1e-6)          # -> 0 as C -> inf


def test_exec_time_scales_inversely_with_capacity(comp):
    W = 4.2e8
    assert float(comp.exec_time_s(W, 2e9)) == pytest.approx(2.0 * float(comp.exec_time_s(W, 4e9)))


def test_exec_time_vectorized(comp):
    W = np.array([1e8, 5e8, 1e9])
    C = 2e9
    t = comp.exec_time_s(W, C)
    assert t.shape == (3,)
    assert np.allclose(t, W / C)


# --- §4.5 (B6): linear execution energy --------------------------------------
def test_exec_energy_linear_in_work(comp):
    e1 = float(comp.exec_energy_j(1e9))
    e2 = float(comp.exec_energy_j(2e9))
    assert e2 == pytest.approx(2.0 * e1)               # strictly proportional
    assert float(comp.exec_energy_j(0.0)) == 0.0       # no work -> no energy


def test_exec_energy_value(comp):
    # 1e9 cycles * 3.6e-6 J/cycle = 3600 J
    assert float(comp.exec_energy_j(1e9)) == pytest.approx(3600.0)


def test_exec_energy_independent_of_capacity(comp):
    # Linear model: energy depends only on cycles, not on clock speed.
    # (No capacity argument exists; this documents the modeling choice.)
    assert float(comp.exec_energy_j(7e8)) == pytest.approx(comp.energy_per_cycle_j * 7e8)


# --- guards ------------------------------------------------------------------
def test_zero_capacity_raises(comp):
    with pytest.raises(ValueError, match="C_j"):
        comp.exec_time_s(1e9, 0.0)


def test_negative_work_raises(comp):
    with pytest.raises(ValueError):
        comp.exec_time_s(-1.0, 2e9)
    with pytest.raises(ValueError):
        comp.exec_energy_j(-1.0)
