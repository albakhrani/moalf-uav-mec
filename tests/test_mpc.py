"""Tests for MPC (src/moalf/optimizers/mpc.py), spec §8.

MPC has no closed-form optimum, so these check HAND-COMPUTABLE BEHAVIORS, not
exact values: the UAV moves toward a static device and settles near it; with two
devices it settles between them (shifted by demand ω); v_max and fixed altitude
are respected; no objective weights are held; and — connecting entry 24 — in the
strong-SNR regime (task/energy position-independent) motion is coverage-driven.
"""

from __future__ import annotations

import io
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.objective import Objective, Term
from moalf.optimizers.mpc import MPC
from moalf.system_model.coverage import gaussian_coverage

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def mpc(config) -> MPC:
    return MPC.from_config(config)


@pytest.fixture(scope="module")
def obj(config) -> Objective:
    return Objective.from_config(config)


@pytest.fixture(scope="module")
def sigma(config) -> float:
    return config["objective"]["coverage_radius_sigma_m"]


# --- helpers -----------------------------------------------------------------
def coverage_eval(devices, sigma, omega=None, task_const=0.0, energy_const=0.0):
    """evaluate_positions(positions) -> raw {TASK, ENERGY, COVERAGE}.

    task/energy are CONSTANTS (position-independent), modelling the strong-SNR
    regime (entry 24) where rate is saturated so latency barely depends on UAV
    position; coverage = -D(p) is the only position-dependent term.
    """
    def ev(positions):
        D = gaussian_coverage(positions, devices, sigma, omega)
        return {Term.TASK: task_const, Term.ENERGY: energy_const, Term.COVERAGE: -D}
    return ev


def run(mpc, proj, ev, start, ticks, is_feasible=None):
    pos = np.array([start], dtype=float)  # one UAV (M=1)
    traj = [pos[0].copy()]
    for _ in range(ticks):
        v = mpc.plan(proj, ev, 0, pos, is_feasible=is_feasible)
        pos[0] = pos[0] + v * mpc.dt
        traj.append(pos[0].copy())
    return pos[0], np.array(traj)


# --- single device: approach and settle --------------------------------------
def test_single_device_approach_and_settle(mpc, obj, sigma):
    device = np.array([[200.0, 150.0]])
    proj = obj.projection("mpc")
    ev = coverage_eval(device, sigma)
    start = np.array([0.0, 0.0])
    final, traj = run(mpc, proj, ev, start, ticks=60)

    d_start = np.linalg.norm(start - device[0])
    d_final = np.linalg.norm(final - device[0])
    assert d_final < d_start                 # moved toward it
    assert d_final < 5.0                     # settled essentially on it (well within σ)
    # settled: last few steps barely move
    assert np.linalg.norm(traj[-1] - traj[-5]) < 5.0


def test_first_move_points_toward_device(mpc, obj, sigma):
    device = np.array([[300.0, 0.0]])
    proj = obj.projection("mpc")
    ev = coverage_eval(device, sigma)
    v = mpc.plan(proj, ev, 0, np.array([[0.0, 0.0]]))
    assert v[0] > 0.0                          # heads in +x toward the device
    assert abs(v[1]) < 1e-9                    # straight toward it


# --- two devices: settle between, shifted by demand --------------------------
def test_two_devices_equal_settles_near_midpoint(mpc, obj, sigma):
    devices = np.array([[60.0, 0.0], [-60.0, 0.0]])  # within 2σ -> single peak at midpoint
    proj = obj.projection("mpc")
    ev = coverage_eval(devices, sigma)
    final, _ = run(mpc, proj, ev, np.array([150.0, 100.0]), ticks=60)
    assert np.linalg.norm(final - np.array([0.0, 0.0])) < 10.0  # near midpoint


def test_two_devices_demand_weight_shifts_settle_point(mpc, obj, sigma):
    devices = np.array([[60.0, 0.0], [-60.0, 0.0]])
    omega = np.array([[1.0], [3.0]])  # device at x=-60 is 3x more in demand
    proj = obj.projection("mpc")
    ev = coverage_eval(devices, sigma, omega=omega)
    final, _ = run(mpc, proj, ev, np.array([150.0, 80.0]), ticks=60)
    assert final[0] < 0.0  # settles on the higher-demand (left) side
    assert np.linalg.norm(final - devices[1]) < np.linalg.norm(final - devices[0])


# --- velocity / altitude constraints (eq 3, A5) ------------------------------
def test_velocity_never_exceeds_vmax(mpc, obj, sigma):
    device = np.array([[400.0, 400.0]])
    proj = obj.projection("mpc")
    ev = coverage_eval(device, sigma)
    pos = np.array([[10.0, 5.0]])
    for _ in range(20):
        v = mpc.plan(proj, ev, 0, pos)
        assert np.linalg.norm(v) <= mpc.v_max + 1e-9
        pos[0] = pos[0] + v * mpc.dt


def test_motion_is_2d_fixed_altitude(mpc, obj, sigma):
    device = np.array([[100.0, 100.0]])
    proj = obj.projection("mpc")
    v = mpc.plan(proj, coverage_eval(device, sigma), 0, np.array([[0.0, 0.0]]))
    assert v.shape == (2,)                          # only (x, y); no z control
    assert mpc.altitude_m == pytest.approx(100.0)   # H fixed from config (A5)


# --- no-own-weights structural guard + projection contract -------------------
def test_mpc_has_no_objective_weights(mpc):
    names = {f.name for f in fields(MPC)}
    assert names == {"horizon", "v_max", "dt", "altitude_m", "n_directions", "n_speeds"}
    for forbidden in ("w1_task", "w2_energy", "w6_coverage", "weights", "_weights"):
        assert not hasattr(mpc, forbidden)


def test_plan_requires_mpc_projection(mpc, obj, sigma):
    ev = coverage_eval(np.array([[0.0, 0.0]]), sigma)
    with pytest.raises(ValueError, match="mpc"):
        mpc.plan(obj.projection("apso"), ev, 0, np.array([[1.0, 1.0]]))


def test_plan_rejects_raw_callable(mpc, sigma):
    with pytest.raises(TypeError):
        mpc.plan(lambda r: 0.0, lambda p: {}, 0, np.array([[1.0, 1.0]]))


def test_from_config_rejects_non_direct(config):
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    bad["mpc"] = dict(config["mpc"], formulation="lqr_qp")
    with pytest.raises(ValueError, match="direct"):
        MPC.from_config(bad)


# --- coverage-driven regime (entry 24): motion ignores flat task/energy ------
def test_motion_is_coverage_driven_in_strong_snr_regime(mpc, obj, sigma):
    # Strong-SNR regime: task & energy are position-INDEPENDENT (large constants);
    # only coverage varies with position. The UAV must still seek the device,
    # demonstrating the trajectory is coverage-driven (NOT rate/channel-driven).
    # Connects to entry-24 prediction (38% route result may have weak leverage
    # here). NOTE only — not acted on.
    device = np.array([[180.0, -120.0]])
    proj = obj.projection("mpc")
    ev = coverage_eval(device, sigma, task_const=1.0e9, energy_const=1.0e9)
    final, _ = run(mpc, proj, ev, np.array([0.0, 0.0]), ticks=60)
    assert np.linalg.norm(final - device[0]) < 5.0  # coverage still pulls it home


# --- no-fly feasibility (spec §8.2) ------------------------------------------
def test_respects_no_fly_zone(mpc, obj, sigma):
    device = np.array([[200.0, 0.0]])
    proj = obj.projection("mpc")
    ev = coverage_eval(device, sigma)
    center = np.array([50.0, 0.0])
    radius = 20.0

    def is_feasible(p):
        return float(np.linalg.norm(p - center)) > radius

    final, traj = run(mpc, proj, ev, np.array([0.0, 0.0]), ticks=60, is_feasible=is_feasible)
    # never entered the no-fly disk
    assert np.all(np.linalg.norm(traj - center, axis=1) > radius - 1e-9)
    # still made progress toward the device (went around)
    assert np.linalg.norm(final - device[0]) < np.linalg.norm(np.array([0.0, 0.0]) - device[0])
