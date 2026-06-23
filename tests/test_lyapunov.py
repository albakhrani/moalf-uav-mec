"""Tests for the Lyapunov controller (src/moalf/optimizers/lyapunov.py), spec §10.

Covers the two-tier Lyapunov value/drift/drift-plus-penalty and — Increment 1 —
the drift-plus-penalty assignment biasing (§10.1): hand-checkable routing off a
loaded compute queue, load spreading, and the V·penalty influence.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.optimizers.lyapunov import LyapunovController

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_from_config(config):
    c = LyapunovController.from_config(config)
    assert c.V == config["lyapunov"]["penalty_weight_V"]
    assert c.c_Q == config["lyapunov"]["tier_scaling_c_Q"]


# --- eqs 32'/33'/34' ---------------------------------------------------------
def test_two_tier_lyapunov_value():
    c = LyapunovController(V=1.0, c_Q=0.25)
    qr = [3.0, 4.0]      # Σ qr² = 25
    qc = [2.0]           # Σ qc² = 4
    assert c.lyapunov_value(qr, qc) == pytest.approx(0.5 * (25 + 0.25 * 4))


def test_drift_and_drift_plus_penalty():
    c = LyapunovController(V=2.0, c_Q=1.0)
    assert c.drift(10.0, 13.0) == pytest.approx(3.0)
    assert c.drift_plus_penalty(3.0, 5.0) == pytest.approx(3.0 + 2.0 * 5.0)


def test_evaluate_reports_drift(config):
    c = LyapunovController.from_config(config)
    rec = c.evaluate([1.0], [0.0], [2.0], [0.0], penalty=0.0)
    assert rec["drift"] == pytest.approx(rec["L_after"] - rec["L_before"])
    assert rec["L_after"] > rec["L_before"]


# --- §10.1 biasing (hand-checkable) ------------------------------------------
def test_biasing_routes_off_loaded_compute_queue():
    # pure drift (V=0): a device must NOT be routed to the loaded UAV (index 1)
    c = LyapunovController(V=0.0, c_Q=1.0)
    qc = np.array([0.0, 10.0, 0.0])
    inc = lambda i, j: 1.0
    pen = lambda i, j: 0.0
    a = c.biased_assignment([0], [0, 1, 2], qc, inc, pen)
    assert a[0] != 1 and a[0] in (0, 2)


def test_biasing_spreads_load_across_uavs():
    # pure drift, equal queues, constant incoming: three devices should spread to
    # three distinct UAVs (greedy running-backlog accounting).
    c = LyapunovController(V=0.0, c_Q=1.0)
    qc = np.zeros(3)
    inc = lambda i, j: 1.0
    pen = lambda i, j: 0.0
    a = c.biased_assignment([0, 1, 2], [0, 1, 2], qc, inc, pen)
    assert set(a.values()) == {0, 1, 2}


def test_biasing_respects_penalty_term():
    # pure penalty (c_Q=0): route to the cheapest-objective UAV regardless of queue
    c = LyapunovController(V=1.0, c_Q=0.0)
    qc = np.array([0.0, 0.0, 0.0])
    inc = lambda i, j: 1.0
    pen = lambda i, j: 0.0 if j == 2 else 5.0
    a = c.biased_assignment([0], [0, 1, 2], qc, inc, pen)
    assert a[0] == 2
