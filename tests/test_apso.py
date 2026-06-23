"""Tests for APSO (src/moalf/optimizers/apso.py), spec §9.

Covers: construction from config, adaptive inertia decay, convergence to a known
optimum on simple test functions, the structural guard that APSO holds NO
objective weights, and that allocation optimization is scored only through the
'apso' projection of the single objective.
"""

from __future__ import annotations

import io
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.objective import Objective, Term
from moalf.optimizers.apso import APSO, PSOResult

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def apso(config) -> APSO:
    return APSO.from_config(config)


@pytest.fixture(scope="module")
def obj(config) -> Objective:
    return Objective.from_config(config)


# --- construction / hyperparameters ------------------------------------------
def test_from_config(apso, config):
    a = config["apso"]
    assert apso.swarm_size == a["swarm_size"]
    assert apso.inertia_start == a["inertia_start"]
    assert apso.inertia_end == a["inertia_end"]
    assert apso.cognitive_c1 == a["cognitive_c1"]
    assert apso.social_c2 == a["social_c2"]
    assert apso.max_iterations == a["max_iterations"]


def test_inertia_decays_linearly(apso):
    n = apso.max_iterations
    assert apso.inertia(0, n) == pytest.approx(apso.inertia_start)
    assert apso.inertia(n - 1, n) == pytest.approx(apso.inertia_end)
    mids = [apso.inertia(i, n) for i in range(n)]
    assert all(a >= b for a, b in zip(mids, mids[1:]))  # non-increasing
    assert apso.inertia(0, 1) == pytest.approx(apso.inertia_start)  # degenerate guard


# --- convergence to a known optimum ------------------------------------------
def test_converges_on_sphere(apso):
    rng = np.random.default_rng(0)
    res = apso.minimize(lambda x: float(np.sum(x * x)), [-5, -5, -5], [5, 5, 5], rng)
    assert isinstance(res, PSOResult)
    assert res.best_fitness < 1e-2
    assert np.allclose(res.best_position, 0.0, atol=0.1)


def test_converges_on_shifted_optimum(apso):
    rng = np.random.default_rng(1)
    target = np.array([2.5, -1.0])
    res = apso.minimize(lambda x: float(np.sum((x - target) ** 2)), [-5, -5], [5, 5], rng)
    assert res.best_fitness < 1e-2
    assert np.allclose(res.best_position, target, atol=0.1)


def test_history_is_monotone_nonincreasing(apso):
    rng = np.random.default_rng(2)
    res = apso.minimize(lambda x: float(np.sum(x * x)), [-5, -5], [5, 5], rng)
    assert all(a >= b for a, b in zip(res.history, res.history[1:]))


def test_respects_box_bounds(apso):
    rng = np.random.default_rng(3)
    # optimum at 0 lies outside [2, 5]; best must stay in-box at the boundary
    res = apso.minimize(lambda x: float(np.sum(x * x)), [2, 2], [5, 5], rng)
    assert np.all(res.best_position >= 2.0 - 1e-9)
    assert np.all(res.best_position <= 5.0 + 1e-9)


# --- structural guard: APSO holds NO objective weights -----------------------
def test_apso_has_no_objective_weights(apso):
    names = {f.name for f in fields(APSO)}
    assert names == {
        "swarm_size", "inertia_start", "inertia_end",
        "cognitive_c1", "social_c2", "max_iterations",
    }
    # none of the objective weight keys may appear as an attribute
    for forbidden in ("w1_task", "w2_energy", "w3_completion", "w5_util", "w6_coverage",
                      "weights", "_weights"):
        assert not hasattr(apso, forbidden)


def test_from_config_ignores_objective_weights(config):
    # Even if someone slips weights into the apso config block, APSO won't carry them.
    poisoned = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    poisoned["apso"] = dict(config["apso"], w2_energy=99.0)
    a = APSO.from_config(poisoned)
    assert not hasattr(a, "w2_energy")
    assert {f.name for f in fields(APSO)}.isdisjoint({"w2_energy"})


# --- allocation scored ONLY through the 'apso' projection --------------------
def test_optimize_allocation_uses_projection_and_finds_optimum(apso, obj):
    proj = obj.projection("apso")  # terms {TASK, ENERGY, UTIL}

    # raw terms convex in the allocation x=(x0,x1) on [0,1]^2:
    #   J~task = (x0-0.3)^2, J~energy = (x1-0.6)^2, J~util = -x0
    # projection.value = w_task*(x0-0.3)^2 + w_energy*(x1-0.6)^2 - w_util*x0
    def evaluate_terms(x):
        return {
            Term.TASK: obj.scale(Term.TASK) * (x[0] - 0.3) ** 2,
            Term.ENERGY: obj.scale(Term.ENERGY) * (x[1] - 0.6) ** 2,
            Term.UTIL: -obj.scale(Term.UTIL) * x[0],
        }

    rng = np.random.default_rng(7)
    res = apso.optimize_allocation(proj, evaluate_terms, [0.0, 0.0], [1.0, 1.0], rng)

    # analytic optimum from the MASTER weights (proves the master weighting is used)
    w_task = obj.weight(Term.TASK)
    w_energy = obj.weight(Term.ENERGY)
    w_util = obj.weight(Term.UTIL)
    x0_opt = min(max(0.3 + w_util / (2.0 * w_task), 0.0), 1.0)
    x1_opt = 0.6
    assert np.allclose(res.best_position, [x0_opt, x1_opt], atol=0.05)

    # the reported fitness is exactly the projection value at the best position
    assert res.best_fitness == pytest.approx(proj.value(evaluate_terms(res.best_position)))


def test_optimize_allocation_rejects_wrong_projection(apso, obj):
    rng = np.random.default_rng(0)
    ev = lambda x: {Term.TASK: 0.0, Term.ENERGY: 0.0, Term.COMPLETION: 0.0, Term.UTIL: 0.0}
    with pytest.raises(ValueError, match="apso"):
        apso.optimize_allocation(obj.projection("morl"), ev, [0.0], [1.0], rng)


def test_optimize_allocation_requires_a_projection(apso):
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        # passing a raw callable instead of a Projection must fail (no inline weights)
        apso.optimize_allocation(lambda r: 0.0, lambda x: {}, [0.0], [1.0], rng)
