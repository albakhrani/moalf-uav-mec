"""Tests for the keystone objective module (src/moalf/objective.py).

Validates the single-objective architecture (spec §5, §6, D1):
  - all five normalized terms come out O(1) under representative, independently
    computed config-driven inputs (the check that the S_energy and S_coverage
    fixes wired through and no term dominates);
  - projections return the correct §6 subsets, and a projection's weights are
    IDENTICAL to the master's (not redefined);
  - there is structurally only one weighting scheme (no w_4, immutable weights,
    single source).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml

from moalf.objective import Accrual, Objective, Projection, Role, Term, WH_TO_J

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def obj(config) -> Objective:
    return Objective.from_config(config)


# --- helper: representative raw term totals, computed INDEPENDENTLY of obj ----
def representative_raw(config) -> dict:
    """Physically plausible per-run raw term totals derived straight from config
    primitives (NOT from obj.scales), so they genuinely test the scale magnitudes.
    """
    net, uav, task = config["network"], config["uav"], config["task"]
    N = net["num_iot_devices"]
    M = net["num_uavs"]
    Th = net["num_time_steps"]
    dt = net["time_step_s"]
    lam = 0.5 * (task["generation_rate_tps"]["low"] + task["generation_rate_tps"]["high"])
    tau = 0.5 * (task["deadline_s"]["low"] + task["deadline_s"]["high"])
    w_cyc = 0.5 * (task["compute_req_megacycles"]["low"] + task["compute_req_megacycles"]["high"]) * 1e6
    e_c_j = uav["compute_energy_rate_wh_per_cycle"] * WH_TO_J
    p_flight = uav["flight_energy_rate_w"]
    n_task = N * lam * Th

    return {
        # tasks complete in ~half their deadline on average
        Term.TASK: 0.5 * n_task * tau,
        # all tasks executed (compute) + UAVs moving 70% of slots (flight)
        Term.ENERGY: n_task * w_cyc * e_c_j + 0.7 * M * p_flight * dt * Th,
        # ~85% of tasks on time (negative: it is a reward term)
        Term.COMPLETION: -0.85 * n_task,
        # ~50% average utilization across UAV-slots
        Term.UTIL: -0.5 * M * Th,
        # ~80% device-coverage every slot over the horizon
        Term.COVERAGE: -0.8 * N * Th,
    }


# --- O(1) normalization (the core check) -------------------------------------
def test_each_normalized_term_is_order_unity(obj, config):
    raw = representative_raw(config)
    norm = obj.normalized_terms(raw)
    for term, val in norm.items():
        assert 0.1 <= abs(val) <= 10.0, f"{term.name}: J̃ = {val} not O(1)"


def test_coverage_scale_includes_horizon(obj, config):
    # Direct guard for entry-28 fix: if S_coverage were N (not N*T_h),
    # J̃_coverage would be ~800, not ~0.8.
    raw = representative_raw(config)
    assert abs(obj.normalize(Term.COVERAGE, raw[Term.COVERAGE])) < 2.0


def test_energy_scale_is_total_not_flight_only(obj, config):
    # entry-26 fix: S_energy must include compute (~40x flight). A flight-only
    # scale would make J̃_energy ~40.
    raw = representative_raw(config)
    assert abs(obj.normalize(Term.ENERGY, raw[Term.ENERGY])) < 5.0


# --- master objective value --------------------------------------------------
def test_master_value_sums_all_five_terms(obj, config):
    raw = representative_raw(config)
    expected = sum(obj.weight(t) * obj.normalize(t, raw[t]) for t in Term)
    assert obj.value(raw) == pytest.approx(expected)


def test_master_value_requires_all_terms(obj, config):
    raw = representative_raw(config)
    del raw[Term.COVERAGE]
    with pytest.raises(KeyError):
        obj.value(raw)


# --- projections (spec §6) ---------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("morl", (Term.TASK, Term.ENERGY, Term.COMPLETION, Term.UTIL)),  # {1,2,3,5}
        ("mpc", (Term.TASK, Term.ENERGY, Term.COVERAGE)),                # {1,2,6}
        ("apso", (Term.TASK, Term.ENERGY, Term.UTIL)),                   # {1,2,5}
    ],
)
def test_projection_subsets_match_spec_section_6(obj, name, expected):
    proj = obj.projection(name)
    assert proj.terms == expected


def test_morl_excludes_coverage_and_migration(obj):
    morl = obj.projection("morl")
    assert Term.COVERAGE not in morl.terms
    # migration has no Term at all
    assert all(t.value != 4 for t in morl.terms)


def test_projection_value_matches_partial_sum(obj, config):
    raw = representative_raw(config)
    for name in ("morl", "mpc", "apso"):
        proj = obj.projection(name)
        expected = sum(obj.weight(t) * obj.normalize(t, raw[t]) for t in proj.terms)
        assert proj.value(raw) == pytest.approx(expected)


def test_projection_weights_are_identical_to_master(obj):
    for name in ("morl", "mpc", "apso"):
        proj = obj.projection(name)
        for t in proj.terms:
            # same numeric value AND sourced from the same objective instance
            assert proj.weights[t] == obj.weights[t]
        assert proj.objective is obj  # single source, not a copy


def test_unknown_projection_raises(obj):
    with pytest.raises(KeyError):
        obj.projection("genetic")


def test_custom_projection_requires_term_members(obj):
    with pytest.raises(TypeError):
        obj.projection(["task", "energy"])  # strings, not Term


# --- single weighting scheme (structural guards) -----------------------------
def test_no_w4_term_exists():
    values = [t.value for t in Term]
    assert 4 not in values
    assert len(Term) == 5
    assert set(values) == {1, 2, 3, 5, 6}


def test_from_config_rejects_w4_migration(config):
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    bad["objective"] = dict(config["objective"], w4_migration=1.0)
    with pytest.raises(ValueError, match="w4_migration"):
        Objective.from_config(bad)


def test_master_weights_are_immutable(obj):
    with pytest.raises(TypeError):
        obj.weights[Term.TASK] = 999.0  # MappingProxyType is read-only


def test_projection_weights_are_immutable(obj):
    proj = obj.projection("morl")
    with pytest.raises(TypeError):
        proj.weights[Term.TASK] = 999.0


def test_mutating_weight_view_does_not_affect_objective(obj):
    snapshot = dict(obj.weights)
    leaked = dict(obj.weights)
    leaked[Term.TASK] = 12345.0
    assert obj.weights[Term.TASK] == snapshot[Term.TASK]  # unchanged


def test_all_named_projections_share_one_weight_source(obj):
    projs = [obj.projection(n) for n in ("morl", "mpc", "apso")]
    assert all(p.objective is obj for p in projs)
    # the only weights anywhere are obj's; projections expose the same values
    for p in projs:
        for t in p.terms:
            assert p.weights[t] is obj.weights[t] or p.weights[t] == obj.weights[t]


# --- sign convention (spec §5.2): costs positive, rewards negative -----------
def test_role_registry_matches_spec_section_52(obj):
    assert obj.role(Term.TASK) is Role.COST
    assert obj.role(Term.ENERGY) is Role.COST
    assert obj.role(Term.COMPLETION) is Role.REWARD
    assert obj.role(Term.UTIL) is Role.REWARD
    assert obj.role(Term.COVERAGE) is Role.REWARD


def test_representative_raw_signs_match_roles(obj, config):
    """A flipped reward sign would pass the O(1) magnitude check but invert the
    optimizer's intent — guard it here."""
    raw = representative_raw(config)
    for term, value in raw.items():
        if obj.role(term) is Role.COST:
            assert value >= 0.0, f"{term.name} is a COST but raw {value} < 0"
        else:
            assert value <= 0.0, f"{term.name} is a REWARD but raw {value} > 0"


def test_normalization_preserves_sign(obj, config):
    # scales are strictly positive, so J̃_m keeps the sign of J_m (role).
    raw = representative_raw(config)
    for term, value in raw.items():
        norm = obj.normalize(term, value)
        assert (norm >= 0.0) == (obj.role(term) is Role.COST)
        assert obj.scale(term) > 0.0


# --- accrual basis registry (hardening: explicit per-task vs per-slot) --------
def test_accrual_registry_is_correct(obj):
    assert obj.accrual(Term.TASK) is Accrual.PER_TASK
    assert obj.accrual(Term.COMPLETION) is Accrual.PER_TASK
    assert obj.accrual(Term.UTIL) is Accrual.PER_SLOT
    assert obj.accrual(Term.COVERAGE) is Accrual.PER_SLOT
    assert obj.accrual(Term.ENERGY) is Accrual.MIXED


def test_scales_unchanged_after_accrual_refactor(obj, config):
    """The accrual refactor is structure-only: values must equal the §5.3 formulas."""
    net, uav, task = config["network"], config["uav"], config["task"]
    N, M, Th, dt = (net["num_iot_devices"], net["num_uavs"],
                    net["num_time_steps"], net["time_step_s"])
    lam = 0.5 * (task["generation_rate_tps"]["low"] + task["generation_rate_tps"]["high"])
    tau = 0.5 * (task["deadline_s"]["low"] + task["deadline_s"]["high"])
    w_cyc = 0.5 * (task["compute_req_megacycles"]["low"] + task["compute_req_megacycles"]["high"]) * 1e6
    e_c_j = uav["compute_energy_rate_wh_per_cycle"] * WH_TO_J
    pf = uav["flight_energy_rate_w"]
    n_task = N * lam * Th

    assert obj.scale(Term.TASK) == pytest.approx(n_task * tau)
    assert obj.scale(Term.COMPLETION) == pytest.approx(n_task)
    assert obj.scale(Term.UTIL) == pytest.approx(M * Th)
    assert obj.scale(Term.COVERAGE) == pytest.approx(N * Th)
    assert obj.scale(Term.ENERGY) == pytest.approx(n_task * w_cyc * e_c_j + M * pf * dt * Th)
