"""objective — THE single objective function for the whole framework (spec §5; D1).

This module is the structural heart of the spec. There is exactly ONE objective
(the 5-term normalized weighted sum, spec §5.1), and the weights live ONLY here.
Every optimizer (MORL, MPC, APSO) scores candidate decisions through a
**projection** of this objective — a named subset of its terms (spec §6) that
carries *these* weights and *these* normalizations. By construction an optimizer
receives no weights to store and has nowhere to define a second weighting scheme.

Master objective (spec §5.1, supersedes eq 14):

    min  J = Σ_{m ∈ {1,2,3,5,6}}  w_m · J̃_m ,    J̃_m = J_m / S_m

The five active terms (spec §5.2). Index m=4 (J_migration) is RETIRED with
migration (A4): there is no Term for it and no w_4 anywhere.

    m=1 TASK        latency            J_task        (per-task accrual)   §5.2
    m=2 ENERGY      total energy       J_energy      (compute+flight)     §5.2
    m=3 COMPLETION  on-time count      J_completion  (per-task accrual)   §5.2
    m=5 UTIL        utilization        J_util        (per-slot accrual)   §5.2
    m=6 COVERAGE    Gaussian coverage  J_coverage    (per-slot accrual)   §5.2

Normalization reference scales S_m (spec §5.3, with the corrected total-energy
S_energy — entry 26 — and the corrected per-slot S_coverage = N·T_h — entry 28).
All J̃_m are dimensionless and O(1) under representative operation.

Projections (spec §6):
    morl -> {TASK, ENERGY, COMPLETION, UTIL}   (m∈{1,2,3,5})
    mpc  -> {TASK, ENERGY, COVERAGE}           (m∈{1,2,6})
    apso -> {TASK, ENERGY, UTIL}               (m∈{1,2,5})

All parameters are read from config (no hard-coded values).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Union

WH_TO_J = 3600.0  # 1 watt-hour = 3600 joules (matches computation/energy modules)


class Accrual(Enum):
    """How a term's magnitude accumulates over a run — fixes the reference scale's
    leading factor (spec §5.3). Making this explicit prevents accrual-basis errors
    (the class of bug behind the S_energy and S_coverage corrections, §15 entries
    26 & 28).

    PER_TASK  -> scale = n_task_hat * unit_scale   (counted once per task)
    PER_SLOT  -> scale = T_h        * unit_scale   (accumulates every time slot)
    MIXED     -> scale = explicit sum of per-task and per-slot parts
    """

    PER_TASK = "per_task"
    PER_SLOT = "per_slot"
    MIXED = "mixed"


class Role(Enum):
    """A term's role in the minimization (spec §5.2 sign convention).

    COST   -> we want it small; its raw value enters POSITIVE (e.g. J_task, J_energy).
    REWARD -> we want it large; its raw value enters NEGATIVE (e.g. J_completion,
              J_util, J_coverage carry a leading minus in §5.2).
    """

    COST = "cost"
    REWARD = "reward"


class Term(Enum):
    """The five active objective terms; Enum *value* is the master weight index.

    Index 4 (migration) is intentionally absent — migration was dropped (A4),
    so there is structurally no Term(4) and no w_4 in the system.
    """

    TASK = 1
    ENERGY = 2
    COMPLETION = 3
    UTIL = 5
    COVERAGE = 6


# config key that supplies each term's weight — the single naming of weights
_WEIGHT_KEY: Mapping[Term, str] = {
    Term.TASK: "w1_task",
    Term.ENERGY: "w2_energy",
    Term.COMPLETION: "w3_completion",
    Term.UTIL: "w5_util",
    Term.COVERAGE: "w6_coverage",
}

# accrual basis of each term's reference scale (spec §5.3; entries 26, 28)
_ACCRUAL: Mapping[Term, Accrual] = {
    Term.TASK: Accrual.PER_TASK,
    Term.COMPLETION: Accrual.PER_TASK,
    Term.UTIL: Accrual.PER_SLOT,
    Term.COVERAGE: Accrual.PER_SLOT,
    Term.ENERGY: Accrual.MIXED,  # compute (per-task) + flight (per-slot)
}

# cost/reward role of each term (spec §5.2 sign convention)
_ROLE: Mapping[Term, Role] = {
    Term.TASK: Role.COST,
    Term.ENERGY: Role.COST,
    Term.COMPLETION: Role.REWARD,
    Term.UTIL: Role.REWARD,
    Term.COVERAGE: Role.REWARD,
}

# spec §6 named projections — subsets of THIS objective (never separate weights)
_PROJECTION_TERMS: Mapping[str, tuple] = {
    "morl": (Term.TASK, Term.ENERGY, Term.COMPLETION, Term.UTIL),  # {1,2,3,5}
    "mpc": (Term.TASK, Term.ENERGY, Term.COVERAGE),                # {1,2,6}
    "apso": (Term.TASK, Term.ENERGY, Term.UTIL),                   # {1,2,5}
}


def _uniform_mean(spec: Mapping[str, Any]) -> float:
    """Mean of a {dist: uniform[/_int], low, high} config block."""
    return 0.5 * (float(spec["low"]) + float(spec["high"]))


@dataclass(frozen=True)
class Projection:
    """A read-only view of a subset of the master objective (spec §6).

    A Projection holds NO weights of its own: every score is computed from the
    parent :class:`Objective`'s weights and scales. This is the structural
    guarantee that no optimizer can introduce a second weighting scheme.
    """

    objective: "Objective"
    name: str
    terms: tuple

    def value(self, raw_terms: Mapping[Term, float]) -> float:
        """Σ_{m in subset} w_m · (J_m / S_m), using the PARENT's weights/scales."""
        o = self.objective
        missing = [t for t in self.terms if t not in raw_terms]
        if missing:
            raise KeyError(f"projection '{self.name}' missing raw terms: {missing}")
        return sum(o.weight(t) * (float(raw_terms[t]) / o.scale(t)) for t in self.terms)

    @property
    def weights(self) -> Mapping[Term, float]:
        """Read-only weights, fetched live from the parent (not a redefinition)."""
        return MappingProxyType({t: self.objective.weight(t) for t in self.terms})


@dataclass(frozen=True)
class Objective:
    """The single master objective (spec §5). Build via :meth:`from_config`.

    The only weights and the only normalization scales in the system live on
    this instance. Optimizers must obtain a :class:`Projection` via
    :meth:`projection` and score through it.
    """

    _weights: Mapping[Term, float]
    _scales: Mapping[Term, float]

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Objective":
        obj = config["objective"]
        # structural guard: migration is gone — no w_4 may exist (A4)
        if "w4_migration" in obj:
            raise ValueError(
                "config.objective.w4_migration present — migration was dropped (A4); "
                "no w_4 may exist anywhere."
            )

        weights = {t: float(obj[_WEIGHT_KEY[t]]) for t in Term}

        net = config["network"]
        uav = config["uav"]
        task = config["task"]
        N = float(net["num_iot_devices"])
        M = float(net["num_uavs"])
        Th = float(net["num_time_steps"])
        dt = float(net["time_step_s"])
        lam_bar = _uniform_mean(task["generation_rate_tps"])
        tau_bar = _uniform_mean(task["deadline_s"])
        w_bar_cycles = _uniform_mean(task["compute_req_megacycles"]) * 1e6  # Mcyc -> cycles
        e_c_j = float(uav["compute_energy_rate_wh_per_cycle"]) * WH_TO_J
        p_flight = float(uav["flight_energy_rate_w"])

        n_task_hat = N * lam_bar * Th  # expected tasks per run (§5.3)

        # Scales are built from an EXPLICIT accrual basis (spec §5.3): the leading
        # factor is n_task_hat for PER_TASK terms and T_h for PER_SLOT terms, so a
        # new term cannot silently get the wrong basis (the bug class behind the
        # S_energy / S_coverage corrections, §15 entries 26 & 28).
        def per_task(unit_scale: float) -> float:
            return n_task_hat * unit_scale

        def per_slot(unit_scale: float) -> float:
            return Th * unit_scale

        unit_scale: Mapping[Term, float] = {
            Term.TASK: tau_bar,                 # PER_TASK: ~ one task's latency budget (s)
            Term.COMPLETION: 1.0,               # PER_TASK: one completion (count)
            Term.UTIL: M,                       # PER_SLOT: full utilization across M UAVs
            Term.COVERAGE: N,                   # PER_SLOT: all N devices covered (entry 28)
            # ENERGY is MIXED: its per-task and per-slot parts have different units
            Term.ENERGY: w_bar_cycles * e_c_j,  # PER_TASK part: one task's compute energy (J)
        }
        energy_per_slot_unit = M * p_flight * dt  # PER_SLOT part: flight power across UAVs (J/slot)

        scales = {}
        for term in Term:
            accrual = _ACCRUAL[term]
            if accrual is Accrual.PER_TASK:
                scales[term] = per_task(unit_scale[term])
            elif accrual is Accrual.PER_SLOT:
                scales[term] = per_slot(unit_scale[term])
            elif accrual is Accrual.MIXED:  # ENERGY only: compute (per-task) + flight (per-slot)
                scales[term] = per_task(unit_scale[term]) + per_slot(energy_per_slot_unit)
            else:  # pragma: no cover - exhaustive over Accrual
                raise ValueError(f"unhandled accrual {accrual} for {term}")
            if scales[term] <= 0.0:
                raise ValueError(f"reference scale S[{term.name}] must be > 0, got {scales[term]}")

        return cls(_weights=dict(weights), _scales=dict(scales))

    # ---- accessors (single source of truth) ---------------------------------
    def weight(self, term: Term) -> float:
        return self._weights[term]

    def scale(self, term: Term) -> float:
        return self._scales[term]

    @staticmethod
    def role(term: Term) -> Role:
        """Cost/reward role of a term (spec §5.2 sign convention)."""
        return _ROLE[term]

    @staticmethod
    def accrual(term: Term) -> Accrual:
        """Accrual basis of a term's reference scale (spec §5.3)."""
        return _ACCRUAL[term]

    @property
    def weights(self) -> Mapping[Term, float]:
        """Read-only view of THE weights (the only weighting scheme that exists)."""
        return MappingProxyType(dict(self._weights))

    @property
    def scales(self) -> Mapping[Term, float]:
        return MappingProxyType(dict(self._scales))

    # ---- normalization (spec §5.3) ------------------------------------------
    def normalize(self, term: Term, raw: float) -> float:
        """J̃_m = J_m / S_m."""
        return float(raw) / self._scales[term]

    def normalized_terms(self, raw_terms: Mapping[Term, float]) -> Mapping[Term, float]:
        return {t: self.normalize(t, raw_terms[t]) for t in raw_terms}

    # ---- master value (spec §5.1) -------------------------------------------
    def value(self, raw_terms: Mapping[Term, float]) -> float:
        """J = Σ_{m∈{1,2,3,5,6}} w_m · J̃_m over ALL five active terms."""
        missing = [t for t in Term if t not in raw_terms]
        if missing:
            raise KeyError(f"master objective missing raw terms: {missing}")
        return sum(self.weight(t) * self.normalize(t, raw_terms[t]) for t in Term)

    # ---- projections (spec §6) ----------------------------------------------
    def projection(self, which: Union[str, Iterable[Term]]) -> Projection:
        """Return a :class:`Projection` (subset view) of this objective.

        ``which`` is a named projection ('morl' | 'mpc' | 'apso', spec §6) or an
        explicit iterable of Terms. The projection carries THIS objective's
        weights and scales — it cannot hold its own.
        """
        if isinstance(which, str):
            key = which.lower()
            if key not in _PROJECTION_TERMS:
                raise KeyError(
                    f"unknown projection '{which}'; known: {sorted(_PROJECTION_TERMS)}"
                )
            terms = _PROJECTION_TERMS[key]
            name = key
        else:
            terms = tuple(which)
            bad = [t for t in terms if not isinstance(t, Term)]
            if bad:
                raise TypeError(f"projection terms must be Term members; got {bad}")
            name = "custom"
        return Projection(objective=self, name=name, terms=terms)
