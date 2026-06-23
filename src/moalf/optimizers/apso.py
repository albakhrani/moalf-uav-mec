"""apso — Adaptive Particle Swarm Optimization for resource allocation.

Implements the APSO optimizer from the authoritative spec (notes/corrected_spec.md
§9), eqs (30)-(31), with the adaptive linear-inertia rule (B18):

    eq (30)  v_id <- w_in * v_id + c1*r1*(pbest_id - x_id) + c2*r2*(gbest_id - x_id)
    eq (31)  x_id <- x_id + v_id
    w_in : linear decay inertia_start -> inertia_end over the iterations (B18)

D1 / single-objective architecture (spec §6): APSO **holds no objective weights**.
It scores candidate allocations ONLY through the ``'apso'`` projection of the one
:class:`~moalf.objective.Objective` (terms {task, energy, util} = m∈{1,2,5}),
calling ``projection.value(...)``. The weighting it optimizes is therefore the
master weighting, by construction — there is no field on this class to hold a
second scheme. Its own ``inertia/c1/c2`` are *swarm* hyperparameters, not
objective weights.

All hyperparameters are read from config (no hard-coded values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from moalf.objective import Projection, Term


@dataclass(frozen=True)
class PSOResult:
    """Outcome of a PSO run."""

    best_position: np.ndarray
    best_fitness: float
    history: list  # best fitness per iteration
    iterations: int


@dataclass(frozen=True)
class APSO:
    """Adaptive PSO (spec §9). Build via :meth:`from_config`.

    Fields are PSO *swarm* hyperparameters only — deliberately NO objective
    weights live here (see module docstring / D1).
    """

    swarm_size: int
    inertia_start: float
    inertia_end: float
    cognitive_c1: float
    social_c2: float
    max_iterations: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "APSO":
        a = config["apso"]
        return cls(
            swarm_size=int(a["swarm_size"]),
            inertia_start=float(a["inertia_start"]),
            inertia_end=float(a["inertia_end"]),
            cognitive_c1=float(a["cognitive_c1"]),
            social_c2=float(a["social_c2"]),
            max_iterations=int(a["max_iterations"]),
        )

    # ---- adaptive inertia (B18) ---------------------------------------------
    def inertia(self, iteration: int, n_iter: int) -> float:
        """Linear inertia decay inertia_start -> inertia_end over ``n_iter`` steps."""
        if n_iter <= 1:
            return self.inertia_start
        frac = min(max(iteration / (n_iter - 1), 0.0), 1.0)
        return self.inertia_start + (self.inertia_end - self.inertia_start) * frac

    # ---- generic PSO minimizer (eqs 30-31) ----------------------------------
    def minimize(
        self,
        fitness: Callable[[np.ndarray], float],
        lower,
        upper,
        rng: np.random.Generator,
        *,
        max_iterations: int | None = None,
    ) -> PSOResult:
        """Minimize a scalar ``fitness(x)`` over the box [lower, upper].

        Pure optimizer: knows nothing about objectives or weights. ``fitness`` is
        any callable mapping a position vector to a real number (lower is better).
        """
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(hi < lo):
            raise ValueError("upper must be >= lower elementwise")
        dim = lo.size
        n_iter = int(self.max_iterations if max_iterations is None else max_iterations)
        span = hi - lo

        # init positions in-box, velocities small relative to the box
        pos = lo + rng.random((self.swarm_size, dim)) * span
        vel = (rng.random((self.swarm_size, dim)) * 2.0 - 1.0) * (0.1 * span)
        vmax = span  # velocity clamp

        pbest = pos.copy()
        pbest_f = np.array([float(fitness(p)) for p in pos])
        g_idx = int(np.argmin(pbest_f))
        gbest = pbest[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])

        history = [gbest_f]
        for it in range(n_iter):
            w = self.inertia(it, n_iter)
            r1 = rng.random((self.swarm_size, dim))
            r2 = rng.random((self.swarm_size, dim))
            vel = (
                w * vel
                + self.cognitive_c1 * r1 * (pbest - pos)
                + self.social_c2 * r2 * (gbest - pos)
            )
            vel = np.clip(vel, -vmax, vmax)
            pos = np.clip(pos + vel, lo, hi)  # eq (31), constrained to the box

            f = np.array([float(fitness(p)) for p in pos])
            improved = f < pbest_f
            pbest[improved] = pos[improved]
            pbest_f[improved] = f[improved]

            g_idx = int(np.argmin(pbest_f))
            if float(pbest_f[g_idx]) < gbest_f:
                gbest = pbest[g_idx].copy()
                gbest_f = float(pbest_f[g_idx])
            history.append(gbest_f)

        return PSOResult(best_position=gbest, best_fitness=gbest_f,
                         history=history, iterations=n_iter)

    # ---- allocation optimization via the 'apso' projection (spec §6, §9) -----
    def optimize_allocation(
        self,
        projection: Projection,
        evaluate_terms: Callable[[np.ndarray], Mapping[Term, float]],
        lower,
        upper,
        rng: np.random.Generator,
        *,
        max_iterations: int | None = None,
    ) -> PSOResult:
        """Find the allocation minimizing the objective, scored ONLY via ``projection``.

        ``projection`` must be the master objective's ``'apso'`` projection. APSO
        never reads weights — it calls ``projection.value(evaluate_terms(x))``, so
        the weighting is the master weighting by construction. ``evaluate_terms``
        maps an allocation vector to that allocation's raw objective-term values
        (supplied later by the simulation/system-model layer).
        """
        if not isinstance(projection, Projection):
            raise TypeError("optimize_allocation requires an objective Projection (no inline weights)")
        if projection.name != "apso":
            raise ValueError(
                f"APSO must consume the 'apso' projection, got '{projection.name}'"
            )

        def fitness(x: np.ndarray) -> float:
            return projection.value(evaluate_terms(x))

        return self.minimize(fitness, lower, upper, rng, max_iterations=max_iterations)
