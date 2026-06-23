"""mpc — Model Predictive Control for UAV trajectory (spec §8).

Direct receding-horizon controller (B7 — the LQR-QP eqs 39-41 are NOT used; their
matrices/sets are unspecified in the paper). Each tick it optimizes UAV `j`'s
motion over the prediction horizon N_h to minimize the 'mpc' projection of the
single objective (terms {task, energy, coverage} = m∈{1,2,6}, spec §6), then
applies the first velocity and re-plans (spec §8.2).

Reconciled motion model (D2):
  - state transition (eq 2): p_j(τ+1) = p_j(τ) + v_j(τ)·Δt
  - the gradient-ascent rule (62) is folded in via the smooth Gaussian coverage
    term (A3) inside the objective — MPC has no separate coverage mover.
  - motion is 2-D at fixed altitude H (A5): positions are (x, y); z never changes.
  - speed constraint (eq 3): ‖v_j(τ)‖₂ ≤ v_max.

D1 / single-objective architecture (spec §6): MPC **holds no objective weights**.
It scores only through the ``'mpc'`` projection (``projection.value(...)``); same
contract as APSO — it requires a Projection and rejects any projection but 'mpc'.

This implementation uses a constant-velocity rollout over the horizon and searches
the feasible velocity disk (a polar grid including the zero/“hold” velocity).
Because we re-plan every tick and apply only the first velocity, the realized
trajectory approaches a target and settles (the hold velocity wins once no move
improves the horizon coverage). All parameters are read from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from moalf.objective import Projection, Term


@dataclass(frozen=True)
class MPC:
    """Direct receding-horizon trajectory controller (spec §8). Build via
    :meth:`from_config`. Fields are trajectory hyperparameters only — NO
    objective weights live here (D1)."""

    horizon: int          # N_h (prediction horizon, slots)
    v_max: float          # speed cap (m/s), eq (3)
    dt: float             # Δt (s)
    altitude_m: float     # H (A5) — fixed; informational (motion is 2-D)
    n_directions: int = 24
    n_speeds: int = 11

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MPC":
        m = config["mpc"]
        formulation = m.get("formulation", "direct")
        if formulation != "direct":
            raise ValueError(
                f"mpc.formulation = {formulation!r}; only 'direct' is implemented "
                "(spec §8.2 / B7). The LQR-QP form (eqs 39-41) is excluded."
            )
        return cls(
            horizon=int(m["prediction_horizon"]),
            v_max=float(config["uav"]["max_velocity_mps"]),
            dt=float(config["network"]["time_step_s"]),
            altitude_m=float(config["channel"]["uav_altitude_m"]),
        )

    # ---- candidate velocities over the feasible disk -------------------------
    def candidate_velocities(self) -> list:
        """Polar grid of velocities with ‖v‖ ≤ v_max, including the zero (hold)
        velocity so 'settle in place' is always an option."""
        cands = [np.zeros(2)]
        speeds = np.linspace(0.0, self.v_max, self.n_speeds)
        for k in range(self.n_directions):
            theta = 2.0 * np.pi * k / self.n_directions
            d = np.array([np.cos(theta), np.sin(theta)])
            for s in speeds:
                if s > 0.0:
                    cands.append(s * d)
        return cands

    # ---- horizon cost of holding velocity v (constant-velocity rollout) ------
    def _rollout_cost(self, v, projection, evaluate_positions, uav_index, positions):
        pos = positions.copy()
        p = pos[uav_index].copy()
        total = 0.0
        for _ in range(self.horizon):
            p = p + v * self.dt            # eq (2), 2-D; altitude fixed (A5)
            pos[uav_index] = p
            total += projection.value(evaluate_positions(pos))
        return total

    # ---- plan: optimize this UAV's next velocity (spec §8.2) ----------------
    def plan(
        self,
        projection: Projection,
        evaluate_positions: Callable[[np.ndarray], Mapping[Term, float]],
        uav_index: int,
        positions,
        *,
        is_feasible: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> np.ndarray:
        """Return the velocity to apply this slot for UAV ``uav_index``.

        ``evaluate_positions(all_positions)`` returns the per-slot raw objective
        terms {TASK, ENERGY, COVERAGE} for a given (M,2) array of UAV positions.
        Scoring goes ONLY through ``projection`` (the 'mpc' projection); MPC holds
        no weights. ``is_feasible(pos)`` (optional) rejects first-step positions
        in no-fly zones (spec §8.2).
        """
        if not isinstance(projection, Projection):
            raise TypeError("MPC.plan requires an objective Projection (no inline weights)")
        if projection.name != "mpc":
            raise ValueError(f"MPC must consume the 'mpc' projection, got '{projection.name}'")

        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        best_v = np.zeros(2)
        best_cost = None
        for v in self.candidate_velocities():
            if np.linalg.norm(v) > self.v_max + 1e-9:   # eq (3)
                continue
            if is_feasible is not None and not bool(is_feasible(pos[uav_index] + v * self.dt)):
                continue
            cost = self._rollout_cost(v, projection, evaluate_positions, uav_index, pos)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_v = v
        return best_v
