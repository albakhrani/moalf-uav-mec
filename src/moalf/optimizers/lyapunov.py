"""lyapunov — two-tier drift-plus-penalty stability controller (spec §10).

Implements the combined Lyapunov function, realized drift, and drift-plus-penalty
term over the TWO-TIER backlog Θ = ({Q^r_i}, {Q^c_j}) (spec §10.1, A1):

    eq (32')  L(Θ) = ½ [ Σ_i Q^r_i²  +  c_Q · Σ_j Q^c_j² ]
    eq (33')  Δ(t) = L(Θ(t+1)) − L(Θ(t))           (realized, one-sample)
    eq (34')  drift-plus-penalty = Δ(t) + V · P_pen(t)

`c_Q` puts bits² (radio) and cycles² (compute) on one scale; `V` is the
drift/penalty trade-off (both from config). `P_pen(t)` is the per-slot value of
the single objective (spec §5), supplied by the caller (it owns the weights — D1).

This controller computes the stability diagnostics and exposes an `adjust` hook
(spec §11 step 5). The empirical-stability stance (§10.1) holds: stability is
*measured* (bounded time-average backlog), never claimed as proven (§10.2 TODO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LyapunovController:
    """Two-tier drift-plus-penalty controller (spec §10). Build via :meth:`from_config`."""

    V: float        # drift/penalty trade-off weight
    c_Q: float      # tier-scaling constant (bits² vs cycles²)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LyapunovController":
        ly = config["lyapunov"]
        return cls(V=float(ly["penalty_weight_V"]), c_Q=float(ly["tier_scaling_c_Q"]))

    # ---- eq (32'): combined two-tier Lyapunov function ----------------------
    def lyapunov_value(self, qr, qc) -> float:
        qr = np.asarray(qr, dtype=float)
        qc = np.asarray(qc, dtype=float)
        return 0.5 * (float(np.sum(qr * qr)) + self.c_Q * float(np.sum(qc * qc)))

    # ---- eq (33'): realized drift across one slot ---------------------------
    def drift(self, l_before: float, l_after: float) -> float:
        return l_after - l_before

    # ---- eq (34'): drift-plus-penalty ---------------------------------------
    def drift_plus_penalty(self, drift: float, penalty: float) -> float:
        return drift + self.V * penalty

    # ---- spec §11 step 5: evaluate stability, (skeleton) adjust hook ---------
    def evaluate(self, qr_before, qc_before, qr_after, qc_after, penalty: float) -> dict:
        """Return the slot's stability diagnostics."""
        l_before = self.lyapunov_value(qr_before, qc_before)
        l_after = self.lyapunov_value(qr_after, qc_after)
        d = self.drift(l_before, l_after)
        return {
            "L_before": l_before,
            "L_after": l_after,
            "drift": d,
            "drift_plus_penalty": self.drift_plus_penalty(d, penalty),
        }

    def biased_assignment(self, devices, uavs, qc, incoming_fn, penalty_fn) -> dict:
        """Per-slot drift-plus-penalty assignment minimization (spec §10.1, eq 34').

        For each device, choose the serving UAV that minimizes the *marginal*
        two-tier drift-plus-penalty:

            score(i -> j) = c_Q · Q^c_j · a_ij        (compute-tier drift term)
                          + V · penalty(i, j)         (objective penalty, eq 34')

        where ``a_ij = incoming_fn(i, j)`` is the cycles device i would hand to UAV
        j this slot, and ``penalty_fn(i, j)`` is the 'morl'-projection objective
        cost of serving i at j (the weights live on the Objective — D1; this
        controller supplies only V and c_Q). Assignments are made greedily with a
        running projected backlog ``qc_proj`` so that committing work to a UAV
        raises its cost for subsequent devices — i.e. the term `Q^c_j·A^c_j` makes
        the controller spread load off congested compute queues, bending drift down.

        Lower V -> drift term dominates -> stronger load-spreading (more stable);
        higher V -> objective dominates. V and c_Q are frozen (§18); biasing
        reduces drift because that is what drift-plus-penalty does, not via tuning.
        """
        qc_proj = {j: float(qc[j]) for j in uavs}
        assignment: dict = {}
        for i in devices:
            best_j, best_score = None, float("inf")
            for j in uavs:
                a = incoming_fn(i, j)
                drift_term = self.c_Q * qc_proj[j] * a
                score = drift_term + self.V * penalty_fn(i, j)
                if score < best_score:
                    best_score, best_j = score, j
            assignment[i] = best_j
            qc_proj[best_j] += incoming_fn(i, best_j)  # running backlog -> spread load
        return assignment
