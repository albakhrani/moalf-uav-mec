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

    def adjust(self, decisions):
        """Stability adjustment hook (spec §11 step 5).

        SKELETON: identity (no-op) — the per-slot drift-plus-penalty *minimization*
        that biases MORL/MPC/APSO decisions (spec §10.1) is the next integration
        increment. Returned unchanged so the pipeline is wired and reviewable.
        """
        return decisions
