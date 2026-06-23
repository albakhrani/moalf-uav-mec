"""objective — THE single objective function for the whole framework.

There is exactly ONE objective in this repository (the weighted multiobjective
cost, eq 14). Every optimizer (MORL, MPC, APSO, Lyapunov penalty) consumes a
PROJECTION of this objective and its weights — never a second weighting scheme
. Source of truth: notes/corrected_spec.md. Equation references:
notes/method_extraction.md §2.H (eq 14-20).

Stub: logic intentionally not implemented yet (awaiting finalized spec).
"""

raise NotImplementedError(
    "objective not implemented yet — awaiting finalized notes/corrected_spec.md"
)
