# Implementation notes and deviations from the published paper

This document records where this reimplementation differs from the published
MOALF-UAV-MEC paper (IEEE Internet of Things Journal, vol. 12, no. 12, 2025), and
why. The published description contains several internal inconsistencies and
under-specified quantities; building a *single, runnable, self-consistent* system
required resolving them. Each decision below is an engineering/modeling choice
made for consistency and reproducibility, not a change to the paper's intent.

The authoritative specification this code implements is
[`notes/corrected_spec.md`](notes/corrected_spec.md); the equation-level extraction
of the paper is in [`notes/method_extraction.md`](notes/method_extraction.md) and
[`notes/results_extraction.md`](notes/results_extraction.md). Section references
below (e.g. §5, eq 14) point into the spec.

---

## 1. Objective consolidation (one coherent objective)

The paper applies several *different* weighted objective subsets across its
components (a 6-term master objective, a 4-term MORL reward, a 3-term MPC
objective, a 3-term APSO fitness) plus two run-time weight-adaptation rules, with
inconsistent weight indexing. This cannot be a single optimization problem.

- **One objective, defined once** (§5, supersedes eq 14): the 6-term weighted sum
  is the sole objective. Every optimizer consumes a **named projection** of it
  (the subset of terms its decision affects), carrying the *same* weights and
  normalizations. The architecture makes a second weighting scheme structurally
  impossible — an optimizer is handed a projection that computes scores from the
  one objective's weights and holds none of its own.
- **Fixed weights** (not learned). The paper's adaptive-weight rules (eqs 55, 59)
  were removed for reproducibility; weighting variation is studied instead via a
  pre-registered weight sweep.
- **Migration dropped** (objective → 5 active terms): the migration decision
  `y_ijkl`, its objective term, its constraint, and its two undefined parameters
  were thinly specified and carried no reported result, so they are out of scope.
- **MORL is scalarized** (fixed-preference weighted sum); the "multiobjective"
  character is provided by the weight sweep, not an online Pareto solver — this is
  stated plainly rather than claimed as Pareto optimization.
- **Coverage term rewritten** (§5.2): the paper's coverage term (eq 20) is
  mathematically degenerate (it evaluates to a constant independent of UAV
  position and so cannot drive trajectory). It is replaced by a smooth Gaussian
  coverage metric (eq 63), which is differentiable and actually optimizable. This
  is an erratum-level correction to the published formula.

## 2. Two-tier queue model and stability layer

The paper never writes the queue dynamics that its Lyapunov analysis presumes.

- **Two coupled queues** (§4.7): a per-device **radio** backlog (bits awaiting
  offload) and a per-UAV **compute** backlog (cycles awaiting execution), joined
  by an explicit, conservation-exact hand-off: transmitted bits become compute
  cycles via each task's own work intensity `r = W/L`. Stability requires both
  tiers bounded.
- **Lyapunov as a two-tier drift-plus-penalty controller** (§10): the controller
  biases offload assignment toward draining congested compute queues. Stability is
  reported as *empirically demonstrated* (bounded backlog in simulation); a closed
  drift-bound proof is left as a clearly-marked open item, not claimed.
- **Radio service discipline**: priority-by-urgency (drains tasks by the paper's
  urgency metric, eq 5), aligning the queue with the deadline objective.

## 3. Parameter and modeling choices for unspecified quantities

Where the paper is silent, values were chosen on physical/literature grounds and
frozen before any comparison to the paper's numbers (full table in spec §14.2):

- **Channel** (eqs 6–7): reference gain `β0 = −30 dB`, path-loss exponent `α = 2`,
  noise PSD `N0 = −174 dBm/Hz`, bandwidth `B = 1 MHz`, altitude `H = 100 m`
  (fixed; motion is 2-D, consistent with the stated 400×400 m area).
- **Compute energy**: linear model `E = e_c·W` using Table IV's per-cycle figure
  (the paper's alternative quadratic form, eq 9, has no stated coefficient).
- **APSO** allocates the *split* of each UAV's full compute capacity across its
  tasks (it does not throttle total capacity — a UAV always runs its CPU); under
  contention the split minimizes the projection, giving `f ∝ √remaining`.
- **MPC** uses a direct receding-horizon formulation (the paper's LQR-QP form,
  eqs 39–41, references matrices/sets it never specifies) and honors the
  configured re-planning cadence.
- **Normalization corrections** (§5.3): objective terms are normalized to be
  comparable (O(1)). Two reference scales were corrected on magnitude grounds so a
  term could not silently dominate — energy normalized by *total* (compute +
  flight) energy, and coverage by a per-slot-accrued scale. Per-task vs per-slot
  accrual is made explicit in code.
- **Lyapunov trade-off `V`** rescaled from a placeholder `1.0` to
  `≈ 6.9×10¹⁶` on **unit-commensurability** grounds — derived from the measured
  ratio of typical drift to typical penalty at the real-config operating point, so
  the objective penalty is commensurate with the stability drift. Derived from
  system scales only, before observing any performance metric.
- **Symbol disambiguation**: the paper reuses several symbols (α, λ, T, Q, R, τ, …)
  for multiple quantities; each is given a single meaning and secondary uses are
  renamed (spec §3).

## 4. Reproduction findings

Running the consistent system on the paper's frozen parameters (30 runs, seeds
42–71) yields the comparison in [`results/comparison_table.md`](results/comparison_table.md):

- **Task completion rate reproduces** (98.4% vs the paper's 94.5% — Tier 1, and
  slightly exceeding the claim).
- **Throughput (+55%) and route reduction (−38%) do not reproduce at the stated
  parameters** (Tier 3), with an explanation rather than a bare miss: the stated
  configuration is a **lightly-loaded, saturated-SNR regime** (compute demand is
  about one third of capacity; queues are empty ~99% of the time; per-link SNR is
  saturated cell-wide). In that regime baselines already perform near ceiling, so
  there is little throughput to gain, and UAV position has weak leverage, so routes
  are not minimized. The throughput advantage of the learned policy **does** appear
  under genuine load (a labelled high-load context run shows ~+17% over a random
  baseline).
- **Operating-load context**: the multiobjective response is directionally
  coherent for the offload/allocation objectives (boosting a weight improves its
  own outcome — latency, energy-efficiency, completion, utilization), confirming
  the weighting is meaningful and not thrashing. Trajectory is coverage-driven in
  this saturated-SNR regime.

**Manuscript implication.** Completion can be reported as reproduced; the true
operating load should be reported honestly (lightly loaded); and the
throughput/route headline figures should either be re-derived against the paper's
own baseline algorithms under a stated, more-stressed load, or tempered. No frozen
value was tuned to close any gap.

---

*A full, dated, decision-by-decision log (44 entries) is maintained in
[`notes/corrected_spec.md`](notes/corrected_spec.md) §15 for provenance.*
