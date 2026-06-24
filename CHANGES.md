# Implementation Notes and Design Decisions

This document records the design decisions made while building a single, runnable,
self-consistent implementation of the MOALF-UAV-MEC framework (IEEE Internet of
Things Journal, vol. 12, no. 12, 2025), and explains where the implementation
necessarily differs from the published description. Some quantities in the paper
are specified in more than one place with different forms, and some are left open;
turning the description into one executable system required choosing a consistent
resolution for each. The notes below document those choices so the code is fully
traceable back to the paper.

The authoritative specification this code implements is
[`notes/corrected_spec.md`](notes/corrected_spec.md); section references below
(e.g. §5, eq 14) point into it.

---

## 1. A single coherent objective

The paper describes the objective in several places with different term subsets
(a six-term master objective, and narrower subsets used by the MORL reward, the
MPC objective, and the APSO fitness), along with run-time weight-adaptation rules.
For a single, well-defined optimization problem, these are consolidated into one
objective:

- **One objective, defined once** (§5, from eq 14): the six-term weighted sum is
  the sole objective. Each optimizer consumes a **named projection** of it — the
  subset of terms its decision variables affect — carrying the same weights and
  normalizations. The implementation keeps the weights in one place; an optimizer
  receives a projection that scores against those weights and holds none of its
  own, so the system has a single weighting throughout.
- **Fixed weights.** The weights are held constant for a run; weighting variation
  is studied separately through a pre-registered weight sweep rather than adapted
  online. This keeps runs reproducible.
- **Migration omitted** (objective uses five active terms). The task-migration
  decision is lightly specified — its parameters are not given and no separate
  migration result is reported — so it is omitted here to keep the model
  self-contained.
- **MORL as fixed-preference scalarization.** With fixed weights, the MORL reward
  is a weighted-sum scalarization; the multiobjective behavior is demonstrated via
  the weight sweep rather than an online Pareto solver. This is stated explicitly
  so the method's scope is clear.
- **Smooth coverage term** (§5.2): the published coverage term reduces to a
  constant independent of UAV position, so a smooth Gaussian coverage form (eq 63)
  is used instead, giving a differentiable objective so that trajectory
  optimization is well-posed.

## 2. Two-tier queue model and stability layer

The queue dynamics underlying the Lyapunov analysis are not written out in the
paper, so they are defined explicitly here:

- **Two coupled queues** (§4.7): a per-device **radio** backlog (bits awaiting
  offload) and a per-UAV **compute** backlog (cycles awaiting execution), joined
  by a conservation-exact hand-off in which transmitted bits become compute cycles
  via each task's own work intensity `r = W/L`. Stability is defined as both tiers
  remaining bounded.
- **Lyapunov drift-plus-penalty controller** (§10): the controller biases offload
  assignment toward draining congested compute queues. Stability is reported as
  empirically demonstrated (bounded backlog in simulation); a closed-form
  drift-bound derivation is noted as an open item rather than claimed.
- **Radio service discipline**: priority-by-urgency (using the paper's urgency
  metric, eq 5), which aligns queue service with the deadline objective.

## 3. Choices for quantities the paper leaves open

Where the paper does not give a value, one was chosen on physical or
standard-practice grounds and frozen before any comparison to the paper's numbers
(full table in spec §14.2):

- **Channel** (eqs 6–7): reference gain `β0 = −30 dB`, path-loss exponent `α = 2`,
  noise PSD `N0 = −174 dBm/Hz`, bandwidth `B = 1 MHz`, altitude `H = 100 m`
  (fixed; motion is treated as 2-D, consistent with the stated 400×400 m area).
- **Compute energy**: a linear model `E = e_c·W` using the per-cycle figure given
  in Table IV (the alternative quadratic form, eq 9, does not include a
  coefficient).
- **APSO** allocates the split of each UAV's compute capacity across its assigned
  tasks (a UAV runs its processor at full capacity; APSO decides the division).
  Under contention the split minimizes the projection, which yields `f ∝ √remaining`.
- **MPC** uses a direct receding-horizon formulation; the paper's LQR-style
  quadratic-program form (eqs 39–41) references matrices and sets that are not
  specified. A re-planning cadence is honored to keep computation modest.
- **Objective normalization** (§5.3): terms are normalized to a comparable scale
  (order unity) so that no single term dominates purely because of units. Two
  reference scales were set to match the magnitude of the quantity they normalize
  — energy by total (compute + flight) energy, and coverage by a per-slot-accrued
  scale — and the per-task vs per-slot accrual is made explicit in code.
- **Lyapunov trade-off `V`** is set to `≈ 6.9×10¹⁶` on unit-commensurability
  grounds: it is derived from the measured ratio of typical drift to typical
  objective penalty at the nominal operating point, so the two are comparable. It
  is derived from system scales only, before any performance metric is observed.
- **Symbol conventions**: a few symbols are reused in the paper for more than one
  quantity; each is given a single meaning here and secondary uses are renamed
  (spec §3).

## 4. Operating-regime analysis and reproduction

To interpret the reproduction, I characterized the operating regime that the
frozen parameters produce, then ran the consistent system on those parameters
(30 runs, seeds 42–71). Results are in
[`results/comparison_table.md`](results/comparison_table.md):

- **Task completion rate** matches the paper closely (≈ 98% against the reported
  94.5%).
- **Relative-improvement metrics (throughput increase, route reduction)** are
  less pronounced under the frozen parameters. The analysis shows why: the stated
  configuration corresponds to a **lightly-loaded regime** — compute demand is
  roughly one third of capacity, the compute queues are empty most of the time,
  and per-link SNR is high across the cell. In that regime a simple baseline
  already performs near the ceiling, so there is limited headroom for a relative
  throughput gain, and UAV position has limited influence on link quality, so
  route length is not a strong lever. Under a higher offered load the learned
  policy's throughput advantage becomes more pronounced (a labelled high-load run
  shows roughly +17% over a random baseline). See
  [`results/comparison_table.md`](results/comparison_table.md) for the per-metric
  detail and the high-load context block.
- **Weighting behavior**: a directional check confirms the multiobjective response
  is coherent for the offload/allocation objectives — increasing a weight improves
  its own outcome (latency, energy-efficiency, completion, utilization). Trajectory
  is coverage-driven in this high-SNR regime.

**For a manuscript revision**, these notes support reporting the completion result
as reproduced, stating the operating load explicitly, and presenting the
throughput and route metrics in the context of operating load and baseline choice
(for example, evaluated under a higher load or against the paper's own baseline
algorithms). All parameter values were frozen on physical grounds before
comparison; none were adjusted to match a reported figure.

---

*A full, dated, decision-by-decision log is maintained in
[`notes/corrected_spec.md`](notes/corrected_spec.md) §15 for provenance.*
