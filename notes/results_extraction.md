# Results-Section Extraction — MOALF-UAV-MEC (Equations 35–70)

**Paper:** "MOALF-UAV-MEC: Adaptive Multiobjective Optimization for UAV-Assisted Mobile Edge Computing in Dynamic IoT Environments," IEEE IoT Journal 12(12), 15 June 2025.
**Scope of this file:** equations (35)–(70), spanning **Section V (Complexity Analysis, p.8–10)**, the **integration/adaptation mechanisms (p.10)**, and **Section VI–VII (Simulation & Comparative Analysis, p.12–15)**.
**Companion file:** [`method_extraction.md`](method_extraction.md) covers (1)–(34) (system model + framework). Equation/section cross-refs point there.

> **Conventions** (same as the method file)
> - `(n)` matches the paper's numbering.
> - "Depends-on" = the reported number / table / figure that the equation backs.
> - ⚠️ = ambiguity or **conflict with the Section III/IV definitions**; all are collected in [§4 Conflicts](#4-conflicts-with-the-section-iiiiv-definitions).
> - No values are invented. Silence in the paper is recorded, not filled.

---

## 0. TL;DR — the big picture before the equations

The results sections introduce **a second layer of math that the method sections (Sec. III/IV) never mentioned.** Much of it is *empirical/descriptive* (regression fits, characteristic-response models) rather than the optimization machinery being implemented. Several equations **redefine quantities** or **introduce new control rules** that conflict with the framework as specified. The most important conflicts:

- The objective **weights `w_i` are declared time-varying and learned** here (eq 55, 59), but were fixed constants in (14).
- **Three mutually-incompatible UAV-motion rules** now exist: kinematic (2), MPC-control (28/39), and **gradient-ascent (62)**.
- **Task assignment** is done by MORL `x_ijk` (27) in the method, but by a **JSQ softmax policy (64)** in the results.
- Heavy **symbol overloading**: `α, β, γ, δ, λ, τ, T, Q, R` each carry 2–4 different meanings between sections.

Treat (35)–(52) as legitimate complexity bookkeeping; treat (55)–(70) as **post-hoc descriptive models of the reported outputs**, not as additional components to simulate — unless you confirm otherwise (see [§5 Questions](#5-open-questions-for-you)).

---

## 1. Section V — Complexity Analysis (35–52)

### MORL complexity (p.8)
**(35)** `T_MORL = O(|S| · |A| · E)` — time complexity.
- `|S|`,`|A|` — state/action space sizes; `E` — number of training episodes (⚠️ `E` also = edge-server set in (Sec. III-A)).
- **Assumption:** tabular Q-learning, one update per (state,action) per episode.
- **Depends-on:** "scalability" narrative (Sec. V-A) and Table X "Recovery/convergence" comparisons.

**(36)** `dQ(s,a)/dt = α[ R(s,a) + γ·max_{a'} Q(s',a') − Q(s,a) ]` — continuous-time form of the Q-update.
- Same `α` (learning rate, 0.001) and `γ` (0.99) as (27).
- ⚠️ This is a **continuous ODE restatement** of the discrete update (27); it is used only to argue convergence "as a function of α and γ," not implemented.
- **Depends-on:** convergence-rate claims, Fig. 5(a) convergence analysis.

**(37)** `S_MORL = O(|S| · |A|)` — space (Q-table size).
- **Depends-on:** memory-footprint discussion; final total space (52).

### MPC complexity (p.9)
**(38)** `T_MPC = O(M · N³)` — time.
- `M` — number of UAVs (=5); `N` — prediction horizon.
- **Assumption:** per-step quadratic program solved with cubic cost in horizon `N`.
- **Depends-on:** the p.10 claim "cubic relationship with N underscores computational intensity"; prediction-horizon-vs-cost figure (p.10); total complexity (51).

**(39)** MPC QP solved each step:
`min_u Σ_{k=0}^{N−1} [ x_kᵀ Q x_k + u_kᵀ R u_k ] + x_Nᵀ P x_N`.
- `x_k` — state, `u_k` — control input at step `k`; `Q,R` — state/input weight matrices; `P` — terminal weight.
- ⚠️ `Q` here = LQR state-cost matrix, **not** the queue `Q_i` of (32)/(48); `R` here = input-cost matrix, **not** the data rate `R_{ij}` of (7).
**(40)** Dynamics: `x_{k+1} = A x_k + B u_k`, `k = 0,…,N−1` — linear state-space model.
**(41)** Constraints: `x_k ∈ X`, `u_k ∈ U`, `k = 0,…,N−1`.
- ⚠️ `A,B` matrices, state set `X`, input set `U` are **never specified** (and `U` collides with the UAV set `U`). The QP that the MPC "actually solves" is therefore not reproducible.
**(42)** `S_MPC = O(M · N)` — space.
- **Depends-on (39)–(42):** Table VII trajectory results; horizon/cost trade-off discussion.

### APSO complexity (p.9)
**(43)** `T_APSO = O(P · D · I)` — time. `P` particles, `D` problem dimension, `I` iterations.
**(44)** `v_{id}^{t+1} = w·v_{id}^t + c₁·r₁·(p_{id} − x_{id}^t) + c₂·r₂·(p_{gd} − x_{id}^t)` — velocity update.
**(45)** `x_{id}^{t+1} = x_{id}^t + v_{id}^{t+1}` — position update.
- ⚠️ (44)–(45) **restate (30)–(31)** with index notation (`p_{id}` = personal best, `p_{gd}` = global best). Same model; consistent. `w,c₁,c₂,P,I` still unspecified (see method file §6).
**(46)** `S_APSO = O(P · D)` — space.
- **Depends-on:** resource-allocation scalability (Sec. V-C); total complexity.

### Lyapunov complexity (p.9)
**(47)** `T_LYP = O(Q · T)` — time. `Q` = number of queues, `T` = time horizon.
- ⚠️ `Q` here = *count of queues*; in (32)/(48) `Q_i` = *queue backlog value*. Same letter, different role.
**(48)** `L(Q(t)) = ½ · Σ_{i=1}^{Q} Q_i²(t)` — **restates (32)** (upper limit now written `Q`).
**(49)** `Δ(t) = E[ L(Q(t+1)) − L(Q(t)) | Q(t) ]` — **restates (33)**. Consistent.
**(50)** `S_LYP = O(Q)` — space.
- **Depends-on:** stability-overhead discussion; total complexity.

### System totals (p.10)
**(51)** `T_TOTAL = O(M · N³ · T)` — dominated by MPC term × horizon.
- **Depends-on:** the central scalability claim (Sec. V) and the p.10 note that a distributed MPC could reduce it to `O(M·(N/K)³·T)` for `K` distributed compute nodes.
**(52)** `S_TOTAL = O(|S| · |A|)` — dominated by the MORL Q-table.
- **Depends-on:** memory-scalability claim.

---

## 2. Integration & real-time adaptation mechanisms (53–57, p.10)

**(53)** Timescale separation: `Δt_MORL = k₁·Δt_base`, `Δt_MPC = k₂·Δt_base`, `Δt_APSO = k₃·Δt_base`, with `k₁ > k₂ > k₃ > 0` (positive integers).
- Ensures: MORL explores slowest, MPC covers the horizon, APSO adapts fastest.
- ⚠️ `k₁,k₂,k₃` values **not given**; `Δt_base` relation to the `Δt = 1 s` of Table IV unstated.
- **Depends-on:** "hierarchical timing / stability" narrative; implementation-considerations (Sec. V-E).

**(54)** Inter-algorithm message: `M_{ij}(t) = {Φ_{ij}, s_{ij}(t), a_{ij}(t), π_{ij}(t)}`.
- `Φ_{ij}` protocol id, `s_{ij}` state info, `a_{ij}` action recommendations, `π_{ij}` priority level.
- **Depends-on:** the cross-algorithm coordination claims; no numeric result directly cites it.

**(55)** Adaptive weight update: `w_i(t+1) = w_i(t) + α·∇J_i(t) + β·ΔQ_i(t)`.
- `w_i(t)` — weight of algorithm/objective `i` at `t`; `α` — performance-sensitivity parameter; `β` — queue-stability parameter; `∇J_i` — performance gradient; `ΔQ_i` — queue-length variation.
- ⚠️⚠️ **Makes the objective weights time-varying and learned.** Directly conflicts with (14)/(28)/(29), where `w₁..w₆` are fixed constants. Also `α,β` reused (α was learning rate / path-loss exponent).
- **Depends-on:** "adaptive integration" results, Fig. 10 (adaptive performance vs MAPPO/MA-DRL), the "weights continuously optimized" claim on p.12.

**(56)** Integration efficiency ratio:
`η_int = (J_integrated / J_baseline) · (τ_baseline / τ_integrated) · (R_integrated / R_baseline)`.
- Product of three ratios: (1) performance improvement, (2) computational efficiency (response-time ratio), (3) resource utilization.
- ⚠️ `R` here = resource utilization; `τ` here = computation/response time (not deadline). Overloaded.
- **Depends-on:** **Table III** (Integration Performance Metrics): convergence 57.1%, resource efficiency 17.9%, stability 15.9%, adaptation 29.2%.

**(57)** Real-time adaptation: `Δu(t) = K(t)·[ e_s(t); e_p(t); e_r(t) ]ᵀ`.
- `e_s` state-tracking error, `e_p` performance deviation, `e_r` resource-allocation error; `K(t)` adaptive gain matrix.
- **Depends-on:** real-time adaptation mechanism (Sec. V-J), Fig. 11 overall-performance comparison.

---

## 3. Section VI–VII — Simulation & comparative-analysis equations (58–70)

> These appear *inside the results narrative*. Most are **descriptive models of observed behavior** (regressions, characteristic responses), not optimization components from Sec. IV.

**(58)** Multifactor priority function: `π(τ) = w₁·U(τ) + w₂·C(τ) + w₃·D(τ) + w₄·R(τ)`.
- `U(τ)` task urgency (function of deadline proximity + priority), `C(τ)` computational complexity (from CPU cycles + data size), `D(τ)` current resource distribution across UAVs/servers, `R(τ)` network-reliability factor (link stability + bandwidth).
- ⚠️ Reuses `w₁..w₄` (4th distinct weight scheme). `U(τ)` likely the urgency of (5) but with different arguments; `R(τ)` is a reliability *factor*, not the rate `R_{ij}` of (7). `τ` used as a generic time index here, not a deadline.
- **Depends-on:** task-management performance, **Table V** (completion 94.5%, latency 142 ms, util 87.5%, net eff 92.8%) and the "24.06–39.57% latency reduction" claim.

**(59)** Softmax objective weighting: `w_i = exp(λ_i·Q_i) / Σ_{j=1}^{N} exp(λ_j·Q_j)`.
- `λ_i` — sensitivity to queue length `Q_i` for objective `i`.
- ⚠️⚠️ **Second, different** adaptive-weight rule (conflicts with (55) *and* with fixed (14)). `λ_i` reused (was Poisson arrival rate in (4)).
- **Depends-on:** "system maintains optimal performance even under varying workloads" (p.12); Tables V/VI.

**(60)** Energy–completion regression: `E = α·T^β + γ + δ·R(d)`.
- Fitted constants: `α = 10.5` (base energy coefficient), `β = 0.85` (nonlinearity), `γ = 5000` (system operation cost), `δ = 0.15` (distance-dependent reliability factor). Fit quality `R² = 0.97`.
- `T` here = **task completion** (count), `R(d)` = distance-dependent reliability. Units: `E` in energy units (unspecified).
- ⚠️ Pure empirical fit of outputs; `α,β,γ,δ,T,R` all overloaded relative to Sec. III/IV.
- **Depends-on:** the "energy ∝ task completion" relationship claim (p.13), energy-efficiency narrative; not a control input.

**(61)** Trajectory multiobjective vector: `min f(p) = [ f₁(p), f₂(p), f₃(p), f₄(p) ]`, subject to `g(p) ≤ 0`, `h(p) = 0`, where:
- `f₁(p) = Σ_{t=1}^{T} ‖p(t+1) − p(t)‖` — path-length minimization (m).
- `f₂(p) = Σ_{t=1}^{T} ( E_flight(t) + E_comp(t) )` — energy consumption.
- `f₃(p) = Σ_{i=1}^{N} max_{j∈M} I( R_{ij} ≥ R_min )` — coverage (count of covered devices).
- `f₄(p) = Π_{t=1}^{T} ( 1 − P_fail(p(t)) )` — network reliability (product of per-slot survival probs).
- ⚠️⚠️ A **4-objective trajectory problem** that differs from the MPC objective (28) `{task,energy,coverage}` *and* from the master objective (14). This is presented as "the trajectory optimization problem," yet Sec. IV said MPC (28) handles trajectory. Which is implemented?
- **Depends-on:** **Table VII** (path length −38%, energy 82%, coverage 98.2%, link stability 95.5%, load balance 96%).

**(62)** Gradient-ascent trajectory update: `p_j(t+1) = p_j(t) + η·∇H(p_j(t))`.
- `H(p_j(t))` — coverage-quality metric; `η` — adaptation rate.
- ⚠️⚠️ **Third UAV-motion rule.** Conflicts with kinematic update (2) `p_j(t+1)=p_j(t)+v_j(t)·Δt` and with MPC control (28/39). All three claim to position the UAV.
- **Depends-on:** "9.5% link-stability improvement," coverage 98.2% (p.14).

**(63)** Coverage/demand model: `D(p) = Σ_{i=1}^{N} Σ_{j=1}^{M} w_{ij}·exp( −‖p_j − q_i‖² / (2σ²) )`.
- `w_{ij}` — demand weight of location `i` for UAV `j`; `σ` — controls coverage radius (Gaussian).
- ⚠️ `σ` value **not given**; `w_{ij}` demand weights unspecified; `D` here = coverage demand, not the IoT-device set `D` of Sec. III-A.
- **Depends-on:** coverage-optimization result 98.2% area coverage, Fig. 2(a) spatial distribution, Fig. 3(a/b).

**(64)** Join-shortest-queue assignment: `P(UAV_i) = exp(−Q_i/T + λ·R_i) / Σ_j exp(−Q_j/T + λ·R_j)`.
- `Q_i` current queue length at UAV `i`; `T` — **system temperature parameter** controlling load balance; `R_i` link reliability; `λ` — network-sensitivity parameter (adaptively adjusted from SDN feedback).
- ⚠️⚠️ **New assignment policy not in Sec. IV** — offloading there is the MORL decision `x_ijk` (27). `T` reused as a softmax temperature (was time horizon / task completion). `λ` reused again. `R_i` here = reliability, not rate.
- **Depends-on:** load-balancing 96% efficiency, Fig. 3(c) load distribution, Table VI resource-utilization.

**(65)** Hierarchical adaptive control: `Δu(t) = K(t)·[ e_s(t); e_p(t); e_r(t) ]ᵀ` — **restates (57)**.
**(66)** Gain update: `K(t) = K_0 + α·∇K_p·J(t) + β·ΔQ(t)`.
- `K_0` base gain, `α,β` adaptation parameters (overloaded yet again), `∇K_p` gain gradient, `ΔQ` queue variation.
- **Depends-on (65)–(66):** service-quality tiers (>95% / 75–90% / ≥75%), 7-step recovery, **Table VIII** resilience.

**(67)** Load characteristic response: `L(t) = L₀ + A·exp(−t/τ) + B·sin(ωt + φ)`.
- `L₀` baseline load; `A·exp(−t/τ)` transient (τ = transient-response time); `B·sin(ωt+φ)` periodic fluctuation; parameters "adaptively tuned from historical data."
- ⚠️ `τ` reused (was deadline / response time). Descriptive model of observed load, not a control law.
- **Depends-on:** Fig. 3(d) maintenance/load response; adaptation-mechanism evidence (p.14–15).

**(68)** *(System Resilience metric — Sec. VII, p.15)* `S(t) = α₁·S_local(t) + α₂·S_network(t) + α₃·S_global(t)`.
- `S_local` UAV-level stability, `S_network` network resilience, `S_global` system-wide performance; `α₁,α₂,α₃` adaptively adjusted weights.
- ⚠️ `α₁,α₂,α₃` (more overloaded α's). *(Equation appears between 67 and 70; carries no printed number in some renderings — labeled 68 here per sequence.)*
- **Depends-on:** **Table VIII** resilience (performance 100%/98%/99.5%, link stability, recovery 7 steps).

**(69)** *(Recovery characteristic — Sec. VII, p.15)* `R(t) = R_max·(1 − e^{−t/t_const}) + Σ_{i} β_i·f_i(t)`.
- `R_max` maximum recovery level; `t_const` recovery time constant; `β_i` weights; `f_i(t)` recovery modes.
- ⚠️ `R` reused (rate / reliability / resource → now recovery). Descriptive.
- **Depends-on:** "96.8% efficiency restoration," recovery-elasticity 55% (Table VIII).

**(70)** Predictive fault management: `P_fail(t+Δt) = Σ_{i=1}^{M} w_i·F_i(t)·e^{−λ_i·Δt}`.
- `F_i(t)` different fault indicators; `w_i` learned weights; `λ_i` temporal relevance of indicator `i`.
- ⚠️ `λ_i` reused (arrival rate / softmax sensitivity / now temporal-decay rate); `w_i` yet another weight set.
- **Depends-on:** "92% early-fault detection accuracy," minimal-disruption claim, Table VIII recovery, Table IX/X comparative resilience.

---

## 4. Conflicts with the Section III/IV definitions

Ordered by how much they would change a re-implementation:

| # | Equation(s) | Conflict | Impact |
|---|---|---|---|
| C1 | **(55), (59)** vs **(14)** | Objective weights are **fixed** in (14) but **time-varying & learned** here — and (55) (gradient) and (59) (softmax) are **two different rules** for the same weights. | High — changes what is optimized every step. |
| C2 | **(62)** vs **(2)** vs **(28)/(39)** | **Three incompatible UAV-motion rules**: kinematic velocity update, MPC control sequence, and coverage-gradient ascent. | High — which one moves the UAV in the sim? |
| C3 | **(64)** vs **(27)/(11)** | Task assignment via **JSQ softmax (64)** vs MORL decision `x_ijk` (27) / delay-energy argmin (11). | High — two different offloading deciders. |
| C4 | **(61)** vs **(28)** vs **(14)** | Trajectory objective set differs three ways: `{path,energy,coverage,reliability}` (61) vs `{task,energy,coverage}` (28) vs the 6-term master (14). | Medium-high. |
| C5 | **(58)** weights `w₁..w₄`; **(70),(66),(55)** weights `w_i` | At least **five distinct weight schemes** across the paper (master, MORL, MPC, APSO, priority, resilience, fault). No stated mapping. | Medium — central tuning unknowns. |
| C6 | **Symbol overloading** | `α` = {learning rate, path-loss exp, perf-sensitivity (55,66), energy coeff 10.5 (60)}; `λ` = {Poisson rate (4), queue sensitivity (59,64), temporal decay (70)}; `T` = {time horizon, task completion (60), softmax temperature (64)}; `Q` = {queue backlog (32), queue count (47), LQR cost matrix (39)}; `R` = {data rate (7), input-cost matrix (39), resource util (56), reliability (58,64), recovery (69)}; `τ` = {deadline (5), response/transient time (56,67)}; `E` = {edge-server set, episodes (35)}. | Medium — must be disambiguated per equation before coding. |
| C7 | **(36)** vs **(27)** | Continuous-time ODE vs discrete Q-update — an approximation used only for the convergence argument, not implemented. | Low (analysis only). |
| C8 | **(48)/(49)** vs **(32)/(33)** | Identical restatements (consistent) — but `Q` upper-limit notation overloads queue-value `Q_i`. | Low (cosmetic). |
| C9 | **(44)/(45)** vs **(30)/(31)** | Restatement of PSO update; consistent model, different notation. | Low. |
| C10 | **(60),(67),(69)** | Empirical regression / characteristic-response models presented inside results; not derivable from the system model and not part of the controller. Risk of being mistaken for components to simulate. | Low-medium — clarify their role. |

---

## 5. Open questions for you

(In addition to the 10 in [`method_extraction.md` §7](method_extraction.md).)

1. **Are the weights fixed or learned?** If learned, which rule governs them — (55) gradient, (59) softmax, or both at different layers? (C1)
2. **Which UAV-motion rule is canonical** — (2), MPC (28/39), or gradient ascent (62)? Are they meant to compose (e.g. MPC sets a target, gradient refines)? (C2)
3. **Who assigns tasks** — MORL `x_ijk` (27/11) or the JSQ policy (64)? If both, in what order? (C3)
4. **Which trajectory objective** to implement — (61) or (28)? And `σ` in coverage (63), `R_min`, and demand weights `w_{ij}`? (C4)
5. **Are (60), (67), (69) outputs or inputs?** I read them as *post-hoc fits of measured results* (note `R²=0.97` in (60)). Confirm I should **not** code them as system dynamics — only as validation targets if at all.
6. **MPC QP internals** (39)–(41): state/input matrices `A,B`, cost matrices `Q,R,P`, and the sets `X,U` are unspecified. Do you have them, or should the trajectory layer be implemented from (61)/(62) instead of the LQR-style QP?
7. **Timescale constants** `k₁>k₂>k₃` (53) and their relation to `Δt = 1 s` — values?
8. **Resilience/fault models** (68)–(70): are these simulated mechanisms (inject failures, predict, recover) or only reported metrics? The Table-IV "UAV failure prob 0.05 / no-fly-zone 0.1" suggest injection is intended — confirm the recovery loop you want.

---

*Status: equations (1)–(70) are now fully extracted across [`method_extraction.md`](method_extraction.md) (1–34) and this file (35–70). No simulation code written. Awaiting your answers on the conflicts above before any implementation, per your instruction not to guess.*
