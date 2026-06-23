# corrected_spec.md — Authoritative Specification for MOALF-UAV-MEC

**Status:** authoritative source of truth for this repository .
Implement *this file*, not the raw paper. Where this spec and the paper disagree, this spec wins.

**Derived from:** the paper "MOALF-UAV-MEC" (IEEE IoT Journal 12(12), 2025), as extracted in
[`method_extraction.md`](method_extraction.md) (eqs 1–34) and [`results_extraction.md`](results_extraction.md) (eqs 35–70),
resolved by five author decisions (D1–D5, recorded in §1).

**How to read this file**
- 🔲 **SIGN-OFF** = a value or modeling choice proposed here but that needs the author's explicit approval. All are also listed together in [§16](#16-consolidated-sign-off-checklist). Nothing marked SIGN-OFF is settled.
- **PROPOSED** = a concrete default supplied so the system is codeable now; always paired with a one-line justification and a SIGN-OFF flag.
- **paper-stated** = taken verbatim from the paper (Table IV unless noted).
- "supersedes (n)" = this section replaces/redefines paper equation *n*.

> ⚠️ The author has **not** yet signed off on any PROPOSED value. A coder may implement the *structure* and the paper-stated values immediately, but every PROPOSED number is provisional until §16 is cleared.

---

## 1. Author decisions this spec encodes

| # | Decision | Effect |
|---|---|---|
| **D1** | Eq (14), the 6-term weighted sum, is THE single objective. Weights are **fixed constants**, not learned. Each optimizer optimizes a **projection** of (14) with the *same* weights/meaning. | §5, §6. Eqs (55),(59) removed as live mechanisms. |
| **D2** | **MPC owns UAV motion.** Kinematic (2) = MPC's internal state-transition; gradient-ascent (62) = folded into MPC's coverage objective term. One motion rule. | §8. Reconciles (2),(28),(62). |
| **D3** | **MORL owns offloading** (`x_ijk`). Join-shortest-queue softmax (64) is a **comparison baseline**, not live. | §7, §13. (64) demoted. |
| **D4** | Regression fits (60),(67),(69) are **descriptive of outputs**, never simulated as dynamics. | §12. Excluded. |
| **D5** | **Lyapunov dual-track:** primary = an empirically-validated drift-plus-penalty controller with an **explicitly defined queue update** `Q_i(t+1)` and penalty `P_pen(t)`; secondary = a drift-bound derivation left as a clearly-marked, unproven TODO. | §10. Defines (32)–(34) concretely; no proof claimed. |

---

## 2. Conventions

### 2.1 Internal units (SI). Config files use friendly units; convert on load.
| Quantity | Internal unit | Config unit → internal factor |
|---|---|---|
| time | second (s) | ms ÷1000; "time steps" × `Δt` |
| distance | meter (m) | — |
| energy | **joule (J)** | Wh ×3600 |
| power | watt (W) | — |
| data size | **bit** | MB ×8×10⁶ (decimal MB) |
| compute work | **CPU cycle** | Megacycle ×10⁶ |
| compute rate | cycle/s | GHz ×10⁹ |
| data rate | bit/s | Mb/s ×10⁶ |

> 🔲 **SIGN-OFF (units):** MB taken as decimal (8×10⁶ bits). If you meant MiB (8×2²⁰), say so.

### 2.2 Indexing (fixed for the whole repo)
`i` = IoT device (∈ 𝒟, |𝒟|=N); `j` = UAV (∈ 𝒰, |𝒰|=M); `l` = ground compute node / migration target (∈ 𝒢, |𝒢|=N_G); `k` = task; `t` = time slot (t = 1…T_h), slot length `Δt`.

---

## 3. Symbol table — one meaning per symbol

The paper overloaded several letters (documented in `results_extraction.md` §4 C6). Here each symbol has **exactly one** meaning; secondary uses are **renamed** and the rename is recorded. Renamed symbols are used everywhere below.

| Canonical symbol | Single meaning | Unit | Was also used in paper for → renamed to |
|---|---|---|---|
| `α_lr` | MORL learning rate (0.001) | – | path-loss exp → `α_path`; perf-sensitivity in (55)/(66) → **removed**; energy-fit coef → `a_E` (excluded) |
| `α_path` | channel path-loss exponent | – | (see above) |
| `γ` | MORL discount factor (0.99) | – | energy-fit constant 5000 → `c_E` (excluded) |
| `λ_i(t)` | task arrival rate at device i | task/s | queue-sensitivity in (59)/(64) → `λ_b` (baseline-only, §13); fault decay in (70) → excluded |
| `T_h` | optimization horizon = number of slots (1000) | slots | task-completion count in (60) → `T_c` (excluded); softmax temperature in (64) → `T_sm` (baseline-only) |
| `Q_i(t)` | queue backlog at device i | bit | "number of queues" in (47) → `N_Q`; LQR state-cost matrix in (39) → `Q_lqr` (unused, §8.4) |
| `Q_rl(s,a)` | MORL action-value function | – | distinct from queue `Q_i` (paper reused `Q`) |
| `R_ij(t)` | achievable data rate, link i→j | bit/s | LQR input-cost matrix → `R_lqr` (unused); resource-util ratio in (56) → `η_int` (excluded); reliability in (58)/(64) → `θ_rel`/baseline; recovery in (69) → excluded |
| `τ_{i,k}` | task deadline | s | transient/response time in (56)/(67) → `t_resp` (excluded) |
| `ρ_{i,k}` | task priority (1–5) | – | — |
| `θ_rel` | link-reliability threshold (0.95) | prob | (paper called it a "threshold"; distinct from `R_min`) |
| `κ_j` | compute energy coefficient (§9 model) | see §9 | timescale multipliers `k1,k2,k3` → `m_MORL,m_MPC,m_APSO` |
| `η_j` | UAV harvested power | W | gradient step in (62) → folded into MPC (no symbol); integration-efficiency in (56) → excluded |
| `β0` | reference channel gain at 1 m | – | energy-fit exponent 0.85 → `b_E` (excluded); queue-stability param in (55) → removed |
| `δ_l` | migration delay to node l | s | energy-fit coef 0.15 → `d_E` (excluded) |
| `ε_l` | migration fixed cost to node l | J | — |
| `P_tx` | IoT device transmit power (0.1 W) | W | Lyapunov penalty → `P_pen`; PSO particle count → `n_P` |
| `V` | Lyapunov drift/penalty weight | – | LQR input set `𝒰` clashes with UAV set 𝒰 → LQR unused (§8.4) |
| `N` | number of IoT devices (50) | – | MPC horizon → `N_h`; Poisson arrival count in (4) → `n_arr` |
| `w_m` | objective weight, m∈{1..6} | – | PSO inertia → `w_in`; demand weight in (63) → `ω_ij` |
| `σ_cov` | Gaussian coverage radius (§5, J_coverage) | m | — |
| `𝒢, N_G` | ground edge-server set & count (3) | – | paper `E` also = energy & episodes → energy stays `E_j`, episodes → `n_ep` |
| `E_j(t)` | UAV j stored energy | J | — |
| `n_ep` | MORL training episodes | – | — |

> Any symbol not in this table keeps its `method_extraction.md` meaning. A coder must not introduce a new meaning for a listed symbol.

---

## 4. System model (deterministic core) — build first

### 4.1 Network (supersedes III-A)
3-D space; `N=50` devices 𝒟, `M=5` UAVs 𝒰, `N_G=3` edge nodes 𝒢. Horizon `T_h=1000` slots of `Δt=1 s`. Device positions `q_i` fixed per run (see §14 init).

### 4.2 UAV motion (supersedes (1),(2),(3); see also D2/§8)
- State transition (MPC-internal): `p_j(t+1) = p_j(t) + v_j(t)·Δt`  *(eq 2)*.
- Speed constraint: `‖v_j(t)‖₂ ≤ v_max`  *(eq 3, v_max = 10 m/s)*.
- Continuous form (1) is descriptive only; the discrete update (2) is implemented. Altitude `z_j` may be fixed at `H` (§14) → effectively 2-D motion. 🔲 **SIGN-OFF (2-D vs 3-D motion).**

### 4.3 Task generation (supersedes (4),(5))
- Arrivals: per device i, homogeneous Poisson with rate `λ_i` (tasks/s), `λ_i ~ Uniform(0.1,0.3)` drawn once per run. Number arriving in a slot: `n_arr ~ Poisson(λ_i·Δt)`. **The "nonhomogeneous/time-varying" wording in the paper is not modeled** — no `λ_i(t)` profile is given. 🔲 **SIGN-OFF (homogeneous λ).**
- Each task `T_{i,k} = (L_{i,k}, W_{i,k}, τ_{i,k}, ρ_{i,k}, t^a_{i,k})`, sampled from §14 distributions.
- Urgency (eq 5), used by MORL state features only: `U_{i,k}(t) = ρ_{i,k}·τ_{i,k} / (τ_{i,k} − (t − t^a_{i,k}))`; clip when denominator ≤ 0 (deadline passed → task dropped).

### 4.4 Channel (supersedes (6),(7))
- Gain: `h_ij(t) = β0·(‖p_j(t) − q_i‖₂² + H²)^(−α_path/2)·ξ_ij(t)`  *(eq 6)*, `ξ_ij` ~ Rician(K=15 dB).
- Rate: `R_ij(t) = B·log₂(1 + P_tx·h_ij(t)/(N₀·B))`  *(eq 7)*.

### 4.5 Computation (supersedes (8),(9))
- Exec time: `t_exec(i,k,j) = W_{i,k} / C_j`  *(eq 8)*.
- Exec energy — **contradiction to resolve:** eq (9) is quadratic `E = κ_j·C_j²·W`; Table IV gives a **linear** rate `1×10⁻⁹ Wh/cycle`. These disagree.
  - **Implemented (PROPOSED):** linear model `E_exec(i,k,j) = e_c · W_{i,k}`, `e_c = 1×10⁻⁹ Wh/cycle = 3.6×10⁻⁶ J/cycle` (paper-stated number). Rationale: it is the only *numeric* energy figure the paper provides; the quadratic (9) has no coefficient value.
  - 🔲 **SIGN-OFF (compute-energy model):** linear (Table IV) vs quadratic (eq 9). If quadratic, supply `κ_j`.

### 4.6 UAV energy (supersedes (10),(12) — these conflict)
- The paper gives two updates: (10) `…+η_j·Δt` and (12) `…+min{η_j·P_harv,E_max}` with `P_harv` undefined.
- **Implemented:** `E_j(t+1) = min{ E_j(t) − E_cons,j(t) + η_j·Δt , E_max }`  *(eq 10 form)*; `η_j = 5 W`, `E_max = 1000 Wh`.
- `E_cons,j(t) = P_flight·Δt·1{UAV moving} + Σ_{i,k} x_ijk(t)·E_exec(i,k,j)`; `P_flight = 100 W`.
- Constraint: `E_j(t) ≥ E_min` (eq 23). Reason for choice: (10) is self-contained; (12)'s `P_harv` is undefined. 🔲 **SIGN-OFF (E_min value, §14).**

### 4.7 Queue dynamics — TWO-TIER (NEW; paper never wrote any queue; required by D5/A1)

The system carries **two coupled backlogs per slot**. Stability (§10) requires **both** to stay bounded. The tiers are joined by an **explicit hand-off**: bits that leave a device's radio queue become compute work arriving at the serving UAV's compute queue.

**Notation & the bits→cycles conversion.** Each task `k` from device `i` carries `L_{i,k}` **bits** (to transmit) and `W_{i,k}` **cycles** (to execute). Its **compute-intensity** is the exact per-task ratio
```
r_{i,k} = W_{i,k} / L_{i,k}        [cycles per bit]
```
`r_{i,k}` is **the** conversion factor that turns transmitted bits (radio tier, §Tier 1) into queued cycles (compute tier, §Tier 2). It is applied **per task** at the hand-off (below) — never as a device-wide average, so a device mixing light tasks (low `W/L`) and heavy tasks (high `W/L`) hands off exactly the cycles its transmitted bits represent. *(A separate representative mean `r̄` appears later only as the unit-balancing constant `c_Q` in the Lyapunov function §10.1; `r̄` converts no actual work.)*

#### Tier 1 — per-device RADIO/transmission backlog `Q^r_i(t)` (unit: **bits**)
Bits generated at device `i` waiting to be offloaded over the uplink.
```
Q^r_i(t+1) = max{ Q^r_i(t) − S^r_i(t), 0 } + A^r_i(t)
```
- Arrivals `A^r_i(t) = Σ_{k : t^a_{i,k}=t} L_{i,k}` (newly generated task bits).
- Transmission service (capacity) `S^r_i(t) = Σ_j x_ij(t)·R_ij(t)·Δt` (bits the uplink can carry this slot; `x_ij(t)=1` iff `i` offloads to UAV `j`, from the MORL decision).
- **Per-task bits transmitted** (needed so the hand-off can convert each task by its own intensity): the device serves its queued tasks in **PRIORITY-BY-URGENCY order — highest `U_{i,k}(t)` first** (eq 5, §4.3), so the radio queue drains in the order that best serves the `J_completion` deadline objective and reuses the urgency already in MORL's state. Ties broken by earliest arrival. *(Approved 2026-06-23; replaces the earlier FIFO default — see §15 entry 23 and the Lyapunov interaction note §10.3.)* Let `b_{i,k}(t) ≥ 0` be the bits of task `k` actually sent in slot `t`, with
  ```
  Σ_k b_{i,k}(t) = D^r_i(t) = min{ Q^r_i(t)+A^r_i(t), S^r_i(t) }   [bits],   b_{i,k}(t) ≤ remaining bits of task k.
  ```
  The aggregate `D^r_i(t)` bits leave Tier 1; the per-task split `{b_{i,k}(t)}` feeds the hand-off below.

#### Hand-off (explicit Tier 1 → Tier 2 coupling — THIS is where bits become cycles)
Each transmitted bit-segment is converted to cycles by **its own task's** intensity `r_{i,k} = W_{i,k}/L_{i,k}`, then summed over the tasks and devices served by UAV `j`:
```
                ┌─ per task: (cycles/bit) · (bits) = cycles ─┐
H_j(t) = Σ_i x_ij(t) · Σ_k  r_{i,k} · b_{i,k}(t)              [cycles]
                            └ W_{i,k}/L_{i,k} ┘
```
- **Unit check:** `r_{i,k}` [cycles/bit] × `b_{i,k}(t)` [bits] = [cycles]; the double sum stays in cycles → total CPU work delivered to UAV `j` this slot.
- **Why per-task, not mean:** if all of device `i`'s tasks shared one intensity `r_i`, this collapses to `H_j = Σ_i x_ij·r_i·D^r_i`. In general the per-task `r_{i,k}` is kept so the cycles handed off equal the **exact** work of the specific bits transmitted — no averaging across heterogeneous tasks.
- **Consistency identity:** if a task `k` is transmitted in full (`Σ_t b_{i,k}(t) = L_{i,k}`), the total cycles it contributes over its transmission is `r_{i,k}·L_{i,k} = (W_{i,k}/L_{i,k})·L_{i,k} = W_{i,k}` — i.e. exactly its CPU requirement enters the compute tier. (Built-in sanity check for the implementation.)
- This term is Tier 1's *service* re-expressed as Tier 2's *arrivals* — the queues are coupled: draining the radio tier necessarily loads the compute tier.

#### Tier 2 — per-UAV COMPUTE backlog `Q^c_j(t)` (unit: **cycles**)
CPU work delivered to UAV `j` waiting for execution.
```
Q^c_j(t+1) = max{ Q^c_j(t) − S^c_j(t), 0 } + A^c_j(t)
```
- Compute arrivals `A^c_j(t) = H_j(t)` [cycles] (the hand-off; the SAME bits that departed Tier 1, each converted to cycles by its own task's `r_{i,k}`).
- Compute service `S^c_j(t) = C_j·Δt` (cycles UAV `j` executes per slot; when APSO subdivides capacity, `Σ_{i,k} f_ijk ≤ C_j`, constraint (22)).

#### Conventions
- Initial conditions: `Q^r_i(0) = 0` ∀i, `Q^c_j(0) = 0` ∀j (§14, B25).
- Units differ by tier (bits in Tier 1, cycles in Tier 2). The **only** bits→cycles conversion of actual work happens at the hand-off, per task, via `r_{i,k}`. §10 separately introduces a scalar `c_Q` to put bits² and cycles² on one scale *inside the Lyapunov function* — that is a unit-balancing constant for the stability metric, **not** a work conversion.
- `Q^r_i` and `Q^c_j` are **the** queues referenced everywhere; the single symbol `Q_i` from the symbol table (§3) now denotes the radio tier `Q^r_i`, and `Q^c_j` is added for the compute tier.
- supersedes: the implicit single queue behind (32),(33); fills the gap noted in `method_extraction.md` §6 #25. Logged in §15 (entry 16).

---

## 5. The single objective (supersedes (14)–(20); D1)

### 5.1 Master objective — defined ONCE
```
min  J = Σ_{m ∈ {1,2,3,5,6}} w_m · J̃_m
```
with the **five active terms** below. `J̃_m` is the **normalized** term (§5.3). Default `w_m = 1.0` for each active `m` (§5.4).

> **Migration dropped (A4).** Term `m=4` (`J_migration`) and weight `w_4` are **removed** from the implemented model, along with the migrate variable `y_ijkl`, constraint (26), and parameters `δ_l`/`ε_l`. Weight indices are **left unrenumbered** (active set `{1,2,3,5,6}`) so that every `w5`/`w6` reference in §6/§8/§9 stays valid. Logged in §15 (entry 19).

### 5.2 The six raw terms (signs preserved from the paper)
| m | Term | Raw definition | supersedes |
|---|---|---|---|
| 1 | `J_task` | `Σ_t Σ_i Σ_j Σ_k x_ijk(t)·(T_trans,ij(t)+t_exec(i,k,j))` — total latency (s) | (15) |
| 2 | `J_energy` | `Σ_t Σ_j ( P_flight·Δt·1{moving} + Σ_{i,k} x_ijk(t)·E_exec(i,k,j) )` — total energy (J) | (16) |
| 3 | `J_completion` | `− Σ_t Σ_i Σ_j Σ_k x_ijk(t)·𝟙(T_comp,ijk(t) ≤ τ_{i,k})` — neg. on-time count | (17) |
| ~~4~~ | ~~`J_migration`~~ | **REMOVED (A4).** Migration is out of the implemented model (no `y_ijkl`, no `δ_l`/`ε_l`). | ~~(18)~~ |
| 5 | `J_util` | `− Σ_t Σ_j ( Σ_{i,k} x_ijk(t)·f_ijk(t) ) / C_j` — neg. utilization | (19) |
| 6 | `J_coverage` | `− D(p(t))` summed over t, where `D(p(t)) = Σ_i Σ_j ω_ij·exp(−‖p_j(t)−q_i‖₂²/(2σ_cov²))` — neg. smooth coverage | (20)+(63) |

- `T_trans,ij(t) = L_{i,k}/R_ij(t)`; `T_comp = T_trans + t_exec`.
- **Coverage rewrite (J_coverage):** the paper's (20) `min{max{𝟙(R_ij≥R_min),1},1}` is degenerate (collapses to 1). It is **replaced** by the smooth Gaussian coverage `D(p)` from (63), which is differentiable (so MPC, §8, can use its gradient — this is exactly where the gradient-ascent (62) is absorbed). `ω_ij` = device demand weight (default 1), `σ_cov` = coverage radius. 🔲 **SIGN-OFF (coverage rewrite + σ_cov, ω_ij).** This is a modeling change, not just a parameter.

### 5.3 Normalization (makes `w_m=1` meaningful; D1 requires it)
Each term is divided by a **fixed reference scale** `S_m` computed once from config (deterministic, reproducible — *not* a running normalizer):
```
J̃_m = J_m / S_m
```
PROPOSED reference scales (all 🔲 **SIGN-OFF**), with `N̂_task = N·λ̄·T_h ≈ 50·0.2·1000 = 10000` expected tasks, `τ̄ = 12.5 s` mean deadline:

| m | `S_m` | Rationale |
|---|---|---|
| 1 task | `N̂_task · τ̄` | total latency if every task took its mean deadline |
| 2 energy | `N̂_task · W̄ · e_c  +  M · P_flight · Δt · T_h` | **total energy = compute + flight** (corrected 2026-06-23; see below and §15 entries 25–26) |
| 3 completion | `N̂_task` | max possible completions |
| ~~4 migration~~ | — | **REMOVED (A4)** — no migration term to normalize |
| 5 util | `M · T_h` | max utilization-slots (per-slot accrual) |
| 6 coverage | `N · T_h` | all devices covered every slot over the horizon (per-slot accrual, like util; corrected 2026-06-23, §15 entry 28) |

with `N̂_task = N·λ̄·T_h ≈ 10000`, `W̄ = 5.5×10⁸ cycles` (mean of `W`), `e_c = 3.6×10⁻⁶ J/cycle`.

**Per-slot vs per-task accrual (why the scales differ).** Terms that accrue **every slot** (energy flight component, `util`, `coverage`) carry a `T_h` factor in their scale; terms counted **once per task** (`task` latency, `completion`) scale with `N̂_task`. The coverage scale was initially `N` (missing `T_h`) — corrected to `N·T_h` so `J̃_coverage` is O(1) like the others (entry 28).

**`S_energy` correction (CORRECTNESS, not tuning — §18).** The earlier flight-only scale `M·P_flight·Δt·T_h ≈ 0.5 MJ` was wrong: the implemented computation model (§4.5) shows **compute energy dominates flight energy by ~40×** (expected compute `≈ N̂_task·W̄·e_c ≈ 19.8 MJ` vs flight `≈ 0.5 MJ`; see §15 entry 25). A flight-only normalizer would leave `J̃_energy ≈ 40` while the other normalized terms are O(1), silently letting energy swamp the objective. The corrected scale uses **total expected per-run energy**:
```
S_energy = N̂_task · W̄ · e_c  +  M · P_flight · Δt · T_h  ≈ 1.98×10⁷ + 0.50×10⁶ ≈ 2.03×10⁷ J  (≈ 20.3 MJ).
```
*Reason for the change:* a normalizer must match the magnitude of the quantity it normalizes; this is independent of any target result (the fix would be identical whatever number the paper reported).

Result: each `J̃_m` is dimensionless and O(1); terms 3,5,6 lie in ~[−1,0]. (Normalization scheme signed off 2026-06-23 with this correction.) Alternative (z-score / min-max over a warm-up) is possible but less reproducible — not chosen.

### 5.4 Weights
- Default: `w_1 = … = w_6 = 1.0`, applied to the **normalized** terms.
- Marked **tunable baseline**; a sensitivity sweep over `w_m` is planned (experiments). 🔲 **SIGN-OFF (default weights = 1.0 and sweep plan).**
- Weights are **fixed for a run** and shared by all optimizers (§6). No optimizer rescales them.

### 5.5 Constraints (supersedes (21)–(25); (26) removed by A4)
(21) `Σ_{j,k} x_ijk(t) ≤ 1`; (22) `Σ_{i,k} x_ijk(t)·f_ijk(t) ≤ C_j`; (23) `E_j(t) ≥ E_min`; (24) `‖v_j(t)‖≤v_max`; (25) `T_comp,ijk(t) ≤ τ_{i,k}`. ~~(26) `Σ_l y_ijkl(t) ≤ 1`~~ — **REMOVED (A4, migration dropped).**

---

## 6. Optimizer projections of the one objective (D1)

Every optimizer minimizes a **subset** of the *same* `Σ w_m J̃_m` — same weights, same normalized terms — restricted to the terms its decision variables affect. There is **no** second weighting scheme.

| Optimizer | Owns decision | Sees terms (projection) | Weights used | supersedes paper subset |
|---|---|---|---|---|
| **MORL** | `x_ijk` (offload only) | `{J̃_task, J̃_energy, J̃_completion, J̃_util}` = m∈{1,2,3,5} | `w1,w2,w3,w5` | (eq 27 reward {1,2,3,5} — now **matches the paper** after dropping migration) |
| **MPC** | `p_j` / `v_j` (trajectory) | `{J̃_task, J̃_energy, J̃_coverage}` = m∈{1,2,6} | `w1,w2,w6` | (28) {1,2,6} — unchanged |
| **APSO** | `f_ijk` (resource alloc) | `{J̃_task, J̃_energy, J̃_util}` = m∈{1,2,5} | `w1,w2,w5` | (29) {1,2,5} — unchanged |

- **Reconciling "num_objectives = 3" (Table IV):** overridden. The master has **5** active terms. APSO and MPC each see 3; MORL sees 4. The Table-IV "3" is **not** used as a system parameter. 🔲 **SIGN-OFF (override Table IV's "3").**
- **Migration dropped (A4):** the paper's literal MORL reward omitted `J_migration`, and migration is now removed entirely (no `y_ijkl`). MORL therefore owns **offloading only**, and its projection `{1,2,3,5}` now **coincides with the paper's stated reward** — the earlier "+migration" deviation no longer exists.
- **MORL is scalarized (A2, hybrid):** because weights are fixed (D1), the multi-objective reward collapses to the single scalar `R_MORL = −Σ_{m∈{1,2,3,5}} w_m J̃_m(s,a)` (per-step contribution). The "multiobjective" character is provided **not** by a Pareto solver here but by the §5.4 **weight-sweep** (the sweep is the Pareto exploration). 🔲 **SIGN-OFF (scalarized MORL + weight-sweep as Pareto evidence).**

---

## 7. MORL — offloading (supersedes (27), Alg. 2; D3; migration dropped by A4)

- **Role:** choose, each MORL tick, `x_ijk(t)` (which UAV serves each pending task) to maximize `R_MORL` (§6). *(No migration decision — A4.)*
- **State** `s(t)`: per-UAV `(p_j, E_j, C_j)`, per-device radio backlog `Q^r_i` and per-UAV compute backlog `Q^c_j` (§4.7), link rates `R_ij`, and task urgencies `U_{i,k}` (eq 5).
- **State encoding — compute backlog is RELATIVE (load fraction), not absolute (2026-06-23, §15 entry 35).** In the implemented per-decision offload state, each UAV's compute backlog is encoded as its **share of the total** `Q^c_j / Σ_j Q^c_j` (uniform `1/M` when all-empty), *not* an absolute `Q^c_j / Q^c_max`. Reason: an absolute normalizer **saturates to 1.0** once backlogs exceed `Q^c_max` (which happens under load), making every UAV look equally full and destroying the load-balancing signal MORL must learn. The relative encoding exposes which UAV is less loaded at any backlog scale. (Capacity `C_j` and energy `E_j` use absolute min–max / fraction encodings; only the compute backlog is relative.)
- **In-loop reward = Lyapunov biasing as reward shaping (Increment 2, §10.1).** When MORL is trained inside the full loop, its per-decision reward is `−(c_Q·Q^c_j·a_ij + V·penalty_ij)` — the per-slot drift-plus-penalty cost (eq 34'). MORL therefore *learns* the stability-aware policy rather than being hard-overridden by `biased_assignment` (which would leave it nothing to learn). The objective penalty enters via the 'morl' projection (weights on the `Objective`, D1).
- **Action** `a(t)`: offload assignment `x` for currently-pending tasks, subject to constraint (21).
- **Update (eq 27):** Q-learning TD target `R_MORL + γ·max_{a'} Q(s',a')`, learned by a DQN (below).
- **Exploration:** ε-greedy, `ε: 1.0 → ×0.995/episode`.
- **Hyperparams:** `α_lr=0.001, γ=0.99, replay=10000, batch=32, n_ep` (§14).
- **Architecture — DQN function approximator (RESOLVED 2026-06-23, author's call; was the last open §16 item).** The continuous/high-dimensional state rules out a tabular Q-table; a **DQN** (MLP Q-network + target network + experience replay) is used, which matches the replay-buffer / batch / ε-decay hyperparameters already in config. Framework: **PyTorch (pinned)**. Both the PyTorch and NumPy RNGs are seeded from the frozen seed (B27=42); training runs on CPU for determinism. *Exact bit-reproducibility requires the same framework build* (documented in the module). *(§15 entry 29)*
- **D1 contract (same as APSO/MPC):** the DQN agent **holds no objective weights**. Its scalar reward is `R_MORL = −(value of the 'morl' projection)` (terms {task, energy, completion, util} = m∈{1,2,3,5}), computed by the environment via `projection.value(...)`. Because the projection pulls weights from the one `Objective`, re-running training under a different weight vector (the §5.4 weight-sweep, A2) is just a re-run with a different `Objective` — the agent is weight-agnostic. Offload-only action space (A4: no `y_ijkl`).

### 7.1 Pre-registered MORL success criteria (decided BEFORE training, per §18)

These are the bar for "MORL works," fixed before any number is seen (anti-tuning):
1. **Learning curve** trends upward and roughly plateaus (the agent improves, then stabilizes).
2. **Beats baselines:** the learned greedy policy beats **random** offloading **and** the **JSQ softmax baseline** (eq 64 / §13). *If it cannot beat JSQ, that is reported as a FINDING, not a pass.*
3. **Hand-checkable corner case:** sensible behavior in an obvious situation — e.g. a **near-depleted UAV is offloaded to less** than an equivalent charged UAV.

**Hyperparameter stance (binding):** all hyperparameters not paper-stated (network width/depth, optimizer, target-update cadence, etc.) are set on **standard grounds and FROZEN before** checking whether the paper's 94.5% completion reproduces. They are **never** tuned toward that target (§18). Whether 94.5% reproduces is a separate, post-hoc comparison reported honestly under the §18 tiers. *(§15 entry 30)*

---

## 8. MPC — trajectory (supersedes (1),(2),(28),(39)–(41),(62); D2)

### 8.1 One motion rule
The three paper rules are reconciled here: **(2)** is the internal state-transition model; **(28)** is the objective; **(62)**'s coverage gradient is *inside* the objective via `J_coverage` (§5.2). There is no standalone gradient mover and no standalone kinematic mover.

### 8.2 Problem solved each MPC tick (receding horizon `N_h`)
```
min_{v_j(t..t+N_h−1)}  Σ_{τ=t}^{t+N_h−1} [ w1·J̃_task(τ) + w2·J̃_energy(τ) + w6·J̃_coverage(τ) ]
s.t.  p_j(τ+1) = p_j(τ) + v_j(τ)·Δt      (eq 2, internal model)
      ‖v_j(τ)‖₂ ≤ v_max                  (eq 3)
      p_j(τ) ∈ feasible region (no-fly zones, §14)
apply v_j(t); re-plan next tick.
```
- Coverage term `J̃_coverage` is smooth in `p` (§5.2), so gradient-based or sampling-based MPC both work.
- `T_trans`/rate terms couple trajectory to `J_task` (closer UAV → higher `R_ij`).

### 8.3 Horizon
`N_h` = MPC prediction horizon; complexity O(M·N_h³) (eq 38) ⇒ keep modest. PROPOSED `N_h = 10` slots. 🔲 **SIGN-OFF (N_h).**

### 8.4 Linear-QP form (39)–(41) is NOT implemented as written
The paper's LQR-style QP needs `A,B,Q_lqr,R_lqr,P_lqr` and sets `𝒳,𝒰` that the paper never specifies, and its `Q_lqr/R_lqr` collide with queue/rate symbols. We implement the **direct** formulation in §8.2 instead. `Q_lqr,R_lqr,P_lqr` are therefore unused. 🔲 **SIGN-OFF (skip the LQR-QP form).**

---

## 9. APSO — resource allocation (supersedes (29),(30),(31), Alg. 4)

- **Role — allocates the SPLIT of a UAV's full compute capacity, NOT total capacity (clarified 2026-06-23, §15 entry 31).** Given MORL's assignment `x` and MPC's positions `p`, APSO chooses continuous `f_ijk ∈ [0,C_j]` with `Σ_{i,k} f_ijk ≤ C_j` (constraint 22) to minimize the APSO projection `w1·J̃_task + w2·J̃_energy + w5·J̃_util` (§6). The `f_ijk` partition each UAV's processor **among the tasks assigned to it**; they do **not** decide whether the processor runs. **A UAV always executes at its full capacity `C_j`** (its compute service is `S^c_j = min(Q^c_j, C_j·Δt)`, §4.7); APSO only decides *how that capacity is divided across tasks* (affecting per-task latency/completion), never the aggregate throughput. Setting `Σf < C_j` (idling the CPU) is not an admissible interpretation.
- **Particle** = a full `{f_ijk}` vector. **Updates (eqs 30,31):**
  `v_id ← w_in·v_id + c1·r1·(pbest_id − x_id) + c2·r2·(gbest_id − x_id)`; `x_id ← x_id + v_id`; clip to `[0,C_j]`.
- **"Adaptive":** the paper never specifies the adaptation rule. PROPOSED: linearly decay inertia `w_in: 0.9→0.4` over iterations (standard). 🔲 **SIGN-OFF (adaptation rule).**
- **Hyperparams (all PROPOSED, §14):** swarm `n_P`, `w_in`, `c1`, `c2`, iterations `n_iter`.
- **Pipeline note:** final `f_ijk` = APSO output (overrides MORL's provisional `f^rl` from Alg. 1 line 7).
- **Implementation status (Increment 3 DONE, 2026-06-23):** the compute tier now tracks a **per-task** backlog at each UAV (`compute_items[j] = {tid: remaining_cycles}`, kept in sync with the aggregate `Q^c_j`). Each slot APSO allocates `C_j` across UAV `j`'s tasks via the 'apso' projection: no contention → serve all (full CPU); contention → APSO splits the scarce capacity (minimizing the projection ⇒ `f ∝ √remaining`, larger tasks get more). Conservation is exact end-to-end (Σf ≤ C_j·Δt, each task drained ≤ its remaining). *(§15 entry 40 / simulation.py)*

---

## 10. Lyapunov — stability (supersedes (32),(33),(34); D5)

The Lyapunov layer now governs the **two-tier** backlog of §4.7 (A1). Let the joint backlog be `Θ(t) = ( {Q^r_i(t)}_i , {Q^c_j(t)}_j )` — all device radio queues and all UAV compute queues.

### 10.1 Primary track — two-tier drift-plus-penalty controller (implement now)
- **Combined Lyapunov function (extends eq 32):**
  ```
  L(Θ(t)) = ½·[ Σ_i Q^r_i(t)²  +  c_Q·Σ_j Q^c_j(t)² ]
  ```
  `c_Q > 0` is a **tier-scaling constant** that makes bits² (Tier 1) and cycles² (Tier 2) comparable inside `L`; PROPOSED `c_Q = r̄⁻²` (r̄ = a representative mean intensity, cycles/bit), so a queued bit and the cycles it becomes carry equal Lyapunov weight. Note `r̄` here is only a unit-balancing scalar for the stability metric — the **actual** work conversion is the exact per-task `r_{i,k}` at the §4.7 hand-off, not `r̄`. 🔲 **SIGN-OFF (c_Q).**
- **Drift (eq 33):** `Δ(t) = E[ L(Θ(t+1)) − L(Θ(t)) | Θ(t) ]`.
- **Penalty:** `P_pen(t) = Σ_{m∈{1,2,3,5,6}} w_m·J̃_m(t)` — the per-slot value of the one (now 5-term) objective (§5; migration term m=4 removed per A4).
- **Per-slot control (eq 34):** among feasible decisions for the slot, choose those minimizing the linearized drift-plus-penalty bound across **both** tiers:
  ```
  Σ_i Q^r_i(t)·( A^r_i(t) − S^r_i(t) )
    +  c_Q·Σ_j Q^c_j(t)·( A^c_j(t) − S^c_j(t) )
    +  V·P_pen(t)
  ```
  with `A^r,S^r,A^c,S^c` from §4.7. **Cross-tier coupling is explicit and must be honored:** `A^c_j(t) = H_j(t)` depends on the radio service `S^r` (transmitted bits), so emptying a device's radio queue *fills* the serving UAV's compute queue. The controller therefore cannot drain one tier in isolation — aggressive offloading that clears `Q^r` can destabilize `Q^c`. Operationally this **biases** the MORL/MPC/APSO decisions: small `V` favors joint backlog draining (balancing radio vs compute load), large `V` favors objective-optimality. In Algorithm 1 (§11) this is the "Lyapunov_check → adjust" stage.
- **Empirical stability claim only — BOTH tiers:** stability = *both* time-average backlogs stay bounded across the run,
  ```
  limsup (1/T_h)Σ_t (1/N)Σ_i Q^r_i(t) < ∞   AND   limsup (1/T_h)Σ_t (1/M)Σ_j Q^c_j(t) < ∞.
  ```
  Report both as **measured**, never as proven. `V` PROPOSED = 1.0, to be swept. 🔲 **SIGN-OFF (V default + sweep).**

### 10.2 Secondary track — two-tier drift-bound derivation (TODO, NOT a proof yet)
> **This subsection is a placeholder. No bound is claimed.** The two-tier model makes the derivation strictly harder than the single-queue case because the tiers are coupled. To close it one must: (a) bound
> `Δ(t) ≤ B + Σ_i Q^r_i(t)·E[A^r_i − S^r_i | Θ] + c_Q·Σ_j Q^c_j(t)·E[A^c_j − S^c_j | Θ]`
> with an explicit constant `B`, where the **cross-terms** arise because `A^c_j = Σ_i x_ij Σ_k r_{i,k}·b_{i,k}` (the per-task hand-off, §4.7) is itself a function of the Tier-1 service (handed-off bits ≤ `S^r_i`); (b) show the §10.1 controller minimizes the RHS jointly over both tiers; (c) derive the `[O(1/V) optimality gap, O(V) backlog]` tradeoff for the **joint** backlog. The coupling means the per-tier rate-stability conditions are **not separable** and `B` must absorb the hand-off cross-terms. The derivation must also carry the §10.3 service discipline as a constraint on the achievable compute-arrival process. Until (a)–(c) are written and checked, **the repo claims empirical stability only.** Tracking: `method_extraction.md` §6 #26.

### 10.3 Service discipline ↔ drift interaction (priority-by-urgency; no stability conflict)
The radio-queue discipline is **priority-by-urgency** (§4.7: highest `U_{i,k}` first). Its interaction with the two-tier drift was checked explicitly (it does **not** conflict with §10.1):
1. **Radio-tier drift unchanged.** The discipline sets only the *order* tasks within `Q^r_i` drain; it does not change the aggregate per-slot service `S^r_i` (it is work-conserving). Hence `Q^r_i·(A^r_i − S^r_i)` is identical to FIFO — the drift sees only totals.
2. **Compute-tier arrival composition is order-dependent.** Because intensities `r_{i,k}` differ across tasks, the order changes *which* cycles enter `A^c_j(t)` each slot. But the long-run compute arrival rate equals `Σ(offloaded W)` regardless of order, so the rate-stability condition — and boundedness of `Q^c_j` — is **unchanged**.
3. **What it costs.** Priority-by-urgency is a *layered scheduling policy*, not a Lyapunov decision variable. It can make the per-slot compute drift non-minimal (the controller no longer freely orders intra-queue work) — i.e. **mildly drift-suboptimal** — but it does not enlarge or shrink the stability region.
4. **Starvation is self-limiting.** By eq 5, `U_{i,k} → ∞` as a task nears its deadline, so an aging low-urgency task's priority rises automatically; it cannot be starved indefinitely. This is why the choice aligns with `J_completion`.

**Net:** compatible with the §10.1 empirical-stability claim; trades a small amount of drift-optimality for deadline alignment. Recorded so the trade is explicit, not hidden.

---

## 11. Orchestration — Algorithm 1 & timescales (supersedes Alg. 1, (53))

Per slot `t` (pipeline order fixed):
1. observe state `s(t)`.
2. **MORL** → `x_ijk(t)` (offload only; §7).
3. **MPC** → `p_j(t)` via `v_j(t)` (§8).
4. **APSO** → `f_ijk(t)` (§9).
5. **Lyapunov** → evaluate two-tier drift; adjust decisions per §10.1 if either backlog tier is unstable.
6. step environment: update **both queue tiers + hand-off** (§4.7), energy (§4.6), positions (§4.2); inject failures/no-fly (§14).
7. learn (MORL update); log metrics.

**Timescales (eq 53):** components may run at different cadences `m_MORL > m_MPC > m_APSO` (in slots). PROPOSED `m_MORL=10, m_MPC=2, m_APSO=1` (MORL re-decides every 10 slots, MPC re-plans every 2, APSO every slot). 🔲 **SIGN-OFF (cadences).** If all =1, the pipeline simply runs every slot.

---

## 12. Excluded from the implemented model (D1, D4)

| Item | Paper eq | Why excluded | Where it may still appear |
|---|---|---|---|
| Adaptive weight update | (55) | D1: weights fixed for reproducibility | nowhere (documented here only) |
| Softmax weight scheme | (59) | D1: second weighting scheme forbidden | nowhere |
| Energy–completion regression | (60) | D4: a fit of outputs, not dynamics (`R²=0.97`) | post-hoc plot of results, with its R² |
| Load characteristic `L(t)` | (67) | D4: descriptive response model | post-hoc characterization |
| Recovery characteristic `R(t)` | (69) | D4: descriptive | post-hoc characterization |

These must **never** drive the simulation state. They are analysis overlays only.

---

## 13. Baselines for experiments (not part of the live system; D3)

To be implemented under `experiments/` for comparison, **not** in `src/moalf/`:
- **JSQ-softmax assignment (eq 64):** `P(UAV_i)= softmax(−Q_i/T_sm + λ_b·θ_rel)`. A baseline offloader contrasted against MORL. `T_sm` (softmax temperature), `λ_b` (baseline sensitivity) are baseline-only symbols.
- Other paper baselines (Table IX): DDPG, NSGA-II, MA-DRL, JTO, MAPPO — out of scope for this spec; add as needed.

---

## 14. Parameter table

**Legend:** `paper-stated` (Table IV / cited section) vs `PROPOSED — needs sign-off`. Every PROPOSED row is 🔲 in §16.

### 14.1 Paper-stated (no decision needed)
| Param | Value | Source |
|---|---|---|
| Area | 400×400 m | Table IV |
| Horizon `T_h` | 1000 slots | Table IV |
| `Δt` | 1 s | Table IV |
| `N` devices | 50 | Table IV |
| `M` UAVs | 5 | Table IV |
| `N_G` edge | 3 | Table IV |
| Network slices | 3 | Table IV |
| `C_j` (UAV) | Uniform(2,5) GHz | Table IV |
| `C_l` (edge) | Uniform(10,20) GHz | Table IV |
| `E_max` | 1000 Wh | Table IV |
| `v_max` | 10 m/s | Table IV |
| `P_flight` | 100 W | Table IV |
| `e_c` (compute energy) | 1×10⁻⁹ Wh/cycle | Table IV |
| `η_j` (harvest) | 5 W | Table IV |
| burst duration / mult. | 20 slots / 2× | Table IV |
| Task gen rate `λ_i` | Uniform(0.1,0.3) /s | Table IV |
| Task size `L` | Uniform(0.1,1) MB | Table IV (prose "0–1" — see changelog) |
| Task work `W` | Uniform(100,1000) Mcyc | Table IV |
| Deadline `τ` | Uniform(5,20) s | Table IV |
| Priority `ρ` | Uniform int(1,5) | Table IV |
| `P_tx` | 0.1 W | Table IV |
| Channel | Rician K=15 dB | Table IV |
| `α_lr,γ,ε,ε-decay` | 0.001, 0.99, 1.0, 0.995 | Table IV |
| replay / batch | 10000 / 32 | Table IV |
| UAV failure prob | 0.05 | Table IV |
| No-fly prob / radius | 0.1 / Uniform(10,50) m | Table IV |
| `θ_rel` link-reliability thr. | 0.95 | Table IV |
| SDN delay / flow table / update | 10 ms / 1000 / 100 ms | Table IV |
| **Number of runs** | **30 per configuration** | **paper §VI methodology** (not unspecified) |

### 14.2 PROPOSED defaults for previously-unspecified parameters (each 🔲 SIGN-OFF)
| Param | PROPOSED value | One-line justification |
|---|---|---|
| `β0` (ref gain @1 m) | −30 dB (1×10⁻³) | typical sub-6 GHz free-space reference gain at 1 m |
| `α_path` | 2.0 | LoS-dominant air-to-ground, consistent with Rician K=15 dB |
| `N₀` | −174 dBm/Hz | standard thermal noise PSD |
| `B` | 1 MHz | Table IV's "1000 Mb/s" is an aggregate capacity, not B; 1 MHz gives plausible per-link rates |
| `H` (altitude) | 100 m | common UAV-BS altitude for a 400 m cell |
| compute-energy model | linear `e_c·W` | only numeric energy figure given (see §4.5; vs eq 9 quadratic) |
| `R_min` (coverage rate) | 1 Mb/s | nominal QoS floor for the eval-side coverage metric |
| `σ_cov` (coverage radius) | 100 m | ≈ ¼ of cell width; sets Gaussian coverage footprint (§5.2) |
| `ω_ij` (demand weight) | 1.0 ∀i,j | uniform demand absent other info |
| ~~`δ_l` (migration delay)~~ | **REMOVED** | migration dropped (A4) |
| ~~`ε_l` (migration fixed cost)~~ | **REMOVED** | migration dropped (A4) |
| `E_min` (reserve) | 100 Wh (10% of E_max) | typical safe-return reserve |
| `V` (Lyapunov) | 6.9×10¹⁶ | unit-commensurability rescale at the λ=0.2 reference (§15 entry 37); was 1.0 |
| `c_Q` (two-tier scaling) | `r̄⁻²` (mean intensity, cycles/bit) | NEW (A1): equalizes bits²/cycles² in the combined Lyapunov fn (§10.1) |
| `N_h` (MPC horizon) | 10 slots | balances foresight vs O(M·N_h³) cost |
| `n_P` (swarm) | 30 | standard PSO swarm size |
| `w_in` (inertia) | 0.9→0.4 decay | standard adaptive inertia |
| `c1, c2` | 1.49, 1.49 | Clerc-constriction standard |
| `n_iter` (APSO) | 100 | standard convergence budget |
| `m_MORL,m_MPC,m_APSO` | 10, 2, 1 slots | MORL slowest, APSO fastest (eq 53 ordering) |
| `n_ep` (MORL episodes) | 1 run = 1 episode of `T_h` slots | simplest mapping; revisit if episodic resets wanted |
| UAV init position | uniform-random in area, `z=H` | no layout given; reproducible via seed |
| UAV init energy `E_j(0)` | `E_max` (full) | standard "start charged" assumption |
| Queue init `Q_i(0)` | 0 | empty system at t=0 |
| Device positions `q_i` | uniform-random in area, fixed per run | no layout given |
| RNG seed | 42 | reproducibility; runs 0–29 use seeds 42…71 |
| weights `w_m` | 1.0 ∀m (normalized terms) | D1 baseline; sweep planned |

> Default weights and normalization are repeated here from §5 for the parameter-file author.

---

## 15. Contradiction changelog (for the manuscript revision)

One line each: **paper said → we chose → why.**

1. **Objective weights:** paper used 4 different subsets + 2 adaptive rules (55,59) → single fixed 6-term objective (14), optimizers see projections → reproducibility, one source of truth. *(D1)*
2. **"num_objectives = 3" (Table IV):** stated 3 → overridden to 6-term master; optimizers see 3–5 → matches the actual six-term cost. *(D1)*
3. **MORL reward terms:** paper {task,energy,completion,util} → **unchanged** (migration later dropped, see entry 19) → MORL projection now coincides with the paper's stated reward. *(D1/D3; superseded detail in entry 19)*
4. **UAV motion:** three rules (2),(28),(62) → one MPC scheme; (2)=model, (62)=coverage-gradient inside objective → eliminates triple control of position. *(D2)*
5. **Task assignment:** MORL `x` vs JSQ softmax (64) → MORL owns it; (64)=baseline → one decider. *(D3)*
6. **Compute energy:** quadratic (9) vs linear Table-IV rate → linear (uses the only stated number) → consistency; flagged. *(§4.5)*
7. **UAV energy harvest:** (10) `η·Δt` vs (12) `min{η·P_harv,…}` (P_harv undefined) → (10) → self-contained. *(§4.6)*
8. **Coverage term:** degenerate `min{max{𝟙,1},1}` (20) → smooth Gaussian `−D(p)` (63) → differentiable; absorbs (62); non-trivial → flagged. *(§5.2)*
9. **Queue dynamics:** never written → explicit, now **two-tier** (radio + compute, see entry 16) → required to implement Lyapunov. *(D5/§4.7; superseded by entry 16)*
10. **Lyapunov proof:** paper implies stability guarantees → we claim **empirical** stability only; bound is a marked TODO → no proof exists yet. *(D5/§10.2)*
11. **Regression fits (60,67,69):** read as system behavior → reclassified as output fits, excluded from dynamics → they describe results, not mechanism. *(D4)*
12. **Arrival process:** called "nonhomogeneous/time-varying" → modeled homogeneous (static rate draw) → no `λ_i(t)` profile given; flagged. *(§4.3)*
13. **Task size range:** Table IV `Uniform(0.1,1) MB` vs prose "0–1 MB" → use Table IV → table is the parameter source. *(§14)*
14. **Symbol overloading** (α,λ,T,Q,R,τ,E,β,δ,η,…): one meaning each; secondary uses renamed → see §3 → removes ambiguity. *(§3)*
15. **MPC QP internals (39–41):** matrices/sets unspecified → use direct formulation (§8.2), skip LQR-QP → unimplementable as written; flagged. *(§8.4)*

### Sign-off decisions — 2026-06-23 (author cleared §16; two changes from proposal)

16. **Queue model → TWO-TIER (A1, CHANGE):** the proposed single per-device transmission-only queue → **two coupled backlogs** — per-device **radio** `Q^r_i` (bits) and per-UAV **compute** `Q^c_j` (cycles) — with an **explicit per-task hand-off** (`A^c_j = H_j = Σ_i x_ij Σ_k r_{i,k}·b_{i,k}`, where `r_{i,k}=W_{i,k}/L_{i,k}` cycles/bit: each transmitted bit becomes cycles via *its own task's* intensity). Stability now requires **both** bounded. §4.7 and §10 rewritten; §10.2 derivation now spans both tiers with non-separable cross-terms. *Why:* the author's central stability claim is congestion at **both** the radio and the UAV-compute stage, which a transmission-only queue cannot represent. *(A1/D5/§4.7/§10)*
17. **MORL framing → scalarization + weight-sweep (A2, hybrid):** ship MORL as **fixed-preference weighted-sum scalarization**; provide the "multiobjective/Pareto" evidence via the §5.4 **weight-sweep**, not a Pareto solver. **Manuscript action:** describe MORL as *fixed-preference scalarized multiobjective RL with a weight-sweep Pareto exploration* — do **not** claim an online Pareto-front method. *(A2/§6/§7)*
18. **Coverage term → Gaussian, eq (20) needs ERRATUM (A3):** keep the smooth Gaussian `J_coverage = −D(p)`. **Manuscript action:** paper eq (20) `min{max{𝟙(R_ij≥R_min),1},1}` is **degenerate — it evaluates to the constant `−N`** independent of UAV positions, so it cannot have produced the reported coverage results. Issue an **erratum** correcting (20) and stating the actual coverage metric used. *(A3/§5.2)*
19. **Migration DROPPED entirely (A4, CHANGE):** remove the migrate variable `y_ijkl`, the objective term `J_migration` (objective → **5 active terms**, weight indices left as `{1,2,3,5,6}`, `w_4` retired), constraint (26), the §5.3 migration normalization scale, and parameters `δ_l`/`ε_l`. *Why:* migration was thinly specified and vestigial (`δ_l`,`ε_l` undefined, no migration result reported); dropping it removes two unknowns and simplifies the model. **Manuscript action:** either remove migration from the formulation or mark it explicitly out-of-scope for this reproduction. *(A4/§5/§6/§7)*
20. **Parameters `δ_l`, `ε_l` removed (B8/B9):** previously PROPOSED (0.05 s, 0.5 J) → **removed** as moot, consequent to migration being dropped (entry 19). *(A4/B8/B9)*
21. **Seed frozen permanent (B27, anti-tuning):** RNG seed 42 (runs 0–29 → seeds 42…71) is **frozen permanently** under the §18 freeze-before-compare rule; it may not be changed to influence any reproduced metric. *(B27/§18)*
22. **2-D motion confirmed (A5):** fixed altitude `H`; `H` enters only as a constant in the channel gain. No change from proposal. *(A5/§4.2)*
23. **Radio-queue service discipline → PRIORITY-BY-URGENCY (2026-06-23):** the §4.7 flag resolved to draining tasks by descending urgency `U_{i,k}` (eq 5), highest first, ties by earliest arrival — *not* FIFO. *Why:* aligns the queue with the `J_completion` deadline objective and reuses the urgency already in MORL's state. Checked against the Lyapunov drift (§10.3): **no stability conflict** — work-conserving so radio-tier drift is unchanged and the compute-tier rate condition is preserved; the discipline is mildly drift-suboptimal but not destabilizing, and starvation is self-limiting via eq 5. *(§4.7/§10.3)*

> **Net structural changes from this sign-off:** (a) one queue → two coupled queues; (b) six objective terms → five (migration gone); (c) radio queue drains by urgency. All other §16 items approved as proposed. Config folded 2026-06-23; implementation begun (channel first).

### Predictions (logged pre-experiment, for honest comparison later)

24. **PREDICTION — 2026-06-23 (before any trajectory run):** With the as-approved channel constants (B1–B5), the channel model yields **SNR ≈ 51–64 dB cell-wide** (measured from the implemented `channel.py` across 10–400 m, `β0=−30 dB`, `α_path=2`, `P_tx=0.1 W`, `B=1 MHz`, `H=100 m`). The radio link is therefore **not the bottleneck**; the compute tier dominates. **Prediction:** the paper's trajectory/route-optimization result (**38% route reduction**) may show **weak leverage in this regime**, because UAV position barely affects an already-saturated link — so a consistent re-implementation could land at **Tier 3** (divergent) on that specific metric. To be **checked when the trajectory experiment runs**. **Per §18 this is NOT to be addressed by tuning channel params**; the only admissible response is to re-examine, on physical grounds, whether the paper's intended bandwidth / path-loss / cell-size differ from the proposed B1–B5 (e.g. a wider cell, higher `α`, or smaller `B` would make position matter more). Logged so the prediction predates the result. *(§18 / B1–B5 / channel.py)*

### Findings & corrections (post-implementation)

25. **FINDING — 2026-06-23 (from `computation.py` sanity check):** With the paper-stated `e_c = 1×10⁻⁹ Wh/cycle (= 3.6×10⁻⁶ J/cycle)`, expected **compute energy ≈ 19.8 MJ/run** (`N̂_task·W̄·e_c`, with `N̂_task≈10⁴`, `W̄=5.5×10⁸ cyc`) vs **flight energy ≈ 0.5 MJ/run** (`M·P_flight·Δt·T_h`) — **compute dominates flight by ~40×**. The author **confirms `e_c` is intentional (not an erratum)**. *Consequence:* the original flight-only `S_energy` normalizer mis-scaled `J̃_energy` by ~40× (it would read ~40 while other normalized terms are O(1)). *(computation.py / §4.5 / §5.3)*
26. **CORRECTION — 2026-06-23 (`S_energy`, correctness not tuning, §18):** §5.3 `S_energy` changed from flight-only `M·P_flight·Δt·T_h` to **total expected energy** `N̂_task·W̄·e_c + M·P_flight·Δt·T_h ≈ 2.03×10⁷ J (≈20.3 MJ)`, so `J̃_energy` is O(1) like the other normalized terms. **Reason:** a normalizer must match the magnitude of the quantity it normalizes — *independent of any target result* (the fix is identical whatever number the paper reported). Follows directly from entry 25. *(§5.3 / §18)*
27. **PREDICTION — 2026-06-23 (before the objective/optimizer runs):** because compute energy dominates, `J_energy` is now **primarily a measure of compute volume**, which is in **direct tension with `J̃_completion`** (completing more tasks necessarily costs more compute energy). The **`J̃_energy` vs `J̃_completion` weight balance (`w2` vs `w3`) is therefore expected to be highly influential** on the trade-off the system finds. To be **explored via the planned §5.4 weight-sweep**, reported as a Pareto curve — **NOT** resolved by tuning toward any paper metric (§18). Logged so the expectation predates the sweep. *(§5.4 / §18 / objective.py-to-come)*
28. **CORRECTION — 2026-06-23 (`S_coverage`, found while building objective.py; correctness not tuning, §18):** §5.3 `S_coverage` changed from `N` to **`N · T_h`**. *Reason:* `J_coverage = −Σ_t D(p(t))` accrues **every slot** over the horizon (like `util` and flight energy), so its magnitude is ~`N·T_h`; the `N`-only scale would have left `J̃_coverage ≈ T_h = 1000` rather than O(1), dominating the objective ~1000×. Same class as the S_energy fix (entry 26): a normalizer must match the magnitude of what it normalizes, independent of any target result. Establishes the per-slot-accrual (`×T_h`) vs per-task (`×N̂_task`) rule now documented in §5.3. *(§5.3 / §18)*
29. **DECISION — 2026-06-23 (MORL architecture; last open §16 item resolved):** MORL is a **DQN function approximator** (PyTorch, pinned), not a tabular Q-table — the state is continuous/high-dimensional, and DQN matches the config replay/batch/ε-decay hyperparameters. RNGs (torch + numpy) seeded from the frozen seed 42; CPU for determinism; exact reproducibility requires the same framework build. The agent holds **no objective weights** (D1): reward = −value of the 'morl' projection; weight-agnostic so the §5.4 weight-sweep just re-runs it. Offload-only (A4). *(§7 / author's call)*
30. **PRE-REGISTRATION — 2026-06-23 (MORL success criteria, before training, §18):** the bar for "MORL works" is fixed in §7.1 *before seeing any number*: (1) learning curve trends up and plateaus; (2) beats random AND the JSQ baseline (failing to beat JSQ is a FINDING, not a pass); (3) sensible corner-case behavior (near-depleted UAV offloaded-to less). Non-paper hyperparameters are set on standard grounds and **frozen before** any 94.5%-completion comparison; never tuned toward it. *(§7.1 / §18)*
31. **CLARIFICATION — 2026-06-23 (APSO role; spec made authoritative over code):** APSO allocates the **split** of a UAV's full compute capacity `C_j` across its assigned tasks (`Σf_ijk ≤ C_j`); it does **NOT** throttle total capacity — a UAV always runs its CPU at `C_j` (`S^c_j = min(Q^c_j, C_j·Δt)`). *Trigger:* the skeleton's placeholder APSO proxy minimized a per-slot objective whose energy term (per-slot raws ÷ per-run `S_energy`) dominated, driving the capacity fraction to ~0 and unphysically idling the CPUs — which also masked the Lyapunov effect. Fixed to full-capacity service (a correctness fix, not tuning). §9 updated to state this; full per-task `f_ijk` allocation is Increment 3. *(§9 / simulation.py)*
32. **FINDING — 2026-06-23 (frozen config is lightly loaded):** under the **frozen real config**, expected compute demand ≈ `N·λ̄·W̄ ≈ 50·0.2·550 = 5500 Mcyc/slot` vs total capacity ≈ `M·C̄·Δt ≈ 5·3.5 GHz·1 s = 17500 Mcyc/slot` — demand is only **~1/3 of capacity**, so the system is **lightly loaded** and queues barely grow even *without* Lyapunov biasing (measured: Q^c≈0 with biasing off, near-stable). The two-tier stability mechanism is **demonstrably effective under high load** (shown via a test-local `λ=0.6` fixture: Q^c OFF≈57k vs ON≈1k Mcyc) but is **rarely BINDING under the paper's actual parameters**. **Manuscript implication:** if the paper frames *stability-under-load* as a headline result, the revision should report that the frozen scenario is lightly loaded (demand ~⅓ capacity), so the stability machinery is mostly slack there. **NOT addressed by raising λ** — the light load is what the parameters give; reported honestly rather than tuned toward a more impressive narrative (§18). *(§14 / experiments + simulation tests)*
33. **FINDING — 2026-06-23 (MORL training; JSQ baseline is near-random here):** at full training the DQN clearly beat both baselines (mean reward DQN≈1.86 vs JSQ≈0.83 vs random≈0.82), but **JSQ ≈ random** — both far below the DQN. *Cause:* the JSQ baseline (eq 64 / §13) routes by **queue length only** and is blind to the **energy and compute-capacity constraints that dominate this system** (a depleted or low-capacity UAV is a bad choice even with the shortest queue). So shortest-queue carries little signal here. **Manuscript implication:** if the paper benchmarks against JSQ, the revision should note that in this energy/capacity-constrained regime JSQ is **near-random**, so "beats JSQ" is a weak bar — this is an *incomplete-comparison context*, **not an error** in the paper. A fairer baseline would be energy/capacity-aware. To revisit once the full Algorithm-1 simulation provides the §18.4 comparison. *(§13 / experiments/train_morl.py)*
34. **FINDING — 2026-06-23 (V=1.0 makes the drift term swamp the penalty → MORL near objective-blind; logged BEFORE any V change):** in the in-loop drift-plus-penalty reward `−(c_Q·Q^c_j·a + V·penalty)`, with frozen `V=1.0` the drift term is ~`c_Q·Q^c·a ≈ 6.4e-5·1e10·1e9 ≈ 6e14` while `V·penalty ≈ 1·O(10⁻³)` — the drift **swamps the objective penalty by ~17 orders of magnitude**. Consequence: in-loop MORL currently learns **stability-dominated** routing and is **near objective-blind**. **This threatens the A2 multiobjective claim** — a weight-sweep may come out nearly **flat** (the objective weights barely move the policy) until `V` is rescaled for unit-commensurability. **Action plan (no tuning):** (i) run a weight-sweep *diagnostic* with `V=1.0` still frozen to confirm/refute flatness behaviorally (§15 entry 36 / below); (ii) if confirmed, rescale `V` on **principled unit-commensurability grounds, computed before observing any completion metric** — never tuned to a target (§18). *(§7 / §10.1 / A2 / §18)*
35. **CHANGE — 2026-06-23 (MORL state encoding: compute backlog → relative load fraction):** the per-decision offload state now encodes each UAV's compute backlog as its **share of total** `Q^c_j/Σ Q^c_j` rather than absolute `Q^c_j/Q^c_max`. *Reason:* the absolute normalizer **saturated to 1.0 under load** (backlogs exceed `Q^c_max`), so all UAVs looked equally full and the load-balancing signal vanished — in-loop MORL would not learn. The relative encoding exposes relative load at any scale and is what enabled the 7× in-loop backlog reduction. Changes *what MORL observes* (documented in §7). *(§7 / simulation.py)*
36. **DIAGNOSTIC RESULT — 2026-06-23 (weight-sweep CONFIRMS objective-blindness at V=1.0; behavioral, before any V change):** the weight-sweep harness (`experiments/weight_sweep.py`) trained MORL in-loop under 4 meaningfully different objective weight vectors (balanced, task-/completion-/energy-heavy) with **V=1.0 frozen**, and compared the learned greedy policies on a probe battery. Result at **both** loads: the policies **learned** (differed from the untrained init by **17%** of probes at real λ=0.2, **73.5%** at high λ=0.6) but showed **0.0% cross-weight divergence** — every weight vector produced the *identical* learned policy. **Behaviorally confirms entry 34** and extends it: MORL is objective-blind not only at high load (drift swamps penalty) but at the real config's light load too (the penalty is non-discriminative when capacity is ample). **The A2 multiobjective/weight-sweep evidence is currently empty** with V=1.0. **Decision pending (author):** rescale `V` for unit-commensurability on principled grounds, **computed before observing any completion metric, never tuned to a target** (§18) — then re-run this exact diagnostic; a non-flat sweep is the prerequisite for any A2 Pareto claim. *(§7.1 / A2 / §18 / experiments/weight_sweep.py)*

> **MPC runtime (Increment-2 follow-up, 2026-06-23):** the loop now honors the configured `m_MPC` cadence (B21 = re-plan every 2 slots, hold velocity between), per spec §11 — this dropped per-slot time 297→180 ms (0.61×) at no fidelity cost. Further MPC rollout vectorization is deferred (not needed at 180 ms/slot for current diagnostics; revisit if the §18.4 full runs demand it).

37. **DERIVATION — 2026-06-23 (V rescaled 1.0 → 6.9×10¹⁶ on unit-commensurability grounds; frozen before any metric):** to make the objective penalty commensurate with the Lyapunov drift in the in-loop reward `−(c_Q·Q^c·a + V·penalty)`, `V` is set so the two terms are comparable **at a stated reference operating point**. **Reference point:** real config load **λ=0.2**; "typical steady-state backlog" = per-UAV compute backlog **conditional on the compute tier being non-empty**, `Q^c_active ≈ 3.9×10⁸ cycles` (the all-slots typical is 0 — the queue is empty **98.9%** of the time at λ=0.2 — so balancing uses the active-backlog level, the only regime where drift is nonzero and the trade-off is live). **Computation:** `drift_ref = c_Q·Q^c_active·a_ref ≈ 1.38×10¹³`; `penalty_ref ≈ 2.0×10⁻⁴` (measured load-independent); **`V = drift_ref/penalty_ref ≈ 6.9×10¹⁶`**. Robust to reference load (V spans 6.9e16→4.4e17 over λ=0.2→0.6, same order). **Derived from system scales (backlog, incoming cycles, penalty), NOT from any completion/performance metric (§18); frozen.** *Caveat:* at λ=0.2 drift=0 for 98.9% of decisions, where reward=`−V·penalty` is uniform in V → V cannot change the light-load policy; the rescale targets the moderate-to-high-load regime where drift>0. Post-rescale diagnostic: see entry 38. *(§10.1 / §14.2 / A2 / §18)*
38. **DIAGNOSTIC RESULT — 2026-06-23 (post-rescale sweep is NON-FLAT → A2 multiobjective evidence now exists):** re-running the exact entry-36 weight-sweep with the rescaled **V=6.9×10¹⁶** (frozen): cross-weight policy divergence rose from **0.0% → 23.5%** at real load (λ=0.2) and **0.0% → 68.8%** at high load (λ=0.6) — both **OBJECTIVE-SENSITIVE** (different weight vectors now learn different policies; all still learned, 12–85% off the untrained init). The success criterion (entry 37 / §7.1: sensitivity in the regime where objectives conflict, moderate-to-high load) is **met** — and sensitivity even appeared at light load (the rare congestion events now carry enough objective signal). **The A2 multiobjective / weight-sweep Pareto evidence is now available** (was empty at V=1.0). At very high backlog the drift still dominates V·penalty (stability correctly takes over under stress) — the intended drift-plus-penalty behavior. *(A2 / §7.1 / §18 / experiments/weight_sweep.py)*
39. **FINDING — 2026-06-23 (consolidated: the frozen real config is a LIGHTLY-LOADED regime):** four independent measurements agree that under the frozen parameters the system is substantially **underloaded** relative to a "high-demand UAV-MEC" framing: (i) compute queue **empty ~98.9%** of the time at λ=0.2 (this turn); (ii) compute demand only **~1/3 of capacity** (entry 32: ~5500 vs ~17500 Mcyc/slot); (iii) radio **SNR saturated cell-wide** (51–64 dB, entry 24) so links are never the bottleneck; (iv) **harvesting negligible** vs consumption (~0.24%, energy finding). **Manuscript implication:** the paper's actual parameters describe a **less-stressed regime** than the high-demand framing implies; the revision should report the true operating load (demand ≈ ⅓ capacity, queues mostly empty) honestly — the stability/optimization machinery is mostly slack at these parameters. *(Reporting only — config NOT changed; §18.)* *(§14 / entries 24, 32, 33, 37)*
40. **INCREMENT 3 — 2026-06-23 (per-task APSO `f_ijk` allocation):** replaced the aggregate full-capacity compute service with a **per-task** compute backlog (`compute_items[j]={tid:cycles}`) and a real APSO allocation of `C_j` across each UAV's tasks (consuming the 'apso' projection; D1). No-contention → all served (full CPU); contention → APSO splits scarce capacity, minimizing the projection ⇒ `f ∝ √remaining` (verified: 2e9 vs 5e8 task → 2:1 capacity split). Capacity constraint `Σf ≤ C_j·Δt` and exact end-to-end conservation hold (hand-off identity + per-task drain). Runtime: APSO now does real per-task work under contention (timing reported with the increment). *(§9 / simulation.py / tests/test_apso_alloc.py)*
41. **PRE-STATED EXPECTATIONS — 2026-06-23 (directional-coherence check, BEFORE running; CHECK 1):** boosting each objective weight one-at-a-time (others fixed) at moderate-to-high load should move *its own* outcome in the expected direction, proving the non-flat sweep (entry 38) is **coherent** multiobjective response, not thrashing: `w1_task↑ → mean task latency ↓`; `w2_energy↑ → total energy consumed ↓`; `w3_completion↑ → on-time completion rate ↑`; `w5_util↑ → mean compute utilization ↑`; `w6_coverage↑ → mean coverage ↑`. *Caveat (pre-noted):* total **compute** energy ≈ `e_c × total executed work` is largely policy-invariant (no task dropping in the model); `w2` can lower energy mainly via **flight** (MPC moving UAVs less), so a weak/flat `w2→energy` would be a model property, NOT incoherence. **Incoherent = an outcome moving the WRONG way** → STOP & report (entry 42). Result logged in entry 42. *(CHECK 1 / A2 / §18)*
42. **DIAGNOSTIC RESULT — 2026-06-23 (CHECK 1 directional sweep: 4/5 coherent, no thrashing; 1 objective un-exercised — STOPPED per protocol):** boosting each weight 10× at λ=0.6 (others fixed), own-outcome vs the all-1 baseline. **Coherent (right direction):** `w1_task↑ → mean latency 4.199→4.162 (DOWN ✓)`; `w3_completion↑ → completion 0.674→0.923 (UP ✓)`; `w5_util↑ → util 0.714→0.860 (UP ✓)`; `w2_energy↑ → energy-per-completed-task 2144→2014 J (DOWN ✓)`. The w2 case **looked wrong on *total* energy (2.79e6→3.58e6, UP)** but that was the pre-noted metric conflation: w2↑ also raised completion (0.673→0.969, far more work done), so total energy rose while **per-task efficiency improved** — correct once measured as J/completed-task. **No outcome moved the WRONG way → no thrashing; the multiobjective response is directionally coherent.** **One objective un-exercised:** `w6_coverage↑ → coverage 88.39→88.39 (FLAT)` because MPC's `_mpc_evaluate_positions` holds task=energy=0 (entry-24 strong-SNR simplification), so MPC is *already* 100% coverage-driven and `w6` is a uniform-scale no-op (coverage saturated, not traded off). **Structural, not incoherent. Per §18: NOT tuned.** Decision pending (author): give MPC non-trivial task/energy terms so coverage genuinely trades off (would make `w6` directional). The offload/allocation multiobjectivity (w1/w2/w3/w5) is **substantiated**. *(CHECK 1 / A2 / §8.2 / §18)*
43. **DECISION + FINDING — 2026-06-23 (accept coverage-driven trajectory; A2 rests on the four offload/allocation objectives):** author accepted option (a): do **NOT** enrich MPC. The **A2 multiobjective claim is substantiated for the offload/allocation objectives** — `w1` latency, `w2` energy-efficiency, `w3` completion, `w5` utilization — all directionally coherent (entry 42). The **trajectory is coverage-driven** because saturated cell-wide SNR (entry 24) leaves UAV position with essentially **nothing to trade against** in `J_task`/`J_energy`; `w6` is therefore un-exercised. This is a **finding consistent with the load/SNR picture (entries 24/32/39), not a patched deficiency.** **Manuscript implication:** describe trajectory optimization as **coverage-driven** in this regime, and rest the *multiobjective* framing on the four offload/allocation objectives. *(A2 / §8 / entry 24 / §18)*

---

## 16. Consolidated sign-off checklist

**✅ SIGNED OFF 2026-06-23** (decisions recorded in `signoff_worksheet.md` and §15 entries 16–22). Two items changed from proposal (A1 two-tier queue, A4 drop migration); all others approved as proposed. Boxes below reflect that sign-off.

**Modeling choices (change behavior):**
- [x] §4.5 compute-energy model: **linear** (Table IV) — approved (B6)
- [x] §4.6 energy update: (10) form adopted — approved
- [x] §5.2 **coverage rewrite** to smooth Gaussian `−D(p)` — approved; eq (20) flagged for **erratum** (A3/§15-18)
- [x] §5.3 normalization scheme + reference scales `S_m` — approved; **migration scale m=4 removed** (A4)
- [x] §5.4 default weights `w_m = 1.0` + sweep — approved (now **5 active terms**)
- [x] §6 MORL — **migration DROPPED** (A4): MORL = offload-only, projection `{1,2,3,5}`; override Table-IV "3 objectives"; **scalarized** + weight-sweep Pareto evidence (A2)
- [x] §4.7 / §10 **TWO-TIER queue** (radio + compute) with hand-off — **CHANGED** per A1
- [x] §7 MORL architecture — **DQN function approximator** (PyTorch), resolved 2026-06-23 (entry 29); success criteria pre-registered (§7.1, entry 30)
- [x] §8.4 skip the LQR-QP form (39–41); use direct MPC (§8.2) — approved (B7)
- [x] §10 Lyapunov: empirical-stability stance; `V` default + sweep — approved; now two-tier
- [x] §4.2 / §4.3 **2-D** motion (fixed `H`) and homogeneous `λ_i` — approved (A5)
- [x] §9 APSO adaptation rule (inertia decay) — approved (B18)

**Proposed parameter values (§14.2):** β0, α_path, N₀, B, H, R_min, σ_cov, ω_ij, E_min, V, **c_Q** (new, two-tier scaling), N_h, n_P, w_in, c1, c2, n_iter, m_MORL/m_MPC/m_APSO, n_ep, UAV/device init positions, E_j(0), `Q^r_i(0)`/`Q^c_j(0)`, seed — **all approved**. ~~δ_l, ε_l~~ **removed** (migration dropped, B8/B9). Seed **frozen permanent** (B27/§18).

> (Resolved-item note retained for history.) One item remains genuinely open: §7 tabular-vs-approximator is an implementation-time decision, not a value. Per the author's order of operations, approved values are **not yet folded into `config/default.yaml`** and **no code is written** — both are held until the author reviews the rewritten §4.7 and §10.

---

## 17. Section → superseded-equation map

| Spec § | Supersedes (paper eq) |
|---|---|
| §4.1 network | III-A |
| §4.2 motion | (1),(2),(3) |
| §4.3 tasks | (4),(5) |
| §4.4 channel | (6),(7) |
| §4.5 computation | (8),(9) |
| §4.6 energy | (10),(12) |
| §4.7 queue (two-tier) | (new; underlies 32–34) |
| §5 objective | (14),(15),(16),(17),(19),(20),(63) — **(18) migration dropped** |
| §5.5 constraints | (21)–(25) — **(26) removed** |
| §6 projections | (27 reward),(28),(29) scopes |
| §7 MORL | (27), Alg. 2 |
| §8 MPC | (1),(2),(28),(39),(40),(41),(62) |
| §9 APSO | (29),(30),(31), Alg. 4 |
| §10 Lyapunov | (32),(33),(34), Alg. 5 |
| §11 orchestration | Alg. 1, (53) |
| §12 excluded | (55),(59),(60),(67),(69) |
| §13 baselines | (64) |

## 18. Reproduction Criteria and Anti-Tuning Protocol

**Binding rules for this repository.** This section governs how reproduction is
judged and how parameters/choices may be set. It is not advisory: a result that
violates the Anti-Tuning Protocol is invalid regardless of how well it matches
the paper. the project rules "Verification stance" points here as the controlling
protocol for every session.

### 18.1 Goal
Reproduce the paper's headline numbers — the **38% UAV route reduction**, the
**55% throughput increase**, the **94.5% task completion rate**, and the other
**Table V–X** metrics — using the *single consistent system* defined by this
spec (not the paper's internally-inconsistent system).

### 18.2 Success criteria (tiered)
The bar for success is **Tier 2**, not exact replication.

| Tier | Meaning | Status |
|---|---|---|
| **Tier 1 — exact match** | Reproduced value lands on the paper's digits (within tight tolerance). | **Bonus, not the bar.** Welcome if it happens; never required, never engineered. |
| **Tier 2 — directional + order-of-magnitude** | Same direction and correct magnitude band: a *substantial* route reduction in a broad band around 38%, a *large* throughput gain, a *high* completion rate, etc. | **THE SUCCESS BAR.** Tier 2 validates the paper's **claims** even when exact digits differ. |
| **Tier 3 — divergent** | The consistent system yields dramatically different results. | **Reported honestly as a FINDING** — i.e. the paper's original numbers came from the inconsistent system. Never hidden, never tuned away. |

- "Broad band" is deliberate: Tier 2 asks whether the *claim* holds (route gets
  much shorter, throughput rises a lot, completion is high), not whether a
  specific percentage recurs.
- A Tier-3 outcome is a legitimate, publishable result about the original work,
  not a failure of this repo. It must be stated plainly with its likely cause.

### 18.3 Anti-Tuning Protocol (core integrity rule)
1. **Freeze-before-compare.** Every unspecified parameter — the ~25 PROPOSED
   values in §16 (β0, α_path, N₀, B, H, σ_cov, ω_ij, δ_l, ε_l, E_min, V, N_h,
   n_P, w_in, c1, c2, n_iter, timescales m_*, n_ep, init conditions, seed, …) —
   MUST be frozen to its documented value and **committed to git BEFORE any
   comparison run** against the paper's numbers. The committing diff is the
   record that values predate the comparison.
2. **Physical/literature grounds only.** A parameter value is chosen on
   physical, mathematical, or cited-literature grounds. A value may **NEVER** be
   changed because the change moves a result closer to the paper's number.
3. **Post-hoc changes are gated.** Any change to a frozen parameter after a
   comparison run requires BOTH: (a) a stated physical/mathematical reason, and
   (b) a logged entry in the §15 changelog (date, old→new, reason).
   **"To match 38%" (or any target) is never a valid reason and must be
   refused** — by the author and by any agent acting in this repo.
4. **Modeling choices follow the same rule.** The four contradiction flags
   (§4.5 compute energy, §4.6 energy update, §5.2 coverage rewrite, §8.4
   MPC form) and the ~11 behavioral choices in §16 are decided on
   **correctness alone**, never on whether they hit a target metric.
5. **Parameter influence is explored ONLY via a pre-registered sweep.** Studying
   how a parameter affects a result is allowed only by varying that one
   parameter across a **documented range** and reporting the **full curve**
   (the §5.4 sensitivity sweep is the template). Selecting the point on that
   curve that best matches the paper and adopting it as "the" value is tuning
   and is forbidden.

> Operational test for any value change, ask: *"Would I make this exact change
> if the paper had reported a different number?"* If no, the change is tuning
> and must be refused.

### 18.4 Required validation artifact
Every experiment that maps to a paper claim MUST emit a **side-by-side
comparison table** — the repo's primary validation output — with exactly these
columns:

| metric | paper's claimed value | reproduced value | absolute gap | relative gap | tier achieved |
|---|---|---|---|---|---|

- One row per mapped claim (Table V–X metrics, plus the headline 38% / 55% /
  94.5%).
- `relative gap = (reproduced − claimed) / claimed`.
- `tier achieved ∈ {1, 2, 3}` per §18.2.
- This table MUST be **regenerable by the one-command runner** (the runner in
  README's "Reproducing the results"), so anyone can reproduce the comparison,
  not just the figures.
- The reproduced column is reported **as computed** — never edited toward the
  claimed column.

### 18.5 What this section supersedes / relates to
- Operationalizes corrected_spec.md "Verification stance" (honest reporting, no tuning).
- Uses the §16 sign-off list as the freeze manifest and the §15 changelog as the
  post-hoc-change log.
- Tier-3 honesty is the same stance taken for the Lyapunov proof (§10.2): claim
  only what the consistent system actually demonstrates.

---

*End of spec. Awaiting sign-off on §16 before any implementation proceeds.*
