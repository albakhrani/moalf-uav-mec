# §16 Sign-Off Worksheet — MOALF-UAV-MEC

**Purpose:** clear the [§16 sign-off gate](corrected_spec.md#16-consolidated-sign-off-checklist) in `corrected_spec.md`. Per the project's working rules, **no implementation code is written until every item below carries an `AUTHOR DECISION:`**. This worksheet does not change the spec; it mirrors §16 so you can decide each item in one place. When you've filled the decision lines, I'll fold approved values into `config/default.yaml`, log any changes in the §15 changelog, and only then begin coding (system_model first).

**How to use:** fill the `AUTHOR DECISION:` line under each item. "OK" / "approve" accepts the proposed value. Anything else is treated as a change and gets a §15 changelog entry. Per §18.3, a value may never be set to hit a target number.

---

## ⚠️ Read first — note on §5.3 normalization (affects how you read everything below)

The objective normalizes each term `J̃_m = J_m / S_m` by a **fixed reference scale** `S_m` (§5.3). Those scales **bake in assumptions**, and the biggest is that **energy is flight-dominated** (`S_energy = M·P_flight·Δt·T_h`, ignoring compute energy). If, in practice, compute energy or migration cost turns out non-negligible, the chosen `S_m` will **distort which objective term dominates** the weighted sum — e.g. an under-scaled term silently outweighs the others even though all `w_m = 1.0`.

**Implication for sign-off:** the `w_m = 1.0` "equal weighting" is only as equal as the `S_m` make it. If results look odd (one objective seems ignored or pathologically dominant), the **first thing to revisit is §5.3**, before touching weights or parameters. This is flagged here so a strange result later isn't mistaken for a parameter problem when it's a normalization problem. (The normalization scheme itself is a Tier-B item, §B.13.)

---

# TIER A — DECIDE CAREFULLY

*These change what the system **is**, or imply a manuscript-level correction. They are not just defaults. Decide each deliberately.*

---

### A1. Queue semantics — per-device, transmission-limited (§4.7)

- **Spec section:** §4.7 (queue dynamics), underpinning §10 (Lyapunov).
- **Proposed:** `Q_i(t+1) = max{Q_i(t) − S_i(t), 0} + A_i(t)`, one queue **per IoT device i**, measured in **bits**, where service `S_i(t) = Σ_j x_ij(t)·R_ij(t)·Δt` is **transmission-limited** (the offload uplink rate), and arrivals `A_i(t) = Σ_k L_{i,k}` for tasks arriving that slot.
- **Justification:** the paper never wrote a queue update at all; this is the minimal, standard Lyapunov backlog. Per-device + transmission-limited is the simplest form that makes the drift-plus-penalty controller (§10.1) well-defined.
- **What stability claim this supports vs. does NOT cover** (confirm this matches your paper's stability story):
  - ✅ **Supported:** stability of the **per-device offload-transmission backlog** — i.e. each device's uplink queue of bits waiting to be sent to a UAV does not grow unbounded. The "queues bounded" claim is about the **radio/offload path**.
  - ❌ **NOT covered:** **UAV-side compute-queue congestion** — tasks piling up *at the UAV* waiting for CPU cycles (the `W/C_j` execution backlog). Also not covered: edge-node (migration target) queues, and any end-to-end (transmit+compute) queue. If your manuscript's stability narrative is about **compute congestion at the UAVs**, this queue model is the wrong abstraction and must change before coding.
- **Consequence if changed:**
  - *Switch to UAV-side compute queues* (`Q_j`, service `= C_j·Δt/W̄`): the Lyapunov controller would prioritize draining busy UAVs instead of busy radios; changes which decisions the §10.1 bias nudges, and changes what "stable" means in every results table. Larger refactor of §4.7 + §10.
  - *Add a second queue tier (radio **and** compute)*: most faithful to a real MEC pipeline, but doubles the queue state and complicates the drift expression; the §10.2 derivation TODO gets harder.
  - *Switch bits→tasks as the unit*: simpler integers, but loses the rate-coupling that ties trajectory (better channel) to faster draining.
- **AUTHOR DECISION:** **CHANGE.** Reject the single per-device transmission-only queue. Adopt a **TWO-TIER** model: (1) per-device **radio/transmission** backlog `Q^r_i` (bits waiting to offload) and (2) per-UAV **compute** backlog `Q^c_j` (CPU work waiting for execution at the UAV). Stability = **BOTH** bounded. Explicit hand-off: bits leaving a device's radio queue ARRIVE at the serving UAV's compute queue. §4.7 and §10 rewritten to this model; the §10.2 derivation TODO now spans both tiers with cross-terms. Logged in §15.

---

### A2. Scalarized MORL vs. the paper's "multiobjective" framing (§6)

- **Spec section:** §6 (optimizer projections), §7 (MORL).
- **Proposed:** because weights are fixed (D1), the MORL reward collapses to a **single scalar** `R_MORL = −Σ_{m∈{1,2,3,4,5}} w_m·J̃_m`. "Multiobjective" survives only as the *structure* of the projection, not as a Pareto solver.
- **Justification:** fixed weights + weighted-sum scalarization is the standard, reproducible reading of "multiobjective RL with fixed preferences." A true Pareto front is unnecessary if the preference vector is fixed.
- **Tension to resolve (this is a manuscript question, not just code):** the paper repeatedly bills the method as **"multiobjective reinforcement learning."** A pure scalarization is, strictly, **single-objective RL on a weighted sum** — a reviewer could call the "multiobjective" label inaccurate. Your options (I am **not** choosing for you):
  - **(i) Soften the manuscript framing:** describe MORL as "fixed-preference scalarized multiobjective RL" / "weighted-sum scalarization." Minimal code, honest, but concedes the method isn't Pareto-multiobjective.
  - **(ii) Implement a genuine multiobjective version:** keep a **vector-valued** reward and either (a) learn a Pareto set / Pareto-front approximation, or (b) run a **weight-sweep** producing multiple policies. Preserves the "multiobjective" claim literally, but is materially more code and changes §7, and the front then needs presenting in results.
  - **(iii) Hybrid:** ship scalarized as the primary system (reproducible), and add the weight-sweep (already planned in §5.4) as the "multiobjective" evidence — the sweep *is* the Pareto exploration.
- **Consequence if changed:** (i) is near-zero code but a manuscript edit; (ii) is the largest single scope increase in the whole spec; (iii) adds the sweep harness but no new solver. Whichever you pick determines whether §7 stays as-is.
- **AUTHOR DECISION:** **APPROVE option (iii) — hybrid.** Ship scalarized MORL as the primary system; the §5.4 weight-sweep provides the multiobjective/Pareto evidence. Manuscript to describe MORL as **fixed-preference scalarization with a weight-sweep Pareto exploration** (recorded in §15).

---

### A3. Coverage rewrite — eq (20) is degenerate (errata-level) (§5.2)

- **Spec section:** §5.2 (J_coverage), feeding §8 (MPC).
- **Proposed:** replace the paper's coverage term `J_coverage = −Σ_i min{max{𝟙(R_ij ≥ R_min),1},1}` with the **smooth Gaussian** `J_coverage = −Σ_i Σ_j ω_ij·exp(−‖p_j−q_i‖²/(2σ_cov²))` (from eq 63).
- **Justification:** the paper's eq (20) is **mathematically degenerate** — `𝟙(·) ∈ {0,1}`, so `max{𝟙,1} ≡ 1` and `min{1,1} ≡ 1`; the term is the **constant `−N`** regardless of UAV positions. It contributes **zero gradient** and cannot drive trajectory. The Gaussian is the differentiable surrogate that makes coverage actually optimizable (and is where eq 62's gradient-ascent is absorbed).
- **This is a manuscript-level correction, not just a code choice:** eq (20) as printed is an **error (errata)**. Whatever the paper *reported* for coverage was produced by some other, unstated computation — not by eq (20) as written. You will likely need to (a) issue a correction/erratum for eq (20), and (b) state the actual coverage metric used. The repo cannot reproduce (20) verbatim because (20) computes a constant.
- **Consequence if changed:**
  - *Keep eq (20) verbatim:* coverage term is constant → MPC ignores coverage entirely → UAVs optimize only latency+energy → trajectory/coverage results (Table VII, 98.2% coverage) cannot be reproduced at all. Not viable.
  - *Use a different coverage metric* (e.g. rate-coverage count `Σ_i 𝟙(max_j R_ij ≥ R_min)`): also valid and closer to the paper's *intent*, but **non-differentiable** → forces sampling-based MPC, not gradient-based. Decide jointly with A5 and the MPC method (§8.4).
  - `σ_cov`, `ω_ij` values are Tier B (§B.10–B.11), but the **rewrite itself** is Tier A.
- **AUTHOR DECISION:** **APPROVE the Gaussian coverage rewrite.** Paper eq (20) is degenerate (computes the constant `−N`); §15 flags it as requiring an **ERRATUM** in the manuscript that records the actual coverage metric used.

---

### A4. MORL owns migration — or drop migration entirely (§6)

- **Spec section:** §6 (projections), §7 (MORL action space), term `J_migration` (§5.2 m=4).
- **Proposed:** MORL owns both offload `x_ijk` **and** migration `y_ijkl`, and its projection **includes** `J_migration` (the paper's literal 4-term reward omitted it, leaving migration ownerless).
- **Justification:** a migrate decision must be made *somewhere*; MORL is the only discrete-decision component, so it is the natural owner, and an owner should see its own cost.
- **Alternative — drop migration entirely (consider seriously):** in the paper, migration (`y_ijkl`, eq 18, `J_migration`) is **thinly specified and may be vestigial** — `δ_l`, `ε_l` are undefined, no migration result is singled out, and the migration targets `l ∈ 𝒢` barely appear. If migration carries no weight in your intended story, the cleanest choice is to **remove `y`, `J_migration`, and constraint (26) from the implemented model** and note migration as out-of-scope. This deletes two unspecified parameters (`δ_l`, `ε_l`) and one objective term, simplifying everything.
- **Consequence if changed:**
  - *Drop migration:* objective becomes 5-term; MORL action space shrinks to offload only; `δ_l`, `ε_l`, `S_migration` normalization (§5.3 m=4) all vanish; constraint (26) removed. Simpler and more reproducible, at the cost of not modeling a paper feature.
  - *Keep but move ownership elsewhere* (e.g. a rule-based migrator): possible, but adds a component the paper doesn't describe.
  - *Keep in MORL (proposed):* most faithful to eq 14's 6 terms; requires `δ_l`, `ε_l` values (Tier B §B.8–B.9).
- **AUTHOR DECISION:** **DROP migration entirely.** Remove `y_ijkl`, `J_migration` (objective → **5 active terms**), constraint (26), and the need for `δ_l`/`ε_l`. §5 (incl. normalization §5.3 losing scale m=4), §6, §7 updated; B8/B9 removed. Logged in §15.

---

### A5. 2-D vs 3-D UAV motion (§4.2)

- **Spec section:** §4.2 (motion), coupled to §4.4 (channel `H` term) and §8 (MPC trajectory).
- **Proposed:** effectively **2-D** motion — UAVs move in the horizontal plane at a **fixed altitude `z_j = H`**; the vertical term enters only as the constant `H` inside the channel gain (eq 6).
- **Justification:** the paper's simulation area is described as **"400 m × 400 m"** (2-D), Table IV gives no altitude range, and fixing altitude is standard for UAV-BS coverage studies.
- **Consequence if changed (effect on channel `H` and trajectory):**
  - *Fixed-altitude 2-D (proposed):* `H` is a **constant** in `h_ij = β0·(‖p_j−q_i‖²+H²)^(−α/2)·ξ`. The horizontal distance alone varies the channel; trajectory optimization is a 2-D problem; MPC control is `(v_x, v_y)`. Smaller search space, faster, matches "400×400."
  - *Full 3-D:* `z_j` becomes a **decision variable**; `H` is no longer constant — UAVs could **descend to improve channel** (lower `H` → higher gain) and trade that against flight energy to climb. This adds a real and interesting coupling (altitude vs. link quality vs. energy), enlarges the MPC state/control by one dimension, and needs an **altitude range + vertical flight-energy model** that the paper does not give. More faithful to "3-D space `A ⊂ ℝ³`" wording, but more unspecified parameters and more compute.
- **AUTHOR DECISION:** **APPROVE 2-D motion** at fixed altitude `H` (`H` enters only as a constant in the channel term).

---

# TIER B — CLEAR QUICKLY

*Defensible defaults chosen on physical/literature/paper-stated grounds. No single right answer; I expect these approved as-is. Each: proposed value · source · short note.*

| # | Item (spec §) | Proposed value | Source / basis | Note | AUTHOR DECISION |
|---|---|---|---|---|---|
| B1 | `β0` reference channel gain @1 m (§4.4) | −30 dB (1×10⁻³) | typical sub-6 GHz free-space ref. gain at 1 m | sets absolute SNR scale; only relative effect matters given fixed `P_tx` | **APPROVE** |
| B2 | `α_path` path-loss exponent (§4.4) | 2.0 | LoS-dominant air-to-ground, consistent w/ Rician K=15 dB | higher α → faster gain decay w/ distance → coverage shrinks | **APPROVE** |
| B3 | `N₀` noise PSD (§4.4) | −174 dBm/Hz | standard thermal noise floor | with B sets noise power `N₀·B` | **APPROVE** |
| B4 | `B` bandwidth (§4.4) | 1 MHz | Table IV's "1000 Mb/s" is aggregate capacity, **not** B; 1 MHz gives plausible per-link rates | drives absolute throughput numbers; revisit if rates look off vs. paper | **APPROVE** |
| B5 | `H` altitude (§4.4/§4.2) | 100 m | common UAV-BS altitude for a 400 m cell | constant if A5 = 2-D; couples to A5 | **APPROVE** (A5 = 2-D ⇒ `H` constant) |
| B6 | compute-energy model (§4.5) | **linear** `E_exec = e_c·W`, `e_c=1e-9 Wh/cycle` | only numeric energy figure in paper (Table IV); eq 9 quadratic has no coefficient | resolves the (9)-vs-Table-IV contradiction toward the stated number | **APPROVE** |
| B7 | skip LQR-QP form (§8.4) | **skip** eqs (39)–(41); use direct MPC (§8.2) | `A,B,Q_lqr,R_lqr,P_lqr,𝒳,𝒰` never specified in paper | unimplementable as written; direct form is equivalent in intent | **APPROVE** |
| B8 | `δ_l` migration delay (§5.2) | 0.05 s | ≈ SDN ctrl delay (10 ms) + transfer | **moot if A4 drops migration** | **REMOVED** — A4 drops migration |
| B9 | `ε_l` migration fixed cost (§5.2) | 0.5 J | small fixed per-migration overhead | **moot if A4 drops migration** | **REMOVED** — A4 drops migration |
| B10 | `σ_cov` coverage radius (§5.2) | 100 m | ≈ ¼ cell width; sets Gaussian footprint | larger σ → flatter coverage gradient → UAVs spread less | **APPROVE** |
| B11 | `ω_ij` demand weight (§5.2) | 1.0 ∀i,j | uniform demand absent other info | non-uniform demand would bias UAVs toward hotspots | **APPROVE** |
| B12 | `E_min` energy reserve (§4.6) | 100 Wh (10% of E_max) | typical safe-return reserve | binds constraint (23); too high starves operation | **APPROVE** |
| B13 | normalization scheme + `S_m` (§5.3) | fixed reference scales per §5.3 table | deterministic & reproducible (vs. running normalizer) | **see top-of-file note**: bakes in flight-dominated energy; revisit if a term dominates oddly | **APPROVE** (migration scale m=4 removed per A4) |
| B14 | default weights `w_m` (§5.4) | 1.0 ∀m (on normalized terms) | D1 baseline; equal preference | meaning depends on B13; sweep planned | **APPROVE** (now 5 active terms) |
| B15 | `V` Lyapunov weight (§10.1) | 1.0 | neutral start; to be swept | small V → favor stability; large V → favor objective | **APPROVE** |
| B16 | `N_h` MPC horizon (§8.3) | 10 slots | foresight vs. O(M·N_h³) cost | longer horizon → better trajectory, cubic cost growth | **APPROVE** |
| B17 | `n_P` APSO swarm size (§9) | 30 | standard PSO swarm | larger → better allocation, linear cost | **APPROVE** |
| B18 | `w_in` APSO inertia (§9) | 0.9 → 0.4 decay | standard adaptive inertia (this is the "adaptive" rule) | the undefined "adaptive PSO" rule, made concrete | **APPROVE** |
| B19 | `c1, c2` APSO coefficients (§9) | 1.49, 1.49 | Clerc constriction standard | balanced cognitive/social pull | **APPROVE** |
| B20 | `n_iter` APSO iterations (§9) | 100 | standard convergence budget | per-slot APSO cost ∝ n_P·n_iter | **APPROVE** |
| B21 | timescale cadences `m_MORL,m_MPC,m_APSO` (§11) | 10, 2, 1 slots | eq (53) ordering `k1>k2>k3`; MORL slowest | all =1 → everything runs every slot (simpler) | **APPROVE** |
| B22 | `n_ep` MORL episodes (§7) | 1 run = 1 episode of `T_h` slots | simplest mapping | revisit if episodic resets/curriculum wanted | **APPROVE** |
| B23 | UAV init positions (§4.2) | uniform-random in area, `z=H` | no layout given; reproducible via seed | alternative: fixed grid for determinism | **APPROVE** |
| B24 | UAV init energy `E_j(0)` (§4.6) | `E_max` (full charge) | standard "start charged" assumption | starting low would stress harvesting early | **APPROVE** |
| B25 | queue init `Q_i(0)` (§4.7) | 0 | empty system at t=0 | standard cold start | **APPROVE** (both tiers: `Q^r_i(0)=Q^c_j(0)=0`) |
| B26 | device positions `q_i` (§4.1) | uniform-random in area, fixed per run | no layout given | re-drawn per seed across the 30 runs | **APPROVE** |
| B27 | RNG seed (§14.2) | 42 (runs 0–29 → seeds 42…71) | reproducibility | the 30-run average (paper-stated) spans these seeds | **APPROVE — FROZEN PERMANENT** per §18 anti-tuning protocol |
| B28 | MB unit convention (§2.1) | decimal, 1 MB = 8×10⁶ bits | SI/decimal default | MiB (8×2²⁰) ≈ 5% larger tasks if you prefer | **APPROVE** |

---

## Items that are paper-stated (no decision needed — listed for completeness)

These are **not** sign-off items; they come straight from Table IV / paper §VI and are already fixed: area 400×400 m, `T_h`=1000, `Δt`=1 s, `N`=50, `M`=5, `N_G`=3, slices=3, `C_j`~U(2,5) GHz, `C_l`~U(10,20) GHz, `E_max`=1000 Wh, `v_max`=10 m/s, `P_flight`=100 W, `η_j`=5 W, burst 20 slots/2×, task gen ~U(0.1,0.3)/s, `L`~U(0.1,1) MB, `W`~U(100,1000) Mcyc, `τ`~U(5,20) s, `ρ`~U{1,5}, `P_tx`=0.1 W, Rician K=15 dB, MORL `α_lr`=0.001/`γ`=0.99/`ε`=1.0/decay=0.995/replay=10000/batch=32, failure prob 0.05, no-fly 0.1 / U(10,50) m, `θ_rel`=0.95, SDN 10 ms/1000/100 ms, **runs=30**.

---

## Sign-off summary (fill when done)

- Tier A decided: ☑ A1 (CHANGE→two-tier) ☑ A2 (hybrid iii) ☑ A3 (approve) ☑ A4 (drop migration) ☑ A5 (approve 2-D)
- Tier B decided: ☑ B1–B28 approved as proposed, EXCEPT **B8, B9 REMOVED** (migration dropped) and **B27 FROZEN PERMANENT** (anti-tuning).
- Normalization note acknowledged (§5.3 / top of file): ☑
- **Cross-coupling resolved:** A4 ⇒ B8/B9 removed; A5 = 2-D ⇒ B5 `H` constant; A3 = Gaussian coverage ⇒ B7 direct-MPC + B10/B11 (`σ_cov`, `ω_ij`) stand.

**Signed off:** 2026-06-23. Two changes from proposal (A1 two-tier queue, A4 drop migration); all else approved as proposed.

> Status after this turn (per author's order of operations): worksheet filled; spec §4.7 and §10 rewritten to the two-tier model; A4 migration-removal applied to §5–§7; §15 changelog entries added. **NOT done yet (held for review):** folding values into `config/default.yaml` and any implementation. Nothing is coded before the author reviews the rewritten §4.7 and §10.
