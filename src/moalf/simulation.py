"""simulation — Algorithm-1 per-slot loop (spec §11). INTEGRATION SKELETON.

Wires the four components in the spec §11 order, per time slot:

    1. observe state
    2. MORL  -> offload assignment x_ij   (which UAV serves each device's work)
    3. MPC   -> UAV velocities/positions  (coverage-driven trajectory)
    4. APSO  -> compute-capacity allocation
    5. Lyapunov -> two-tier drift diagnostics + adjust hook
    6. step environment: arrivals -> radio queue -> per-task hand-off (bits->cycles)
       -> compute queue -> compute service; energy update (eq 10); coverage metric
    7. (learning deferred — see notes)

SKELETON SCOPE (reviewable structure before any paper comparison):
  - MORL and MPC are fully functional (assignment + movement).
  - APSO is called in order but on a simplified per-UAV capacity-fraction problem;
    the full per-task f_ijk allocation is the next increment.
  - Lyapunov computes real two-tier drift; its decision-biasing adjust is a no-op
    hook for now (spec §10.1 minimization is the next increment).
  - MORL learning (spec §11 step 7) is deferred; the skeleton exercises forward
    decisions only.
  - No §18.4 paper-comparison table is produced here (by request).

The TWO-TIER hand-off (spec §4.7) is implemented exactly and conserved:
bits leaving a device's radio queue become compute cycles at the serving UAV via
each task's own intensity r_{i,k}=W/L; a fully transmitted task contributes
exactly W cycles (the r·L=W identity). The accompanying tests verify this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np

from moalf.objective import Objective, Term
from moalf.optimizers.apso import APSO
from moalf.optimizers.lyapunov import LyapunovController
from moalf.optimizers.morl import MORLAgent
from moalf.optimizers.mpc import MPC
from moalf.system_model.channel import ChannelModel
from moalf.system_model.computation import ComputationModel
from moalf.system_model.coverage import gaussian_coverage
from moalf.system_model.energy import EnergyModel

MB_TO_BITS = 8.0e6
MCYCLES_TO_CYCLES = 1.0e6


@dataclass
class Task:
    """A task awaiting offload (spec §4.3). Tracks remaining bits for partial tx."""

    device: int
    bits: float            # L_{i,k}
    cycles: float          # W_{i,k}
    deadline_s: float      # τ_{i,k}
    priority: float        # ρ_{i,k}
    arrival_t: int         # t^a_{i,k}
    remaining_bits: float

    @property
    def intensity(self) -> float:
        """r_{i,k} = W/L (cycles/bit) — the bits->cycles conversion (spec §4.7)."""
        return self.cycles / self.bits

    def urgency(self, t: int) -> float:
        """U_{i,k}(t) = ρ·τ / (τ − (t − t^a)) (eq 5); ∞ once the deadline is reached."""
        remaining = self.deadline_s - (t - self.arrival_t)
        if remaining <= 0:
            return float("inf")
        return self.priority * self.deadline_s / remaining


@dataclass
class SlotRecord:
    """Per-slot diagnostics (conservation + stability + metrics)."""

    t: int
    arrivals_bits: float
    transmitted_bits: float
    handoff_cycles: float
    handoff_cycles_check: float   # Σ b_{i,k}·r_{i,k}, independent recomputation
    executed_cycles: float
    qr_before: float
    qr_after: float
    qc_before: float
    qc_after: float
    coverage: float
    total_energy_j: float
    lyapunov: dict = field(default_factory=dict)
    assignment: dict = field(default_factory=dict)
    mean_reward: float = 0.0   # set by train_step (Increment 2)


@dataclass
class SimState:
    t: int
    uav_pos: np.ndarray          # (M, 2) m
    uav_energy_j: np.ndarray     # (M,) J
    uav_capacity_cps: np.ndarray # (M,) cycles/s
    device_pos: np.ndarray       # (N, 2) m
    radio_q: List[List[Task]]    # per-device task lists (bits), Q^r_i
    compute_q_cycles: np.ndarray # (M,) cycles, Q^c_j


class Simulation:
    """Algorithm-1 integration skeleton (spec §11)."""

    def __init__(self, config: Mapping[str, Any], *, seed: Optional[int] = None,
                 offload_policy: str = "morl", lyapunov_biasing: bool = False,
                 reward_scale: Optional[float] = None):
        self.config = config
        if seed is None:
            seed = int(config["run"]["seed"])
        self.rng = np.random.default_rng(seed)
        # base offload policy ('morl' = untrained DQN greedy; 'nearest' = best channel)
        # and whether the Lyapunov drift-plus-penalty biasing (Increment 1) is ON.
        self.offload_policy = offload_policy
        self.lyapunov_biasing = lyapunov_biasing

        # deterministic system-model layer
        self.channel = ChannelModel.from_config(config)
        self.computation = ComputationModel.from_config(config)
        self.energy = EnergyModel.from_config(config)

        # the single objective + per-optimizer projections (weights live only here)
        self.objective = Objective.from_config(config)
        self.proj_mpc = self.objective.projection("mpc")
        self.proj_apso = self.objective.projection("apso")
        self.proj_morl = self.objective.projection("morl")

        # the four optimizers
        net = config["network"]
        self.M = int(net["num_uavs"])
        self.N = int(net["num_iot_devices"])
        self.dt = float(net["time_step_s"])
        self.area = np.asarray(net["area_m"], dtype=float)
        self.morl = MORLAgent.from_config(config, state_dim=3 * self.M + 2,
                                          action_dim=self.M, seed=seed)
        self.mpc = MPC.from_config(config)
        self.apso = APSO.from_config(config)
        self.lyapunov = LyapunovController.from_config(config)

        # task/channel params
        task = config["task"]
        self.lam_lo, self.lam_hi = task["generation_rate_tps"]["low"], task["generation_rate_tps"]["high"]
        self.L_lo, self.L_hi = task["input_size_mb"]["low"], task["input_size_mb"]["high"]
        self.W_lo, self.W_hi = task["compute_req_megacycles"]["low"], task["compute_req_megacycles"]["high"]
        self.d_lo, self.d_hi = task["deadline_s"]["low"], task["deadline_s"]["high"]
        self.rho_lo, self.rho_hi = task["priority"]["low"], task["priority"]["high"]
        self.lam = self.rng.uniform(self.lam_lo, self.lam_hi, self.N)  # per-device Poisson rate
        self.sigma_cov = float(config["objective"]["coverage_radius_sigma_m"])
        self.e_max_j = float(config["uav"]["energy_capacity_wh"]) * 3600.0
        self.c_lo, self.c_hi = config["uav"]["compute_capacity_ghz"]["low"], config["uav"]["compute_capacity_ghz"]["high"]

        # reward scale for in-loop MORL training: a uniform, POLICY-INVARIANT
        # constant that brings the drift-plus-penalty marginal cost to ~O(1) for
        # stable DQN gradients (it does not change the argmax/policy; numerical
        # conditioning only, not result-tuning).
        cbar_cps = 0.5 * (self.c_lo + self.c_hi) * 1e9
        drift_ref = self.lyapunov.c_Q * (self.M * cbar_cps * self.dt) * (cbar_cps * self.dt)
        self.reward_scale = float(reward_scale) if reward_scale is not None \
            else 1.0 / max(drift_ref, 1e-300)

        self.state = self._init_state(seed)

    # ---- initial conditions (spec §14) --------------------------------------
    def _init_state(self, seed) -> SimState:
        uav_pos = self.rng.uniform(0, self.area, size=(self.M, 2))         # B23
        device_pos = self.rng.uniform(0, self.area, size=(self.N, 2))      # B26
        energy = np.full(self.M, self.e_max_j)                             # B24: start full
        capacity = self.rng.uniform(self.c_lo, self.c_hi, self.M) * 1e9    # GHz -> cyc/s
        return SimState(
            t=0,
            uav_pos=uav_pos,
            uav_energy_j=energy,
            uav_capacity_cps=capacity,
            device_pos=device_pos,
            radio_q=[[] for _ in range(self.N)],                          # Q^r_i(0)=0 (B25)
            compute_q_cycles=np.zeros(self.M),                            # Q^c_j(0)=0 (B25)
        )

    # ---- helpers ------------------------------------------------------------
    def inject_task(self, device: int, bits: float, cycles: float,
                    deadline_s: float, priority: float = 1.0) -> None:
        """Insert a task directly (for tests / controlled scenarios)."""
        self.state.radio_q[device].append(
            Task(device, bits, cycles, deadline_s, priority, self.state.t, bits))

    def _offload_state(self, device: int, head: Task) -> np.ndarray:
        """Per-device offload-decision features for MORL (matches the agent's dim)."""
        s = self.state
        c = (s.uav_capacity_cps / 1e9 - self.c_lo) / (self.c_hi - self.c_lo)
        e = s.uav_energy_j / self.e_max_j
        # RELATIVE compute-load feature (load fraction across UAVs): exposes which
        # UAV is less loaded at ANY absolute backlog scale — an absolute Q^c/qmax
        # saturates to 1 under load and hides the balance signal MORL must learn.
        total_qc = float(s.compute_q_cycles.sum())
        q = (s.compute_q_cycles / total_qc) if total_qc > 0 else np.full(self.M, 1.0 / self.M)
        w = (head.cycles / MCYCLES_TO_CYCLES - self.W_lo) / (self.W_hi - self.W_lo)
        d = (head.deadline_s - self.d_lo) / (self.d_hi - self.d_lo)
        return np.concatenate([c, e, q, [w, d]]).astype(np.float32)

    def _device_intensity(self, i: int) -> float:
        """Mean work intensity r̄_i = ΣW/ΣL (cycles/bit) of device i's queued tasks."""
        q = self.state.radio_q[i]
        tot_bits = sum(tk.remaining_bits for tk in q)
        tot_cyc = sum(tk.remaining_bits * tk.intensity for tk in q)
        return tot_cyc / tot_bits if tot_bits > 0 else 0.0

    def _predicted_handoff_cycles(self, i: int, j: int) -> float:
        """Predicted cycles device i would hand to UAV j this slot (for biasing).

        Approximate (uses pre-arrival backlog and mean intensity); the ACTUAL
        hand-off is computed exactly per task in the env step, so this prediction
        affects only the *decision*, never the bit/cycle accounting.
        """
        s = self.state
        gain = self.channel.large_scale_gain(s.uav_pos[j], s.device_pos[i])
        budget_bits = float(self.channel.rate(gain)) * self.dt
        avail_bits = sum(tk.remaining_bits for tk in s.radio_q[i])
        return min(budget_bits, avail_bits) * self._device_intensity(i)

    def _assignment_penalty(self, i: int, j: int) -> float:
        """Objective penalty (via the 'morl' projection) of serving device i at UAV j.

        Weights come from the Objective (D1); the Lyapunov layer supplies only V/c_Q.
        """
        s = self.state
        a = self._predicted_handoff_cycles(i, j)          # cycles
        c_j = s.uav_capacity_cps[j]
        head = max(s.radio_q[i], key=lambda tk: tk.urgency(self.state.t))
        latency = (s.compute_q_cycles[j] + a) / c_j        # s (approx completion time)
        exec_e = a * self.computation.energy_per_cycle_j   # J
        available = s.uav_energy_j[j]
        completed = (latency <= head.deadline_s) and (available >= exec_e)
        util = min(1.0, c_j * head.deadline_s / (s.compute_q_cycles[j] + a + 1.0))
        raw = {
            Term.TASK: latency,
            Term.ENERGY: exec_e,
            Term.COMPLETION: -1.0 if completed else 0.0,
            Term.UTIL: -util,
        }
        return self.proj_morl.value(raw)

    def _mpc_evaluate_positions(self, positions: np.ndarray):
        """Raw {task, energy, coverage} for an MPC candidate position set.

        SKELETON: task/energy held constant (strong-SNR regime, entry 24 —
        verified that motion is coverage-driven there); coverage = −D(positions).
        """
        D = gaussian_coverage(positions, self.state.device_pos, self.sigma_cov)
        return {Term.TASK: 0.0, Term.ENERGY: 0.0, Term.COVERAGE: -D}

    # ---- assignment (spec §11 steps 2 & 5) ----------------------------------
    def _assign(self, devices: list, t: int) -> dict:
        """Base offload assignment (+ Lyapunov biasing if enabled). Used by step();
        train_step() instead lets the MORL agent choose with reward shaping."""
        s = self.state
        base: dict = {}
        for i in devices:
            head = max(s.radio_q[i], key=lambda tk: tk.urgency(t))
            if self.offload_policy == "nearest":
                gains = [float(self.channel.large_scale_gain(s.uav_pos[j], s.device_pos[i]))
                         for j in range(self.M)]
                base[i] = int(np.argmax(gains))
            else:  # "morl": DQN greedy (trained in-loop by train_step, Increment 2)
                base[i] = self.morl.greedy(self._offload_state(i, head))
        if self.lyapunov_biasing and devices:
            return self.lyapunov.biased_assignment(
                devices, list(range(self.M)), s.compute_q_cycles,
                incoming_fn=self._predicted_handoff_cycles,
                penalty_fn=self._assignment_penalty)
        return base

    # ---- one slot (Algorithm 1) ---------------------------------------------
    def step(self) -> SlotRecord:
        s = self.state
        t = s.t
        devices = [i for i in range(self.N) if s.radio_q[i]]   # steps 1-2
        assignment = self._assign(devices, t)                  # steps 2 & 5
        return self._advance(assignment)                       # steps 3,4,6,7

    # ---- Increment 2: in-loop MORL training step ----------------------------
    def train_step(self, epsilon: float) -> SlotRecord:
        """One slot where the MORL agent CHOOSES the offload assignment (ε-greedy)
        and LEARNS from the per-decision drift-plus-penalty reward (spec §10.1
        biasing realized as reward shaping — so MORL *learns* the stability-aware
        policy rather than being overridden). The agent then runs against the full
        coupled system via :meth:`_advance` (MPC, APSO, env, two-tier queues)."""
        s = self.state
        t = s.t
        devices = [i for i in range(self.N) if s.radio_q[i]]
        qc_proj = s.compute_q_cycles.astype(float).copy()      # running backlog (like biased_assignment)
        assignment: dict = {}
        rewards = []
        for i in devices:
            head = max(s.radio_q[i], key=lambda tk: tk.urgency(t))
            state_vec = self._offload_state(i, head)
            j = self.morl.act(state_vec, epsilon)              # ε-greedy choice
            assignment[i] = j
            a_cyc = self._predicted_handoff_cycles(i, j)
            # marginal drift-plus-penalty cost (eq 34'); reward = -cost (scaled, policy-invariant)
            cost = (self.lyapunov.c_Q * qc_proj[j] * a_cyc
                    + self.lyapunov.V * self._assignment_penalty(i, j))
            r = -cost * self.reward_scale
            self.morl.remember(state_vec, j, r, state_vec, True)  # contextual: done per decision
            self.morl.learn()
            rewards.append(r)
            qc_proj[j] += a_cyc                                  # commit -> spread next device
        self.morl.decay_epsilon()
        rec = self._advance(assignment)
        rec.mean_reward = float(np.mean(rewards)) if rewards else 0.0
        return rec

    # ---- shared environment advance (spec §11 steps 3,4,6,7) ----------------
    def _advance(self, assignment: dict) -> SlotRecord:
        s = self.state
        t = s.t
        qr_before = sum(task.remaining_bits for q in s.radio_q for task in q)
        qc_before = float(np.sum(s.compute_q_cycles))

        # --- 3. MPC: coverage-driven velocity per UAV, update positions ------
        moved = np.zeros(self.M, dtype=bool)
        for j in range(self.M):
            v = self.mpc.plan(self.proj_mpc, self._mpc_evaluate_positions, j, s.uav_pos)
            s.uav_pos[j] = s.uav_pos[j] + v * self.dt
            moved[j] = bool(np.linalg.norm(v) > 0.0)

        # --- 4. APSO: compute-capacity allocation (wired; per-task split applied
        #         in Increment 3 — the aggregate compute queue has no split to act
        #         on yet, so its result is not used to throttle full CPU service). -
        _alloc_frac = self._apso_allocate()

        # (Lyapunov stability diagnostics are computed after the env update; the
        #  drift-plus-penalty *biasing* of the assignment was applied in step 2/5.)

        # --- 6. environment update ------------------------------------------
        arrivals_bits = self._arrivals(t)

        transmitted_bits, handoff_per_uav, handoff_check = self._radio_service(assignment, t)
        for j in range(self.M):
            s.compute_q_cycles[j] += handoff_per_uav[j]                    # hand-off -> Q^c

        # Compute service S^c_j = min(Q^c_j, C_j·dt): a UAV always runs its CPU at
        # full capacity. APSO (alloc_frac) governs the per-task SPLIT of C_j across
        # a UAV's tasks — that allocation is applied in Increment 3 (the skeleton
        # tracks an aggregate compute queue, so the split has nothing to act on yet);
        # it must NOT throttle total capacity. (`alloc_frac` is computed/ wired above.)
        executed = np.zeros(self.M)
        for j in range(self.M):
            ex = min(s.compute_q_cycles[j], s.uav_capacity_cps[j] * self.dt)  # S^c_j, full CPU
            s.compute_q_cycles[j] -= ex
            executed[j] = ex

        # energy update (eq 10): flight (if the UAV actually moved) + compute(executed), harvest
        for j in range(self.M):
            s.uav_energy_j[j] = float(
                self.energy.step(s.uav_energy_j[j], executed[j], 1.0 if moved[j] else 0.0))

        coverage = gaussian_coverage(s.uav_pos, s.device_pos, self.sigma_cov)

        # --- diagnostics / Lyapunov drift -----------------------------------
        qr_after = sum(task.remaining_bits for q in s.radio_q for task in q)
        qc_after = float(np.sum(s.compute_q_cycles))
        lyap = self.lyapunov.evaluate(
            qr_before=[qr_before], qc_before=[qc_before],
            qr_after=[qr_after], qc_after=[qc_after], penalty=0.0)

        s.t += 1
        return SlotRecord(
            t=t,
            arrivals_bits=arrivals_bits,
            transmitted_bits=transmitted_bits,
            handoff_cycles=float(np.sum(handoff_per_uav)),
            handoff_cycles_check=handoff_check,
            executed_cycles=float(np.sum(executed)),
            qr_before=qr_before, qr_after=qr_after,
            qc_before=qc_before, qc_after=qc_after,
            coverage=coverage,
            total_energy_j=float(np.sum(s.uav_energy_j)),
            lyapunov=lyap,
            assignment=assignment,
        )

    # ---- environment sub-steps ----------------------------------------------
    def _arrivals(self, t: int) -> float:
        """Poisson task arrivals per device (spec §4.3). Returns total bits added."""
        added = 0.0
        for i in range(self.N):
            n = self.rng.poisson(self.lam[i] * self.dt)
            for _ in range(int(n)):
                L = self.rng.uniform(self.L_lo, self.L_hi) * MB_TO_BITS
                W = self.rng.uniform(self.W_lo, self.W_hi) * MCYCLES_TO_CYCLES
                d = self.rng.uniform(self.d_lo, self.d_hi)
                rho = self.rng.integers(self.rho_lo, self.rho_hi + 1)
                self.state.radio_q[i].append(Task(i, L, W, d, float(rho), t, L))
                added += L
        return added

    def _radio_service(self, assignment: dict, t: int):
        """Transmit bits per device toward its assigned UAV, PRIORITY-BY-URGENCY
        (spec §4.7), and convert transmitted bits to compute cycles per task
        (the explicit hand-off H_j = Σ_i x_ij Σ_k r_{i,k} b_{i,k})."""
        s = self.state
        handoff = np.zeros(self.M)
        handoff_check = 0.0
        transmitted_total = 0.0
        for i in range(self.N):
            if i not in assignment or not s.radio_q[i]:
                continue
            j = assignment[i]
            gain = self.channel.large_scale_gain(s.uav_pos[j], s.device_pos[i])  # deterministic
            budget = float(self.channel.rate(gain)) * self.dt  # bits this slot
            # serve in descending urgency; ties -> earliest arrival
            order = sorted(s.radio_q[i], key=lambda tk: (-tk.urgency(t), tk.arrival_t))
            for task in order:
                if budget <= 0:
                    break
                b = min(task.remaining_bits, budget)
                task.remaining_bits -= b
                budget -= b
                transmitted_total += b
                handoff[j] += b * task.intensity            # bits -> cycles (per task)
                handoff_check += b * task.intensity
            # drop fully transmitted tasks from the radio queue
            s.radio_q[i] = [tk for tk in s.radio_q[i] if tk.remaining_bits > 1e-9]
        return transmitted_total, handoff, handoff_check

    def _apso_allocate(self) -> np.ndarray:
        """APSO compute-capacity allocation (spec §9), SKELETON form.

        Allocates a per-UAV capacity fraction in [0,1]^M minimizing the 'apso'
        projection of a simple latency/util/energy proxy. The full per-task f_ijk
        allocation is the next increment; here it demonstrates the wired call and
        yields fractions used to scale compute service.
        """
        s = self.state
        Qc = s.compute_q_cycles
        Ccps = s.uav_capacity_cps

        def evaluate_terms(frac):
            frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
            served = frac * Ccps + 1.0
            latency = float(np.sum((Qc + 1.0) / served))                 # s: backlog / served rate
            cycles_served = frac * Ccps * self.dt
            energy = float(np.sum(self.computation.exec_energy_j(cycles_served)))  # J (e_c-consistent units)
            util = float(np.sum(frac))                                   # capacity used (reward)
            return {Term.TASK: latency, Term.ENERGY: energy, Term.UTIL: -util}

        res = self.apso.optimize_allocation(
            self.proj_apso, evaluate_terms,
            lower=np.zeros(self.M), upper=np.ones(self.M), rng=self.rng)
        return np.clip(res.best_position, 0.0, 1.0)
