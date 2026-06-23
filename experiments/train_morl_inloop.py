"""Increment 2: in-loop MORL (DQN) training against the COMPLETE coupled system.

Unlike experiments/train_morl.py (standalone synthetic offload env), this trains
MORL inside the real Algorithm-1 loop: MPC moves UAVs, the two-tier queues evolve,
energy/coverage update, and MORL chooses the offload assignment each slot. The
**Lyapunov drift-plus-penalty biasing is realized as reward shaping** (spec §10.1):
MORL's per-decision reward is −(c_Q·Q^c_j·a + V·penalty), the per-slot cost the
Lyapunov framework minimizes — so MORL *learns* the stability-aware policy that
Increment 1 computed analytically (rather than being hard-overridden, which would
leave it nothing to learn). Full compute capacity (a UAV always runs its CPU).

Learning curve = periodic greedy evaluation backlog on a fresh fixed system
(trends DOWN as the policy improves) — a cleaner signal than the per-slot reward,
whose magnitude drifts with the growing in-episode backlog.

Honest notes:
  - High-load fixture (λ=0.6) so stability is binding (the frozen config is lightly
    loaded — §15 entry 32 — and would show little to learn). This is a training
    fixture, NOT the frozen real config and NOT tuning a paper number.
  - RUNTIME: MPC dominates (~85% of each slot, ~250 ms for M=5 plans; ~320 ms/slot
    total). This is the bottleneck flagged for the vectorization decision.
  - reward_scale is a uniform, policy-invariant conditioning constant.
  - NO §18.4 paper-comparison table (waits for Increment 3).

Usage:  python experiments/train_morl_inloop.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import yaml

from moalf.simulation import Simulation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
SEED = 7
TRAIN_CHUNK = 40
N_CHUNKS = 5
EVAL_SLOTS = 30
EPS_DECAY = 0.99


def high_load(config):
    return {**config,
            "task": {**config["task"],
                     "generation_rate_tps": {"dist": "uniform", "low": 0.6, "high": 0.6}}}


def eval_backlog(agent, cfg, slots, seed=9090):
    sim = Simulation(cfg, seed=seed, offload_policy="morl", lyapunov_biasing=False)
    sim.morl = agent
    saved_eps, agent.epsilon = agent.epsilon, 0.0
    qc, max_ident = [], 0.0
    for _ in range(slots):
        rec = sim.step()
        qc.append(sim.state.compute_q_cycles.sum() / 1e6)
        max_ident = max(max_ident, abs(rec.handoff_cycles - rec.handoff_cycles_check))
    agent.epsilon = saved_eps
    return float(np.mean(qc)), max_ident


def main():
    config = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    cfg = high_load(config)
    sim = Simulation(cfg, seed=SEED, offload_policy="morl", lyapunov_biasing=False)
    print(f"in-loop MORL training (N={sim.N}, M={sim.M}, lambda=0.6), reward_scale={sim.reward_scale:.2e}")

    # baseline: untrained policy
    base_backlog, _ = eval_backlog(sim.morl, cfg, EVAL_SLOTS)
    print(f"untrained eval backlog: {base_backlog:.0f} Mcyc")

    learning_curve = []
    eps = 1.0
    t0 = time.time()
    max_ident = 0.0
    for ch in range(N_CHUNKS):
        for _ in range(TRAIN_CHUNK):
            rec = sim.train_step(eps)
            eps = max(0.05, eps * EPS_DECAY)
            max_ident = max(max_ident, abs(rec.handoff_cycles - rec.handoff_cycles_check))
        b, ie = eval_backlog(sim.morl, cfg, EVAL_SLOTS)
        max_ident = max(max_ident, ie)
        learning_curve.append(b)
        print(f"  after {TRAIN_CHUNK*(ch+1):3d} train slots: greedy eval backlog = {b:8.0f} Mcyc")
    dt = time.time() - t0

    print(f"\nlearning curve (eval backlog Mcyc, should trend DOWN): "
          f"{[round(x) for x in learning_curve]}")
    print(f"improvement vs untrained: {base_backlog:.0f} -> {learning_curve[-1]:.0f} "
          f"({learning_curve[-1]/base_backlog:.2f}x)")
    print(f"max hand-off identity error (training+eval): {max_ident:.1e}  (conservation holds)")
    total_slots = TRAIN_CHUNK * N_CHUNKS + EVAL_SLOTS * (N_CHUNKS + 1)
    print(f"runtime: {dt:.0f}s for ~{total_slots} slots ({dt/total_slots*1000:.0f} ms/slot; "
          f"MPC ~85% — bottleneck flagged for vectorization decision)")


if __name__ == "__main__":
    main()
