"""Weight-sweep harness — A2 multiobjective evidence + V diagnostic (spec §5.4 / §7.1).

Trains MORL in-loop under several objective weight vectors (the agent is
weight-agnostic; only the Objective changes — D1/A2) and reports whether the
LEARNED POLICY changes with the weights. Used here as a **V diagnostic**
(V=1.0 STILL FROZEN, §18): if the policy barely changes, that behaviorally
confirms the objective-blindness hypothesis (§15 entry 34); if it changes, the
objective is influencing the policy at this load.

Run at the REAL config's load (λ default ≈ 0.2), per the author's instruction —
NOT the λ=0.6 stress fixture.

Usage:  python experiments/weight_sweep.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml

from moalf.simulation import Simulation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
TRAIN_SLOTS = 90
N_PROBES = 400
SEED = 7

# meaningfully different weight vectors (diagnostic; the real default stays all-1)
WEIGHT_VECTORS = {
    "balanced":        {},                       # all 1.0
    "task-heavy":      {"w1_task": 10.0},        # favors high capacity / low backlog
    "completion-heavy":{"w3_completion": 10.0},  # favors UAVs that COMPLETE (need energy+capacity)
    "energy-heavy":    {"w2_energy": 10.0},      # (note: per §penalty, exec energy is per-task, not per-UAV)
}


def cfg_with_weights(config, overrides):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["objective"] = dict(config["objective"], **overrides)
    return cfg


def train_policy(config, overrides, slots, seed):
    sim = Simulation(cfg_with_weights(config, overrides), seed=seed,
                     offload_policy="morl", lyapunov_biasing=False)
    eps = 1.0
    for _ in range(slots):
        sim.train_step(eps)
        eps = max(0.05, eps * 0.97)
    sim.morl.epsilon = 0.0
    return sim


def probe_states(sim, n, rng):
    """Construct discriminating probe states: UAVs vary in capacity and energy
    independently (incl. high-capacity/low-energy), light backlog."""
    states = []
    for _ in range(n):
        caps = rng.uniform(sim.c_lo, sim.c_hi, sim.M)
        ens = rng.uniform(0.05, 1.0, sim.M)
        sim.state.uav_capacity_cps = caps * 1e9
        sim.state.uav_energy_j = ens * sim.e_max_j
        sim.state.compute_q_cycles = np.zeros(sim.M)
        # a representative pending task on device 0
        from moalf.simulation import Task, MB_TO_BITS, MCYCLES_TO_CYCLES
        head = Task(0, 0.5 * MB_TO_BITS, 500 * MCYCLES_TO_CYCLES, 12.0, 3.0, 0, 0.5 * MB_TO_BITS)
        vec = sim._offload_state(0, head)
        states.append((vec, caps, ens))
    return states


def run_at_load(config, lam, label):
    cfg = {**config, "task": {**config["task"],
           "generation_rate_tps": {"dist": "uniform", "low": lam, "high": lam}}}
    print(f"\n===== load: {label} (lambda={lam}) =====")
    sims = {name: train_policy(cfg, ov, TRAIN_SLOTS, SEED) for name, ov in WEIGHT_VECTORS.items()}
    # an UNTRAINED reference (same seed/init) to detect whether training changed anything
    untrained = Simulation(cfg_with_weights(cfg, {}), seed=SEED, offload_policy="morl")
    untrained.morl.epsilon = 0.0

    rng = np.random.default_rng(1234)
    probes = probe_states(sims["balanced"], N_PROBES, rng)
    vecs = [p[0] for p in probes]
    choices = {n: np.array([s.morl.greedy(v) for v in vecs]) for n, s in sims.items()}
    untr = np.array([untrained.morl.greedy(v) for v in vecs])

    names = list(WEIGHT_VECTORS)
    print("did training move each policy off its (identical) init?  (disagreement vs untrained)")
    for n in names:
        moved = np.mean(choices[n] != untr) * 100
        print(f"  {n:18s}: {moved:5.1f}% of probes differ from untrained")
    base = choices["balanced"]
    max_div = max(np.mean(choices[n] != base) for n in names if n != "balanced") * 100
    learned = max(np.mean(choices[n] != untr) for n in names) * 100
    print(f"max cross-weight divergence (vs balanced): {max_div:.1f}%")
    if learned < 5:
        verdict = "INCONCLUSIVE — training barely moved any policy off init (weak signal at this load)"
    elif max_div < 5:
        verdict = "OBJECTIVE-BLIND — policies learned but weights do NOT change them"
    else:
        verdict = "OBJECTIVE-SENSITIVE — weights change the learned policy"
    print(f"VERDICT: {verdict}")
    return max_div, learned


def main():
    config = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    print(f"weight-sweep V-diagnostic, V={config['lyapunov']['penalty_weight_V']} FROZEN; "
          f"{len(WEIGHT_VECTORS)} policies x {TRAIN_SLOTS} in-loop slots each")
    run_at_load(config, lam=0.2, label="REAL config")          # the author's requested load
    run_at_load(config, lam=0.6, label="HIGH (learning happens)")  # contrast where learning is known to occur
    print("\nReminder: V=1.0 frozen; this diagnostic INFORMS (does not perform) any V decision (sec 18).")


if __name__ == "__main__":
    main()
