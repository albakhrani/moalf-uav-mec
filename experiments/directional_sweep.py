"""CHECK 1 — directional-coherence sweep (spec §15 entries 41/42).

Proves the non-flat weight-sweep (entry 38) is COHERENT multiobjective response,
not thrashing: boosting each objective weight one-at-a-time (others fixed) should
move ITS OWN outcome in the pre-stated direction (entry 41). Run at moderate-to-
high load where objectives genuinely conflict. V frozen.

Pre-stated expectations (entry 41):
  w1_task↑       -> mean task latency      DOWN
  w2_energy↑     -> total energy consumed  DOWN   (may be weak: compute energy ~ policy-invariant)
  w3_completion↑ -> on-time completion     UP
  w5_util↑       -> mean compute util      UP
  w6_coverage↑   -> mean coverage          UP

Usage:  python experiments/directional_sweep.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml

from moalf.simulation import Simulation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
LAM = 0.6                # moderate-to-high load (clear objective conflict)
TRAIN_SLOTS = 90
EVAL_SLOTS = 60
BOOST = 10.0
SEED = 7

# (weight key, outcome metric, expected direction for an INCREASE in the weight)
CASES = [
    ("w1_task",       "mean_latency_s",   "down"),
    ("w2_energy",     "energy_consumed_j","down"),
    ("w3_completion", "completion_rate",  "up"),
    ("w5_util",       "mean_util",        "up"),
    ("w6_coverage",   "mean_coverage",    "up"),
]


def cfg_with(config, lam, overrides):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    c["task"] = dict(config["task"], generation_rate_tps={"dist": "uniform", "low": lam, "high": lam})
    c["objective"] = dict(config["objective"], **overrides)
    return c


def run_outcome(config, lam, overrides):
    sim = Simulation(cfg_with(config, lam, overrides), seed=SEED,
                     offload_policy="morl", lyapunov_biasing=False)
    eps = 1.0
    for _ in range(TRAIN_SLOTS):                 # train in-loop under these weights
        sim.train_step(eps)
        eps = max(0.05, eps * 0.97)
    sim.morl.epsilon = 0.0
    sim.reset_metrics()
    for _ in range(EVAL_SLOTS):                  # measure outcomes under the greedy policy
        sim.step()
    return sim.metrics_summary()


def main():
    config = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    print(f"directional-coherence check at lambda={LAM} (moderate-high), V frozen; "
          f"boost={BOOST}x, train {TRAIN_SLOTS} / eval {EVAL_SLOTS} slots")
    base = run_outcome(config, LAM, {})         # all weights = 1
    print("\nbaseline (all weights = 1):")
    for k in ("mean_latency_s", "energy_consumed_j", "completion_rate", "mean_util", "mean_coverage"):
        print(f"    {k:18s} = {base[k]:.4g}")

    print(f"\n{'weight boosted':16s} {'metric':18s} {'baseline':>12s} {'boosted':>12s} {'dir':>6s} {'expect':>7s} {'OK?':>5s}")
    all_ok = True
    for wkey, metric, expect in CASES:
        out = run_outcome(config, LAM, {wkey: BOOST})
        b, v = base[metric], out[metric]
        direction = "up" if v > b else ("down" if v < b else "flat")
        ok = (direction == expect)
        # flat/uncontrollable energy is acceptable (pre-noted caveat), not incoherent
        coherent = ok or (wkey == "w2_energy" and direction in ("down", "flat"))
        all_ok = all_ok and coherent
        flag = "OK" if ok else ("~ok" if coherent else "WRONG")
        print(f"{wkey:16s} {metric:18s} {b:12.4g} {v:12.4g} {direction:>6s} {expect:>7s} {flag:>5s}")

    print(f"\nVERDICT: {'COHERENT (multiobjective response is directionally correct)' if all_ok else 'INCOHERENT — an outcome moved the WRONG way (STOP & report)'}")


if __name__ == "__main__":
    main()
