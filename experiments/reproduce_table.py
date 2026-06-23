"""§18.4 PAPER-COMPARISON TABLE — the moment of truth (spec §18).

Runs the consistent MOALF system on the FROZEN REAL CONFIG (lambda=0.2, all
signed-off values) over the paper's 30-run protocol (seeds 42-71), and compares
the headline metrics against the paper's claims with the six-column §18.4 table:
  metric | paper-claimed | reproduced (mean±sd) | abs gap | rel gap | tier

Tiers (§18.2): 1=exact (bonus), 2=directional+order-of-magnitude (the bar),
3=divergent (honest finding). §18 absolute: nothing is tuned toward a paper
number — whatever comes out is reported.

Baselines (stated): throughput/completion improvement is MOALF vs a RANDOM
offload policy; route reduction is MOALF's total UAV travel vs a straight-flight-
to-device-centroid baseline. (The paper's own baselines are other algorithms,
out of scope; these are clearly-labelled stand-ins.)

A clearly-labelled high-load (lambda=0.6) context block is included — it is NOT
the paper's config and is not the reproduction.

Usage:  python experiments/reproduce_table.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml

from moalf.simulation import Simulation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
SEEDS = list(range(42, 72))         # paper-stated 30 runs
TRAIN_SEED = 7                      # policy trained once on a held-out seed
TRAIN_SLOTS = 80
EVAL_SLOTS = 50

PAPER = {"route_reduction_pct": 38.0, "throughput_increase_pct": 55.0, "completion_pct": 94.5}


def cfg_at(config, lam):
    return {**config, "task": {**config["task"],
            "generation_rate_tps": {"dist": "uniform", "low": lam, "high": lam}}}


def train_moalf(config, lam):
    sim = Simulation(cfg_at(config, lam), seed=TRAIN_SEED,
                     offload_policy="morl", lyapunov_biasing=True)
    eps = 1.0
    for _ in range(TRAIN_SLOTS):
        sim.train_step(eps)
        eps = max(0.05, eps * 0.97)
    return sim.morl


def eval_run(config, lam, seed, policy, agent=None):
    sim = Simulation(cfg_at(config, lam), seed=seed,
                     offload_policy=policy, lyapunov_biasing=(policy == "morl"))
    if agent is not None:
        sim.morl = agent
        sim.morl.epsilon = 0.0
    uav0 = sim.state.uav_pos.copy()
    centroid = sim.state.device_pos.mean(axis=0)
    base_path = float(np.sum(np.linalg.norm(uav0 - centroid, axis=1)))  # straight-flight baseline route
    sim.reset_metrics()
    for _ in range(EVAL_SLOTS):
        sim.step()
    m = sim.metrics_summary()
    m["completed"] = sim.metrics["completed"]
    m["path_len_m"] = sim.metrics["path_len_m"]
    m["base_path_m"] = base_path
    return m


def collect(config, lam, seeds):
    agent = train_moalf(config, lam)
    moalf = [eval_run(config, lam, s, "morl", agent) for s in seeds]
    rand = [eval_run(config, lam, s, "random") for s in seeds]
    return moalf, rand


def stats(xs):
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std())


def tier(reproduced, claimed, *, higher_is_match_direction=True):
    """§18.2 tiers from the relative gap and direction."""
    if claimed == 0:
        return "—"
    rel = (reproduced - claimed) / abs(claimed)
    if abs(rel) <= 0.10:
        return "1 (exact)"
    # directional + order of magnitude: same sign and within ~[0.3x, 3x] of claim
    same_dir = (reproduced > 0) == (claimed > 0)
    if same_dir and 0.3 * abs(claimed) <= abs(reproduced) <= 3.0 * abs(claimed):
        return "2 (directional)"
    return "3 (divergent)"


def report_block(label, config, lam, seeds, is_paper_config):
    moalf, rand = collect(config, lam, seeds)
    tag = "PRIMARY — frozen real config" if is_paper_config else "CONTEXT ONLY — NOT the paper config"
    print(f"\n===== {label}  (lambda={lam}, {len(seeds)} runs)  [{tag}] =====")

    comp_m, comp_s = stats([m["completion_rate"] * 100 for m in moalf])
    thru_moalf, _ = stats([m["completed"] for m in moalf])
    thru_rand, _ = stats([m["completed"] for m in rand])
    thr_inc = [( (a["completed"] - b["completed"]) / b["completed"] * 100 if b["completed"] else 0.0)
               for a, b in zip(moalf, rand)]
    thr_m, thr_s = stats(thr_inc)
    route_red = [((m["base_path_m"] - m["path_len_m"]) / m["base_path_m"] * 100 if m["base_path_m"] > 0 else 0.0)
                 for m in moalf]
    rr_m, rr_s = stats(route_red)

    rows = [
        ("Task completion rate (%)", PAPER["completion_pct"], comp_m, comp_s),
        ("Throughput increase vs random (%)", PAPER["throughput_increase_pct"], thr_m, thr_s),
        ("UAV route reduction (%)", PAPER["route_reduction_pct"], rr_m, rr_s),
    ]
    hdr = f"{'metric':34s} {'paper':>8s} {'reproduced':>16s} {'abs gap':>9s} {'rel gap':>9s} {'tier':>16s}"
    print(hdr); print("-" * len(hdr))
    for name, claimed, rm, rs in rows:
        absg = rm - claimed
        relg = absg / abs(claimed) * 100 if claimed else float("nan")
        print(f"{name:34s} {claimed:8.1f} {rm:9.1f}+/-{rs:4.1f} {absg:9.1f} {relg:8.1f}% {tier(rm, claimed):>16s}")

    # context absolutes (no paper target)
    lat, _ = stats([m["mean_latency_s"] for m in moalf])
    util, _ = stats([m["mean_util"] for m in moalf])
    epc, _ = stats([m["energy_consumed_j"] / max(m["completed"], 1) for m in moalf])
    cov, _ = stats([m["mean_coverage"] for m in moalf])
    print(f"  context (MOALF absolutes): mean_latency={lat:.2f}s  util={util:.2f}  "
          f"energy/task={epc:.0f}J  coverage={cov:.1f}  | random completion="
          f"{stats([m['completion_rate']*100 for m in rand])[0]:.1f}%")


def main():
    config = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    print("§18.4 reproduction — frozen config, 30-run protocol (seeds 42-71). V/all values FROZEN; nothing tuned.")
    report_block("REAL CONFIG", config, 0.2, SEEDS, is_paper_config=True)
    report_block("HIGH LOAD", config, 0.6, list(range(42, 52)), is_paper_config=False)
    print("\n(§18: results reported as computed. Per-metric findings noted in the spec / response.)")


if __name__ == "__main__":
    main()
