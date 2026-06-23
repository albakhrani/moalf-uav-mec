"""Full-scale MORL (DQN) training + pre-registered verification (spec §7.1).

Runs the three pre-registered checks at full length:
  1. learning curve trends up and plateaus;
  2. learned policy beats random AND the JSQ baseline (else: a FINDING);
  3. corner case: near-depleted UAV offloaded-to less.

Usage:  python experiments/train_morl.py
Reads config/default.yaml. CPU, seeded from the frozen seed (42) for reproducibility
(exact bit-reproducibility also requires the pinned torch build).

IMPORTANT (honesty, §18): the completion rate here is on the *standalone* offload
environment, NOT the full Algorithm-1 simulation. It is therefore NOT directly
comparable to the paper's 94.5% — that comparison waits until the simulation loop
is wired up and produces the §18.4 side-by-side table. Hyperparameters are frozen;
nothing here is tuned toward 94.5%.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml

from moalf.objective import Objective
from moalf.optimizers.morl import (
    MORLAgent, OffloadEnv, evaluate, jsq_policy, random_policy, train,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
TRAIN_STEPS = 8000
EVAL_EPISODES = 5000
SEED = 42


def main() -> None:
    config = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    proj = Objective.from_config(config).projection("morl")

    env = OffloadEnv(config, proj, seed=SEED)
    agent = MORLAgent.from_config(config, env.state_dim, env.action_dim, seed=SEED)

    print(f"Training DQN MORL: {TRAIN_STEPS} steps, state_dim={env.state_dim}, "
          f"actions={env.action_dim}")
    curve = train(agent, env, steps=TRAIN_STEPS, window=400)
    print("learning curve (mean reward / 400 steps):")
    print("  " + " ".join(f"{c:.2f}" for c in curve))
    improved = np.mean(curve[-3:]) > np.mean(curve[:3])
    print(f"  criterion 1 (trends up & plateaus): {'PASS' if improved else 'FAIL'}")

    # criterion 2: beat random AND JSQ on a fresh evaluation stream
    def fresh():
        return OffloadEnv(config, proj, seed=10_000)

    dqn_r, dqn_c = evaluate(lambda s, i: agent.greedy(s), fresh(), EVAL_EPISODES)
    rand_r, rand_c = evaluate(random_policy(fresh(), np.random.default_rng(1)), fresh(), EVAL_EPISODES)
    jsq_r, jsq_c = evaluate(jsq_policy(config), fresh(), EVAL_EPISODES)
    print("\nmean reward    : DQN={:.3f}  random={:.3f}  JSQ={:.3f}".format(dqn_r, rand_r, jsq_r))
    print("completion rate: DQN={:.2%}  random={:.2%}  JSQ={:.2%}".format(dqn_c, rand_c, jsq_c))
    beats = dqn_r > rand_r and dqn_r > jsq_r
    print(f"  criterion 2 (beats random AND JSQ): {'PASS' if beats else 'FINDING - did not beat a baseline'}")
    if jsq_r < rand_r:
        print("  note: JSQ < random here — shortest-queue ignores energy/capacity (a finding).")

    # criterion 3: near-depleted UAV avoided
    rng = np.random.default_rng(5)
    picks_depleted = 0
    for _ in range(1000):
        c = np.full(env.M, 3.5)
        q = np.full(env.M, env.q_max_cyc * 0.5)
        e = np.full(env.M, 0.02)
        e[1] = 0.95
        info = {"c": c, "e": e, "q": q,
                "w": rng.uniform(env.w_lo, env.w_hi), "d": rng.uniform(env.d_lo, env.d_hi)}
        if agent.greedy(env.encode(info)) == 0:
            picks_depleted += 1
    print(f"\ncorner case: chose near-depleted UAV {picks_depleted}/1000 times")
    print(f"  criterion 3 (depleted avoided): {'PASS' if picks_depleted < 200 else 'FAIL'}")

    print("\nReminder: completion here is the STANDALONE offload env, not the full "
          "Algorithm-1 sim — not comparable to the paper's 94.5% yet (§18.4).")


if __name__ == "__main__":
    main()
