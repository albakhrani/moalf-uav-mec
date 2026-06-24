# MOALF-UAV-MEC Simulation

A simulation of the **MOALF-UAV-MEC** framework for UAV-assisted mobile edge
computing. Multiple IoT devices generate computational tasks that are offloaded
to a fleet of UAVs acting as edge servers. The framework coordinates four
controllers over a discrete-time simulation: **multiobjective reinforcement
learning (MORL)** decides task-offload assignments, **model predictive control
(MPC)** plans UAV trajectories, **adaptive particle swarm optimization (APSO)**
allocates each UAV's compute capacity across its tasks, and a **Lyapunov
drift-plus-penalty** layer keeps the system's queues stable. Tasks flow through a
**two-tier queue model** — a per-device radio (transmission) backlog and a
per-UAV compute backlog — so the bits→cycles pipeline is modelled and conserved
end to end.

## Architecture

The system is organized around a single shared objective, a deterministic
system-model layer, four optimizers, and a per-slot simulation loop.

**Single objective with per-optimizer projections.** There is exactly one
objective function (a normalized, weighted sum of task latency, energy,
completion, utilization, and coverage terms), and the objective weights live in
exactly one place. Each optimizer scores candidate decisions through a *named
projection* of that objective — the subset of terms its decision variables affect
— that carries the same weights and normalizations:

- `morl` → {task, energy, completion, utilization}
- `mpc`  → {task, energy, coverage}
- `apso` → {task, energy, utilization}

A projection holds no weights of its own; it computes scores from the parent
objective. By construction no optimizer can introduce a second weighting scheme,
so the whole system optimizes one coherent objective.

**System-model layer** (`src/moalf/system_model/`) provides the deterministic
physics, each reading its parameters from configuration:

- `channel` — log-distance + Rician channel gain and Shannon achievable rate.
- `computation` — task execution time and energy.
- `energy` — UAV battery dynamics (consumption, harvesting, capacity cap).
- `coverage` — smooth Gaussian coverage metric used by the trajectory objective.

**Optimizers** (`src/moalf/optimizers/`):

- `morl` — a DQN agent that chooses offload assignments and is trained in-loop.
- `mpc` — a direct receding-horizon controller that plans 2-D UAV motion at fixed
  altitude.
- `apso` — particle-swarm allocation of each UAV's compute capacity across its
  tasks.
- `lyapunov` — a two-tier drift-plus-penalty controller that biases assignment to
  keep both queue tiers bounded.

**Simulation loop** (`src/moalf/simulation.py`) ties these together. Each time
slot, in order: observe state → MORL assigns offloads → MPC moves UAVs → APSO
allocates compute → the Lyapunov layer adjusts for stability → the environment
advances (task arrivals, radio transmission, the per-task bits→cycles hand-off
into the compute tier, compute execution, energy, and coverage). The two queue
tiers are updated with exact conservation of the work passing through them.

## Repository layout

```
moalf-uav-mec/
├── config/default.yaml          # all parameters (single source for the run)
├── src/moalf/
│   ├── system_model/            # channel, computation, energy, coverage
│   ├── optimizers/              # morl, mpc, apso, lyapunov
│   ├── objective.py             # the single objective; optimizers use projections
│   └── simulation.py            # per-slot simulation loop (two-tier queues)
├── experiments/                 # runnable experiment scripts
├── tests/                       # pytest suite
└── notes/                       # specification and design notes
```

## Installation

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

Dependencies are pinned (numpy, pyyaml, torch, pytest).

## Running

Run the test suite:

```bash
pytest
```

Run an experiment (scripts read `config/default.yaml`):

```bash
PYTHONPATH=src python experiments/<script>.py
```

## Citation

If you use this software, see [CITATION.cff](CITATION.cff) for citation metadata.

## License

MIT — see [LICENSE](LICENSE).
