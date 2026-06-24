# MOALF-UAV-MEC: Reimplementation and Reproduction Analysis

A from-scratch, test-verified reimplementation of the **MOALF-UAV-MEC** framework —
adaptive multiobjective optimization for UAV-assisted mobile edge computing —
combining **multiobjective reinforcement learning (MORL)**, **model predictive
control (MPC)**, **adaptive particle swarm optimization (APSO)**, and **Lyapunov
drift-plus-penalty** control over a **two-tier (radio + compute) queue model**.
It is built for reproducibility and analysis: a single internally-consistent
specification, one coherent objective shared by all optimizers, exact end-to-end
conservation of the bits→cycles task pipeline, and an honest reproduction of the
published paper's headline results. The reference paper is A. A. Al-Bakhrani, M.
Li, M. S. Obaidat, and G. A. Ameran, *"MOALF-UAV-MEC: Adaptive Multiobjective
Optimization for UAV-Assisted Mobile Edge Computing in Dynamic IoT Environments,"*
IEEE Internet of Things Journal, vol. 12, no. 12, pp. 20736–20756, 2025.

## Repository layout

```
moalf-uav-mec/
├── config/default.yaml          # all parameters (paper-stated or documented choices)
├── src/moalf/
│   ├── system_model/            # channel, computation, energy, coverage models
│   ├── optimizers/              # morl (DQN), mpc, apso, lyapunov
│   ├── objective.py             # the single objective; optimizers consume projections
│   └── simulation.py            # Algorithm-1 per-slot loop (two-tier queues)
├── experiments/                 # reproduction + diagnostic runners
├── tests/                       # pytest suite (system model, optimizers, conservation)
├── results/comparison_table.md  # generated §18.4 paper-comparison table
└── notes/                       # authoritative corrected spec + sign-off worksheet
```

The authoritative behavior spec is `notes/corrected_spec.md`; deviations from the
published paper are summarized in [CHANGES.md](CHANGES.md).

## Installation

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

Dependencies are pinned (numpy, pyyaml, torch, pytest). Training runs on CPU and
is seeded; exact bit-reproducibility of the MORL results also requires matching
the pinned PyTorch build.

## Reproducing the results

Regenerate the paper-comparison table (frozen config, 30-run protocol):

```bash
PYTHONPATH=src python experiments/reproduce_table.py
```

This writes/prints the §18.4 comparison (also saved in
[`results/comparison_table.md`](results/comparison_table.md)). Run the test suite
with:

```bash
pytest
```

## Results

On the paper's frozen configuration (30 runs, seeds 42–71), comparing the
consistent reimplementation against the published headline metrics:

| metric | paper-claimed | reproduced (mean ± sd) | tier |
|---|---|---|---|
| Task completion rate | 94.5% | **98.4% ± 0.5** | 1 — reproduced (exceeds) |
| Throughput increase vs baseline | +55% | +0.6% ± 5.3 | 3 — regime-dependent |
| UAV route reduction | −38% | −11.9% ± 15.3 | 3 — regime-dependent |

**Task completion reproduces and slightly exceeds the published value.** The
throughput and route-reduction figures do **not** reproduce at the stated
parameters — not as a bare miss, but because the published configuration is a
**lightly-loaded, saturated-SNR regime**: compute demand is ~⅓ of capacity, queues
are empty ~99% of the time, and per-link SNR is saturated cell-wide. In that
regime baselines already run near ceiling (little throughput to gain) and UAV
position has weak leverage (routes are not minimized). The learned policy's
throughput advantage **does** appear under genuine load (a labelled high-load run
shows ~+17% over a random baseline). Full per-metric analysis and the high-load
context block are in [`results/comparison_table.md`](results/comparison_table.md).

## Relationship to the published paper

This is a **consistent reimplementation and analysis**, not the original code. The
published description contains internal inconsistencies and under-specified
quantities; making it a single runnable system required documented modeling
decisions — consolidating the several per-component weightings into **one coherent
objective**, defining an explicit **two-tier queue model** for the Lyapunov layer,
correcting a degenerate coverage term, and fixing values the paper leaves open on
physical grounds. These choices, and the reproduction findings above, are recorded
in detail in [CHANGES.md](CHANGES.md). The aim is a faithful, transparent basis for
reuse and for an honest revision of the manuscript.

## Citation

If you use this software, please cite both the software and the original paper;
see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
