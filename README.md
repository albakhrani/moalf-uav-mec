# MOALF-UAV-MEC: Reference Simulation

A reproducible, publication-grade reference implementation of **MOALF-UAV-MEC**,
the adaptive multiobjective optimization framework for UAV-assisted mobile edge
computing in dynamic IoT environments, as described in the paper *"MOALF-UAV-MEC:
Adaptive Multiobjective Optimization for UAV-Assisted Mobile Edge Computing in
Dynamic IoT Environments"* (IEEE Internet of Things Journal, vol. 12, no. 12,
2025). The aim is a clean implementation that someone can clone and run to
regenerate the paper's figures and tables — built from a single, internally
consistent specification rather than the raw paper, which contains documented
contradictions.

> **Status: under construction.** This repository is currently a skeleton. No
> simulation, optimizer, or system-model logic is implemented yet — the logic
> files are stubs pending a finalized specification. No results have been
> produced. Nothing here reproduces the paper's numbers yet, and this README
> will not claim otherwise until it does.

## Installation

> TODO: finalize once the package has runnable code.

```bash
# (placeholder)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Reproducing the results

> TODO: provide a single-command runner that regenerates all figures/tables
> into `results/`. Not available yet.

```bash
# (placeholder — not implemented)
# python -m moalf.simulation --config config/default.yaml
```

## Repository layout

```
moalf-uav-mec/
├── config/default.yaml        # all parameters (paper-stated or marked UNSPECIFIED)
├── src/moalf/
│   ├── system_model/          # channel, computation, energy models (stubs)
│   ├── optimizers/            # MORL, MPC, APSO, Lyapunov (stubs)
│   ├── objective.py           # the single objective function (stub)
│   └── simulation.py          # main loop / Algorithm 1 (stub)
├── tests/                     # pytest suite
├── experiments/               # runnable experiment scripts
├── results/figures/           # generated figures (tracked); other results ignored
├── notes/                     # equation extraction + corrected spec (source of truth)
└── paper/                     # the source PDF
```

See [corrected_spec.md] for the development contract (one objective, no
hard-coded parameters, spec as source of truth).

## Citation

If you use this software, please cite both the software and the original paper.
See [CITATION.cff](CITATION.cff) for machine-readable metadata. The method is
described in:

> A. A. Al-Bakhrani, M. Li, M. S. Obaidat, and G. A. Ameran, "MOALF-UAV-MEC:
> Adaptive Multiobjective Optimization for UAV-Assisted Mobile Edge Computing in
> Dynamic IoT Environments," *IEEE Internet of Things Journal*, vol. 12, no. 12,
> pp. 20736–20756, 2025.

## License

MIT — see [LICENSE](LICENSE).
