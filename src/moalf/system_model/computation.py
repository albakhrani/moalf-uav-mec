"""computation — task execution time and (linear) execution energy.

Implements the system model's computation equations from the authoritative spec
(notes/corrected_spec.md §4.5), which supersede paper eqs (8)-(9):

    eq (8)  execution time:    t_exec(i,k,j) = W_{i,k} / C_j
    §4.5    execution energy:  E_exec(i,k,j) = e_c * W_{i,k}        (LINEAR model, B6)

where (symbols per spec §3):
    W_{i,k}  task compute requirement   (cycles)
    C_j      UAV compute capacity       (cycles/s)
    e_c      per-cycle compute energy   (config: uav.compute_energy_rate_wh_per_cycle, Wh/cycle)

Decision B6 (§16): the LINEAR energy model `E_exec = e_c * W` is implemented
(the only numeric energy figure the paper gives, Table IV). The quadratic
eq (9) `E = kappa * C^2 * W` is NOT used — it has no stated coefficient.

Internal units are SI: cycles, seconds, joules. Config supplies e_c in Wh/cycle;
it is converted to J/cycle on load (1 Wh = 3600 J). All values are read from
config (no hard-coded parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

WH_TO_J = 3600.0  # 1 watt-hour = 3600 joules


@dataclass(frozen=True)
class ComputationModel:
    """Execution time (eq 8) and linear execution energy (§4.5, B6).

    Construct via :meth:`from_config`. ``energy_per_cycle_j`` is in joules/cycle.
    """

    energy_per_cycle_j: float   # e_c, converted from Wh/cycle to J/cycle

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ComputationModel":
        comp = config.get("computation", {})
        model = comp.get("energy_model", "linear")
        if model != "linear":
            raise ValueError(
                f"computation.energy_model = {model!r}; only 'linear' is implemented "
                "(spec §4.5 / sign-off B6). The quadratic eq (9) is excluded."
            )
        e_c_wh = float(config["uav"]["compute_energy_rate_wh_per_cycle"])
        return cls(energy_per_cycle_j=e_c_wh * WH_TO_J)

    # ---- eq (8): execution time ---------------------------------------------
    def exec_time_s(self, work_cycles, capacity_cps) -> np.ndarray:
        """Execution time t_exec = W / C_j (seconds), eq (8).

        ``work_cycles`` (W, cycles) and ``capacity_cps`` (C_j, cycles/s) broadcast.
        Requires positive capacity (a UAV with zero compute cannot execute).
        """
        W = np.asarray(work_cycles, dtype=float)
        C = np.asarray(capacity_cps, dtype=float)
        if np.any(C <= 0.0):
            raise ValueError("compute capacity C_j must be > 0 (cycles/s)")
        if np.any(W < 0.0):
            raise ValueError("work W must be >= 0 (cycles)")
        return W / C

    # ---- §4.5 (B6): linear execution energy ---------------------------------
    def exec_energy_j(self, work_cycles) -> np.ndarray:
        """Execution energy E_exec = e_c * W (joules), linear model (§4.5, B6).

        Note: independent of the clock C_j in this model — energy scales only with
        the number of cycles, by the single per-cycle figure e_c.
        """
        W = np.asarray(work_cycles, dtype=float)
        if np.any(W < 0.0):
            raise ValueError("work W must be >= 0 (cycles)")
        return self.energy_per_cycle_j * W
