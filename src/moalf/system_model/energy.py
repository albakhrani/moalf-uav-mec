"""energy — UAV battery dynamics: consumption, harvesting, capacity cap.

Implements the system model's energy equations from the authoritative spec
(notes/corrected_spec.md §4.6), which adopt paper eq (10) and drop eq (12):

    eq (10)  E_j(t+1) = min{ E_j(t) - E_cons,j(t) + eta_j * dt , E_max }
    E_cons,j(t) = P_flight * dt * 1{moving} + e_c * cycles_executed_j(t)
    constraint (23):  E_j(t) >= E_min   (checked, not clamped — see note)

Coupling to the two-tier queue (§4.7): the compute-energy term is driven by the
cycles the UAV actually drains from its compute backlog this slot — i.e. the
Tier-2 compute service S^c_j(t) (cycles), whose arrivals came through the
per-task bits->cycles hand-off. ``cycles_executed`` is that S^c_j; this module
consumes it, it does not compute it (the queue/sim does).

Notes:
  - Per spec, eq (10) has NO lower clamp at 0: energy may fall below E_min (or 0),
    which signals infeasibility. The E_min reserve (constraint 23) is enforced by
    the controller, not by the dynamics; ``is_depleted`` reports a violation.
  - Compute energy reuses the same linear per-cycle figure e_c as
    computation.py (single source via ComputationModel), so the two modules
    cannot drift apart.

Internal units: joules, watts, seconds. Config supplies Wh; converted on load
(1 Wh = 3600 J). All values are read from config (no hard-coded params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from moalf.system_model.computation import ComputationModel, WH_TO_J


@dataclass(frozen=True)
class EnergyModel:
    """UAV energy update (eq 10) with linear flight + compute consumption.

    Construct via :meth:`from_config`. Energies in joules, powers in watts.
    """

    harvest_power_w: float      # eta_j (eq 10)
    flight_power_w: float       # P_flight
    capacity_j: float           # E_max
    min_reserve_j: float        # E_min (constraint 23)
    dt_s: float                 # Delta t
    computation: ComputationModel  # supplies e_c for compute energy

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EnergyModel":
        en = config.get("energy", {})
        harvest_model = en.get("harvest_model", "linear_eta_dt")
        if harvest_model != "linear_eta_dt":
            raise ValueError(
                f"energy.harvest_model = {harvest_model!r}; only 'linear_eta_dt' "
                "is implemented (spec §4.6, eq 10). The eq (12) P_harv form is excluded."
            )
        uav = config["uav"]
        return cls(
            harvest_power_w=float(uav["harvest_rate_w"]),
            flight_power_w=float(uav["flight_energy_rate_w"]),
            capacity_j=float(uav["energy_capacity_wh"]) * WH_TO_J,
            min_reserve_j=float(en["min_reserve_wh"]) * WH_TO_J,
            dt_s=float(config["network"]["time_step_s"]),
            computation=ComputationModel.from_config(config),
        )

    # ---- consumption / harvest components -----------------------------------
    def harvested_j(self) -> float:
        """Energy harvested in one slot: eta_j * dt (J), eq (10)."""
        return self.harvest_power_w * self.dt_s

    def consumed_j(self, cycles_executed, moving) -> np.ndarray:
        """Energy consumed in one slot (J): flight (if moving) + compute.

        ``cycles_executed`` = S^c_j, the compute-tier cycles drained this slot
        (§4.7). ``moving`` is a 0/1 (or bool) flag, broadcastable.
        """
        cycles = np.asarray(cycles_executed, dtype=float)
        if np.any(cycles < 0.0):
            raise ValueError("cycles_executed must be >= 0")
        flight = self.flight_power_w * self.dt_s * np.asarray(moving, dtype=float)
        compute = self.computation.exec_energy_j(cycles)
        return flight + compute

    # ---- eq (10): one-slot energy update ------------------------------------
    def step(self, energy_j, cycles_executed, moving) -> np.ndarray:
        """Update battery one slot, eq (10): min{E - E_cons + eta*dt, E_max}.

        No lower clamp (per spec); falling below E_min/0 signals infeasibility.
        """
        E = np.asarray(energy_j, dtype=float)
        new_E = E - self.consumed_j(cycles_executed, moving) + self.harvested_j()
        return np.minimum(new_E, self.capacity_j)

    # ---- constraint (23) ----------------------------------------------------
    def is_depleted(self, energy_j) -> np.ndarray:
        """True where E_j < E_min (constraint 23 violated)."""
        return np.asarray(energy_j, dtype=float) < self.min_reserve_j
