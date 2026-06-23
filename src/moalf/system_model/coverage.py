"""coverage — smooth Gaussian coverage metric D(p) (spec §5.2, A3; eq 63).

The coverage objective term is J_coverage = −D(p) (spec §5.2), with

    D(p) = Σ_i Σ_j  ω_ij · exp( −‖p_j − q_i‖₂² / (2 σ_cov²) )      (eq 63)

This replaces the degenerate indicator coverage of paper eq (20) (A3): D is
smooth and differentiable, so it can actually drive UAV trajectory (the MPC
coverage term). Positions are horizontal (x, y); UAV altitude is fixed (A5) and
does not enter the *horizontal* coverage geometry.

Symbols (spec §3): q_i device position, p_j UAV position, ω_ij demand weight,
σ_cov coverage radius. Config: objective.coverage_radius_sigma_m, .coverage_demand_weight.
Deterministic; no hard-coded parameters.
"""

from __future__ import annotations

import numpy as np


def gaussian_coverage(uav_xy, device_xy, sigma_cov: float, omega=None) -> float:
    """Total smooth coverage D(p), eq (63).

    Parameters
    ----------
    uav_xy : array (M, 2) or (2,)   horizontal UAV positions p_j
    device_xy : array (N, 2) or (2,) horizontal device positions q_i
    sigma_cov : float                coverage radius σ_cov (m), > 0
    omega : None | float | array (N, M)
        demand weights ω_ij; None -> 1.0 for all pairs.

    Returns
    -------
    float : D(p) = Σ_i Σ_j ω_ij · exp(−‖p_j−q_i‖²/(2σ²))
    """
    if sigma_cov <= 0.0:
        raise ValueError("sigma_cov must be > 0")
    U = np.atleast_2d(np.asarray(uav_xy, dtype=float))      # (M, 2)
    Q = np.atleast_2d(np.asarray(device_xy, dtype=float))   # (N, 2)
    diff = Q[:, None, :] - U[None, :, :]                    # (N, M, 2)
    d2 = np.sum(diff * diff, axis=-1)                       # (N, M)
    kern = np.exp(-d2 / (2.0 * sigma_cov * sigma_cov))      # (N, M)
    if omega is None:
        return float(np.sum(kern))
    W = np.asarray(omega, dtype=float)
    return float(np.sum(W * kern))
