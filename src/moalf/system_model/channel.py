"""channel — UAV <-> IoT wireless channel model.

Implements the system model's channel equations from the authoritative spec
(notes/corrected_spec.md §4.4), which supersede paper eqs (6)-(7):

    eq (6)  large-scale gain:  h_ij = beta0 * (||p_j - q_i||^2 + H^2)^(-alpha/2) * xi_ij
    eq (7)  achievable rate:   R_ij = B * log2(1 + P_tx * h_ij / (N0 * B))

where (symbols per spec §3 symbol table):
    beta0     reference channel gain at 1 m       (config: channel.reference_gain_beta0_db, dB)
    alpha_path path-loss exponent                 (config: channel.path_loss_exponent_alpha)
    H         UAV altitude, fixed (2-D motion, A5) (config: channel.uav_altitude_m, m)
    xi_ij     Rician small-scale fading, E[xi]=1   (config: channel.rician_k_db)
    B         channel bandwidth                    (config: channel.bandwidth_hz, Hz)
    N0        noise power spectral density         (config: channel.noise_psd_dbm_per_hz, dBm/Hz)
    P_tx      device transmit power                (config: channel.iot_tx_power_w, W)

All values are read from config (no hard-coded parameters). The
two unit conversions used here (dB -> linear, dBm/Hz -> W/Hz) are documented
physical formulas, not magic constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


# --- documented unit conversions ---------------------------------------------
def db_to_linear(db: float) -> float:
    """Decibels -> linear ratio:  10^(dB/10)."""
    return 10.0 ** (db / 10.0)


def dbm_per_hz_to_w_per_hz(dbm_per_hz: float) -> float:
    """Noise PSD dBm/Hz -> W/Hz:  10^((dBm - 30)/10)  (the -30 converts dBm -> dBW)."""
    return 10.0 ** ((dbm_per_hz - 30.0) / 10.0)


@dataclass(frozen=True)
class ChannelModel:
    """Log-distance + Rician channel (spec §4.4, eqs 6-7).

    Construct via :meth:`from_config`. All attributes are in internal SI units
    (linear gains, W, Hz, m).
    """

    beta0: float            # linear reference gain at 1 m (eq 6)
    alpha_path: float       # path-loss exponent (eq 6)
    altitude_m: float       # H, UAV altitude (eq 6)
    bandwidth_hz: float     # B (eq 7)
    noise_psd_w_per_hz: float  # N0 (eq 7)
    tx_power_w: float       # P_tx (eq 7)
    rician_k_linear: float  # Rician K factor (linear) for xi_ij

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ChannelModel":
        """Build from the parsed config dict (expects a top-level 'channel' section)."""
        ch = config["channel"]
        return cls(
            beta0=db_to_linear(ch["reference_gain_beta0_db"]),
            alpha_path=float(ch["path_loss_exponent_alpha"]),
            altitude_m=float(ch["uav_altitude_m"]),
            bandwidth_hz=float(ch["bandwidth_hz"]),
            noise_psd_w_per_hz=dbm_per_hz_to_w_per_hz(ch["noise_psd_dbm_per_hz"]),
            tx_power_w=float(ch["iot_tx_power_w"]),
            rician_k_linear=db_to_linear(ch["rician_k_db"]),
        )

    @property
    def noise_power_w(self) -> float:
        """Total in-band noise power  N0 * B  (W)."""
        return self.noise_psd_w_per_hz * self.bandwidth_hz

    # ---- eq (6): large-scale channel gain -----------------------------------
    def large_scale_gain(self, uav_pos, dev_pos) -> np.ndarray:
        """Deterministic (fading-free) channel gain h_ij with xi_ij = 1, eq (6).

        ``uav_pos`` / ``dev_pos`` are horizontal (x, y) coordinates in metres;
        the fixed altitude ``H`` supplies the vertical separation. Shapes broadcast
        over the leading axes, so this works for a single pair or whole grids.
        """
        uav = np.asarray(uav_pos, dtype=float)
        dev = np.asarray(dev_pos, dtype=float)
        horiz = uav[..., :2] - dev[..., :2]
        dist_sq = np.sum(horiz * horiz, axis=-1)            # ||p_j - q_i||^2
        slant_sq = dist_sq + self.altitude_m ** 2           # + H^2
        return self.beta0 * slant_sq ** (-self.alpha_path / 2.0)

    # ---- Rician small-scale fading (xi_ij, normalised E[xi] = 1) -------------
    def sample_fading(self, shape, rng: np.random.Generator) -> np.ndarray:
        """Rician power-gain samples xi with E[xi] = 1 (K from config).

        xi = x^2 + y^2 with x ~ N(s, sigma^2), y ~ N(0, sigma^2),
        s = sqrt(K/(K+1)), sigma^2 = 1/(2(K+1)) so E[xi] = s^2 + 2 sigma^2 = 1.
        """
        K = self.rician_k_linear
        s = np.sqrt(K / (K + 1.0))
        sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
        x = rng.normal(s, sigma, size=shape)
        y = rng.normal(0.0, sigma, size=shape)
        return x * x + y * y

    def channel_gain(self, uav_pos, dev_pos, rng: np.random.Generator | None = None) -> np.ndarray:
        """Full channel gain h_ij, eq (6). With ``rng=None`` fading xi=1 (mean power,
        deterministic); with an RNG, multiply by a Rician fading sample."""
        gain = self.large_scale_gain(uav_pos, dev_pos)
        if rng is None:
            return gain
        return gain * self.sample_fading(np.shape(gain), rng)

    # ---- eq (7): SNR and achievable rate ------------------------------------
    def snr(self, gain) -> np.ndarray:
        """Receive SNR  P_tx * h_ij / (N0 * B)  (linear, dimensionless)."""
        return self.tx_power_w * np.asarray(gain, dtype=float) / self.noise_power_w

    def rate(self, gain) -> np.ndarray:
        """Achievable rate R_ij = B * log2(1 + SNR), eq (7), in bit/s."""
        return self.bandwidth_hz * np.log2(1.0 + self.snr(gain))
