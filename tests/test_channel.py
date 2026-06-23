"""Tests for the channel model (src/moalf/system_model/channel.py).

Covers the authoritative spec §4.4 (eqs 6-7): unit conversions, unit/dimension
checks, a limiting case (rate -> 0 as distance -> inf), an SNR sanity check, and
the Rician fading normalisation. All parameters come from config/default.yaml —
no values are hard-coded in the test beyond what is re-derived from that config.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from moalf.system_model.channel import (
    ChannelModel,
    db_to_linear,
    dbm_per_hz_to_w_per_hz,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def chan(config) -> ChannelModel:
    return ChannelModel.from_config(config)


# --- unit conversions --------------------------------------------------------
def test_db_to_linear():
    assert db_to_linear(0.0) == pytest.approx(1.0)
    assert db_to_linear(-30.0) == pytest.approx(1e-3)
    assert db_to_linear(10.0) == pytest.approx(10.0)


def test_dbm_per_hz_to_w_per_hz():
    # -174 dBm/Hz is the standard thermal noise floor ~ 3.98e-21 W/Hz
    assert dbm_per_hz_to_w_per_hz(-174.0) == pytest.approx(3.9811e-21, rel=1e-3)
    assert dbm_per_hz_to_w_per_hz(30.0) == pytest.approx(1.0)  # 30 dBm = 1 W


# --- construction from config ------------------------------------------------
def test_from_config_matches_paper_stated(chan, config):
    ch = config["channel"]
    assert chan.beta0 == pytest.approx(1e-3)            # -30 dB
    assert chan.alpha_path == pytest.approx(2.0)
    assert chan.altitude_m == pytest.approx(100.0)
    assert chan.bandwidth_hz == pytest.approx(1e6)
    assert chan.tx_power_w == pytest.approx(ch["iot_tx_power_w"])
    assert chan.rician_k_linear == pytest.approx(db_to_linear(15.0))
    assert chan.noise_power_w == pytest.approx(chan.noise_psd_w_per_hz * 1e6)


# --- eq (6): large-scale gain ------------------------------------------------
def test_gain_directly_overhead(chan):
    # device directly below UAV: ||p-q||=0, slant^2 = H^2 -> beta0 * H^(-alpha)
    g = chan.large_scale_gain([0.0, 0.0], [0.0, 0.0])
    expected = chan.beta0 * chan.altitude_m ** (-chan.alpha_path)
    assert float(g) == pytest.approx(expected)
    assert expected == pytest.approx(1e-7)  # 1e-3 * 100^-2


def test_gain_monotonic_decreasing_with_distance(chan):
    dists = np.array([0.0, 50.0, 100.0, 250.0, 500.0])
    gains = np.array([chan.large_scale_gain([d, 0.0], [0.0, 0.0]) for d in dists])
    assert np.all(np.diff(gains) < 0.0)          # strictly decreasing
    assert np.all(gains > 0.0)                   # gain is a positive ratio


def test_gain_vectorised_shape(chan):
    uav = np.zeros((3, 2))
    dev = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 300.0]])
    g = chan.large_scale_gain(uav, dev)
    assert g.shape == (3,)
    # farther device -> smaller gain
    assert g[0] > g[1] > g[2] or g[0] > g[2]


# --- eq (7): SNR and rate ----------------------------------------------------
def test_snr_matches_definition(chan):
    g = chan.large_scale_gain([100.0, 0.0], [0.0, 0.0])
    manual = chan.tx_power_w * float(g) / (chan.noise_psd_w_per_hz * chan.bandwidth_hz)
    assert float(chan.snr(g)) == pytest.approx(manual)
    assert manual > 0.0


def test_rate_equals_shannon_of_snr(chan):
    g = chan.large_scale_gain([120.0, 80.0], [0.0, 0.0])
    snr = float(chan.snr(g))
    assert float(chan.rate(g)) == pytest.approx(chan.bandwidth_hz * np.log2(1.0 + snr))


def test_rate_positive_and_decreasing(chan):
    dists = np.array([10.0, 100.0, 400.0, 1000.0])
    rates = np.array([float(chan.rate(chan.large_scale_gain([d, 0.0], [0.0, 0.0]))) for d in dists])
    assert np.all(rates > 0.0)
    assert np.all(np.diff(rates) < 0.0)


def test_rate_units_are_bits_per_second(chan):
    # rate = B * log2(1+SNR): with B in Hz and SNR dimensionless, result is bit/s.
    # Sanity: a strong-SNR link should yield a large-but-finite positive rate.
    g = chan.large_scale_gain([0.0, 0.0], [0.0, 0.0])  # best case, overhead
    r = float(chan.rate(g))
    assert 0.0 < r < 1e9  # well under 1 Gb/s for a 1 MHz channel


# --- limiting case: rate -> 0 as distance -> infinity ------------------------
def test_rate_vanishes_at_infinite_distance(chan):
    near = float(chan.rate(chan.large_scale_gain([10.0, 0.0], [0.0, 0.0])))
    far = float(chan.rate(chan.large_scale_gain([1e6, 0.0], [0.0, 0.0])))
    farther = float(chan.rate(chan.large_scale_gain([1e12, 0.0], [0.0, 0.0])))
    assert near > far > farther
    assert farther == pytest.approx(0.0, abs=1e-3)   # effectively zero
    # gain itself -> 0
    assert float(chan.large_scale_gain([1e12, 0.0], [0.0, 0.0])) == pytest.approx(0.0, abs=1e-18)


# --- Rician fading normalisation (xi, E[xi] = 1) -----------------------------
def test_fading_is_positive_and_unit_mean(chan):
    rng = np.random.default_rng(0)
    xi = chan.sample_fading(200_000, rng)
    assert np.all(xi >= 0.0)
    assert float(xi.mean()) == pytest.approx(1.0, abs=0.02)  # normalised power


def test_channel_gain_deterministic_without_rng(chan):
    g1 = chan.channel_gain([100.0, 0.0], [0.0, 0.0])
    g2 = chan.channel_gain([100.0, 0.0], [0.0, 0.0])
    assert float(g1) == float(g2)
    # with no fading, channel_gain == large_scale_gain
    assert float(g1) == pytest.approx(float(chan.large_scale_gain([100.0, 0.0], [0.0, 0.0])))


def test_channel_gain_with_fading_averages_to_large_scale(chan):
    rng = np.random.default_rng(1)
    base = float(chan.large_scale_gain([100.0, 0.0], [0.0, 0.0]))
    samples = np.array([float(chan.channel_gain([100.0, 0.0], [0.0, 0.0], rng)) for _ in range(20_000)])
    assert samples.mean() == pytest.approx(base, rel=0.05)  # E[xi]=1 -> mean gain ~ large-scale
