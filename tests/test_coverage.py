"""Tests for the Gaussian coverage metric (system_model/coverage.py), spec §5.2/eq 63."""

from __future__ import annotations

import math

import numpy as np
import pytest

from moalf.system_model.coverage import gaussian_coverage


def test_max_when_uav_on_device():
    # single uav/device coincident -> exp(0) = 1
    assert gaussian_coverage([0.0, 0.0], [0.0, 0.0], sigma_cov=100.0) == pytest.approx(1.0)


def test_decreases_with_distance():
    d0 = gaussian_coverage([0.0, 0.0], [0.0, 0.0], 100.0)
    d1 = gaussian_coverage([50.0, 0.0], [0.0, 0.0], 100.0)
    d2 = gaussian_coverage([200.0, 0.0], [0.0, 0.0], 100.0)
    assert d0 > d1 > d2 > 0.0


def test_known_value_at_one_sigma():
    # at distance = sigma, kernel = exp(-1/2)
    val = gaussian_coverage([100.0, 0.0], [0.0, 0.0], sigma_cov=100.0)
    assert val == pytest.approx(math.exp(-0.5))


def test_sums_over_devices_and_uavs():
    # two coincident uav-device pairs far apart -> ~2.0
    uav = [[0.0, 0.0], [1000.0, 1000.0]]
    dev = [[0.0, 0.0], [1000.0, 1000.0]]
    assert gaussian_coverage(uav, dev, 100.0) == pytest.approx(2.0, abs=1e-6)


def test_omega_weighting_scales_contribution():
    base = gaussian_coverage([0.0, 0.0], [0.0, 0.0], 100.0)
    weighted = gaussian_coverage([0.0, 0.0], [0.0, 0.0], 100.0, omega=3.0)
    assert weighted == pytest.approx(3.0 * base)


def test_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_coverage([0.0, 0.0], [0.0, 0.0], sigma_cov=0.0)
