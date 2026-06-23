"""Config integrity tests for config/default.yaml.

Primary guard: no numeric field is silently stored as a *string* (the
`1.0e6`-parses-as-"1.0e6" class of bug, where PyYAML keeps an unsigned-exponent
float as text). Also checks there are no leftover nulls and that key numeric
fields are real numbers.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with io.open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _looks_numeric(s: str) -> bool:
    """True if a string would parse as a number (i.e. it *should* have been numeric)."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _walk(node, path=""):
    """Yield (dotted_path, leaf_value) for every scalar leaf in a nested dict/list."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for idx, v in enumerate(node):
            yield from _walk(v, f"{path}[{idx}]")
    else:
        yield path, node


def test_no_number_stored_as_string(config):
    """Every string leaf must be a genuine label, not a number-as-text."""
    offenders = [
        (p, v) for p, v in _walk(config)
        if isinstance(v, str) and _looks_numeric(v)
    ]
    assert not offenders, (
        "These config fields are numbers stored as strings (fix the YAML, e.g. "
        f"use 1.0e+6 not 1.0e6): {offenders}"
    )


def test_no_remaining_nulls(config):
    nulls = [p for p, v in _walk(config) if v is None]
    assert not nulls, f"Unresolved (null) config fields remain: {nulls}"


@pytest.mark.parametrize(
    "dotted",
    [
        "channel.bandwidth_hz",
        "channel.reference_gain_beta0_db",
        "channel.noise_psd_dbm_per_hz",
        "channel.path_loss_exponent_alpha",
        "channel.uav_altitude_m",
        "channel.iot_tx_power_w",
        "uav.compute_energy_rate_wh_per_cycle",
        "lyapunov.penalty_weight_V",
        "lyapunov.tier_scaling_c_Q",
        "objective.coverage_rate_threshold_rmin_bps",
        "run.seed",
        "run.num_runs",
    ],
)
def test_known_numeric_fields_are_numbers(config, dotted):
    node = config
    for key in dotted.split("."):
        node = node[key]
    assert isinstance(node, (int, float)) and not isinstance(node, bool), (
        f"{dotted} = {node!r} is not numeric"
    )


def test_distribution_blocks_have_numeric_bounds(config):
    """Every {dist: ...} block must have numeric low/high."""
    def check(node, path=""):
        if isinstance(node, dict):
            if node.get("dist") in {"uniform", "uniform_int"}:
                for b in ("low", "high"):
                    assert isinstance(node.get(b), (int, float)) and not isinstance(node.get(b), bool), (
                        f"{path}.{b} in a dist block is not numeric: {node.get(b)!r}"
                    )
            for k, v in node.items():
                check(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")

    check(config)


def test_migration_keys_removed(config):
    """A4 dropped migration: these keys must not exist."""
    assert "w4_migration" not in config["objective"]
    assert "migration_delay_delta" not in config["objective"]
    assert "migration_fixed_cost_epsilon" not in config["objective"]
    assert "energy_coefficient_kappa" not in config.get("computation", {})
