"""Tests for :func:`rba.config.load_target_config` — mirrors ``load_model_config``.

No network. Round-trips the real ``targets.yaml`` entries end-to-end and covers
the error paths (unknown name, the scalar ``default`` key, no-arg default
resolution).
"""

from __future__ import annotations

import pytest

from rba.config import load_target_config, load_targets_config


# -----------------------------------------------------------------------------
# Entry lookup + name injection.
# -----------------------------------------------------------------------------
def test_load_target_config_returns_entry_with_name() -> None:
    cfg = load_target_config("three_class")
    assert cfg["kind"] == "classification"
    assert cfg["source_columns"] == ["rate_change_bps"]
    assert cfg["encoding"]["rule"] == "sign_of_change"
    assert cfg["classes"] == ["cut", "hold", "hike"]
    assert cfg["name"] == "three_class"


@pytest.mark.parametrize(
    "target_name",
    ["three_class", "magnitude_class", "ordinal", "delta_regression", "level_regression"],
)
def test_load_target_config_covers_every_yaml_entry(target_name: str) -> None:
    """Every non-``default`` key in targets.yaml resolves via load_target_config."""
    cfg = load_target_config(target_name)
    assert cfg["name"] == target_name
    assert "kind" in cfg
    assert "encoding" in cfg


def test_load_target_config_no_arg_resolves_yaml_default() -> None:
    """``load_target_config()`` uses the ``default:`` scalar (currently ``three_class``)."""
    cfg = load_target_config()
    assert cfg["name"] == "three_class"


# -----------------------------------------------------------------------------
# Error paths.
# -----------------------------------------------------------------------------
def test_load_target_config_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="No target named"):
        load_target_config("no_such_target")


def test_load_target_config_default_key_is_not_a_target() -> None:
    """The scalar ``default:`` at the top of targets.yaml must not resolve as a target."""
    with pytest.raises(KeyError, match="No target named"):
        load_target_config("default")


# -----------------------------------------------------------------------------
# Full mapping loader.
# -----------------------------------------------------------------------------
def test_load_targets_config_returns_full_mapping() -> None:
    """``load_targets_config`` returns every entry plus the ``default:`` scalar."""
    config = load_targets_config()
    assert {
        "three_class",
        "magnitude_class",
        "ordinal",
        "delta_regression",
        "level_regression",
    } <= set(config)
    assert config["default"] == "three_class"
