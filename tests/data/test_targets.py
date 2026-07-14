"""Tests for :func:`rba.data.targets.build_target` — end-to-end targets.yaml driver.

No network. Small hand-built frames exercise each of the three encoding rules
(``sign_of_change``, ``discrete_bps``, ``identity``) against the real yaml
entries loaded via ``rba.config.load_target_config``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import load_target_config
from rba.data.targets import build_target


def _decisions(
    change_bps: list[float | None], new_rate: list[float | None] | None = None
) -> pd.DataFrame:
    """A tiny meeting-indexed decision frame carrying the target source columns."""
    if new_rate is None:
        new_rate = [3.0 + (b or 0) / 100.0 for b in change_bps]
    idx = pd.date_range("2024-01-01", periods=len(change_bps), freq="MS", name="observation_date")
    return pd.DataFrame(
        {
            "rate_change_bps": change_bps,
            "new_rate_pct": new_rate,
        },
        index=idx,
    )


# =============================================================================
# sign_of_change (three_class).
# =============================================================================
def test_build_target_three_class_returns_string_labels_by_default() -> None:
    frame = _decisions([-25, 0, 0, 25, 0])
    cfg = load_target_config("three_class")
    y = build_target(frame, cfg)
    assert list(y) == ["cut", "hold", "hold", "hike", "hold"]
    assert y.index.equals(frame.index)
    assert y.name == "three_class"


def test_build_target_three_class_int_labels_returns_signed_ints() -> None:
    frame = _decisions([-25, 0, 25])
    cfg = load_target_config("three_class")
    y = build_target(frame, cfg, int_labels=True)
    np.testing.assert_array_equal(y.to_numpy(), np.array([-1, 0, 1], dtype=np.int64))


def test_build_target_three_class_drops_nan_rows() -> None:
    """NaN sources drop from the output — the harness reindexes X against the shorter y."""
    frame = _decisions([-25, None, 0, 25])
    cfg = load_target_config("three_class")
    y = build_target(frame, cfg)
    assert len(y) == 3
    assert list(y) == ["cut", "hold", "hike"]
    # Dropped row's index isn't in y.
    assert frame.index[1] not in y.index


def test_build_target_missing_source_column_raises_keyerror() -> None:
    frame = pd.DataFrame({"other": [0, 1]})
    cfg = load_target_config("three_class")
    with pytest.raises(KeyError, match="source columns"):
        build_target(frame, cfg)


# =============================================================================
# discrete_bps (magnitude_class / ordinal).
# =============================================================================
def test_build_target_magnitude_class_keeps_only_bin_values() -> None:
    """Values in ``bins`` survive as ints; a +12.5 bp out-of-bin row drops (drop strategy)."""
    frame = _decisions([-50, -25, 0, 25, 50, 12])  # 12 bp is out-of-bin
    cfg = load_target_config("magnitude_class")
    y = build_target(frame, cfg)
    assert len(y) == 5
    np.testing.assert_array_equal(y.to_numpy(), np.array([-50, -25, 0, 25, 50], dtype=np.int64))


def test_build_target_ordinal_matches_magnitude_class_output() -> None:
    """The two share encoding rules; only kind / eval_metrics differ."""
    frame = _decisions([-50, -25, 0, 25, 50])
    y_mag = build_target(frame, load_target_config("magnitude_class"))
    y_ord = build_target(frame, load_target_config("ordinal"))
    np.testing.assert_array_equal(y_mag.to_numpy(), y_ord.to_numpy())


def test_build_target_discrete_bps_drops_nan_rows() -> None:
    frame = _decisions([-25, None, 25])
    cfg = load_target_config("magnitude_class")
    y = build_target(frame, cfg)
    assert list(y.to_numpy()) == [-25, 25]


def test_build_target_discrete_bps_rejects_unknown_strategy() -> None:
    cfg = dict(load_target_config("magnitude_class"))
    cfg["encoding"] = dict(cfg["encoding"], out_of_bin_strategy="nearest")
    frame = _decisions([-25, 25])
    with pytest.raises(ValueError, match="not implemented"):
        build_target(frame, cfg)


# =============================================================================
# identity (delta_regression / level_regression).
# =============================================================================
def test_build_target_delta_regression_passes_through_source() -> None:
    frame = _decisions([-25, 0, 25])
    y = build_target(frame, load_target_config("delta_regression"))
    np.testing.assert_allclose(y.to_numpy(), np.array([-25.0, 0.0, 25.0]))
    assert y.dtype == np.float64


def test_build_target_level_regression_uses_new_rate_pct() -> None:
    """``level_regression`` reads ``new_rate_pct``, not ``rate_change_bps``."""
    frame = _decisions([0, 0, 0], new_rate=[3.5, 3.75, 4.0])
    y = build_target(frame, load_target_config("level_regression"))
    np.testing.assert_allclose(y.to_numpy(), np.array([3.5, 3.75, 4.0]))


def test_build_target_identity_drops_nan_rows() -> None:
    frame = _decisions([-25, None, 25])
    y = build_target(frame, load_target_config("delta_regression"))
    assert len(y) == 2
    assert not y.isna().any()


# =============================================================================
# Error path.
# =============================================================================
def test_build_target_unknown_encoding_rule_raises() -> None:
    cfg = {
        "name": "custom",
        "source_columns": ["rate_change_bps"],
        "encoding": {"rule": "bogus"},
    }
    with pytest.raises(ValueError, match="unknown encoding rule"):
        build_target(_decisions([0]), cfg)
