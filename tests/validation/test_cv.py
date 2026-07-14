"""Tests for :class:`rba.validation.WalkForwardSplit` — the §7 CV splitter.

No network. Covers fold enumeration for both expanding and rolling windows,
the sklearn-compatible signature (``split`` / ``get_n_splits`` taking ``y`` and
``groups`` as ignored positional args), the no-leakage invariant, step / test
size / max_train_size interactions, and the constructor's guardrail errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.validation import WalkForwardSplit


# -----------------------------------------------------------------------------
# Basic expanding fold enumeration.
# -----------------------------------------------------------------------------
def test_expanding_basic_folds() -> None:
    """3-row train, 1-row test, expanding, on 6 rows → 3 folds."""
    splitter = WalkForwardSplit(initial_train_size=3, test_size=1)
    folds = list(splitter.split(np.arange(6)))
    assert len(folds) == 3
    assert [f[0].tolist() for f in folds] == [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    assert [f[1].tolist() for f in folds] == [[3], [4], [5]]


def test_expanding_test_window_gt_one() -> None:
    """test_size=2 with step=2 gives non-overlapping test windows."""
    splitter = WalkForwardSplit(initial_train_size=4, test_size=2, step=2)
    folds = list(splitter.split(np.arange(10)))
    # test starts: 4, 6, 8 → three full-window folds.
    assert len(folds) == 3
    assert [f[1].tolist() for f in folds] == [[4, 5], [6, 7], [8, 9]]


def test_expanding_step_advances_test_window() -> None:
    """step=3 skips two rows between test windows."""
    splitter = WalkForwardSplit(initial_train_size=2, test_size=1, step=3)
    folds = list(splitter.split(np.arange(9)))
    # test starts: 2, 5, 8.
    assert [f[1].tolist() for f in folds] == [[2], [5], [8]]


# -----------------------------------------------------------------------------
# Rolling window.
# -----------------------------------------------------------------------------
def test_rolling_window_drops_earliest_rows() -> None:
    """Rolling with max_train_size=3 keeps only the last 3 rows before each test."""
    splitter = WalkForwardSplit(
        initial_train_size=3, test_size=1, expanding=False, max_train_size=3
    )
    folds = list(splitter.split(np.arange(7)))
    # test starts: 3, 4, 5, 6 → each train window has length 3.
    assert [f[0].tolist() for f in folds] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert [f[1].tolist() for f in folds] == [[3], [4], [5], [6]]


def test_rolling_default_window_falls_back_to_initial_train_size() -> None:
    """max_train_size=None with expanding=False rolls at initial_train_size."""
    splitter = WalkForwardSplit(initial_train_size=4, test_size=1, expanding=False)
    folds = list(splitter.split(np.arange(7)))
    for train_idx, _ in folds:
        assert len(train_idx) == 4


# -----------------------------------------------------------------------------
# get_n_splits agrees with split().
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n, initial, test, step",
    [(6, 3, 1, 1), (10, 4, 2, 2), (9, 2, 1, 3), (5, 3, 3, 1), (5, 5, 1, 1)],
)
def test_get_n_splits_matches_enumeration(n: int, initial: int, test: int, step: int) -> None:
    splitter = WalkForwardSplit(initial_train_size=initial, test_size=test, step=step)
    expected = len(list(splitter.split(np.arange(n))))
    assert splitter.get_n_splits(np.arange(n)) == expected


# -----------------------------------------------------------------------------
# No-leakage invariant + full-fold guarantee.
# -----------------------------------------------------------------------------
def test_test_indices_are_strictly_after_train() -> None:
    """CONTEXT.md Invariant #3 as a property: max(train) < min(test) in every fold."""
    splitter = WalkForwardSplit(initial_train_size=5, test_size=3, step=2)
    for train_idx, test_idx in splitter.split(np.arange(30)):
        assert train_idx.max() < test_idx.min()
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_trailing_partial_fold_is_dropped() -> None:
    """A tail smaller than ``test_size`` is dropped — never emit partial folds."""
    splitter = WalkForwardSplit(initial_train_size=3, test_size=3)
    folds = list(splitter.split(np.arange(8)))
    # test starts at 3 (window [3, 6)); next would be at 4 (window [4, 7)); next 5 → [5, 8) fits.
    # step=1: 3, 4, 5 → three full folds. No fold at start=6 (window would be [6, 9), n=8).
    assert len(folds) == 3
    for _, test_idx in folds:
        assert len(test_idx) == 3


def test_insufficient_data_returns_no_folds() -> None:
    """If ``initial_train_size + test_size > n``, no fold fits — return empty iterator."""
    splitter = WalkForwardSplit(initial_train_size=10, test_size=1)
    assert list(splitter.split(np.arange(5))) == []
    assert splitter.get_n_splits(np.arange(5)) == 0


# -----------------------------------------------------------------------------
# sklearn-compatibility surface.
# -----------------------------------------------------------------------------
def test_split_accepts_y_and_groups_positionally() -> None:
    """Sklearn-shape ``split(X, y=None, groups=None)`` — extra args ignored, no crash."""
    splitter = WalkForwardSplit(initial_train_size=2, test_size=1)
    X = np.arange(4)
    y = pd.Series([0, 1, 0, 1])
    groups = np.zeros(4)
    folds_no_y = list(splitter.split(X))
    folds_with_y = list(splitter.split(X, y, groups))
    assert len(folds_no_y) == len(folds_with_y)
    for (t1, s1), (t2, s2) in zip(folds_no_y, folds_with_y, strict=True):
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(s1, s2)


def test_split_reads_only_length_of_X() -> None:
    """Row values are never touched — a DataFrame gives the same folds as an ndarray."""
    n = 8
    df = pd.DataFrame({"a": np.random.default_rng(12).standard_normal(n)})
    arr = np.arange(n)
    splitter = WalkForwardSplit(initial_train_size=3, test_size=1)
    df_folds = list(splitter.split(df))
    arr_folds = list(splitter.split(arr))
    assert len(df_folds) == len(arr_folds)
    for (t_df, s_df), (t_arr, s_arr) in zip(df_folds, arr_folds, strict=True):
        np.testing.assert_array_equal(t_df, t_arr)
        np.testing.assert_array_equal(s_df, s_arr)


def test_split_is_stateless_and_repeatable() -> None:
    """Re-invoking split() yields identical fold sequences."""
    splitter = WalkForwardSplit(initial_train_size=3, test_size=1)
    first = list(splitter.split(np.arange(6)))
    second = list(splitter.split(np.arange(6)))
    for (t1, s1), (t2, s2) in zip(first, second, strict=True):
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(s1, s2)


# -----------------------------------------------------------------------------
# Constructor guardrails.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_train_size": 0}, "initial_train_size"),
        ({"initial_train_size": -1}, "initial_train_size"),
        ({"initial_train_size": 3, "test_size": 0}, "test_size"),
        ({"initial_train_size": 3, "step": 0}, "step"),
        ({"initial_train_size": 3, "max_train_size": 0}, "max_train_size"),
    ],
)
def test_constructor_rejects_nonpositive_sizes(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WalkForwardSplit(**kwargs)
