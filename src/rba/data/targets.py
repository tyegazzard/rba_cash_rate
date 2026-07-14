"""Canonical target builder driven by ``src/rba/config/targets.yaml``.

Transforms a meeting-indexed frame (typically ``data/processed/master.parquet``
via ``rba.data.align`` — or any frame carrying the target's ``source_columns``)
into the aligned ``y`` vector for a single named target entry in
``targets.yaml``. The single entry point is :func:`build_target`, dispatched on
``encoding.rule`` to the three supported rules — ``sign_of_change`` (three_class),
``discrete_bps`` (magnitude_class / ordinal), and ``identity`` (delta_regression
/ level_regression) — so a §7 evaluation-harness caller can request ``y`` by
target name instead of hand-building it per model run.

Rows dropped
------------
The returned ``pd.Series`` is aligned to the *surviving* subset of ``frame.index``:

- ``sign_of_change`` / ``identity`` — drops rows where the source column is NaN.
- ``discrete_bps`` — with ``out_of_bin_strategy: drop`` (the only mode used in
  ``targets.yaml`` v1), drops rows whose source value isn't in the ``bins`` set;
  NaN rows drop for free. Other strategies are not implemented (raise).

Downstream callers align ``X`` back to ``y.index`` (typically
``X.loc[y.index]``) before feeding either into a splitter or model.

Usage::

    from rba.config import load_target_config
    from rba.data.targets import build_target

    cfg = load_target_config("three_class")
    y = build_target(master_frame, cfg)
    X = master_frame.loc[y.index, feature_cols]

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

_SUPPORTED_RULES: tuple[str, ...] = ("sign_of_change", "discrete_bps", "identity")


def build_target(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    int_labels: bool = False,
) -> pd.Series:
    """Construct the ``y`` vector for a single ``targets.yaml`` entry.

    Parameters
    ----------
    frame
        Meeting-indexed frame carrying the target's ``source_columns``.
        Typically the ``align`` master frame, but any frame with the right
        columns works (unit tests use tiny hand-built frames).
    cfg
        A target entry from :func:`rba.config.load_target_config`. Must carry
        ``encoding.rule`` (``sign_of_change`` / ``discrete_bps`` / ``identity``)
        and ``source_columns``.
    int_labels
        Classification-only: return the integer encoding from ``cfg["class_to_int"]``
        instead of the string labels from ``cfg["classes"]``. Only meaningful for
        ``sign_of_change`` (``three_class``) — ``discrete_bps`` targets are
        already integer-valued and ignore this flag; regression targets ignore it.

    Returns
    -------
    pandas.Series
        Aligned to the surviving rows of ``frame.index`` (dropped: NaN sources
        and, for ``discrete_bps``, values outside the bin set).

    Raises
    ------
    KeyError
        If any required source column is missing from ``frame``.
    ValueError
        If ``encoding.rule`` is not one of :data:`_SUPPORTED_RULES`, or if a
        ``discrete_bps`` target requests an unimplemented ``out_of_bin_strategy``.
    """
    encoding = cfg.get("encoding") or {}
    rule = encoding.get("rule")
    source_columns = list(cfg.get("source_columns") or [])

    missing = [col for col in source_columns if col not in frame.columns]
    if missing:
        raise KeyError(
            f"build_target({cfg.get('name', '?')!r}): source columns {missing} not in "
            f"frame; frame has {list(frame.columns)}."
        )

    if rule == "sign_of_change":
        y = _sign_of_change(frame, cfg, int_labels=int_labels)
    elif rule == "discrete_bps":
        y = _discrete_bps(frame, cfg, encoding)
    elif rule == "identity":
        y = _identity(frame, cfg)
    else:
        raise ValueError(
            f"build_target({cfg.get('name', '?')!r}): unknown encoding rule "
            f"{rule!r}; expected one of {_SUPPORTED_RULES}."
        )

    logger.debug(
        "build_target({!r}): rule={}, n_out={}/{}, dtype={}.",
        cfg.get("name"),
        rule,
        len(y),
        len(frame),
        y.dtype,
    )
    return y


def _sign_of_change(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    int_labels: bool,
) -> pd.Series:
    """three_class: ``sign(source)`` → -1 / 0 / +1, then optionally string-labelled.

    NaN rows drop. When ``int_labels=False`` (default) and ``cfg`` carries a
    ``class_to_int`` mapping, the ints are relabelled back to their string form
    (``-1 → 'cut'``, ``0 → 'hold'``, ``+1 → 'hike'``) — the canonical output
    per ``targets.yaml``'s ``classes:`` list.
    """
    source_col = cfg["source_columns"][0]
    values = pd.to_numeric(frame[source_col], errors="coerce").dropna()
    signs = np.sign(values.to_numpy()).astype(np.int64)
    y = pd.Series(signs, index=values.index, name=cfg.get("name", "target"))
    if int_labels or "class_to_int" not in cfg:
        return y
    int_to_label = {int(v): k for k, v in cfg["class_to_int"].items()}
    return y.map(int_to_label).astype("object").rename(cfg.get("name", "target"))


def _discrete_bps(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    encoding: Mapping[str, Any],
) -> pd.Series:
    """magnitude_class / ordinal: keep source rows whose value is an allowed bin.

    Only ``out_of_bin_strategy: drop`` is implemented — every out-of-bin row is
    dropped (NaN rows drop for free). Other strategies raise ``ValueError`` so a
    future ``nearest`` implementation can't ship silently.
    """
    strategy = encoding.get("out_of_bin_strategy", "drop")
    if strategy != "drop":
        raise ValueError(
            f"build_target({cfg.get('name', '?')!r}): out_of_bin_strategy="
            f"{strategy!r} not implemented; only 'drop' is supported."
        )
    bins = list(encoding.get("bins") or [])
    if not bins:
        raise ValueError(
            f"build_target({cfg.get('name', '?')!r}): discrete_bps rule requires "
            f"non-empty ``bins``; got {bins!r}."
        )
    source_col = cfg["source_columns"][0]
    values = pd.to_numeric(frame[source_col], errors="coerce")
    mask = values.isin(bins)
    y = values[mask].astype(np.int64)
    return y.rename(cfg.get("name", "target"))


def _identity(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """delta_regression / level_regression: numeric pass-through, dropping NaN."""
    source_col = cfg["source_columns"][0]
    values = pd.to_numeric(frame[source_col], errors="coerce").dropna().astype(np.float64)
    return values.rename(cfg.get("name", "target"))
