"""Tests for ``rba.models.xrfm`` — the DEFERRED xRFM classifier stub.

No network; tiny inline ``(X, y)`` fixtures. The stub must expose the full
:class:`~rba.models.base.Model` surface (``fit`` / ``predict`` /
``predict_proba`` / ``feature_names_``) so the §7 ``build_model`` factory's
interface check passes, must instantiate (with and without arbitrary kwargs,
which it retains in ``_params``), yet every operational call
(``fit`` / ``predict`` / ``predict_proba``) must raise
:class:`NotImplementedError` whose message points at the ``models.yaml`` note
*"fill in after porting course code"*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.models.xrfm import XRFMClassifierWrapper


def _X(n: int) -> pd.DataFrame:
    """A dummy feature frame — the stub never reads it, only its columns matter."""
    return pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float)})


def _y(n: int) -> pd.Series:
    """A tiny two-class target aligned to ``_X``."""
    return pd.Series((["hold", "hike"] * n)[:n], name="target")


# -----------------------------------------------------------------------------
# Interface presence — build_model's factory check must pass on the class.
# -----------------------------------------------------------------------------
def test_class_exposes_the_full_model_surface() -> None:
    """hasattr on the class (not an instance) — what build_model probes."""
    for member in ("fit", "predict", "predict_proba", "feature_names_"):
        assert hasattr(XRFMClassifierWrapper, member)


# -----------------------------------------------------------------------------
# Instantiation — with and without arbitrary kwargs.
# -----------------------------------------------------------------------------
def test_instantiation_without_kwargs() -> None:
    model = XRFMClassifierWrapper()
    assert model._params == {}


def test_instantiation_ignores_and_stores_arbitrary_kwargs() -> None:
    model = XRFMClassifierWrapper(bandwidth=5.0, reg=0.1, whatever="x")
    assert model._params == {"bandwidth": 5.0, "reg": 0.1, "whatever": "x"}


# -----------------------------------------------------------------------------
# Every operational call raises NotImplementedError pointing at the yaml note.
# -----------------------------------------------------------------------------
def test_fit_raises_not_implemented() -> None:
    """BaseModel.fit stores feature_names_ then calls _fit, which raises."""
    X, y = _X(4), _y(4)
    with pytest.raises(NotImplementedError, match="porting course code"):
        XRFMClassifierWrapper().fit(X, y)


def test_predict_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="porting course code"):
        XRFMClassifierWrapper().predict(_X(3))


def test_predict_proba_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="porting course code"):
        XRFMClassifierWrapper().predict_proba(_X(3))


def test_message_flags_the_model_as_deferred() -> None:
    """The raised message should identify the deferral, not a generic error."""
    with pytest.raises(NotImplementedError, match="deferred"):
        XRFMClassifierWrapper().fit(_X(4), _y(4))
