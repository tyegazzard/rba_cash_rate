"""Tests for the model factory — ``rba.config.load_model_config`` + ``build_model``.

No network. Round-trips the real ``majority_class`` entry from ``models.yaml``
end to end (config → instance → fit → predict), and covers the factory's error
paths (unknown name, the ``defaults`` non-model, missing keys, a non-model class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import load_model_config
from rba.models import Model, build_model
from rba.models.baselines import MajorityClass


# -----------------------------------------------------------------------------
# load_model_config.
# -----------------------------------------------------------------------------
def test_load_model_config_returns_entry_with_name() -> None:
    cfg = load_model_config("majority_class")
    assert cfg["module"] == "rba.models.baselines"
    assert cfg["class"] == "MajorityClass"
    assert cfg["task"] == "classification"
    assert cfg["name"] == "majority_class"


def test_load_model_config_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="No model named"):
        load_model_config("no_such_model")


def test_load_model_config_defaults_section_is_not_a_model() -> None:
    with pytest.raises(KeyError, match="No model named"):
        load_model_config("defaults")


# -----------------------------------------------------------------------------
# build_model — resolution + end-to-end round-trip.
# -----------------------------------------------------------------------------
def test_build_model_resolves_majority_class() -> None:
    model = build_model(load_model_config("majority_class"))
    assert isinstance(model, MajorityClass)


def test_build_model_round_trip_fits_and_predicts() -> None:
    model = build_model(load_model_config("majority_class"))
    X = pd.DataFrame({"f0": np.arange(6, dtype=float)})
    y = pd.Series(["hold"] * 4 + ["hike"] * 2)
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    assert (fitted.predict(X) == "hold").all()
    assert fitted.predict_proba(X).shape == (6, 2)


def test_build_model_missing_keys_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="missing required key"):
        build_model({"class": "MajorityClass"})  # no 'module'


def test_build_model_non_model_class_raises_typeerror() -> None:
    """A resolvable class that isn't a Model is rejected with a clear message."""
    with pytest.raises(TypeError, match="does not implement the Model interface"):
        build_model({"module": "builtins", "class": "object"})
