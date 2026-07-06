"""Model package — the :class:`Model` interface and the :func:`build_model` factory.

``models.yaml`` is the single model registry: every entry carries the ``module`` +
``class`` that names its implementation, so :func:`build_model` resolves a config
to an instance by dynamic import rather than a hardcoded name→class table here.
Re-exports the shared interface (:class:`Model` / :class:`BaseModel`) for
convenience so model authors and the harness can ``from rba.models import Model``.

Typical use (see ``models.yaml`` header)::

    from rba.config import load_model_config
    from rba.models import build_model

    cfg = load_model_config("majority_class")
    model = build_model(cfg)
    model.fit(X_train, y_train)
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from typing import Any

from loguru import logger

from rba.models.base import BaseModel, Model

__all__ = ["BaseModel", "Model", "build_model"]

# The members every model exposes (the :class:`Model` protocol surface). Checked on
# the resolved **class** — never via ``isinstance`` on a fresh instance, because
# ``feature_names_`` raises until ``fit`` and a ``runtime_checkable`` isinstance
# would trip that property getter on an unfitted model.
_REQUIRED_MEMBERS: tuple[str, ...] = ("fit", "predict", "predict_proba", "feature_names_")


def build_model(cfg: Mapping[str, Any]) -> Model:
    """Instantiate the model described by a ``models.yaml`` entry.

    Resolves ``cfg['module']`` + ``cfg['class']`` via :func:`importlib.import_module`
    and instantiates the class with ``cfg['default']`` as keyword arguments — so
    ``models.yaml`` stays the single registry and this factory carries no hardcoded
    model table.

    Parameters
    ----------
    cfg
        A single model entry, e.g. from :func:`rba.config.load_model_config`. Must
        carry ``module`` and ``class``; ``default`` (a kwargs mapping) is optional.

    Returns
    -------
    Model
        A freshly-constructed, **unfitted** model satisfying the :class:`Model`
        protocol.

    Raises
    ------
    KeyError
        If ``cfg`` lacks ``module`` or ``class``.
    TypeError
        If the resolved class does not expose the :class:`Model` interface.
    """
    try:
        module_path = cfg["module"]
        class_name = cfg["class"]
    except KeyError as exc:
        raise KeyError(
            f"Model config missing required key {exc}; got keys {sorted(cfg)}."
        ) from exc

    default_params = dict(cfg.get("default") or {})
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    missing = [member for member in _REQUIRED_MEMBERS if not hasattr(model_cls, member)]
    if missing:
        raise TypeError(
            f"{module_path}.{class_name} does not implement the Model interface; "
            f"missing {missing}."
        )

    model: Model = model_cls(**default_params)
    logger.debug(
        "Built model {!r} from {}.{} with params {}.",
        cfg.get("name", class_name),
        module_path,
        class_name,
        default_params,
    )
    return model
