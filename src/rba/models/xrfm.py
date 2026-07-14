"""xRFM classifier wrapper — a **DEFERRED** §7 model stub.

This module is a placeholder for the xRFM (Recursive Feature Machine) classifier
described in ``src/rba/config/models.yaml`` under ``xrfm_classifier``. The
underlying ``xrfm`` package is **not installed** and the course RFM code has not
yet been ported, so the yaml default block carries the note
*"fill in after porting course code"* and this wrapper cannot be fit or used.

Why it exists at all
--------------------
The §7 ``build_model(cfg)`` factory resolves every model named in ``models.yaml``
by importing its class and checking it exposes the
:class:`~rba.models.base.Model` surface (``fit`` / ``predict`` /
``predict_proba`` / ``feature_names_``). :class:`XRFMClassifierWrapper` subclasses
:class:`~rba.models.base.BaseModel`, so it advertises that full interface and the
factory can resolve ``xrfm_classifier`` today — but any attempt to actually
:meth:`~rba.models.base.BaseModel.fit`, :meth:`predict`, or :meth:`predict_proba`
raises :class:`NotImplementedError` pointing back at the yaml note. Because it is
non-functional, it is **SKIPPED** in the shared-harness parametric model list
(``tests/models/test_shared_harness.py``) rather than exercised like the live
wrappers.

Porting checklist (to un-defer)
-------------------------------
1. Install / vendor the ``xrfm`` package.
2. Port the course RFM training code into :meth:`XRFMClassifierWrapper._fit`,
   :meth:`~XRFMClassifierWrapper.predict`, and
   :meth:`~XRFMClassifierWrapper.predict_proba`, following the
   :mod:`rba.models.sklearn_wrappers` house style (store ``classes_`` in
   sklearn-sorted order, preserve the original label dtype, honour
   ``sample_weight``).
3. Fill in the ``xrfm_classifier`` ``default:`` block in ``models.yaml`` and give
   :meth:`~XRFMClassifierWrapper.__init__` keyword-only params matching it.
4. Add it to the shared-harness parametric list.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rba.models.base import BaseModel

_MESSAGE = (
    "xRFM is deferred: port the course RFM code and fill the models.yaml "
    "'xrfm_classifier' default block (see the yaml note 'fill in after porting "
    "course code'). Until then this model cannot be fit or used."
)


class XRFMClassifierWrapper(BaseModel):
    """**DEFERRED** stub for the xRFM classifier — resolves but does not run.

    Registered so the §7 ``build_model`` factory can import it and confirm it
    exposes the :class:`~rba.models.base.Model` surface (``fit`` / ``predict`` /
    ``predict_proba`` / ``feature_names_``, the last three inherited from
    :class:`~rba.models.base.BaseModel`). Every operational method raises
    :class:`NotImplementedError` carrying :data:`_MESSAGE`, which points at the
    ``models.yaml`` note *"fill in after porting course code"*. It is **SKIPPED**
    in the shared-harness parametric model list because it cannot be fit.

    The constructor accepts and ignores **arbitrary keyword arguments** (the
    ``models.yaml`` ``xrfm_classifier`` default block is currently ``{}``) and
    stashes them in :attr:`_params` so the eventual port can read whatever config
    the yaml grows without changing the factory contract.

    Attributes
    ----------
    _params : dict
        The keyword arguments passed at construction, retained verbatim for the
        future port. Empty while the yaml default block is ``{}``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = dict(kwargs)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Raise — xRFM is deferred and cannot be fit.

        Reached via :meth:`~rba.models.base.BaseModel.fit`, which stores
        ``feature_names_`` before delegating here.

        Raises
        ------
        NotImplementedError
            Always, with :data:`_MESSAGE`.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        raise NotImplementedError(_MESSAGE)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raise — xRFM is deferred and cannot predict.

        Raises
        ------
        NotImplementedError
            Always, with :data:`_MESSAGE`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        raise NotImplementedError(_MESSAGE)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raise — xRFM is deferred and has no class probabilities to give.

        Raises
        ------
        NotImplementedError
            Always, with :data:`_MESSAGE`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        raise NotImplementedError(_MESSAGE)
