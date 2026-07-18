"""scikit-learn classifier wrappers — the §7 linear / tree / kernel / neural models.

Four :class:`~rba.models.base.Model` implementations registered in
``src/rba/config/models.yaml`` under ``rba.models.sklearn_wrappers``:
:class:`LogisticRegressionWrapper` (``logistic_regression``),
:class:`RandomForestClassifierWrapper` (``random_forest_classifier``),
:class:`SVMClassifierWrapper` (``svm_classifier``), and
:class:`MLPClassifierWrapper` (``mlp_classifier``). Each subclasses
:class:`~rba.models.base.BaseModel`, stores ``classes_`` in sklearn-sorted order
(the column order of :meth:`predict_proba`), preserves the original label dtype,
and honours ``sample_weight`` — so the §7 harness drives them exactly like a
baseline.

Two NaN strategies
------------------
The three **dense** classifiers — linear (:class:`LogisticRegressionWrapper`),
kernel (:class:`SVMClassifierWrapper`), and neural
(:class:`MLPClassifierWrapper`) — cannot ingest a NaN matrix and are
scale-sensitive, so each is fit inside a per-fold
:class:`~sklearn.pipeline.Pipeline`
``[FfillImputer -> StandardScaler -> <estimator>]``. This realises
``features.yaml``'s documented ``dense_fallback: ffill`` **leakage-safe**: the
:class:`~rba.models._preprocessing.FfillImputer` and the
:class:`~sklearn.preprocessing.StandardScaler` are both fit **on the training
fold only** (CONTEXT.md Invariant #3 — scaler fit on train only — and Invariant
#4 — imputation inside the Pipeline, per fold). Shared machinery lives in the
private :class:`_DensePipelineClassifier` base.

:class:`RandomForestClassifierWrapper` is **different**: forests split on missing
values natively (scikit-learn 1.4+ missing-value splits — verified on 1.8) and
trees are scale-invariant, so it uses **no** Pipeline — ``X`` is handed to the
forest untouched.

No network anywhere in this module.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from rba.config import RANDOM_SEED
from rba.models._preprocessing import FfillImputer
from rba.models.base import BaseModel


class _DensePipelineClassifier(BaseModel):
    """Shared base for the NaN-intolerant, scale-sensitive sklearn classifiers.

    The linear / kernel / neural estimators (:class:`LogisticRegressionWrapper`,
    :class:`SVMClassifierWrapper`, :class:`MLPClassifierWrapper`) all fit the same
    per-fold :class:`~sklearn.pipeline.Pipeline`
    ``[('impute', FfillImputer()), ('scale', StandardScaler()), ('clf', ...)]`` and
    differ only in the final estimator, which each subclass supplies from
    :meth:`_build_estimator`. Building the pipeline inside :meth:`_fit` (rather than
    the constructor) is what makes the imputer and scaler fit on the **training
    fold only** — the ``dense_fallback: ffill`` dense fill realised leakage-safe
    (CONTEXT.md Invariants #3 and #4).

    Labels are integer-encoded (via :class:`~sklearn.preprocessing.LabelEncoder`)
    before the internal estimator sees them, and decoded back on :meth:`predict`.
    This is **required** for :class:`~sklearn.neural_network.MLPClassifier` with
    ``early_stopping=True`` (the ``models.yaml`` default), whose internal validation
    scoring runs :func:`numpy.isnan` on the predictions and therefore rejects
    non-numeric (string) labels in scikit-learn 1.8; it is a transparent no-op for
    the linear / kernel estimators. ``LabelEncoder`` codes are assigned in
    sorted-label order — exactly the sklearn class ordering the ``predict_proba``
    columns already follow — so :attr:`classes_` stays sklearn-sorted, the proba
    columns stay aligned to it, and :meth:`predict` returns the **original** label
    values / dtype.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels (sklearn convention); the column order of
        :meth:`predict_proba`.
    """

    classes_: np.ndarray

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return a fresh, unfitted final-step estimator for the Pipeline."""
        ...

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit ``[FfillImputer -> StandardScaler -> estimator]`` on this fold.

        The imputer and scaler are fit on ``X`` alone (the training window), so no
        held-out statistic leaks in. Labels are integer-encoded first (see the class
        docstring); ``sample_weight`` is routed to the final estimator via the
        ``clf__sample_weight`` Pipeline convention.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._label_encoder = LabelEncoder()
        y_codes = self._label_encoder.fit_transform(np.asarray(y))
        pipe = Pipeline(
            [
                ("impute", FfillImputer()),
                ("scale", StandardScaler()),
                ("clf", self._build_estimator()),
            ]
        )
        fit_params = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
        self._pipeline = pipe.fit(X, y_codes, **fit_params)
        self.classes_ = self._label_encoder.classes_
        logger.debug(
            "{} fitted: {} class(es) {}, weighted={}.",
            type(self).__name__,
            len(self.classes_),
            self.classes_.tolist(),
            sample_weight is not None,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class label for each row of ``X``.

        The internal estimator predicts integer codes; they are decoded back to the
        original ``y`` values / dtype via the fitted
        :class:`~sklearn.preprocessing.LabelEncoder`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        codes = self._pipeline.predict(X)
        return self._label_encoder.inverse_transform(codes)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates, columns aligned to :attr:`classes_`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._pipeline.predict_proba(X)


class LogisticRegressionWrapper(_DensePipelineClassifier):
    """Multinomial logistic-regression classifier (linear §7 model).

    Fit inside the dense per-fold Pipeline
    ``[FfillImputer -> StandardScaler -> LogisticRegression]``: the linear model
    cannot ingest NaN and its L2 penalty is scale-sensitive, so the leakage-safe
    ``dense_fallback: ffill`` imputation and the standardisation are both fit on the
    training fold only (see :class:`_DensePipelineClassifier`).

    Parameters
    ----------
    penalty : str, default ``"l2"``
        Regularisation norm. ``'l1'`` / ``'elasticnet'`` are unsupported by the
        default ``lbfgs`` solver, so when one is requested with an incompatible
        solver the wrapper transparently switches to ``'saga'`` (which supports
        every penalty). This lets the hyperparameter search sweep ``penalty`` over
        ``[l1, l2]`` (per ``models.yaml``) without also having to co-vary the solver.
    C : float, default 1.0
        Inverse regularisation strength (smaller ⇒ stronger regularisation).
    class_weight : str, dict or None, default ``"balanced"``
        Passed to :class:`~sklearn.linear_model.LogisticRegression` as-is
        (``'balanced'`` reweights inversely to class frequency).
    max_iter : int, default 1000
        Maximum solver iterations. A non-converged fit warns but still predicts.
    solver : str, default ``"lbfgs"``
        Optimisation algorithm.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for the (stochastic) solvers.
    """

    def __init__(
        self,
        *,
        penalty: str = "l2",
        C: float = 1.0,
        class_weight: Any = "balanced",
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._penalty = penalty
        self._C = float(C)
        self._class_weight = class_weight
        self._max_iter = int(max_iter)
        self._solver = solver
        self._random_state = random_state

    # Solvers that cannot fit an L1 / elastic-net penalty; a request for those
    # penalties with one of these solvers is auto-upgraded to ``saga``.
    _L2_ONLY_SOLVERS = frozenset({"lbfgs", "newton-cg", "newton-cholesky", "sag"})

    def _build_estimator(self) -> LogisticRegression:
        """Return an unfitted :class:`~sklearn.linear_model.LogisticRegression`.

        Auto-selects ``saga`` when the requested ``penalty`` (``l1`` / ``elasticnet``)
        is incompatible with the configured solver, so the search can sweep
        ``penalty`` freely.
        """
        solver = self._solver
        if self._penalty in ("l1", "elasticnet") and solver in self._L2_ONLY_SOLVERS:
            logger.debug(
                "LogisticRegressionWrapper: penalty={!r} needs an L1-capable solver; "
                "switching {!r} -> 'saga'.",
                self._penalty,
                solver,
            )
            solver = "saga"
        return LogisticRegression(
            penalty=self._penalty,
            C=self._C,
            class_weight=self._class_weight,
            max_iter=self._max_iter,
            solver=solver,
            random_state=self._random_state,
        )


class RandomForestClassifierWrapper(BaseModel):
    """Random-forest classifier (tree §7 model) — no imputation or scaling.

    Unlike the dense wrappers, a forest **handles NaN natively** (scikit-learn
    1.4+ missing-value splits — verified on 1.8) and is scale-invariant, so there
    is **no** Pipeline: ``X`` is handed to
    :class:`~sklearn.ensemble.RandomForestClassifier` untouched (NaN cells and all).

    Parameters
    ----------
    n_estimators : int, default 500
        Number of trees in the forest.
    max_depth : int or None, default None
        Maximum tree depth; ``None`` grows trees until leaves are pure or hit
        ``min_samples_leaf``.
    min_samples_leaf : int, default 5
        Minimum samples required at a leaf — a mild regulariser.
    class_weight : str, dict or None, default ``"balanced"``
        Passed through as-is to the forest.
    n_jobs : int, default -1
        Parallel jobs for fitting / prediction (``-1`` ⇒ all cores).
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed controlling the bootstrap and per-split feature sampling.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels (sklearn convention); the column order of
        :meth:`predict_proba`.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        class_weight: Any = "balanced",
        n_jobs: int = -1,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._n_estimators = int(n_estimators)
        self._max_depth = None if max_depth is None else int(max_depth)
        self._min_samples_leaf = int(min_samples_leaf)
        self._class_weight = class_weight
        self._n_jobs = int(n_jobs)
        self._random_state = random_state

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the forest directly on ``X`` (NaN handled natively; no scaling).

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            class_weight=self._class_weight,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        logger.debug(
            "{} fitted: {} tree(s), {} class(es) {}, weighted={}.",
            type(self).__name__,
            self._n_estimators,
            len(self.classes_),
            self.classes_.tolist(),
            sample_weight is not None,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class label for each row of ``X`` (original label dtype).

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates, columns aligned to :attr:`classes_`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._model.predict_proba(X)


class SVMClassifierWrapper(_DensePipelineClassifier):
    """Support-vector classifier (kernel §7 model).

    Fit inside the dense per-fold Pipeline
    ``[FfillImputer -> StandardScaler -> SVC]``: the kernel is highly
    scale-sensitive (RBF distances) and cannot ingest NaN, so the leakage-safe
    ``dense_fallback: ffill`` imputation and standardisation are fit on the training
    fold only (see :class:`_DensePipelineClassifier`).

    ``probability=True`` is **required** for :meth:`predict_proba` — it enables
    :class:`~sklearn.svm.SVC`'s internal Platt-scaling cross-validation. A known
    scikit-learn caveat: the Platt-scaled ``predict_proba`` can mildly disagree with
    ``predict`` (a hard-margin ``decision_function`` argmax) for points near the
    decision boundary. This is accepted; the harness scores probabilities from
    ``predict_proba`` and hard labels from ``predict``.

    Parameters
    ----------
    kernel : str, default ``"rbf"``
        Kernel type.
    C : float, default 1.0
        Inverse regularisation strength (penalty on margin violations).
    gamma : str or float, default ``"scale"``
        Kernel coefficient; ``'scale'`` / ``'auto'`` or a positive float.
    class_weight : str, dict or None, default ``"balanced"``
        Passed through to :class:`~sklearn.svm.SVC` as-is.
    probability : bool, default True
        Enable Platt-scaled probability estimates (needed for
        :meth:`predict_proba`).
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for the probability-calibration shuffling.
    """

    def __init__(
        self,
        *,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Any = "scale",
        class_weight: Any = "balanced",
        probability: bool = True,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._kernel = kernel
        self._C = float(C)
        self._gamma = gamma
        self._class_weight = class_weight
        self._probability = bool(probability)
        self._random_state = random_state

    def _build_estimator(self) -> SVC:
        """Return an unfitted :class:`~sklearn.svm.SVC`."""
        return SVC(
            kernel=self._kernel,
            C=self._C,
            gamma=self._gamma,
            class_weight=self._class_weight,
            probability=self._probability,
            random_state=self._random_state,
        )


class MLPClassifierWrapper(_DensePipelineClassifier):
    """Multi-layer-perceptron classifier (neural §7 model).

    Fit inside the dense per-fold Pipeline
    ``[FfillImputer -> StandardScaler -> MLPClassifier]``: the network cannot ingest
    NaN and trains far better on standardised inputs, so the leakage-safe
    ``dense_fallback: ffill`` imputation and standardisation are fit on the training
    fold only (see :class:`_DensePipelineClassifier`).

    :class:`~sklearn.neural_network.MLPClassifier` has **no** ``class_weight``
    argument, so class imbalance is handled by the ``sample_weight`` the harness
    passes to :meth:`~rba.models.base.BaseModel.fit`, which **is** forwarded to the
    network (``sample_weight`` supported in scikit-learn 1.8). With
    ``early_stopping=True`` the estimator holds out an internal stratified
    validation slice and stops when its score stops improving.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default ``(64, 32)``
        Width of each hidden layer. Coerced to a tuple of ``int`` (``models.yaml``
        supplies a list ``[64, 32]``).
    activation : str, default ``"relu"``
        Hidden-layer activation function.
    alpha : float, default 0.0001
        L2 regularisation strength.
    learning_rate_init : float, default 0.001
        Initial learning rate for the Adam optimiser.
    max_iter : int, default 300
        Maximum training epochs. A non-converged fit warns but still predicts.
    early_stopping : bool, default True
        Hold out an internal validation slice and stop early on it.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for weight initialisation and the internal validation split.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: Any = (64, 32),
        activation: str = "relu",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        early_stopping: bool = True,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self._activation = activation
        self._alpha = float(alpha)
        self._learning_rate_init = float(learning_rate_init)
        self._max_iter = int(max_iter)
        self._early_stopping = bool(early_stopping)
        self._random_state = random_state

    def _build_estimator(self) -> MLPClassifier:
        """Return an unfitted :class:`~sklearn.neural_network.MLPClassifier`."""
        return MLPClassifier(
            hidden_layer_sizes=self._hidden_layer_sizes,
            activation=self._activation,
            alpha=self._alpha,
            learning_rate_init=self._learning_rate_init,
            max_iter=self._max_iter,
            early_stopping=self._early_stopping,
            random_state=self._random_state,
        )
