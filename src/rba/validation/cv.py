"""Walk-forward time-series cross-validation splitter.

The single class in this module — :class:`WalkForwardSplit` — is the project's
canonical CV splitter for §7 model evaluation (see CONTEXT.md Invariant #2: no
random k-fold). It duck-types the sklearn ``BaseCrossValidator`` protocol
(``split(X, y=None, groups=None) → Iterator[(train_idx, test_idx)]`` and
``get_n_splits(X, y=None, groups=None) → int``) so it drops into any harness
that already accepts an sklearn splitter, but is a plain class (no inheritance)
to keep the code self-contained.

Semantics
---------
Given ``n = len(X)`` chronologically-ordered rows:

- **Fold ``k = 0``** trains on rows ``[0, initial_train_size)`` and tests on
  the next ``test_size`` rows starting at ``initial_train_size``.
- **Fold ``k > 0``** advances the test window by ``step`` rows; the training
  window either **expands** (``expanding=True``, default) to include every
  earlier row, or **rolls** (``expanding=False``) to a fixed window of length
  ``max_train_size`` (or the same as ``initial_train_size`` when unset).
- Folds are emitted while the test window fits in ``[0, n)``. When the tail
  of the data is smaller than ``test_size``, the trailing partial fold is
  **dropped** — every fold is exactly ``test_size`` wide.

Contract with the §7 harness
----------------------------
- Every ``test_idx`` is strictly after every ``train_idx`` in its fold. No
  leakage (Invariant #3).
- ``get_n_splits(X)`` runs the same enumeration as ``split`` and returns the
  count without materialising the arrays — cheap for a harness that wants to
  pre-log fold count before running.
- The splitter is stateless across ``split()`` calls; re-invoking yields the
  same fold sequence.

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

from loguru import logger
import numpy as np


class WalkForwardSplit:
    """Expanding- or rolling-window walk-forward CV splitter.

    Parameters
    ----------
    initial_train_size : int
        Number of rows in the first fold's training window; the earliest
        possible test-window start. Must be ``>= 1``.
    test_size : int, default 1
        Size of every test window. Must be ``>= 1``. The default of ``1``
        matches the RBA harness's natural cadence (one meeting per fold).
    step : int, default 1
        Number of rows to advance the test-window start between folds. Must be
        ``>= 1``. ``step = test_size`` yields non-overlapping test windows;
        ``step < test_size`` yields overlapping ones (rare).
    expanding : bool, default True
        ``True`` (**default**) — the training window grows to include every
        row before the current test window (classic walk-forward).
        ``False`` — the training window is a *rolling* window of size
        ``max_train_size`` (or ``initial_train_size`` if that's ``None``).
    max_train_size : int or None, default None
        Only used when ``expanding=False``. Caps the rolling training window
        to at most this many rows. Must be ``>= 1`` when set. ``None`` defers
        to ``initial_train_size``.

    Examples
    --------
    >>> import numpy as np
    >>> splitter = WalkForwardSplit(initial_train_size=3, test_size=1)
    >>> for train, test in splitter.split(np.arange(6)):
    ...     print(train.tolist(), test.tolist())
    [0, 1, 2] [3]
    [0, 1, 2, 3] [4]
    [0, 1, 2, 3, 4] [5]
    """

    def __init__(
        self,
        initial_train_size: int,
        *,
        test_size: int = 1,
        step: int = 1,
        expanding: bool = True,
        max_train_size: int | None = None,
    ) -> None:
        if initial_train_size < 1:
            raise ValueError(f"initial_train_size must be >= 1; got {initial_train_size!r}.")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1; got {test_size!r}.")
        if step < 1:
            raise ValueError(f"step must be >= 1; got {step!r}.")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError(f"max_train_size must be None or >= 1; got {max_train_size!r}.")
        self.initial_train_size = int(initial_train_size)
        self.test_size = int(test_size)
        self.step = int(step)
        self.expanding = bool(expanding)
        self.max_train_size = None if max_train_size is None else int(max_train_size)

    # sklearn-compatible signature: (X, y=None, groups=None). y and groups are
    # ignored (walk-forward's fold layout depends only on the row count).
    def split(
        self,
        X: Sized,
        y: Any = None,  # noqa: ARG002 -- sklearn signature
        groups: Any = None,  # noqa: ARG002 -- sklearn signature
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for every fold.

        Test-window starts at ``initial_train_size`` and advances by ``step``;
        the training window either expands (default) or rolls per the
        constructor. Emits only *full* folds — a trailing tail smaller than
        ``test_size`` is dropped so every ``test_idx`` has length ``test_size``.

        Parameters
        ----------
        X
            Any ``Sized`` — the length is what matters. Row values are never
            touched (an ``np.arange(n)`` gives the same folds as the real
            ``X``).
        y, groups
            Ignored — sklearn-compatibility placeholders.

        Yields
        ------
        (train_idx, test_idx) : tuple[numpy.ndarray, numpy.ndarray]
            Integer positional indices into ``X``. ``train_idx`` is
            chronologically before ``test_idx`` (strict inequality in max/min).
        """
        n = len(X)
        for test_start in self._test_starts(n):
            test_end = test_start + self.test_size
            if self.expanding:
                train_start = 0
            else:
                window = self.max_train_size or self.initial_train_size
                train_start = max(0, test_start - window)
            train_idx = np.arange(train_start, test_start, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)
            logger.debug(
                "WalkForwardSplit fold: train=[{}, {}) test=[{}, {}).",
                train_start,
                test_start,
                test_start,
                test_end,
            )
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Sized,
        y: Any = None,  # noqa: ARG002
        groups: Any = None,  # noqa: ARG002
    ) -> int:
        """Number of folds :meth:`split` will emit for a length-``len(X)`` input.

        Runs the same enumeration as :meth:`split` without materialising the
        arrays — cheap for a harness that wants a fold-count sanity check
        before iterating.
        """
        return sum(1 for _ in self._test_starts(len(X)))

    def _test_starts(self, n: int) -> Iterator[int]:
        """Enumerate the ``test_start`` position of every full fold in a length-``n`` frame."""
        test_start = self.initial_train_size
        while test_start + self.test_size <= n:
            yield test_start
            test_start += self.step
