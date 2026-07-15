"""Loughran-McDonald (LM) lexicon scoring — first ``§5`` text feature.

Scores each RBA document (post-meeting statements, minutes, SoMP overviews,
speeches later) against the LM sentiment dictionary and emits per-**document**
features. Meeting-frame point-in-time aggregation is **deliberately deferred** to
a shared text-align step (common to LM + the later custom hawk/dove lexicon +
embeddings), so this module stops at the document level and carries each
document's ``publication_date`` through for that later as-of join.

What it computes
----------------
For each of the seven standard LM categories (Negative, Positive, Uncertainty,
Litigious, Strong_Modal, Weak_Modal, Constraining), per document:

- ``lm_<category>_count`` — raw number of tokens in that category.
- ``lm_<category>_normalized_count`` — category tokens per 1000 total tokens
  (a scale-free frequency, robust to document length).

Plus two document-level features:

- ``lm_total_words`` — total tokenized word count (document length).
- ``lm_tone`` — polarity ``(positive − negative) / (positive + negative)``.
  LM literature warns raw *Positive* counts are frequently negated, so tone is a
  documented, optional composite kept separate from the raw category counts. The
  degenerate case ``positive + negative == 0`` (no determinable directional
  signal) yields ``NaN`` — never ``±inf`` and never a misleading ``0`` — so a
  document with no polarity words is not silently scored neutral.

``ratio`` (category / total, a fraction) is also supported by the aggregation
dispatch for completeness, but is **not** enabled in ``features.yaml`` — it is
redundant given ``count`` + ``normalized_count``.

Empty / whitespace text
-----------------------
A document with no tokens scores every ``count`` and ``normalized_count`` as
``0`` (not ``NaN``, not a crash); ``lm_total_words`` is ``0`` and ``lm_tone`` is
``NaN`` (undefined polarity, per the degenerate rule above).

Leakage safety (Invariant #1)
-----------------------------
Scoring is a pure function of a single document's text — no cross-document or
cross-meeting information enters, so it is leakage-safe by construction. The
strict ``publication_date < meeting_date`` rule is applied later, at the shared
text-align step, exactly as :func:`rba.data.align.as_of_levels` does for numeric
series (statements/minutes/SoMP: most-recent prior document; speeches:
many-to-one prior aggregation). This module only *carries* ``publication_date``.

No network: the LM dictionary is read from the local cache populated once by
:mod:`rba.features.text.lexicon_data`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR, load_features_config
from rba.features.text import lexicon_data
from rba.features.text.lexicon_data import LM_CATEGORIES, load_lm_lexicon

# Default per-category aggregations (``features.yaml`` text.lexicon.aggregations).
# ``ratio`` is intentionally omitted (redundant given count + normalized_count).
DEFAULT_AGGREGATIONS: tuple[str, ...] = ("count", "normalized_count")

# Tokens-per-N scale for the ``normalized_count`` aggregation (per 1000 words).
_PER_N_WORDS = 1000.0

# Feature-name prefix + the two document-level composite feature names.
_FEATURE_PREFIX = "lm"
_TOTAL_WORDS_FEATURE = "lm_total_words"
_TONE_FEATURE = "lm_tone"

# The two categories tone is derived from.
_POSITIVE = "positive"
_NEGATIVE = "negative"

# Deterministic tokenizer: maximal runs of ASCII letters, case-folded. Numbers,
# punctuation and markup (e.g. "4.35", "per cent", "&nbsp;") never match, so the
# token stream is stable regardless of the document's formatting.
_TOKEN_RE = re.compile(r"[a-z]+")

# Only the LM lexicon is in scope for this builder; other configured lexicons
# (e.g. the not-yet-built custom hawk/dove dictionary) are warn-skipped.
_LM_LEXICON_NAME = "loughran_mcdonald"

# Identifier / metadata columns emitted per document, ahead of the score columns.
_DOCUMENT_SOURCE = "document_source"
_DECISION_DATE = "decision_date"
_PUBLICATION_DATE = "publication_date"
_DOCUMENT_ID = "document_id"
_ID_COLUMNS: tuple[str, ...] = (_DOCUMENT_SOURCE, _DECISION_DATE, _PUBLICATION_DATE, _DOCUMENT_ID)

# Document-frame columns the builder reads (per the text-source Parquet schema).
_BODY_COLUMN = "body_text"
_URL_COLUMN = "url"


# -----------------------------------------------------------------------------
# Pure scorer.
# -----------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercased alphabetic tokens (deterministic).

    Shapes
    ------
    Returns: (n_tokens,) — one entry per maximal run of ASCII letters.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def count_categories(
    tokens: Sequence[str],
    lexicon: Mapping[str, frozenset[str]],
    *,
    categories: Sequence[str] = LM_CATEGORIES,
) -> dict[str, int]:
    """Count how many ``tokens`` fall in each requested LM category.

    Parameters
    ----------
    tokens
        Tokenized document (see :func:`tokenize`).
    lexicon
        ``{word: frozenset(category, ...)}`` from
        :func:`rba.features.text.lexicon_data.load_lm_lexicon`.
    categories
        Categories to count (defaults to all seven LM categories).

    Returns
    -------
    dict[str, int]
        ``{category: raw_count}`` for every requested category (0 where none).
    """
    counts = {category: 0 for category in categories}
    for token in tokens:
        member = lexicon.get(token)
        if not member:
            continue
        for category in member:
            if category in counts:
                counts[category] += 1
    return counts


def score_text(
    text: str,
    lexicon: Mapping[str, frozenset[str]],
    *,
    categories: Sequence[str] = LM_CATEGORIES,
    aggregations: Sequence[str] = DEFAULT_AGGREGATIONS,
    emit_tone: bool = True,
) -> dict[str, float]:
    """Score one document against the LM lexicon.

    Pure and case-insensitive; out-of-lexicon tokens are ignored. Empty or
    whitespace-only ``text`` yields all-zero counts / normalized counts,
    ``lm_total_words == 0`` and ``lm_tone == NaN`` (see module docstring).

    Parameters
    ----------
    text
        The document text (e.g. a Parquet ``body_text`` cell).
    lexicon
        ``{word: frozenset(category, ...)}`` mapping.
    categories
        LM categories to score (defaults to all seven).
    aggregations
        Per-category aggregations to emit — any of ``count`` /
        ``normalized_count`` / ``ratio`` (defaults to count + normalized_count).
    emit_tone
        Whether to emit ``lm_tone`` (requires ``positive`` and ``negative`` in
        ``categories``; otherwise skipped).

    Returns
    -------
    dict[str, float]
        ``{feature_name: value}`` in a deterministic order:
        ``lm_<category>_<aggregation>`` for each category × aggregation, then
        ``lm_total_words``, then (optionally) ``lm_tone``.
    """
    tokens = tokenize(text)
    total = len(tokens)
    counts = count_categories(tokens, lexicon, categories=categories)

    scores: dict[str, float] = {}
    for category in categories:
        count = counts[category]
        for aggregation in aggregations:
            name = f"{_FEATURE_PREFIX}_{category}_{aggregation}"
            scores[name] = _aggregate(count, total, aggregation)

    scores[_TOTAL_WORDS_FEATURE] = float(total)

    if emit_tone:
        if _POSITIVE in counts and _NEGATIVE in counts:
            scores[_TONE_FEATURE] = _tone(counts[_POSITIVE], counts[_NEGATIVE])
        else:
            logger.debug(
                "Tone requested but 'positive'/'negative' not both scored; skipping lm_tone."
            )
    return scores


def _aggregate(count: int, total: int, aggregation: str) -> float:
    """Map a raw category ``count`` to the requested aggregation value.

    ``count`` → raw count; ``normalized_count`` → per-1000-words;
    ``ratio`` → fraction of total. A zero-token document yields ``0.0`` for the
    frequency forms (never a divide-by-zero).
    """
    if aggregation == "count":
        return float(count)
    if aggregation == "normalized_count":
        return (count / total) * _PER_N_WORDS if total else 0.0
    if aggregation == "ratio":
        return count / total if total else 0.0
    raise ValueError(
        f"Unknown LM aggregation {aggregation!r}; expected one of "
        f"count / normalized_count / ratio."
    )


def _tone(positive: int, negative: int) -> float:
    """LM polarity ``(pos − neg) / (pos + neg)``; ``NaN`` when ``pos + neg == 0``.

    The degenerate case means the document carried no directional polarity words,
    so tone is *undefined* rather than neutral — returning ``NaN`` avoids a
    misleading ``0`` and never blows up to ``±inf``.
    """
    denominator = positive + negative
    if denominator == 0:
        return float("nan")
    return (positive - negative) / denominator


# -----------------------------------------------------------------------------
# Config accessors (mirror rba.features.{lags,rolling,changes,surprises}).
# -----------------------------------------------------------------------------
def _text_section(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("text", {}) or {})


def _lexicon_section(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_text_section(config).get("lexicon", {}) or {})


def config_lexicon_enabled(config: Mapping[str, Any]) -> bool:
    """Whether ``text.lexicon`` (and the parent ``text``) group is enabled."""
    text_enabled = bool(_text_section(config).get("enabled", False))
    lexicon_enabled = bool(_lexicon_section(config).get("enabled", False))
    return text_enabled and lexicon_enabled


def config_documents(config: Mapping[str, Any]) -> list[str]:
    """The document artifacts to score, from ``text.documents``.

    Reconciled to the real ``data/external/`` Parquet names in ``features.yaml``
    (``rba_media_releases`` / ``rba_minutes`` / ``rba_somp`` / ``rba_speeches``);
    absent artifacts are warn-skipped at load time.
    """
    return [str(name) for name in (_text_section(config).get("documents", []) or [])]


def config_aggregations(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Per-category aggregations, from ``text.lexicon.aggregations`` or default."""
    aggregations = _lexicon_section(config).get("aggregations")
    if not aggregations:
        return DEFAULT_AGGREGATIONS
    return tuple(str(aggregation) for aggregation in aggregations)


def config_emit_tone(config: Mapping[str, Any]) -> bool:
    """Whether to emit ``lm_tone``, from ``text.lexicon.emit_tone`` (default True)."""
    section = _lexicon_section(config)
    return bool(section.get("emit_tone", True))


def config_lexicon_names(config: Mapping[str, Any]) -> list[str]:
    """Configured lexicon names, from ``text.lexicon.lexicons`` (default LM only)."""
    names = _lexicon_section(config).get("lexicons")
    if not names:
        return [_LM_LEXICON_NAME]
    return [str(name) for name in names]


# -----------------------------------------------------------------------------
# Pure builder over document frames.
# -----------------------------------------------------------------------------
def build_lm_scores(
    documents: Mapping[str, pd.DataFrame],
    lexicon: Mapping[str, frozenset[str]],
    *,
    categories: Sequence[str] = LM_CATEGORIES,
    aggregations: Sequence[str] = DEFAULT_AGGREGATIONS,
    emit_tone: bool = True,
) -> pd.DataFrame:
    """Score every document in every source frame — one output row per document.

    Pure (given ``documents`` + ``lexicon``): no network, no disk, no meeting
    alignment. Each source frame must carry ``body_text`` and ``publication_date``
    (the text-source Parquet schema); ``decision_date`` and ``url`` are carried
    through when present (``url`` becomes ``document_id``). A ``NaN`` / missing
    body is scored as empty text.

    Parameters
    ----------
    documents
        ``{document_source: frame}`` — e.g. the ``rba_media_releases`` /
        ``rba_minutes`` / ``rba_somp`` Parquet frames.
    lexicon
        ``{word: frozenset(category, ...)}`` mapping.
    categories, aggregations, emit_tone
        Passed through to :func:`score_text`.

    Returns
    -------
    pandas.DataFrame
        One row per document, sorted by ``publication_date``. Columns:
        ``document_source``, ``decision_date``, ``publication_date``,
        ``document_id``, then the :func:`score_text` feature columns. Empty when
        no source contributes a document.

    Shapes
    ------
    Returns: (n_documents, 4 + n_score_features).
    """
    rows: list[dict[str, Any]] = []
    for source_name, frame in documents.items():
        if _BODY_COLUMN not in frame.columns or _PUBLICATION_DATE not in frame.columns:
            logger.warning(
                "LM: document source {!r} lacks {!r}/{!r}; skipping.",
                source_name,
                _BODY_COLUMN,
                _PUBLICATION_DATE,
            )
            continue
        for record in frame.itertuples(index=False):
            body = getattr(record, _BODY_COLUMN, None)
            text = "" if body is None or (isinstance(body, float) and pd.isna(body)) else str(body)
            row: dict[str, Any] = {
                _DOCUMENT_SOURCE: source_name,
                _DECISION_DATE: getattr(record, _DECISION_DATE, pd.NaT),
                _PUBLICATION_DATE: getattr(record, _PUBLICATION_DATE),
                _DOCUMENT_ID: getattr(record, _URL_COLUMN, pd.NA),
            }
            row.update(
                score_text(
                    text,
                    lexicon,
                    categories=categories,
                    aggregations=aggregations,
                    emit_tone=emit_tone,
                )
            )
            rows.append(row)

    feature_names = _feature_names(categories, aggregations, emit_tone)
    columns = [*_ID_COLUMNS, *feature_names]
    if not rows:
        logger.info("LM: no documents scored (no sources present).")
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows, columns=columns)
    out[_PUBLICATION_DATE] = pd.to_datetime(out[_PUBLICATION_DATE])
    out = out.sort_values(_PUBLICATION_DATE, kind="stable").reset_index(drop=True)
    logger.info(
        "LM: scored {} documents across {} source(s); {} feature columns.",
        len(out),
        out[_DOCUMENT_SOURCE].nunique(),
        len(feature_names),
    )
    return out


def _feature_names(
    categories: Sequence[str],
    aggregations: Sequence[str],
    emit_tone: bool,
) -> list[str]:
    """The score-feature column names, in the same order :func:`score_text` emits."""
    names = [
        f"{_FEATURE_PREFIX}_{category}_{aggregation}"
        for category in categories
        for aggregation in aggregations
    ]
    names.append(_TOTAL_WORDS_FEATURE)
    if emit_tone and _POSITIVE in categories and _NEGATIVE in categories:
        names.append(_TONE_FEATURE)
    return names


# -----------------------------------------------------------------------------
# Orchestration: load cached documents + dictionary, then score (no network).
# -----------------------------------------------------------------------------
def load_document_frames(
    *,
    config: Mapping[str, Any] | None = None,
    documents: Sequence[str] | None = None,
    external_dir: Path = EXTERNAL_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Read the configured document Parquet artifacts from ``data/external/``.

    Network-free. Reconciled real artifact names come from ``features.yaml``
    ``text.documents``; any artifact whose Parquet is absent (e.g. the
    not-yet-refreshed ``rba_speeches``) is warn-skipped, never fatal.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``{document_source: frame}`` for every artifact present on disk.
    """
    if documents is None:
        config = config if config is not None else load_features_config()
        documents = config_documents(config)

    frames: dict[str, pd.DataFrame] = {}
    for name in documents:
        path = external_dir / f"{name}.parquet"
        if not path.exists():
            logger.warning("LM: document artifact {!r} absent at {}; skipping.", name, path)
            continue
        frames[name] = pd.read_parquet(path)
    logger.info("LM: loaded {} of {} configured document source(s).", len(frames), len(documents))
    return frames


def build_lm_scores_from_cache(
    *,
    config: Mapping[str, Any] | None = None,
    external_dir: Path = EXTERNAL_DATA_DIR,
    lexicon_path: Path = lexicon_data.LM_ARTIFACT_PATH,
) -> pd.DataFrame:
    """End-to-end per-document LM scores from cached artifacts (read-only).

    Loads the document Parquets + the LM dictionary from ``data/external/`` and
    runs :func:`build_lm_scores`. If the LM dictionary cache is absent, warns and
    returns an empty frame (graceful — run the
    :mod:`rba.features.text.lexicon_data` download step first) rather than
    raising.

    Returns
    -------
    pandas.DataFrame
        The per-document score frame (see :func:`build_lm_scores`); empty if the
        dictionary is unavailable or no documents are present.
    """
    config = config if config is not None else load_features_config()

    lexicon_names = config_lexicon_names(config)
    if _LM_LEXICON_NAME not in lexicon_names:
        logger.warning(
            "LM: {!r} not in configured lexicons {}; nothing to do.",
            _LM_LEXICON_NAME,
            lexicon_names,
        )

    if not lexicon_data.lm_dictionary_available(lexicon_path):
        logger.warning(
            "LM: dictionary cache absent at {}; returning empty scores. "
            "Run `python -m rba.features.text.lexicon_data` first.",
            lexicon_path,
        )
        return pd.DataFrame(
            columns=[
                *_ID_COLUMNS,
                *_feature_names(
                    LM_CATEGORIES, config_aggregations(config), config_emit_tone(config)
                ),
            ]
        )

    lexicon = load_lm_lexicon(lexicon_path)
    documents = load_document_frames(config=config, external_dir=external_dir)
    return build_lm_scores(
        documents,
        lexicon,
        categories=LM_CATEGORIES,
        aggregations=config_aggregations(config),
        emit_tone=config_emit_tone(config),
    )
