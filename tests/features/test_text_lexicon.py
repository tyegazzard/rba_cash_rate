"""Tests for ``rba.features.text.lexicon`` + ``lexicon_data``.

No network — the LM dictionary is a tiny synthetic ``word -> categories`` map and
every document frame is built inline. Covers the pure scorer (hand-computed
counts, case-insensitivity, out-of-lexicon tokens, aggregation math, the
empty-text and degenerate-tone rules), the per-document builder, config
reconciliation, and the graceful dictionary-absent path.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from rba.features.text import lexicon_data
from rba.features.text.lexicon import (
    build_lm_scores,
    build_lm_scores_from_cache,
    config_aggregations,
    config_documents,
    config_emit_tone,
    config_lexicon_enabled,
    count_categories,
    load_document_frames,
    score_text,
    tokenize,
)
from rba.features.text.lexicon_data import (
    LM_CATEGORIES,
    lexicon_from_frame,
    load_lm_lexicon,
)


# -----------------------------------------------------------------------------
# Fixtures.
# -----------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    """Capture loguru messages at ALL levels (builder warn-skips / info-logs)."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture
def lexicon() -> dict[str, frozenset[str]]:
    """A tiny hand-built LM lexicon — one or two words per category."""
    by_category = {
        "negative": ["loss", "decline", "weak"],
        "positive": ["gain", "strong", "improve"],
        "uncertainty": ["may", "risk"],
        "litigious": ["court"],
        "strong_modal": ["will", "must"],
        "weak_modal": ["could", "might"],
        "constraining": ["require"],
    }
    mapping: dict[str, frozenset[str]] = {}
    for category, words in by_category.items():
        for word in words:
            mapping[word] = mapping.get(word, frozenset()) | {category}
    return mapping


def _docs(**frames: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return dict(frames)


# -----------------------------------------------------------------------------
# Tokenizer.
# -----------------------------------------------------------------------------
def test_tokenize_lowercases_and_drops_non_alpha() -> None:
    assert tokenize("The rate is 4.35 per cent!") == ["the", "rate", "is", "per", "cent"]


def test_tokenize_empty_and_whitespace() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


# -----------------------------------------------------------------------------
# Pure scorer — hand-computed counts.
# -----------------------------------------------------------------------------
def test_counts_on_known_sentence(lexicon: dict[str, frozenset[str]]) -> None:
    # tokens: the outlook is weak and may decline but growth could improve
    #   weak->neg, decline->neg, may->unc, could->weak_modal, improve->pos
    text = "The outlook is weak and may decline but growth could improve"
    counts = count_categories(tokenize(text), lexicon)
    assert counts["negative"] == 2
    assert counts["positive"] == 1
    assert counts["uncertainty"] == 1
    assert counts["weak_modal"] == 1
    assert counts["litigious"] == 0
    assert counts["strong_modal"] == 0
    assert counts["constraining"] == 0


def test_case_insensitivity(lexicon: dict[str, frozenset[str]]) -> None:
    counts = count_categories(tokenize("WEAK Weak weak"), lexicon)
    assert counts["negative"] == 3


def test_out_of_lexicon_tokens_ignored(lexicon: dict[str, frozenset[str]]) -> None:
    counts = count_categories(tokenize("zzz qqq foo bar"), lexicon)
    assert all(value == 0 for value in counts.values())


def test_multi_category_word_counts_each_category() -> None:
    lex = {"fragile": frozenset({"negative", "uncertainty"})}
    counts = count_categories(tokenize("fragile fragile"), lex)
    assert counts["negative"] == 2
    assert counts["uncertainty"] == 2


# -----------------------------------------------------------------------------
# Aggregation math.
# -----------------------------------------------------------------------------
def test_aggregation_math_count_ratio_normalized(lexicon: dict[str, frozenset[str]]) -> None:
    text = "The outlook is weak and may decline but growth could improve"  # 11 tokens
    scores = score_text(text, lexicon, aggregations=("count", "ratio", "normalized_count"))
    assert scores["lm_negative_count"] == 2.0
    assert scores["lm_total_words"] == 11.0
    assert scores["lm_negative_ratio"] == pytest.approx(2 / 11)
    assert scores["lm_negative_normalized_count"] == pytest.approx(2 / 11 * 1000)


def test_tone_is_polarity(lexicon: dict[str, frozenset[str]]) -> None:
    # pos=1 (improve), neg=2 (weak, decline) -> (1-2)/(1+2) = -1/3
    text = "The outlook is weak and may decline but growth could improve"
    scores = score_text(text, lexicon)
    assert scores["lm_tone"] == pytest.approx(-1 / 3)


def test_default_aggregations_exclude_ratio(lexicon: dict[str, frozenset[str]]) -> None:
    scores = score_text("weak", lexicon)
    assert "lm_negative_count" in scores
    assert "lm_negative_normalized_count" in scores
    assert "lm_negative_ratio" not in scores  # ratio dropped by default


def test_unknown_aggregation_raises(lexicon: dict[str, frozenset[str]]) -> None:
    with pytest.raises(ValueError, match="Unknown LM aggregation"):
        score_text("weak", lexicon, aggregations=("bogus",))


# -----------------------------------------------------------------------------
# Edge cases: empty text and degenerate tone.
# -----------------------------------------------------------------------------
def test_empty_text_scores_zeros_and_nan_tone(lexicon: dict[str, frozenset[str]]) -> None:
    for text in ("", "   \n\t "):
        scores = score_text(text, lexicon)
        assert scores["lm_total_words"] == 0.0
        for category in LM_CATEGORIES:
            assert scores[f"lm_{category}_count"] == 0.0
            assert scores[f"lm_{category}_normalized_count"] == 0.0
        # No directional signal -> tone undefined (NaN), never 0, never inf.
        assert math.isnan(scores["lm_tone"])


def test_tone_nan_when_no_polarity_words(lexicon: dict[str, frozenset[str]]) -> None:
    # Only modal / constraining / uncertainty words -> pos + neg == 0.
    scores = score_text("may could require", lexicon)
    assert scores["lm_uncertainty_count"] == 1.0
    assert scores["lm_weak_modal_count"] == 1.0
    assert scores["lm_constraining_count"] == 1.0
    assert math.isnan(scores["lm_tone"])


def test_tone_skipped_when_categories_lack_polarity(
    lexicon: dict[str, frozenset[str]],
) -> None:
    scores = score_text("weak", lexicon, categories=("uncertainty",), emit_tone=True)
    assert "lm_tone" not in scores  # cannot form tone without pos + neg


def test_emit_tone_false_omits_tone(lexicon: dict[str, frozenset[str]]) -> None:
    scores = score_text("weak improve", lexicon, emit_tone=False)
    assert "lm_tone" not in scores


# -----------------------------------------------------------------------------
# Builder over document frames.
# -----------------------------------------------------------------------------
def _doc_frame(bodies: list[str], start: str = "2020-01-01") -> pd.DataFrame:
    n = len(bodies)
    dates = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame(
        {
            "decision_date": dates,
            "publication_date": dates,
            "url": [f"/doc/{i}" for i in range(n)],
            "body_text": bodies,
        }
    )


def test_builder_one_row_per_document(lexicon: dict[str, frozenset[str]]) -> None:
    docs = _docs(
        rba_media_releases=_doc_frame(["weak decline", "strong improve"]),
        rba_minutes=_doc_frame(["may risk"], start="2020-06-01"),
    )
    out = build_lm_scores(docs, lexicon)
    assert len(out) == 3
    assert list(out.columns)[:4] == [
        "document_source",
        "decision_date",
        "publication_date",
        "document_id",
    ]
    assert "lm_negative_count" in out.columns
    assert "lm_tone" in out.columns


def test_builder_sorted_by_publication_date(lexicon: dict[str, frozenset[str]]) -> None:
    docs = _docs(
        late=_doc_frame(["weak"], start="2021-01-01"),
        early=_doc_frame(["strong"], start="2019-01-01"),
    )
    out = build_lm_scores(docs, lexicon)
    assert out["publication_date"].is_monotonic_increasing


def test_builder_carries_ids_and_scores(lexicon: dict[str, frozenset[str]]) -> None:
    docs = _docs(rba_minutes=_doc_frame(["weak decline may"]))
    out = build_lm_scores(docs, lexicon)
    row = out.iloc[0]
    assert row["document_source"] == "rba_minutes"
    assert row["document_id"] == "/doc/0"
    assert row["lm_negative_count"] == 2.0
    assert row["lm_uncertainty_count"] == 1.0


def test_builder_missing_body_scored_as_empty(lexicon: dict[str, frozenset[str]]) -> None:
    frame = _doc_frame(["weak"])
    frame.loc[0, "body_text"] = None
    out = build_lm_scores(_docs(rba_minutes=frame), lexicon)
    assert out.iloc[0]["lm_total_words"] == 0.0
    assert math.isnan(out.iloc[0]["lm_tone"])


def test_builder_warn_skips_source_without_body(
    lexicon: dict[str, frozenset[str]], loguru_messages: list
) -> None:
    bad = pd.DataFrame({"publication_date": pd.to_datetime(["2020-01-01"])})  # no body_text
    out = build_lm_scores(_docs(broken=bad), lexicon)
    assert out.empty
    assert any("broken" in m and "body_text" in m for m in loguru_messages)


def test_builder_empty_documents_returns_schema_only(
    lexicon: dict[str, frozenset[str]],
) -> None:
    out = build_lm_scores({}, lexicon)
    assert out.empty
    assert "lm_negative_count" in out.columns
    assert "document_source" in out.columns


def test_builder_document_independence(lexicon: dict[str, frozenset[str]]) -> None:
    """Each document's score depends only on its own text (row-wise pure)."""
    base = _docs(src=_doc_frame(["weak", "strong improve"]))
    injected = _docs(src=_doc_frame(["weak", "loss loss loss loss decline"]))
    out_base = build_lm_scores(base, lexicon)
    out_inj = build_lm_scores(injected, lexicon)
    # First document identical text -> identical scores regardless of the second.
    pd.testing.assert_series_equal(
        out_base.iloc[0].drop(labels=["document_id"]),
        out_inj.iloc[0].drop(labels=["document_id"]),
        check_names=False,
    )


# -----------------------------------------------------------------------------
# lexicon_data: mapping + cache load.
# -----------------------------------------------------------------------------
def test_lexicon_from_frame_builds_word_categories() -> None:
    frame = pd.DataFrame(
        {
            "word": ["loss", "gain", "the"],
            "negative": [True, False, False],
            "positive": [False, True, False],
            "uncertainty": [False, False, False],
            "litigious": [False, False, False],
            "strong_modal": [False, False, False],
            "weak_modal": [False, False, False],
            "constraining": [False, False, False],
        }
    )
    mapping = lexicon_from_frame(frame)
    assert mapping["loss"] == frozenset({"negative"})
    assert mapping["gain"] == frozenset({"positive"})
    assert "the" not in mapping  # in no category -> excluded


def test_load_lm_lexicon_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "word": ["loss"],
            **{cat: [cat == "negative"] for cat in LM_CATEGORIES},
        }
    )
    path = tmp_path / "loughran_mcdonald.parquet"
    frame.to_parquet(path, index=False)
    mapping = load_lm_lexicon(path)
    assert mapping == {"loss": frozenset({"negative"})}


def test_load_lm_lexicon_absent_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LM dictionary cache not found"):
        load_lm_lexicon(tmp_path / "missing.parquet")


def test_compact_from_master_csv_keeps_only_category_words() -> None:
    raw = (
        b"Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,"
        b"Weak_Modal,Constraining\n"
        b"LOSS,2009,0,0,0,0,0,0\n"
        b"THE,0,0,0,0,0,0,0\n"
        b"MAY,0,0,2009,0,0,0,0\n"
    )
    compact = lexicon_data._compact_from_master_csv(raw)
    assert set(compact["word"]) == {"loss", "may"}  # THE has no category -> dropped
    assert bool(compact.loc[compact["word"] == "loss", "negative"].iloc[0]) is True


# -----------------------------------------------------------------------------
# Orchestration: document loading + graceful dictionary-absent path.
# -----------------------------------------------------------------------------
def test_load_document_frames_warn_skips_absent(tmp_path: Path, loguru_messages: list) -> None:
    _doc_frame(["weak"]).to_parquet(tmp_path / "rba_minutes.parquet", index=False)
    frames = load_document_frames(documents=["rba_minutes", "rba_speeches"], external_dir=tmp_path)
    assert set(frames) == {"rba_minutes"}  # rba_speeches Parquet absent
    assert any("rba_speeches" in m and "absent" in m for m in loguru_messages)


def test_build_from_cache_absent_dictionary_is_graceful(
    tmp_path: Path, loguru_messages: list
) -> None:
    out = build_lm_scores_from_cache(
        external_dir=tmp_path, lexicon_path=tmp_path / "no_dict.parquet"
    )
    assert out.empty  # no crash
    assert "lm_negative_count" in out.columns
    assert any("dictionary cache absent" in m for m in loguru_messages)


# -----------------------------------------------------------------------------
# Config reconciliation (features.yaml).
# -----------------------------------------------------------------------------
def test_features_yaml_documents_reconciled_to_real_artifacts() -> None:
    from rba.config import load_features_config

    docs = config_documents(load_features_config())
    assert "rba_media_releases" in docs  # real artifact, not rba_post_meeting_statement
    assert "rba_somp" in docs
    assert "rba_post_meeting_statement" not in docs


def test_features_yaml_aggregations_drop_ratio_and_enable_tone() -> None:
    from rba.config import load_features_config

    config = load_features_config()
    assert config_aggregations(config) == ("count", "normalized_count")
    assert config_emit_tone(config) is True
    assert config_lexicon_enabled(config) is True


def test_config_accessors_defaults_when_absent() -> None:
    assert config_aggregations({}) == ("count", "normalized_count")
    assert config_emit_tone({}) is True
    assert config_documents({}) == []
    assert config_lexicon_enabled({}) is False
