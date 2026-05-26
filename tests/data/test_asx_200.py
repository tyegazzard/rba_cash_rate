"""Tests for ``rba.data.sources.asx_200``.

The live ``fetch()`` path hits the Yahoo v8 chart endpoint and is not
exercised here. ``_parse_chart_json``, ``_attach_publication_dates``,
``_build_url``, and ``_to_wide`` are tested in isolation against
synthetic JSON inputs mirroring the real Yahoo response shape
(``chart.result[0].timestamp`` + ``chart.result[0].indicators.quote[0].close``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.asx_200 import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    AsxIndexSeries,
    _attach_publication_dates,
    _build_url,
    _parse_chart_json,
    _to_wide,
)


# Five XJO trade dates: three in AEDT (Nov 1992 was AEDT, summer DST), two
# in 2025 spanning the AEDT → AEST → AEDT transitions. Timestamps are at
# 10:00 market-local Sydney time, encoded as Unix seconds — the same
# convention Yahoo uses for daily bars. The middle "close": null row
# mimics an ASX holiday (Yahoo emits a placeholder timestamp but no
# close value); the parser must drop it.
def _ts(date: str) -> int:
    """Compute a Sydney 10:00 timestamp for a given trade date."""
    ts_syd = pd.Timestamp(date).tz_localize("Australia/Sydney") + pd.Timedelta(hours=10)
    return int(ts_syd.timestamp())


def _build_fixture_payload(
    *,
    symbol: str,
    rows: list[tuple[str, float | None]],
) -> bytes:
    """Build a synthetic Yahoo v8 chart JSON payload.

    Parameters
    ----------
    symbol
        The ticker (e.g. ``^AXJO``). Stamped into ``meta.symbol``.
    rows
        List of ``(trade_date_str, close_or_None)`` tuples — one daily
        bar each.

    Returns
    -------
    bytes
        UTF-8 JSON bytes mirroring the production response shape.
    """
    timestamps = [_ts(d) for d, _ in rows]
    closes = [c for _, c in rows]
    n = len(rows)
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "AUD",
                        "symbol": symbol,
                        "exchangeName": "ASX",
                        "instrumentType": "INDEX",
                        "firstTradeDate": timestamps[0] if timestamps else None,
                        "timezone": "AEDT",
                        "gmtoffset": 39600,
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [None] * n,
                                "high": [None] * n,
                                "low": [None] * n,
                                "close": closes,
                                "volume": [0] * n,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload).encode("utf-8")


_FIXTURE_XJO_ROWS: list[tuple[str, float | None]] = [
    ("1992-11-23", 1455.00),  # AEDT (first Sydney trade date)
    ("1993-01-04", 1576.10),  # AEDT — first inflation-targeting trade
    ("2025-02-14", 8555.80),  # AEDT — Friday
    ("2025-07-04", 8350.00),  # AEST — Friday in winter
    ("2025-07-07", None),  # holiday-style null close — must be dropped
    ("2025-07-08", 8400.00),  # AEST — Tuesday
]
_XJO_SPEC = AsxIndexSeries(series_id="xjo", yahoo_ticker="^AXJO", name="S&P/ASX 200")


# -----------------------------------------------------------------------------
# _parse_chart_json
# -----------------------------------------------------------------------------


def test_parse_chart_json_extracts_close_values() -> None:
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    df = _parse_chart_json(payload, spec=_XJO_SPEC)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 6 rows minus the null-close holiday = 5
    assert len(df) == 5


def test_parse_chart_json_drops_null_close_rows() -> None:
    """ASX-holiday rows arrive as ``close: null`` and must be dropped."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    df = _parse_chart_json(payload, spec=_XJO_SPEC)
    assert pd.Timestamp("2025-07-07") not in df["observation_date"].tolist()


def test_parse_chart_json_recovers_sydney_trade_dates_under_aedt() -> None:
    """During AEDT (Oct–Apr) a 10:00 Sydney timestamp converts to a
    *previous* UTC calendar date. The parser must use the Sydney TZ to
    recover the wall-clock trade date — not UTC."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    df = _parse_chart_json(payload, spec=_XJO_SPEC)
    dates = set(df["observation_date"])
    assert pd.Timestamp("1992-11-23") in dates  # AEDT (Sun 22-Nov was a Sunday)
    assert pd.Timestamp("1993-01-04") in dates  # AEDT
    assert pd.Timestamp("2025-02-14") in dates  # AEDT


def test_parse_chart_json_handles_aest_trade_dates() -> None:
    """During AEST (Apr–Oct) the 10:00 timestamp is at 00:00 UTC same
    day — the parser must still produce the same trade date."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    df = _parse_chart_json(payload, spec=_XJO_SPEC)
    dates = set(df["observation_date"])
    assert pd.Timestamp("2025-07-04") in dates  # AEST
    assert pd.Timestamp("2025-07-08") in dates  # AEST


def test_parse_chart_json_stamps_logical_series_id() -> None:
    payload = _build_fixture_payload(symbol="^AXFJ", rows=[("2013-03-06", 5495.46)])
    spec = AsxIndexSeries(series_id="xfj", yahoo_ticker="^AXFJ", name="Financials")
    df = _parse_chart_json(payload, spec=spec)
    assert (df["series_id"] == "xfj").all()


def test_parse_chart_json_correct_values_for_anchors() -> None:
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    df = _parse_chart_json(payload, spec=_XJO_SPEC)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("1993-01-04")] == pytest.approx(1576.10)
    assert by_date[pd.Timestamp("2025-02-14")] == pytest.approx(8555.80)
    assert by_date[pd.Timestamp("2025-07-08")] == pytest.approx(8400.00)


def test_parse_chart_json_raises_on_upstream_error() -> None:
    payload = json.dumps(
        {"chart": {"result": None, "error": {"code": "Not Found", "description": "..."}}}
    ).encode("utf-8")
    with pytest.raises(ValueError, match="returned error"):
        _parse_chart_json(payload, spec=_XJO_SPEC)


def test_parse_chart_json_raises_on_empty_results() -> None:
    payload = json.dumps({"chart": {"result": [], "error": None}}).encode("utf-8")
    with pytest.raises(ValueError, match="returned no results"):
        _parse_chart_json(payload, spec=_XJO_SPEC)


def test_parse_chart_json_raises_on_zero_timestamps() -> None:
    payload = json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "^AXJO"},
                        "timestamp": [],
                        "indicators": {"quote": [{"close": []}]},
                    }
                ],
                "error": None,
            }
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="zero timestamps"):
        _parse_chart_json(payload, spec=_XJO_SPEC)


def test_parse_chart_json_raises_on_length_mismatch() -> None:
    """If the close array length disagrees with the timestamp array
    length, the upstream schema has shifted under us and the parser
    must refuse to silently mis-align dates and values."""
    payload = json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "^AXJO"},
                        "timestamp": [_ts("2025-07-04"), _ts("2025-07-08")],
                        "indicators": {"quote": [{"close": [8350.0]}]},
                    }
                ],
                "error": None,
            }
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="Length mismatch"):
        _parse_chart_json(payload, spec=_XJO_SPEC)


def test_parse_chart_json_raises_on_missing_close_array() -> None:
    payload = json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "^AXJO"},
                        "timestamp": [_ts("2025-07-04")],
                        "indicators": {"quote": [{}]},
                    }
                ],
                "error": None,
            }
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="missing close array"):
        _parse_chart_json(payload, spec=_XJO_SPEC)


# -----------------------------------------------------------------------------
# _attach_publication_dates — same-day rule (no calendar module)
# -----------------------------------------------------------------------------


def test_attach_publication_dates_equals_observation_date() -> None:
    """ASX index EOD values are published ~16:30 market-local same-day.
    publication_date == observation_date."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    parsed = _parse_chart_json(payload, spec=_XJO_SPEC)
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] == out["observation_date"]).all()


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before stamping — otherwise stale values would leak."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    parsed = _parse_chart_json(payload, spec=_XJO_SPEC).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    parsed = _parse_chart_json(payload, spec=_XJO_SPEC)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


def test_attach_publication_dates_empty_frame_no_raise() -> None:
    """An empty input frame must round-trip without raising."""
    empty = pd.DataFrame(
        {
            "observation_date": pd.Series(dtype="datetime64[ns]"),
            "series_id": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    out = _attach_publication_dates(empty)
    assert out.empty
    assert "publication_date" in out.columns
    assert out["publication_date"].dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_not_after_observation() -> None:
    """Under the same-day rule every publication_date must equal its
    observation_date — never strictly after, never before."""
    payload = _build_fixture_payload(symbol="^AXJO", rows=_FIXTURE_XJO_ROWS)
    parsed = _parse_chart_json(payload, spec=_XJO_SPEC)
    out = _attach_publication_dates(parsed)

    has_pub = out["publication_date"].notna()
    assert has_pub.all()
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days == 0).all(), (
        f"Same-day publication expected; observed lag range "
        f"[{lag_days.min()}, {lag_days.max()}] days"
    )


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def _build_combined_long_df() -> pd.DataFrame:
    """Two series sharing a trade date and one series-only row."""
    xjo_payload = _build_fixture_payload(
        symbol="^AXJO",
        rows=[("2025-07-04", 8350.0), ("2025-07-08", 8400.0)],
    )
    xfj_payload = _build_fixture_payload(
        symbol="^AXFJ",
        rows=[("2025-07-04", 9000.0)],  # XFJ missing 2025-07-08
    )
    xjo_spec = AsxIndexSeries(series_id="xjo", yahoo_ticker="^AXJO", name="XJO")
    xfj_spec = AsxIndexSeries(series_id="xfj", yahoo_ticker="^AXFJ", name="XFJ")
    long_df = pd.concat(
        [
            _parse_chart_json(xjo_payload, spec=xjo_spec),
            _parse_chart_json(xfj_payload, spec=xfj_spec),
        ],
        ignore_index=True,
    )
    return _attach_publication_dates(long_df)


def test_to_wide_schema() -> None:
    long_df = _build_combined_long_df()
    wide = _to_wide(long_df)
    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    assert "xjo" in wide.columns
    assert "xfj" in wide.columns


def test_to_wide_release_date_equals_trade_date() -> None:
    long_df = _build_combined_long_df()
    wide = _to_wide(long_df)
    assert (wide["release_date"] == wide["trade_date"]).all()


def test_to_wide_pivot_emits_nan_for_missing_series_date() -> None:
    """XFJ does not have a value for 2025-07-08 in the fixture; the wide
    pivot should leave NaN there rather than collapsing the row."""
    long_df = _build_combined_long_df()
    wide = _to_wide(long_df)
    row = wide[wide["trade_date"] == pd.Timestamp("2025-07-08")].iloc[0]
    assert pd.notna(row["xjo"])
    assert pd.isna(row["xfj"])


def test_to_wide_column_order_follows_series_registry() -> None:
    """Columns appear in SERIES order between trade_date and release_date."""
    long_df = _build_combined_long_df()
    wide = _to_wide(long_df)
    inner = [c for c in wide.columns if c not in {"trade_date", "release_date"}]
    expected_order = [s.series_id for s in SERIES if s.series_id in {"xjo", "xfj"}]
    assert inner == expected_order


# -----------------------------------------------------------------------------
# _build_url
# -----------------------------------------------------------------------------


def test_build_url_encodes_caret() -> None:
    """``^`` must be percent-encoded to ``%5E`` for safe URL transport."""
    url = _build_url("^AXJO")
    assert "%5EAXJO" in url
    assert "^AXJO" not in url


def test_build_url_includes_full_history_window() -> None:
    url = _build_url("^AXJO")
    assert "period1=0" in url
    assert "period2=9999999999" in url
    assert "interval=1d" in url


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_five_entries() -> None:
    """Headline XJO plus four sector sub-indices."""
    assert len(SERIES) == 5


def test_series_registry_covers_expected_tickers() -> None:
    tickers = {s.yahoo_ticker for s in SERIES}
    assert tickers == {"^AXJO", "^AXFJ", "^AXMJ", "^AXJR", "^AXEJ"}


def test_series_registry_tickers_unique() -> None:
    tickers = [s.yahoo_ticker for s in SERIES]
    assert len(tickers) == len(set(tickers))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_yahoo_tickers() -> None:
    by_id = {s.series_id: s.yahoo_ticker for s in SERIES}
    assert by_id["xjo"] == "^AXJO"
    assert by_id["xfj"] == "^AXFJ"
    assert by_id["xmj"] == "^AXMJ"
    assert by_id["xjr"] == "^AXJR"
    assert by_id["xej"] == "^AXEJ"


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")
