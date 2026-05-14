"""Tests for ``rba.data.sources.rba_f11`` parser and date-convention logic.

The download path (``fetch``) is not exercised here — those tests would hit
the live RBA endpoint. Both helper functions are tested in isolation against
synthetic fixtures that mirror the real RBA HTML layout and the convention
cutover at 2008-02-06.
"""

from __future__ import annotations

import pandas as pd

from rba.data.sources.rba_f11 import (
    CUTOVER_EFFECTIVE_DATE,
    _apply_date_convention,
    _parse,
)

# Synthetic HTML mirroring the real /statistics/cash-rate/ table layout.
# Covers: modern change, modern hold, post-cutover row, pre-cutover row,
# pre-1990 range-target row, and a row with the truncated <td> tail seen in
# very early entries.
_FIXTURE_HTML = b"""
<html><body>
<table id="datatable">
  <thead>
    <tr><th>Effective Date</th><th>Change</th><th>Cash rate target</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">6 May 2026</th>
      <td>+0.25</td>
      <td>4.35</td>
      <td class="links">
        <a aria-label="Media Release 2026-12: Statement" href="/media-releases/2026/mr-26-12.html">Statement</a>
      </td>
    </tr>
    <tr>
      <th scope="row">10 Dec 2025</th>
      <td>0.00</td>
      <td>3.60</td>
      <td class="links">
        <a aria-label="Media Release 2025-33: Statement" href="/media-releases/2025/mr-25-33.html">Statement</a>
        <a aria-label="Minutes of the Monetary Policy Meeting" href="/monetary-policy/rba-board-minutes/2025/2025-12-09.html">Minutes</a>
      </td>
    </tr>
    <tr>
      <th scope="row">6 Feb 2008</th>
      <td>+0.25</td>
      <td>7.00</td>
      <td class="links">
        <a aria-label="Media Release 2008-02: Statement" href="/media-releases/2008/mr-08-02.html">Statement</a>
        <a aria-label="Minutes" href="/monetary-policy/rba-board-minutes/2008/05022008.html">Minutes</a>
      </td>
    </tr>
    <tr>
      <th scope="row">7 Nov 2007</th>
      <td>+0.25</td>
      <td>6.75</td>
      <td class="links">
        <a aria-label="Media Release 2007-20: Statement" href="/media-releases/2007/mr-07-20.html">Statement</a>
      </td>
    </tr>
    <tr>
      <th scope="row">7 Nov 1990</th>
      <td>0.00</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th scope="row">23 Jan 1990</th>
      <td>-0.50 to -1.00</td>
      <td>17.00 to 17.50</td>
      <td class="links"></td>
    </tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_extracts_all_rows() -> None:
    df = _parse(_FIXTURE_HTML)
    assert len(df) == 6
    assert set(df.columns) == {
        "observation_date",
        "change_raw",
        "new_cash_rate_raw",
        "statement_url",
        "minutes_url",
    }


def test_parse_date_formatting() -> None:
    df = _parse(_FIXTURE_HTML)
    dates = pd.to_datetime(df["observation_date"]).dt.strftime("%Y-%m-%d").tolist()
    assert dates == [
        "2026-05-06",
        "2025-12-10",
        "2008-02-06",
        "2007-11-07",
        "1990-11-07",
        "1990-01-23",
    ]


def test_parse_preserves_change_strings_verbatim() -> None:
    df = _parse(_FIXTURE_HTML)
    assert df["change_raw"].tolist() == [
        "+0.25",
        "0.00",
        "+0.25",
        "+0.25",
        "0.00",
        "-0.50 to -1.00",
    ]


def test_parse_preserves_range_rate_strings_verbatim() -> None:
    df = _parse(_FIXTURE_HTML)
    assert df.loc[df["observation_date"] == pd.Timestamp("1990-01-23"),
                  "new_cash_rate_raw"].iloc[0] == "17.00 to 17.50"


def test_parse_extracts_links_when_present() -> None:
    df = _parse(_FIXTURE_HTML)
    row_2025 = df[df["observation_date"] == pd.Timestamp("2025-12-10")].iloc[0]
    assert row_2025["statement_url"] == "/media-releases/2025/mr-25-33.html"
    assert row_2025["minutes_url"] == "/monetary-policy/rba-board-minutes/2025/2025-12-09.html"


def test_parse_handles_rows_without_links_cell() -> None:
    df = _parse(_FIXTURE_HTML)
    # 7 Nov 1990 has no <td> for links at all.
    row_1990_nov = df[df["observation_date"] == pd.Timestamp("1990-11-07")].iloc[0]
    assert pd.isna(row_1990_nov["statement_url"])
    assert pd.isna(row_1990_nov["minutes_url"])


def test_parse_handles_change_only_rows() -> None:
    df = _parse(_FIXTURE_HTML)
    # 7 Nov 2007 has only a Statement link, no Minutes.
    row_2007 = df[df["observation_date"] == pd.Timestamp("2007-11-07")].iloc[0]
    assert row_2007["statement_url"] == "/media-releases/2007/mr-07-20.html"
    assert pd.isna(row_2007["minutes_url"])


def test_apply_date_convention_pre_cutover_unchanged() -> None:
    df = pd.DataFrame({"observation_date": [pd.Timestamp("2005-03-02")]})
    out = _apply_date_convention(df)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2005-03-02")


def test_apply_date_convention_post_cutover_subtracts_one_day() -> None:
    # 2008-02-06 (the cutover row itself) should already be one day after announcement.
    df = pd.DataFrame({"observation_date": [pd.Timestamp("2008-02-06")]})
    out = _apply_date_convention(df)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2008-02-05")


def test_apply_date_convention_emergency_intermeeting() -> None:
    # 2020-03-20 (Fri) — COVID emergency cut, announced Thu 2020-03-19.
    df = pd.DataFrame({"observation_date": [pd.Timestamp("2020-03-20")]})
    out = _apply_date_convention(df)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2020-03-19")


def test_apply_date_convention_cutover_boundary_row_is_inclusive() -> None:
    # The cutover date itself uses the new convention (verified empirically).
    just_before = CUTOVER_EFFECTIVE_DATE - pd.Timedelta(days=1)
    df = pd.DataFrame({"observation_date": [just_before, CUTOVER_EFFECTIVE_DATE]})
    out = _apply_date_convention(df)
    assert out["publication_date"].iloc[0] == just_before  # unchanged
    assert out["publication_date"].iloc[1] == CUTOVER_EFFECTIVE_DATE - pd.Timedelta(days=1)


def test_apply_date_convention_orders_date_columns_first() -> None:
    df = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2026-05-06")],
            "change_raw": ["+0.25"],
            "new_cash_rate_raw": ["4.35"],
            "statement_url": [None],
            "minutes_url": [None],
        }
    )
    out = _apply_date_convention(df)
    assert list(out.columns)[:2] == ["observation_date", "publication_date"]
