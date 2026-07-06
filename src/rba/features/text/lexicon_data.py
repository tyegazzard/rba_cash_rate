"""One-time sourcing + local caching of the Loughran-McDonald (LM) dictionary.

The LM Master Dictionary (Notre Dame SRAF,
https://sraf.nd.edu/loughranmcdonald-master-dictionary/) is the reference data
behind :mod:`rba.features.text.lexicon`. It is **third-party reference data**, so
per the project's data conventions it is downloaded once, hashed (Invariant #5),
and cached under ``data/external/`` (gitignored, regenerated — never committed).

Two clean halves, mirroring the ``fetch`` (network) vs ``align`` (read-only)
split elsewhere in the repo:

- :func:`download_lm_dictionary` — the **only** function that touches the
  network. It pulls the SRAF master CSV, hashes the raw bytes, extracts the
  compact ``word -> categories`` mapping, and writes a small Parquet artifact
  (~4k category words, ~100 KB) plus a provenance sidecar JSON. Run once via
  ``python -m rba.features.text.lexicon_data`` (or the data-refresh step).
- :func:`load_lm_lexicon` — **network-free**; the feature builder's entry point.
  Reads the cached artifact into a ``{word: frozenset[category]}`` mapping.
  Raises :class:`FileNotFoundError` if the cache is absent, with a message
  pointing at the download step.

The SRAF master is hosted on a Google-Drive-style endpoint that periodically
moves; :data:`LM_MASTER_DICTIONARY_URL` is therefore best-effort and may need
updating. Because scoring reads only the local cache, nothing in the feature
path depends on the URL being live.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Canonical LM category names (lowercase, project-internal) mapped to their
# column header in the SRAF master CSV. Order is the canonical LM ordering and is
# the single source of truth for "which categories exist" across the text block.
_MASTER_COLUMN_BY_CATEGORY: dict[str, str] = {
    "negative": "Negative",
    "positive": "Positive",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
    "strong_modal": "Strong_Modal",
    "weak_modal": "Weak_Modal",
    "constraining": "Constraining",
}

#: The seven standard LM sentiment categories (lowercase), in canonical order.
LM_CATEGORIES: tuple[str, ...] = tuple(_MASTER_COLUMN_BY_CATEGORY)

# Compact-artifact + provenance-sidecar locations under data/external/.
LM_ARTIFACT_PATH: Path = EXTERNAL_DATA_DIR / "loughran_mcdonald.parquet"
LM_META_PATH: Path = EXTERNAL_DATA_DIR / "loughran_mcdonald.meta.json"

# The 'Word' column of the master CSV (verbatim header).
_WORD_COLUMN = "Word"

# Direct-download URL for the LM Master Dictionary CSV. The canonical source is
# the Notre Dame SRAF page (https://sraf.nd.edu/loughranmcdonald-master-dictionary/),
# but SRAF serves it from a Google-Drive-style endpoint that requires a
# confirm-token dance and whose file id moves between vintages. This points at a
# stable public mirror of the same master CSV (Word + per-category-year columns);
# override with ``--url`` to pin a specific SRAF vintage. Provenance (source name,
# URL, sha256) is recorded in the sidecar meta on download.
LM_MASTER_DICTIONARY_URL = (
    "https://raw.githubusercontent.com/james-pavlicek/"
    "algorithmic-trading-with-artificial-intelligence/main/"
    "Loughran-McDonald_MasterDictionary.csv"
)


def download_lm_dictionary(
    *,
    url: str = LM_MASTER_DICTIONARY_URL,
    dest: Path = LM_ARTIFACT_PATH,
    meta_path: Path = LM_META_PATH,
    force: bool = False,
) -> Path:
    """Download the SRAF master dictionary and cache the compact artifact.

    The **only** network-touching function in the text block. Fetches the master
    CSV, hashes the raw bytes (Invariant #5), reduces it to the ``word`` +
    per-category boolean columns (keeping only words in ≥1 category), and writes
    the Parquet artifact plus a provenance sidecar.

    Parameters
    ----------
    url
        Master-dictionary CSV URL (defaults to :data:`LM_MASTER_DICTIONARY_URL`).
    dest
        Compact Parquet artifact path (default :data:`LM_ARTIFACT_PATH`).
    meta_path
        Provenance JSON path (default :data:`LM_META_PATH`).
    force
        Re-download even if ``dest`` already exists.

    Returns
    -------
    pathlib.Path
        The path to the written Parquet artifact.
    """
    if dest.exists() and not force:
        logger.info("LM dictionary cache already present at {}; skipping download.", dest)
        return dest

    logger.info("Downloading LM master dictionary from {}", url)
    raw = _http_get(url)
    digest = hashlib.sha256(raw).hexdigest()

    compact = _compact_from_master_csv(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    compact.to_parquet(dest, index=False)

    meta = {
        "artifact": dest.name,
        "source": "Loughran-McDonald Master Dictionary (Notre Dame SRAF)",
        "url": url,
        "sha256_raw_csv": digest,
        "n_category_words": int(len(compact)),
        "categories": list(LM_CATEGORIES),
        "downloaded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.success(
        "Cached LM dictionary: {} category words -> {} (sha256={}).",
        len(compact),
        dest,
        digest,
    )
    return dest


def _http_get(url: str) -> bytes:
    """Fetch ``url`` and return the response bytes (no custom User-Agent)."""
    with urllib.request.urlopen(url) as response:  # noqa: S310 — trusted SRAF URL
        return bytes(response.read())


def _compact_from_master_csv(raw: bytes) -> pd.DataFrame:
    """Reduce the raw master CSV to ``word`` + per-category boolean columns.

    Only words belonging to at least one category are kept (the master lists
    ~86k words; ~4k carry a category), which keeps the cached artifact ~100 KB.
    Words are lowercased to match the scorer's tokenizer.

    Shapes
    ------
    Returns: (n_category_words, 1 + len(LM_CATEGORIES)) — ``word`` plus one bool
    column per category.
    """
    master = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in (_WORD_COLUMN, *_MASTER_COLUMN_BY_CATEGORY.values()) if c not in master.columns]
    if missing:
        raise ValueError(
            f"LM master CSV missing expected column(s): {missing}. "
            f"Got columns: {list(master.columns)[:12]}…"
        )

    out = pd.DataFrame({"word": master[_WORD_COLUMN].astype(str).str.lower()})
    # A word is in a category iff the master's category column is non-zero (the
    # cell holds the year the word was added to that category, else 0).
    for category, master_col in _MASTER_COLUMN_BY_CATEGORY.items():
        out[category] = pd.to_numeric(master[master_col], errors="coerce").fillna(0) != 0

    in_any = out[list(LM_CATEGORIES)].any(axis=1)
    out = out.loc[in_any].drop_duplicates(subset="word").reset_index(drop=True)
    return out


def lm_dictionary_available(path: Path = LM_ARTIFACT_PATH) -> bool:
    """Whether the cached LM dictionary artifact exists (no network)."""
    return path.exists()


def load_lm_lexicon(path: Path = LM_ARTIFACT_PATH) -> dict[str, frozenset[str]]:
    """Load the cached LM dictionary as a ``word -> categories`` mapping.

    Network-free; the feature builder's entry point.

    Parameters
    ----------
    path
        The compact Parquet artifact (default :data:`LM_ARTIFACT_PATH`).

    Returns
    -------
    dict[str, frozenset[str]]
        ``{lowercased_word: frozenset(category, ...)}`` for every category word.

    Raises
    ------
    FileNotFoundError
        If the cache is absent — run
        ``python -m rba.features.text.lexicon_data`` (or the data-refresh step).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"LM dictionary cache not found at {path}. Download it once with "
            f"`python -m rba.features.text.lexicon_data` (network step)."
        )
    frame = pd.read_parquet(path)
    return lexicon_from_frame(frame)


def lexicon_from_frame(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Build a ``word -> categories`` mapping from a compact-artifact frame.

    Pure (no I/O), factored out so tests can exercise the mapping logic on an
    inline frame without touching disk.

    Parameters
    ----------
    frame
        A frame with a ``word`` column plus one boolean column per category in
        :data:`LM_CATEGORIES`.

    Returns
    -------
    dict[str, frozenset[str]]
        ``{word: frozenset(category, ...)}``.
    """
    categories = [c for c in LM_CATEGORIES if c in frame.columns]
    lexicon: dict[str, frozenset[str]] = {}
    for row in frame.itertuples(index=False):
        word = str(getattr(row, "word")).lower()
        member = frozenset(c for c in categories if bool(getattr(row, c)))
        if member:
            lexicon[word] = member
    return lexicon


# -----------------------------------------------------------------------------
# CLI: one-time download.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.features.text.lexicon_data`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.features.text.lexicon_data",
        description="Download + cache the Loughran-McDonald master dictionary.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument("--url", default=LM_MASTER_DICTIONARY_URL, help="Override the source URL.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    download_lm_dictionary(url=args.url, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
