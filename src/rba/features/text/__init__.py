"""Text feature builders for the RBA cash-rate model (``§5`` text block).

This subpackage houses the escalating text-feature tiers described in
``features.yaml`` ``text:``:

- :mod:`rba.features.text.lexicon` — Loughran-McDonald (and, later, a custom
  hawk/dove) lexicon scoring over the RBA document corpus.
- (later) sentence embeddings + an embedding cache.

The dictionary the lexicon scorer needs is sourced once, offline, by
:mod:`rba.features.text.lexicon_data` and cached under ``data/external/`` — the
scoring builders themselves never touch the network (Invariant #1 / #5).
"""

from __future__ import annotations
