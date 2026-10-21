"""Per-reference QC summary collector for the ``qc-summary`` subcommand.

One Parquet row per BAM-header reference joining FL, TC, AP and per-base error
counts/rate into a single table. Aggregation only — the per-read judgements
(full-length, on-target, amplicon-primers, error tally) reuse the already-shared,
already-tested functions in :mod:`sam_tagger.processing` and the error tally in
:class:`sam_tagger.stats.error_stats.ErrorStatsCollector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from sam_tagger.parquet_writer import write_df_to_parquet

if TYPE_CHECKING:
    from sam_tagger.stats.error_stats import ErrorStatsCollector


@dataclass(slots=True)
class _QcCounts:
    total: int = 0
    forward: int = 0
    reverse: int = 0
    fl: int = 0
    on_target: int = 0
    ap_both: int = 0
    ap_start_only: int = 0
    ap_end_only: int = 0
    ap_neither: int = 0


# Output schema (one row per BAM-header reference; zero-read refs included).
# Annotation-free to mirror parquet_writer.py — pyright treats pl.UInt64 as a
# class, so an explicit `dict[str, pl.DataType]` hint would not accept it.
_SCHEMA = {
    "seq_name": pl.Utf8,
    "total_reads": pl.UInt64,
    "forward_reads": pl.UInt64,
    "reverse_reads": pl.UInt64,
    "forward_pct": pl.Float64,
    "fl": pl.UInt64,
    "fl_pct": pl.Float64,
    "on_target": pl.UInt64,
    "on_target_pct": pl.Float64,
    "ap_both": pl.UInt64,
    "ap_start_only": pl.UInt64,
    "ap_end_only": pl.UInt64,
    "ap_neither": pl.UInt64,
    "mismatches": pl.UInt64,
    "insertions": pl.UInt64,
    "deletions": pl.UInt64,
    "aligned_bases": pl.UInt64,
    "error_rate": pl.Float64,
}


class QcSummaryCollector:
    """Accumulates per-reference QC counts; writes the joined summary Parquet."""

    def __init__(self, ref_names: list[str]) -> None:
        # Every header contig gets a row, even with zero reads (FL is the denominator).
        self._counts: dict[str, _QcCounts] = {name: _QcCounts() for name in ref_names}

    def tally_total(self, ref: str) -> None:
        self._counts[ref].total += 1

    def tally_strand(self, ref: str, is_reverse: bool) -> None:
        """Bucket a mapped record by FLAG 0x10 (secondary/supplementary included)."""
        if is_reverse:
            self._counts[ref].reverse += 1
        else:
            self._counts[ref].forward += 1

    def tally_fl(self, ref: str) -> None:
        self._counts[ref].fl += 1

    def tally_on_target(self, ref: str) -> None:
        self._counts[ref].on_target += 1

    def tally_ap(self, ref: str, status: str) -> None:
        """Increment the amplicon-primer bucket for ``status``.

        ``check_amplicon_primers`` returns exactly ``both``/``start_only``/
        ``end_only``/``neither``; any other value is ignored (defensive).
        """
        c = self._counts[ref]
        match status:
            case "both":
                c.ap_both += 1
            case "start_only":
                c.ap_start_only += 1
            case "end_only":
                c.ap_end_only += 1
            case "neither":
                c.ap_neither += 1

    def write_results(self, path: Path, error_collector: ErrorStatsCollector | None) -> None:
        """Build the summary DataFrame and write it to ``path`` as Parquet."""
        columns: dict[str, list[object]] = {name: [] for name in _SCHEMA}
        for seq_name in sorted(self._counts):
            c = self._counts[seq_name]
            total = c.total

            aligned_bases = mismatches = insertions = deletions = 0
            if error_collector is not None:
                counts = error_collector.per_base_counts(seq_name)
                if counts is not None:
                    aligned_bases, mismatches, insertions, deletions = counts
            error_events = mismatches + insertions + deletions

            columns["seq_name"].append(seq_name)
            columns["total_reads"].append(total)
            columns["forward_reads"].append(c.forward)
            columns["reverse_reads"].append(c.reverse)
            columns["forward_pct"].append((c.forward / total) if total else 0.0)
            columns["fl"].append(c.fl)
            columns["fl_pct"].append((c.fl / total) if total else 0.0)
            columns["on_target"].append(c.on_target)
            columns["on_target_pct"].append((c.on_target / total) if total else 0.0)
            columns["ap_both"].append(c.ap_both)
            columns["ap_start_only"].append(c.ap_start_only)
            columns["ap_end_only"].append(c.ap_end_only)
            columns["ap_neither"].append(c.ap_neither)
            columns["mismatches"].append(mismatches)
            columns["insertions"].append(insertions)
            columns["deletions"].append(deletions)
            columns["aligned_bases"].append(aligned_bases)
            columns["error_rate"].append((error_events / aligned_bases) if aligned_bases else 0.0)

        df = pl.DataFrame(columns, schema=_SCHEMA)
        write_df_to_parquet(df, path)
