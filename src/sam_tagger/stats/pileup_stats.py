"""Pileup statistics (DP, AD/ADF/ADR, AF per position) counted on dense arrays."""

from __future__ import annotations

import re
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pysam
from loguru import logger

from sam_tagger.bed import Interval, _merge_intervals_sorted, read_overlaps_intervals
from sam_tagger.parquet_writer import write_df_to_parquet
from sam_tagger.processing import REF_OPS
from sam_tagger.stats.pileup_dense import NO_LAST, SCHEMA, _Plan, _RefCounts, aligned_qualities

_MAX_BATCH = 4096
_MAX_PENDING_READS = 1 << 16
_MAX_PLAN_BYTES = 256 << 20
_NO_QUAL = np.empty(0, dtype=np.uint8)


@dataclass
class _Pending:
    """Reads of one plan that wait for their batch."""

    state: _RefCounts
    plan: _Plan
    qual: list[np.ndarray] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)
    forward: list[bool] = field(default_factory=list)


class PileupStatsCollector:
    """Collects pileup statistics across all references from CS tags.

    Retro-fix algorithm:
    - 1-based ref cursor starting at read's start position
    - Matches: record ref, advance cursor
    - SNPs: record alt, advance cursor
    - Insertions: go back to anchor (ref_idx-1), remove ref, add ins alt, no advance
    - Deletions: go back to anchor (ref_idx-1), remove ref, add del alt, advance cursor
    """

    def __init__(
        self,
        min_base_qual: int,
        ref_names: list[str],
        ref_lengths: dict[str, int],
        faidx_reader: pysam.FastaFile | None = None,
        intervals: dict[str, list[Interval]] | None = None,
    ) -> None:
        # Phred scores stay below 256, so a higher threshold drops every base.
        self._threshold = np.int32(min(max(min_base_qual, 0), 256))
        self._all_refs = list(ref_names)
        self._ref_lengths = dict(ref_lengths)
        self._faidx_reader = faidx_reader
        self._intervals = (
            {name: _merged(ivs) for name, ivs in intervals.items()}
            if intervals is not None
            else None
        )
        self._counts: dict[str, _RefCounts] = {}
        self._pending: dict[tuple[int, str], _Pending] = {}
        self._pending_reads = 0
        self._plans: dict[tuple[int, str], _Plan] = {}
        self._plans_for: str | None = None
        self._plan_bytes = 0
        self._seq = 0

    def tally_from_record(
        self,
        cs_tag: str,
        cigartuples: list[tuple[int, int]],
        record_pos: int,
        qual_scores: Sequence[int] | None,
        ref_name: str,
        is_reverse: bool,
        cs_regex: re.Pattern[str],
    ) -> None:
        if self._intervals is not None:
            intervals = self._intervals.get(ref_name)
            read_end = record_pos + sum(length for op, length in cigartuples if op in REF_OPS)
            if intervals is None or not read_overlaps_intervals(record_pos, read_end, intervals):
                return

        state = self._counts.get(ref_name)
        if state is None:
            state = self._counts[ref_name] = self._new_counts(ref_name)

        if self._plans_for != ref_name:
            self._flush()
            self._plans.clear()
            self._plans_for = ref_name
            self._plan_bytes = 0
        key = (record_pos, cs_tag)
        plan = self._plans.get(key)
        if plan is None:
            plan = state.build_plan(cs_tag, record_pos, cs_regex)
            if self._plan_bytes + plan.est_bytes > _MAX_PLAN_BYTES:
                self._flush()
                self._plans.clear()
                self._plan_bytes = 0
            self._plans[key] = plan
            self._plan_bytes += plan.est_bytes

        self._seq += 1
        qual = self._aligned_qualities(qual_scores, cigartuples, is_reverse)
        if plan.has_indel:
            mask = np.zeros(max(qual.shape[0], plan.read_len) + 1, dtype=bool)
            np.greater_equal(qual, self._threshold, out=mask[: qual.shape[0]])
            last = np.where(mask, self._seq, -1) if plan.has_snp else NO_LAST
            state.apply_single(plan, mask, not is_reverse, self._flush, last)
            return
        pending = self._pending.get(key)
        if pending is None:
            pending = self._pending[key] = _Pending(state, plan)
        pending.qual.append(
            qual[: plan.read_len]
            if qual.shape[0] >= plan.read_len
            else self._padded(qual, plan.read_len)
        )
        pending.lengths.append(qual.shape[0])
        pending.seq.append(self._seq)
        pending.forward.append(not is_reverse)
        self._pending_reads += 1
        if len(pending.qual) >= _MAX_BATCH or self._pending_reads >= _MAX_PENDING_READS:
            self._flush()

    def _new_counts(self, ref_name: str) -> _RefCounts:
        seq = b""
        if self._faidx_reader is not None:
            try:
                text = self._faidx_reader.fetch(ref_name).upper()
            except Exception as e:
                logger.warning("Failed to fetch reference {} — {}", ref_name, e)
            else:
                if not text.isascii():
                    logger.warning(
                        "Reference {} holds characters outside ASCII; the pileup keeps only"
                        " their low bytes",
                        ref_name,
                    )
                seq = text.encode("latin-1", "replace")
        extent = max(self._ref_lengths.get(ref_name, 0), len(seq))
        intervals = self._intervals.get(ref_name) if self._intervals is not None else None
        return _RefCounts(ref_name, seq, extent, intervals)

    def _aligned_qualities(
        self,
        qual_scores: Sequence[int] | None,
        cigartuples: list[tuple[int, int]],
        is_reverse: bool,
    ) -> np.ndarray:
        if qual_scores is None:
            return _NO_QUAL
        if isinstance(qual_scores, array):
            return aligned_qualities(
                np.frombuffer(qual_scores, dtype=np.uint8), cigartuples, is_reverse
            )
        return aligned_qualities(np.array(qual_scores, dtype=np.uint8), cigartuples, is_reverse)

    def _padded(self, qual: np.ndarray, read_len: int) -> np.ndarray:
        """QUAL shorter than the cs tag, filled to the batch width."""
        padded = np.zeros(read_len, dtype=np.uint8)
        padded[: qual.shape[0]] = qual
        return padded

    def _flush(self) -> None:
        """Apply every waiting batch; an indel read calls it when a counter at its anchor is 0."""
        for pending in self._pending.values():
            qual = np.stack(pending.qual)
            passes = qual >= self._threshold
            lengths = np.array(pending.lengths, dtype=np.intp)
            if (lengths < qual.shape[1]).any():
                # A base without a quality entry never passes, whatever the threshold is.
                passes &= lengths[:, None] > np.arange(qual.shape[1])
            last = NO_LAST
            if pending.plan.has_snp:
                seqs = np.array(pending.seq, dtype=np.int64)
                last = np.where(passes, seqs[:, None], -1).max(axis=0)
            if len(pending.forward) == 1:
                pending.state.apply_single(
                    pending.plan, passes[0], pending.forward[0], pass_last=last
                )
                continue
            total = passes.sum(axis=0, dtype=np.int32)
            forward = np.array(pending.forward, dtype=bool)
            pass_fwd = passes[forward].sum(axis=0, dtype=np.int32)
            pending.state.apply(pending.plan, pass_fwd, total - pass_fwd, pass_last=last)
        self._pending.clear()
        self._pending_reads = 0

    def write_results(self, path: Path) -> None:
        logger.info("Writing pileup statistics to {}", path)
        self._flush()
        frames: list[pl.DataFrame] = []
        for name in sorted(self._all_refs):
            state = self._counts.get(name)
            if state is not None and (frame := state.frame()) is not None:
                frames.append(frame)
        df = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
        write_df_to_parquet(df, path)
        logger.info("Pileup statistics successfully written.")


def _merged(intervals: list[Interval]) -> list[Interval]:
    ordered = sorted(intervals, key=lambda iv: iv.start)
    _merge_intervals_sorted(ordered)
    return ordered
