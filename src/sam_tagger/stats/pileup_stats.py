"""Pileup statistics collection (bcftools-mpileup-like).

Already decoupled from the CLI (the constructor takes explicit scalars/paths),
so this is a verbatim port with the package import updated. Implements the
retro-fix algorithm with a 1-based cursor; outputs DP, AD/ADF/ADR, and AF per
position.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pysam
from loguru import logger

from sam_tagger.bed import Interval, position_in_intervals, read_overlaps_intervals
from sam_tagger.parquet_writer import write_df_to_parquet


@dataclass
class _PositionPileup:
    """Holds pileup statistics for a single position."""

    ref_base: str | None = None
    dp: int = 0
    # allele -> (forward_count, reverse_count, total_count)
    allele_counts: dict[str, list[int]] = field(default_factory=dict)

    def record_allele(self, allele: str, is_forward: bool) -> None:
        counts = self.allele_counts.get(allele)
        if counts is None:
            counts = [0, 0, 0]
            self.allele_counts[allele] = counts
        if is_forward:
            counts[0] += 1
        else:
            counts[1] += 1
        counts[2] += 1
        self.dp += 1


@dataclass
class _PileupStatsOutput:
    """Helper for writing pileup stats to Parquet."""

    data: list[tuple[str, int, str, str, int, int, int, int, float]] = field(default_factory=list)
    # (seq_name, pos, ref, alt, dp, adf, adr, ad, af)

    def add(
        self,
        seq_name: str,
        pos: int,
        ref_base: str,
        alt: str,
        dp: int,
        adf: int,
        adr: int,
        ad: int,
        af: float,
    ) -> None:
        self.data.append((seq_name, pos, ref_base, alt, dp, adf, adr, ad, af))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [d[0] for d in self.data],
                "pos": [d[1] for d in self.data],
                "ref": [d[2] for d in self.data],
                "alt": [d[3] for d in self.data],
                "dp": [d[4] for d in self.data],
                "adf": [d[5] for d in self.data],
                "adr": [d[6] for d in self.data],
                "ad": [d[7] for d in self.data],
                "af": [d[8] for d in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "pos": pl.UInt32,
                "ref": pl.Utf8,
                "alt": pl.Utf8,
                "dp": pl.UInt64,
                "adf": pl.UInt64,
                "adr": pl.UInt64,
                "ad": pl.UInt64,
                "af": pl.Float64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


class PileupStatsCollector:
    """Collects pileup statistics across all references from CS tags.

    Uses retro-fix algorithm:
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
        self._stats_by_ref_pos: dict[str, dict[int, _PositionPileup]] = {}
        self._min_base_qual = min_base_qual
        self._all_refs = ref_names
        self._ref_lengths = ref_lengths
        self._faidx_reader = faidx_reader
        self._intervals = intervals

    def _read_overlaps_intervals(self, ref_name: str, read_start: int, read_end: int) -> bool:
        if self._intervals is None:
            return True
        chr_intervals = self._intervals.get(ref_name)
        if chr_intervals is None:
            return False
        return read_overlaps_intervals(read_start, read_end, chr_intervals)

    def _fetch_ref_base(self, ref_name: str, pos_0based: int) -> str:
        if self._faidx_reader is None:
            return "."
        try:
            seq = self._faidx_reader.fetch(ref_name, pos_0based, pos_0based + 1).upper()
            return seq if seq else "."
        except Exception as e:
            logger.warning("Failed to fetch reference base at {}:{} — {}", ref_name, pos_0based, e)
            return "."

    def tally_from_record(
        self,
        cs_tag: str,
        cigartuples: list[tuple[int, int]],
        record_pos: int,
        qual_scores: tuple[int, ...] | list[int] | Sequence[int],
        ref_name: str,
        is_reverse: bool,
        cs_regex: re.Pattern,
        qname: str,
        read_seq: bytes,
    ) -> None:
        # Check interval overlap
        ref_consuming_len = sum(length for op, length in cigartuples if op in (0, 2, 3, 7, 8))
        read_end = record_pos + ref_consuming_len
        if not self._read_overlaps_intervals(ref_name, record_pos, read_end):
            return

        is_forward = not is_reverse
        qual = list(qual_scores)

        def passes_quality_filter(rp: int) -> bool:
            return rp < len(qual) and qual[rp] >= self._min_base_qual

        # Capture intervals for this chromosome
        chr_intervals = self._intervals.get(ref_name) if self._intervals else None

        ref_stats = self._stats_by_ref_pos.setdefault(ref_name, {})

        # 1-BASED cursor
        ref_idx: int = record_pos + 1
        read_pos: int = 0

        clean = cs_tag.removeprefix("cs:Z:")
        ops = list(cs_regex.finditer(clean))

        for caps in ops:
            op_type = caps.group(1)
            val = caps.group(2)

            if op_type == ":":
                length = int(val)
                for i in range(length):
                    abs_ref_pos = ref_idx + i - 1  # convert 1-based to 0-based
                    if chr_intervals is not None and not position_in_intervals(
                        abs_ref_pos, chr_intervals
                    ):
                        continue
                    if passes_quality_filter(read_pos + i):
                        ref_base = self._fetch_ref_base(ref_name, abs_ref_pos)
                        pos_stats = ref_stats.get(abs_ref_pos)
                        if pos_stats is None:
                            pos_stats = _PositionPileup()
                            ref_stats[abs_ref_pos] = pos_stats
                        if pos_stats.ref_base is None:
                            pos_stats.ref_base = ref_base
                        pos_stats.record_allele(ref_base, is_forward)
                ref_idx += length
                read_pos += length

            elif op_type == "*":
                if len(val) >= 2:
                    alt_base_char = val[1].upper()
                    if passes_quality_filter(read_pos):
                        abs_ref_pos = ref_idx - 1
                        pos_stats = ref_stats.get(abs_ref_pos)
                        if pos_stats is None:
                            pos_stats = _PositionPileup()
                            ref_stats[abs_ref_pos] = pos_stats
                        ref_base_char = val[0].upper()
                        pos_stats.ref_base = ref_base_char
                        pos_stats.record_allele(alt_base_char, is_forward)
                ref_idx += 1
                read_pos += 1

            elif op_type == "+":
                inserted_seq = val.upper()
                anchor_pos_1based = ref_idx - 1
                abs_ref_pos = anchor_pos_1based - 1

                if passes_quality_filter(read_pos):
                    pos_stats = ref_stats.get(abs_ref_pos)
                    if pos_stats is None:
                        pos_stats = _PositionPileup()
                        ref_stats[abs_ref_pos] = pos_stats

                    # Remove REF count
                    ref_key = self._fetch_ref_base(ref_name, abs_ref_pos)
                    counts = pos_stats.allele_counts.get(ref_key)
                    if counts is not None:
                        if is_forward and counts[0] > 0:
                            counts[0] -= 1
                        elif not is_forward and counts[1] > 0:
                            counts[1] -= 1
                        if counts[2] > 0:
                            counts[2] -= 1
                        if pos_stats.dp > 0:
                            pos_stats.dp -= 1

                    # Add Insertion Alt
                    ins_key = f"+{inserted_seq}"
                    pos_stats.record_allele(ins_key, is_forward)

                read_pos += len(inserted_seq)
                # Do NOT advance ref_idx

            elif op_type == "-":
                deleted_seq = val.upper()
                del_len = len(deleted_seq)
                anchor_pos_1based = ref_idx - 1
                abs_ref_pos = anchor_pos_1based - 1

                if passes_quality_filter(read_pos):
                    pos_stats = ref_stats.get(abs_ref_pos)
                    if pos_stats is None:
                        pos_stats = _PositionPileup()
                        ref_stats[abs_ref_pos] = pos_stats

                    # Remove REF count
                    ref_key = self._fetch_ref_base(ref_name, abs_ref_pos)
                    counts = pos_stats.allele_counts.get(ref_key)
                    if counts is not None:
                        if is_forward and counts[0] > 0:
                            counts[0] -= 1
                        elif not is_forward and counts[1] > 0:
                            counts[1] -= 1
                        if counts[2] > 0:
                            counts[2] -= 1
                        if pos_stats.dp > 0:
                            pos_stats.dp -= 1

                    # Add Deletion Alt
                    del_key = f"-{deleted_seq}"
                    pos_stats.record_allele(del_key, is_forward)

                # Advance ref_idx over the gap
                ref_idx += del_len
                # Do NOT advance read_pos

    def write_results(self, path: Path) -> None:
        logger.info("Writing pileup statistics to {}", path)
        pileup_stats = _PileupStatsOutput()

        sorted_refs = sorted(self._all_refs)

        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref_pos.get(seq_name)
            if ref_stats is None:
                continue

            sorted_positions = sorted(ref_stats.keys())
            for pos in sorted_positions:
                pos_stats = ref_stats[pos]
                sorted_alleles = sorted(pos_stats.allele_counts.keys())

                for allele in sorted_alleles:
                    counts = pos_stats.allele_counts[allele]
                    adf, adr, ad = counts[0], counts[1], counts[2]
                    dp = pos_stats.dp
                    af = ad / dp if dp > 0 else 0.0
                    ref_base = pos_stats.ref_base or "N"

                    pileup_stats.add(
                        seq_name,
                        pos + 1,  # Convert to 1-based
                        ref_base,
                        allele,
                        dp,
                        adf,
                        adr,
                        ad,
                        af,
                    )

        pileup_stats.write_parquet(path)
        logger.info("Pileup statistics successfully written.")
