"""Error statistics collection (7-file cs-tag error analysis).

Tally/write logic is unchanged from the tested port; only construction was
decoupled — ``ErrorStatsOutputPaths.from_args`` takes an ``ErrorStatsArgs``
(prefix or 7 explicit paths) instead of the old mega-``Cli``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from sam_tagger.cli import ErrorStatsArgs
from sam_tagger.config import ErrorParams
from sam_tagger.parquet_writer import (
    MismatchMatrixStats,
    PositionalErrorStats,
    SingleDeletionStats,
    TypedStats,
)
from sam_tagger.processing import homopolymer_run_span


@dataclass(slots=True)
class _PositionalErrorCounts:
    insertions: int = 0
    deletions: int = 0
    smoothed_deletion: float | None = None
    mismatches: int = 0
    single_deletions: int = 0
    to_a: int = 0
    to_c: int = 0
    to_g: int = 0
    to_t: int = 0


@dataclass(slots=True)
class _PerReadCounts:
    total_reads: int = 0
    reads_with_insertion: int = 0
    reads_with_deletion: int = 0
    reads_with_mismatch: int = 0


@dataclass(slots=True)
class _PerBaseCounts:
    total_aligned_bases: int = 0
    insertions: int = 0
    deletions: int = 0
    mismatches: int = 0


@dataclass(slots=True)
class _PerRefErrorStats:
    positional_errors: dict[int, _PositionalErrorCounts] = field(default_factory=dict)
    # Difference array for per-position depth (samtools depth -J semantics):
    # `:`/`*`/`-` cover their span; `+` covers nothing. Not region-gated.
    depth_diff: dict[int, int] = field(default_factory=dict)
    # 5x5 ACGTN x ACGTN; the N-row (index 4) is tallied but excluded from output
    mismatch_matrix: list[list[int]] = field(default_factory=lambda: [[0] * 5 for _ in range(5)])
    insertion_summary: list[int] = field(default_factory=lambda: [0] * 5)
    deletion_summary: list[int] = field(default_factory=lambda: [0] * 5)
    per_read_summary: _PerReadCounts = field(default_factory=_PerReadCounts)
    per_base_summary: _PerBaseCounts = field(default_factory=_PerBaseCounts)


@dataclass
class ErrorStatsOutputPaths:
    error: Path
    mismatch: Path
    insertion: Path
    deletion: Path
    per_read: Path
    per_base: Path
    single_deletion: Path

    @classmethod
    def from_args(cls, args: ErrorStatsArgs) -> ErrorStatsOutputPaths:
        """Resolve the 7 output paths from either --prefix or 7 explicit paths."""
        if args.prefix is not None:
            prefix = args.prefix

            def _p(ext: str) -> Path:
                # Literal prefix semantics: append .{ext} verbatim. Dotted
                # stems (Sample.Ref) keep every segment; the old Path.stem
                # branch silently dropped the last dotted segment
                # (Sample.Ref -> Sample.error.parquet; found 2026-10-22).
                return Path(f"{prefix}.{ext}")

            return cls(
                error=_p("error.parquet"),
                mismatch=_p("mismatch.parquet"),
                insertion=_p("insertion.parquet"),
                deletion=_p("deletion.parquet"),
                per_read=_p("per_read.parquet"),
                per_base=_p("per_base.parquet"),
                single_deletion=_p("single_deletion.parquet"),
            )
        if all(
            [
                args.error_out,
                args.mismatch_out,
                args.insertion_out,
                args.deletion_out,
                args.per_read_out,
                args.per_base_out,
                args.single_deletion_out,
            ]
        ):
            return cls(
                error=args.error_out,  # type: ignore[arg-type]
                mismatch=args.mismatch_out,  # type: ignore[arg-type]
                insertion=args.insertion_out,  # type: ignore[arg-type]
                deletion=args.deletion_out,  # type: ignore[arg-type]
                per_read=args.per_read_out,  # type: ignore[arg-type]
                per_base=args.per_base_out,  # type: ignore[arg-type]
                single_deletion=args.single_deletion_out,  # type: ignore[arg-type]
            )
        raise ValueError(
            "error-stats requires either --prefix or all seven output paths "
            "(--error-out, --mismatch-out, --insertion-out, --deletion-out, "
            "--per-read-out, --per-base-out, --single-deletion-out)."
        )


def _base_to_idx(base: int) -> int | None:
    b = base | 0x20  # to lowercase
    if b == ord(b"a"):
        return 0
    if b == ord(b"c"):
        return 1
    if b == ord(b"g"):
        return 2
    if b == ord(b"t"):
        return 3
    if b == ord(b"n"):
        return 4
    return None


_BASE_ORDER = ["A", "C", "G", "T"]


class ErrorStatsCollector:
    """Collects error statistics across all references from CS tags.

    ``get_ref_seq`` (optional) maps a reference name to its sequence. When set,
    deletions also accumulate ``smoothed_deletion``: each deleted base spreads
    1/run_len over the maximal same-base reference run containing it, clipped
    to the error inner region. Without it, ``smoothed_deletion`` stays null —
    the cs tag's ``:`` ops carry lengths only, so runs are unknowable without
    the reference.
    """

    def __init__(
        self,
        error_params_map: dict[str, ErrorParams],
        paths: ErrorStatsOutputPaths | None = None,
        get_ref_seq: Callable[[str], str | None] | None = None,
    ) -> None:
        self._stats_by_ref: dict[str, _PerRefErrorStats] = {}
        self._error_params_map = error_params_map
        self._paths = paths
        self._get_ref_seq = get_ref_seq
        self._ref_seq_cache: dict[str, str | None] = {}
        self._warned_ref_mismatch = False

    def per_base_counts(self, ref_name: str) -> tuple[int, int, int, int] | None:
        """Return ``(aligned_bases, mismatches, insertions, deletions)`` for a ref.

        ``None`` means no cs-carrying read mapped to this reference. Lets a caller
        (e.g. ``qc-summary``) reuse the error tally without writing the 7-file output.
        """
        ref_stats = self._stats_by_ref.get(ref_name)
        if ref_stats is None:
            return None
        s = ref_stats.per_base_summary
        return (s.total_aligned_bases, s.mismatches, s.insertions, s.deletions)

    @staticmethod
    def _get_or_create_pos(
        positional_errors: dict[int, _PositionalErrorCounts], key: int
    ) -> _PositionalErrorCounts:
        entry = positional_errors.get(key)
        if entry is None:
            entry = _PositionalErrorCounts()
            positional_errors[key] = entry
        return entry

    def _smooth_deletion(
        self,
        ref_stats: _PerRefErrorStats,
        ref_name: str,
        pos0: int,
        base: int,
        params: ErrorParams,
    ) -> None:
        """Spread 1/run_len over the homopolymer run containing a deleted base.

        No reference access (``_get_ref_seq`` unset) -> no-op, the column stays
        null. Mass landing outside the inner region is dropped, mirroring how
        plain per-position deletions clip.
        """
        if self._get_ref_seq is None:
            return
        if ref_name not in self._ref_seq_cache:
            self._ref_seq_cache[ref_name] = self._get_ref_seq(ref_name)
        ref_seq = self._ref_seq_cache[ref_name]
        if not ref_seq or not (0 <= pos0 < len(ref_seq)):
            return
        # cs tags carry lowercase bases; the fasta sequence is uppercased on fetch.
        base_ch = chr(base).lower()
        start0, end0 = homopolymer_run_span(ref_seq, pos0)
        run_len = end0 - start0
        if run_len <= 1 or ref_seq[pos0].lower() != base_ch:
            if ref_seq[pos0].lower() != base_ch and not self._warned_ref_mismatch:
                self._warned_ref_mismatch = True
                logger.warning(
                    "FASTA base at {}:{} disagrees with the cs-deleted base — "
                    "is --fasta the same reference the BAM was aligned to? "
                    "Further mismatches will pass silently.",
                    ref_name,
                    pos0 + 1,
                )
            entry = ref_stats.positional_errors.get(pos0 + 1)
            if entry is None:
                entry = self._get_or_create_pos(ref_stats.positional_errors, pos0 + 1)
            entry.smoothed_deletion = (entry.smoothed_deletion or 0.0) + 1.0
            return
        share = 1.0 / run_len
        for q in range(start0, end0):
            if params.inner_region_start <= q < params.inner_region_end:
                entry = self._get_or_create_pos(ref_stats.positional_errors, q + 1)
                entry.smoothed_deletion = (entry.smoothed_deletion or 0.0) + share

    @staticmethod
    def _bump_depth(ref_stats: _PerRefErrorStats, start: int, n: int) -> None:
        """Mark [start, start+n) as covered by one more read (difference array)."""
        if n <= 0:
            return
        d = ref_stats.depth_diff
        d[start] = d.get(start, 0) + 1
        d[start + n] = d.get(start + n, 0) - 1

    @staticmethod
    def _depth_array(ref_stats: _PerRefErrorStats, seq_len: int) -> list[int]:
        """Prefix-sum the difference array into per-position depth (0-based index)."""
        depth = [0] * seq_len
        keys = sorted(ref_stats.depth_diff)
        cur = 0
        for k_idx, k in enumerate(keys):
            nxt = keys[k_idx + 1] if k_idx + 1 < len(keys) else seq_len
            cur += ref_stats.depth_diff[k]
            for i in range(max(k, 0), min(nxt, seq_len)):
                depth[i] = cur
        return depth

    def _get_or_create_ref_stats(self, ref_name: str) -> _PerRefErrorStats:
        ref_stats = self._stats_by_ref.get(ref_name)
        if ref_stats is None:
            ref_stats = _PerRefErrorStats()
            self._stats_by_ref[ref_name] = ref_stats
        return ref_stats

    def tally_from_parsed(
        self,
        parsed_ops: list[tuple[str, str]],
        record_pos: int,
        ref_name: str,
        qname: str,
    ) -> None:
        """Tally errors from pre-parsed CS operations (avoids re-regexing)."""
        params = self._error_params_map.get(ref_name)
        if params is None:
            return

        ref_stats = self._get_or_create_ref_stats(ref_name)

        current_pos = record_pos
        start_ref_pos = current_pos

        read_has_insertion = False
        read_has_deletion = False
        read_has_mismatch = False

        for op_type, val in parsed_ops:
            op_pos = current_pos

            is_in_error_region = (
                op_pos >= params.inner_region_start and op_pos < params.inner_region_end
            )

            if op_type == ":":
                self._bump_depth(ref_stats, op_pos, int(val))
                current_pos += int(val)

            elif op_type == "*":
                self._bump_depth(ref_stats, op_pos, 1)
                if is_in_error_region and len(val) >= 2:
                    ref_idx = _base_to_idx(ord(val[0]))
                    read_idx = _base_to_idx(ord(val[1]))
                    if ref_idx is not None and read_idx is not None:
                        ref_stats.mismatch_matrix[ref_idx][read_idx] += 1
                        entry = self._get_or_create_pos(ref_stats.positional_errors, op_pos + 1)
                        entry.mismatches += 1
                        # Destination-base split of the mismatch above; a read-base
                        # N (idx 4) has no column, so to_a+to_c+to_g+to_t <= mismatches.
                        if read_idx == 0:
                            entry.to_a += 1
                        elif read_idx == 1:
                            entry.to_c += 1
                        elif read_idx == 2:
                            entry.to_g += 1
                        elif read_idx == 3:
                            entry.to_t += 1
                        ref_stats.per_base_summary.mismatches += 1
                        read_has_mismatch = True
                current_pos += 1

            elif op_type == "+":
                if is_in_error_region:
                    bases = val.encode()
                    entry = self._get_or_create_pos(ref_stats.positional_errors, op_pos + 1)
                    entry.insertions += 1
                    ref_stats.per_base_summary.insertions += len(bases)
                    read_has_insertion = True
                    for base in bases:
                        idx = _base_to_idx(base)
                        if idx is not None:
                            ref_stats.insertion_summary[idx] += 1

            elif op_type == "-":
                bases = val.encode()
                length = len(bases)
                self._bump_depth(ref_stats, op_pos, length)  # -J: deletions count as covered
                is_single_del = length == 1
                del_end_pos = op_pos + length
                overlap_start = max(op_pos, params.inner_region_start)
                overlap_end = min(del_end_pos, params.inner_region_end)
                has_overlap = overlap_start < overlap_end

                if has_overlap:
                    read_has_deletion = True
                    ref_stats.per_base_summary.deletions += length
                    for i, base in enumerate(bases):
                        pos_of_del = op_pos + i
                        if params.inner_region_start <= pos_of_del < params.inner_region_end:
                            entry = self._get_or_create_pos(
                                ref_stats.positional_errors, pos_of_del + 1
                            )
                            entry.deletions += 1
                            if is_single_del:
                                entry.single_deletions += 1
                            idx = _base_to_idx(base)
                            if idx is not None:
                                ref_stats.deletion_summary[idx] += 1
                            self._smooth_deletion(ref_stats, ref_name, pos_of_del, base, params)
                current_pos += length

        aligned_len = current_pos - start_ref_pos
        ref_stats.per_base_summary.total_aligned_bases += aligned_len

        ref_stats.per_read_summary.total_reads += 1
        if read_has_insertion:
            ref_stats.per_read_summary.reads_with_insertion += 1
        if read_has_deletion:
            ref_stats.per_read_summary.reads_with_deletion += 1
        if read_has_mismatch:
            ref_stats.per_read_summary.reads_with_mismatch += 1

    def write_results(self) -> None:
        if self._paths is None:
            raise ValueError(
                "ErrorStatsCollector.write_results requires output paths "
                "(construct with paths=...)."
            )
        logger.info("Writing all error statistics...")
        paths = self._paths  # narrowed to ErrorStatsOutputPaths by the guard above
        self._write_error_stats(paths)
        self._write_mismatch_stats(paths)
        self._write_insertion_stats(paths)
        self._write_deletion_stats(paths)
        self._write_per_read_stats(paths)
        self._write_per_base_stats(paths)
        self._write_single_deletion_stats(paths)
        logger.info("All error statistics successfully written.")

    def _write_error_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing positional error statistics to {}", paths.error)
        pos_stats = PositionalErrorStats()
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        # null = smoothing unavailable (no --fasta); with a reference, eventless
        # positions read 0.0 like the integer columns.
        if self._get_ref_seq is None:

            def smoothed(v: float | None) -> float | None:
                return v
        else:

            def smoothed(v: float | None) -> float | None:
                return v if v is not None else 0.0

        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref.get(seq_name, default)
            seq_len = self._error_params_map[seq_name].seq_len
            depth = self._depth_array(ref_stats, seq_len)
            for i in range(1, seq_len + 1):
                counts = ref_stats.positional_errors.get(i, _PositionalErrorCounts())
                total = depth[i - 1]
                sm = smoothed(counts.smoothed_deletion)
                sm_rate = None if sm is None else (sm / total if total else 0.0)
                pos_stats.add(
                    seq_name,
                    i,
                    counts.insertions,
                    counts.deletions,
                    sm,
                    counts.mismatches,
                    counts.to_a,
                    counts.to_c,
                    counts.to_g,
                    counts.to_t,
                    total,
                    (counts.insertions / total) if total else 0.0,
                    (counts.deletions / total) if total else 0.0,
                    sm_rate,
                    (counts.mismatches / total) if total else 0.0,
                    (counts.to_a / total) if total else 0.0,
                    (counts.to_c / total) if total else 0.0,
                    (counts.to_g / total) if total else 0.0,
                    (counts.to_t / total) if total else 0.0,
                )
        pos_stats.write_parquet(paths.error)

    def _write_single_deletion_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing single deletion statistics to {}", paths.single_deletion)
        single_del_stats = SingleDeletionStats()
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref.get(seq_name, default)
            seq_len = self._error_params_map[seq_name].seq_len
            for i in range(1, seq_len + 1):
                counts = ref_stats.positional_errors.get(i, _PositionalErrorCounts())
                single_del_stats.add(seq_name, i, counts.deletions, counts.single_deletions)
        single_del_stats.write_parquet(paths.single_deletion)

    def _write_mismatch_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing mismatch matrix to {}", paths.mismatch)
        mismatch_stats = MismatchMatrixStats()
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        # Rust: ROW_ORDER = ['a', 't', 'c', 'g'], COL_ORDER_MAP = [0, 3, 1, 2]
        row_order = [ord(c) for c in "atcg"]
        col_map = [0, 3, 1, 2]
        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref.get(seq_name, default)
            for base_row in row_order:
                row_idx = _base_to_idx(base_row)
                if row_idx is None:
                    continue
                a = ref_stats.mismatch_matrix[row_idx][col_map[0]]
                t = ref_stats.mismatch_matrix[row_idx][col_map[1]]
                c = ref_stats.mismatch_matrix[row_idx][col_map[2]]
                g = ref_stats.mismatch_matrix[row_idx][col_map[3]]
                mismatch_stats.add(seq_name, chr(base_row).upper(), a, t, c, g)
        mismatch_stats.write_parquet(paths.mismatch)

    def _write_insertion_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing insertion summary to {}", paths.insertion)
        typed = TypedStats("insertion")
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref.get(seq_name, default)
            for i, base in enumerate(_BASE_ORDER):
                typed.add(seq_name, base, ref_stats.insertion_summary[i])
        typed.write_parquet(paths.insertion)

    def _write_deletion_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing deletion summary to {}", paths.deletion)
        typed = TypedStats("deletion")
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        for seq_name in sorted_refs:
            ref_stats = self._stats_by_ref.get(seq_name, default)
            for i, base in enumerate(_BASE_ORDER):
                typed.add(seq_name, base, ref_stats.deletion_summary[i])
        typed.write_parquet(paths.deletion)

    def _write_per_read_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing per_read statistics to {}", paths.per_read)
        typed = TypedStats("per_read")
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        for seq_name in sorted_refs:
            summary = self._stats_by_ref.get(seq_name, default).per_read_summary
            typed.add(seq_name, "total", summary.total_reads)
            typed.add(seq_name, "insertion", summary.reads_with_insertion)
            typed.add(seq_name, "deletion", summary.reads_with_deletion)
            typed.add(seq_name, "substitution", summary.reads_with_mismatch)
        typed.write_parquet(paths.per_read)

    def _write_per_base_stats(self, paths: ErrorStatsOutputPaths) -> None:
        logger.info("Writing per_base statistics to {}", paths.per_base)
        typed = TypedStats("per_base")
        sorted_refs = sorted(self._error_params_map.keys())
        default = _PerRefErrorStats()
        for seq_name in sorted_refs:
            summary = self._stats_by_ref.get(seq_name, default).per_base_summary
            typed.add(seq_name, "total", summary.total_aligned_bases)
            typed.add(seq_name, "insertion", summary.insertions)
            typed.add(seq_name, "deletion", summary.deletions)
            typed.add(seq_name, "substitution", summary.mismatches)
        typed.write_parquet(paths.per_base)
