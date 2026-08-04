"""BED interval parsing + interval queries (ported verbatim from py-bam-tagger)."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Interval:
    """A genomic interval with 0-based coordinates."""

    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"Interval start ({self.start}) must be less than end ({self.end})")

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def read_overlaps(self, read_start: int, read_end: int) -> bool:
        return read_start < self.end and self.start < read_end

    def merge(self, other: Interval) -> Interval:
        return Interval(start=min(self.start, other.start), end=max(self.end, other.end))


def parse_bed(path: Path) -> dict[str, list[Interval]]:
    """Parse a BED file and return merged, sorted intervals per chromosome."""
    raw: dict[str, list[Interval]] = {}
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Line {line_num} has fewer than 3 columns")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            if start >= end:
                raise ValueError(f"Line {line_num}: invalid interval (start {start} >= end {end})")
            raw.setdefault(chrom, []).append(Interval(start, end))

    merged: dict[str, list[Interval]] = {}
    for chrom, intervals in raw.items():
        intervals.sort(key=lambda i: i.start)
        _merge_intervals_sorted(intervals)
        merged[chrom] = intervals
    return merged


def _merge_intervals_sorted(intervals: list[Interval]) -> None:
    """Merge overlapping or abutting intervals in a sorted list (in-place)."""
    if len(intervals) <= 1:
        return
    merged: list[Interval] = [intervals[0]]
    for iv in intervals[1:]:
        if iv.start <= merged[-1].end:
            merged[-1] = merged[-1].merge(iv)
        else:
            merged.append(iv)
    intervals[:] = merged


def read_overlaps_intervals(read_start: int, read_end: int, intervals: list[Interval]) -> bool:
    """Binary search check if a read overlaps any interval."""
    idx = bisect_left(intervals, read_start, key=lambda i: i.end)
    for iv in intervals[idx:]:
        if iv.start >= read_end:
            break
        if read_start < iv.end and iv.start < read_end:
            return True
    return False


def position_in_intervals(pos: int, intervals: list[Interval]) -> bool:
    """Binary search check if a position falls within any interval."""
    idx = bisect_left(intervals, pos, key=lambda i: i.end)
    for iv in intervals[idx:]:
        if iv.start > pos:
            break
        if iv.start <= pos < iv.end:
            return True
    return False
