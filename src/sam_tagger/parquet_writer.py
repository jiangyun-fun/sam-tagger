"""Parquet writer module for statistics output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from loguru import logger


def write_df_to_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a polars DataFrame to a Parquet file with Zstd compression."""
    logger.info("Writing Parquet file to {}", path)
    df.write_parquet(path, compression="zstd", compression_level=3)
    logger.info("Successfully wrote Parquet file to {}", path)


@dataclass
class SimpleStats:
    """2-column statistics: (seq_name, count)."""

    stat_name: str
    data: list[tuple[str, int]] = field(default_factory=list)

    def add(self, seq_name: str, count: int) -> None:
        self.data.append((seq_name, count))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _ in self.data],
                "count": [c for _, c in self.data],
            },
            schema={"seq_name": pl.Utf8, "count": pl.UInt64},
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class TypedStats:
    """3-column statistics: (seq_name, type, count)."""

    stat_name: str
    data: list[tuple[str, str, int]] = field(default_factory=list)

    def add(self, seq_name: str, type_name: str, count: int) -> None:
        self.data.append((seq_name, type_name, count))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _ in self.data],
                "type": [t for _, t, _ in self.data],
                "count": [c for _, _, c in self.data],
            },
            schema={"seq_name": pl.Utf8, "type": pl.Utf8, "count": pl.UInt64},
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class PositionalErrorStats:
    """19-column statistics.

    Counts: (seq_name, position, insertion, deletion, smoothed_deletion,
    mismatch, to_a, to_c, to_g, to_t, total). Rates: (insertion_rate,
    deletion_rate, smoothed_deletion_rate, mismatch_rate, to_a_rate,
    to_c_rate, to_g_rate, to_t_rate) — count/total, 0.0 on zero depth.
    ``smoothed_deletion``/``smoothed_deletion_rate`` are Float64 and nullable
    — null when no reference sequence was available.
    """

    data: list[tuple] = field(default_factory=list)

    def add(
        self,
        seq_name: str,
        position: int,
        insertion: int,
        deletion: int,
        smoothed_deletion: float | None,
        mismatch: int,
        to_a: int,
        to_c: int,
        to_g: int,
        to_t: int,
        total: int,
        insertion_rate: float,
        deletion_rate: float,
        smoothed_deletion_rate: float | None,
        mismatch_rate: float,
        to_a_rate: float,
        to_c_rate: float,
        to_g_rate: float,
        to_t_rate: float,
    ) -> None:
        self.data.append(
            (
                seq_name, position, insertion, deletion, smoothed_deletion, mismatch,
                to_a, to_c, to_g, to_t, total, insertion_rate, deletion_rate,
                smoothed_deletion_rate, mismatch_rate, to_a_rate, to_c_rate,
                to_g_rate, to_t_rate,
            )
        )

    def to_dataframe(self) -> pl.DataFrame:
        rows = self.data
        return pl.DataFrame(
            {
                "seq_name": [r[0] for r in rows],
                "position": [r[1] for r in rows],
                "insertion": [r[2] for r in rows],
                "deletion": [r[3] for r in rows],
                "smoothed_deletion": [r[4] for r in rows],
                "mismatch": [r[5] for r in rows],
                "to_a": [r[6] for r in rows],
                "to_c": [r[7] for r in rows],
                "to_g": [r[8] for r in rows],
                "to_t": [r[9] for r in rows],
                "total": [r[10] for r in rows],
                "insertion_rate": [r[11] for r in rows],
                "deletion_rate": [r[12] for r in rows],
                "smoothed_deletion_rate": [r[13] for r in rows],
                "mismatch_rate": [r[14] for r in rows],
                "to_a_rate": [r[15] for r in rows],
                "to_c_rate": [r[16] for r in rows],
                "to_g_rate": [r[17] for r in rows],
                "to_t_rate": [r[18] for r in rows],
            },
            schema={
                "seq_name": pl.Utf8,
                "position": pl.UInt32,
                "insertion": pl.UInt64,
                "deletion": pl.UInt64,
                "smoothed_deletion": pl.Float64,
                "mismatch": pl.UInt64,
                "to_a": pl.UInt64,
                "to_c": pl.UInt64,
                "to_g": pl.UInt64,
                "to_t": pl.UInt64,
                "total": pl.UInt64,
                "insertion_rate": pl.Float64,
                "deletion_rate": pl.Float64,
                "smoothed_deletion_rate": pl.Float64,
                "mismatch_rate": pl.Float64,
                "to_a_rate": pl.Float64,
                "to_c_rate": pl.Float64,
                "to_g_rate": pl.Float64,
                "to_t_rate": pl.Float64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class SingleDeletionStats:
    """4-column statistics: (seq_name, position, deletion, single_deletion)."""

    data: list[tuple[str, int, int, int]] = field(default_factory=list)

    def add(self, seq_name: str, position: int, deletion: int, single_deletion: int) -> None:
        self.data.append((seq_name, position, deletion, single_deletion))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _, _ in self.data],
                "position": [p for _, p, _, _ in self.data],
                "deletion": [d for _, _, d, _ in self.data],
                "single_deletion": [sd for _, _, _, sd in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "position": pl.UInt32,
                "deletion": pl.UInt64,
                "single_deletion": pl.UInt64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class DistributionStats:
    """4-column distribution: (seq_name, metric, value, count)."""

    data: list[tuple[str, str, int, int]] = field(default_factory=list)

    def add(self, seq_name: str, metric: str, value: int, count: int) -> None:
        self.data.append((seq_name, metric, value, count))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _, _ in self.data],
                "metric": [m for _, m, _, _ in self.data],
                "value": [v for _, _, v, _ in self.data],
                "count": [c for _, _, _, c in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "metric": pl.Utf8,
                "value": pl.UInt32,
                "count": pl.UInt64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class RefSummaryStats:
    """3-column per-reference summary: (seq_name, max_min_rate, max_error_rate).

    ``max_min_rate`` = min depth / max depth over the reference positions
    (0.0 when max depth is 0). ``max_error_rate`` = max over positions of
    max(insertion_rate, deletion_rate, mismatch_rate), 0.0 when no errors.
    """

    data: list[tuple[str, float, float]] = field(default_factory=list)

    def add(self, seq_name: str, max_min_rate: float, max_error_rate: float) -> None:
        self.data.append((seq_name, max_min_rate, max_error_rate))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _ in self.data],
                "max_min_rate": [r for _, r, _ in self.data],
                "max_error_rate": [r for _, _, r in self.data],
            },
            schema={"seq_name": pl.Utf8, "max_min_rate": pl.Float64, "max_error_rate": pl.Float64},
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class IndelCompositionStats:
    """Insertion/deletion base composition: one row per reference.

    Columns: seq_name, total, {p}_a, {p}_c, {p}_g, {p}_t and one rate per
    base — count / total (0.0 on zero total), normalized inside the
    sequence. ``prefix`` is ``ins`` or ``del``. N-base events count in
    ``total`` but have no column, so the four rates can sum below 1.
    """

    prefix: str
    data: list[tuple[str, int, int, int, int, int]] = field(default_factory=list)

    def add(
        self, seq_name: str, total: int, a: int, c: int, g: int, t: int
    ) -> None:
        self.data.append((seq_name, total, a, c, g, t))

    def to_dataframe(self) -> pl.DataFrame:
        p = self.prefix
        cols: dict[str, list[object]] = {"seq_name": [], "total": []}
        for b in "acgt":
            cols[f"{p}_{b}"] = []
        for b in "acgt":
            cols[f"{p}_{b}_rate"] = []
        for seq_name, total, a, c, g, t in self.data:
            counts = {"a": a, "c": c, "g": g, "t": t}
            cols["seq_name"].append(seq_name)
            cols["total"].append(total)
            for b in "acgt":
                cols[f"{p}_{b}"].append(counts[b])
            for b in "acgt":
                cols[f"{p}_{b}_rate"].append((counts[b] / total) if total else 0.0)
        schema: dict[str, object] = {"seq_name": pl.Utf8, "total": pl.UInt64}
        for b in "acgt":
            schema[f"{p}_{b}"] = pl.UInt64
        for b in "acgt":
            schema[f"{p}_{b}_rate"] = pl.Float64
        return pl.DataFrame(cols, schema=schema)  # type: ignore[arg-type]

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class PerRefCountsStats:
    """7-column per-reference counts: seq_name, total, insertion, deletion,
    mismatch, and one rate per count — count / total (0.0 on zero total).

    Serves both ``per_base`` (total = aligned bases) and ``per_read``
    (total = tallied reads).
    """

    data: list[tuple[str, int, int, int, int]] = field(default_factory=list)

    def add(
        self, seq_name: str, total: int, insertion: int, deletion: int, mismatch: int
    ) -> None:
        self.data.append((seq_name, total, insertion, deletion, mismatch))

    def to_dataframe(self) -> pl.DataFrame:
        cols: dict[str, list[object]] = {"seq_name": [], "total": []}
        for kind in ("insertion", "deletion", "mismatch"):
            cols[kind] = []
            cols[f"{kind}_rate"] = []
        for seq_name, total, ins, dele, mis in self.data:
            counts = {"insertion": ins, "deletion": dele, "mismatch": mis}
            cols["seq_name"].append(seq_name)
            cols["total"].append(total)
            for kind in ("insertion", "deletion", "mismatch"):
                cols[kind].append(counts[kind])
                cols[f"{kind}_rate"].append((counts[kind] / total) if total else 0.0)
        return pl.DataFrame(
            cols,
            schema={
                "seq_name": pl.Utf8,
                "total": pl.UInt64,
                "insertion": pl.UInt64,
                "deletion": pl.UInt64,
                "mismatch": pl.UInt64,
                "insertion_rate": pl.Float64,
                "deletion_rate": pl.Float64,
                "mismatch_rate": pl.Float64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)


@dataclass
class MismatchLongStats:
    """5-column long-format mismatch table: (seq_name, ref, alt, count, rate).

    ``rate`` = count / the sequence's mismatch total (0.0 on zero total),
    normalized inside the sequence. Rows with N bases or ref == alt are
    not emitted; N mass keeps the rates summing below 1 when present.
    """

    data: list[tuple[str, str, str, int, float]] = field(default_factory=list)

    def add(self, seq_name: str, ref: str, alt: str, count: int, rate: float) -> None:
        self.data.append((seq_name, ref, alt, count, rate))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [r[0] for r in self.data],
                "ref": [r[1] for r in self.data],
                "alt": [r[2] for r in self.data],
                "count": [r[3] for r in self.data],
                "rate": [r[4] for r in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "ref": pl.Utf8,
                "alt": pl.Utf8,
                "count": pl.UInt64,
                "rate": pl.Float64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)
