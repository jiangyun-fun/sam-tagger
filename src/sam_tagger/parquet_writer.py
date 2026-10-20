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
    """10-column statistics.

    Columns: (seq_name, position, insertion, deletion, smoothed_deletion,
    mismatch, to_a, to_c, to_g, to_t). ``smoothed_deletion`` is Float64 and
    nullable — null when no reference sequence was available.
    """

    data: list[tuple[str, int, int, int, float | None, int, int, int, int, int]] = field(
        default_factory=list
    )

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
    ) -> None:
        self.data.append(
            (
                seq_name,
                position,
                insertion,
                deletion,
                smoothed_deletion,
                mismatch,
                to_a,
                to_c,
                to_g,
                to_t,
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
class MismatchMatrixStats:
    """6-column statistics: (seq_name, ref_base, A, T, C, G)."""

    data: list[tuple[str, str, int, int, int, int]] = field(default_factory=list)

    def add(self, seq_name: str, ref_base: str, a: int, t: int, c: int, g: int) -> None:
        self.data.append((seq_name, ref_base, a, t, c, g))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _, _, _, _ in self.data],
                "ref_base": [b for _, b, _, _, _, _ in self.data],
                "A": [a for _, _, a, _, _, _ in self.data],
                "T": [t for _, _, _, t, _, _ in self.data],
                "C": [c for _, _, _, _, c, _ in self.data],
                "G": [g for _, _, _, _, _, g in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "ref_base": pl.Utf8,
                "A": pl.UInt64,
                "T": pl.UInt64,
                "C": pl.UInt64,
                "G": pl.UInt64,
            },
        )

    def write_parquet(self, path: Path) -> None:
        write_df_to_parquet(self.to_dataframe(), path)
