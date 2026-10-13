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
    """9-column statistics.

    Columns: (seq_name, position, insertion, deletion, mismatch, to_a, to_c, to_g, to_t).
    """

    data: list[tuple[str, int, int, int, int, int, int, int, int]] = field(default_factory=list)

    def add(
        self,
        seq_name: str,
        position: int,
        insertion: int,
        deletion: int,
        mismatch: int,
        to_a: int,
        to_c: int,
        to_g: int,
        to_t: int,
    ) -> None:
        self.data.append(
            (seq_name, position, insertion, deletion, mismatch, to_a, to_c, to_g, to_t)
        )

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "seq_name": [n for n, _, _, _, _, _, _, _, _ in self.data],
                "position": [p for _, p, _, _, _, _, _, _, _ in self.data],
                "insertion": [i for _, _, i, _, _, _, _, _, _ in self.data],
                "deletion": [d for _, _, _, d, _, _, _, _, _ in self.data],
                "mismatch": [m for _, _, _, _, m, _, _, _, _ in self.data],
                "to_a": [a for _, _, _, _, _, a, _, _, _ in self.data],
                "to_c": [c for _, _, _, _, _, _, c, _, _ in self.data],
                "to_g": [g for _, _, _, _, _, _, _, g, _ in self.data],
                "to_t": [t for _, _, _, _, _, _, _, _, t in self.data],
            },
            schema={
                "seq_name": pl.Utf8,
                "position": pl.UInt32,
                "insertion": pl.UInt64,
                "deletion": pl.UInt64,
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
