"""Tests for the Parquet writer module, ported from Rust parquet_writer.rs."""

from pathlib import Path

import polars as pl

from sam_tagger.parquet_writer import (
    MismatchMatrixStats,
    PositionalErrorStats,
    SimpleStats,
    SingleDeletionStats,
    TypedStats,
    write_df_to_parquet,
)


def _read_simple_parquet(path: Path) -> dict[str, int]:
    df = pl.scan_parquet(path).collect()
    return dict(zip(df["seq_name"].to_list(), df["count"].to_list(), strict=True))


def _read_typed_parquet(path: Path) -> list[tuple[str, str, int]]:
    df = pl.scan_parquet(path).collect()
    return list(zip(df["seq_name"].to_list(), df["type"].to_list(), df["count"].to_list(), strict=True))


def test_simple_stats_to_dataframe() -> None:
    stats = SimpleStats("test_stat")
    stats.add("ref1", 100)
    stats.add("ref2", 200)
    stats.add("ref1", 50)
    df = stats.to_dataframe()
    assert df.height == 3
    assert df.width == 2
    assert df["seq_name"][0] == "ref1"
    assert df["count"][0] == 100
    assert df["seq_name"][1] == "ref2"
    assert df["count"][1] == 200


def test_simple_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_simple.parquet"
    stats = SimpleStats("FL")
    stats.add("ref1", 42)
    stats.add("ref2", 99)
    stats.add("ref3", 0)
    stats.write_parquet(path)
    assert path.exists()
    result = _read_simple_parquet(path)
    assert result == {"ref1": 42, "ref2": 99, "ref3": 0}


def test_typed_stats_to_dataframe() -> None:
    stats = TypedStats("AP")
    stats.add("ref1", "Both", 10)
    stats.add("ref1", "Neither", 5)
    stats.add("ref2", "Both", 20)
    df = stats.to_dataframe()
    assert df.height == 3
    assert df.width == 3
    assert df["seq_name"][0] == "ref1"
    assert df["type"][0] == "Both"
    assert df["count"][0] == 10


def test_typed_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_typed.parquet"
    stats = TypedStats("AP")
    stats.add("ref1", "Both", 100)
    stats.add("ref1", "Neither", 50)
    stats.add("ref2", "Both", 75)
    stats.write_parquet(path)
    assert path.exists()
    result = _read_typed_parquet(path)
    assert len(result) == 3
    assert ("ref1", "Both", 100) in result
    assert ("ref1", "Neither", 50) in result


def test_positional_error_stats_to_dataframe() -> None:
    stats = PositionalErrorStats()
    stats.add("ref1", 1, 5, 3, 2)
    stats.add("ref1", 2, 0, 1, 0)
    stats.add("ref2", 1, 10, 5, 3)
    df = stats.to_dataframe()
    assert df.height == 3
    assert df.width == 5
    assert df["position"][0] == 1
    assert df["position"][1] == 2


def test_positional_error_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_positional.parquet"
    stats = PositionalErrorStats()
    stats.add("ref1", 10, 5, 3, 2)
    stats.add("ref1", 20, 0, 0, 1)
    stats.write_parquet(path)
    assert path.exists()
    df = pl.scan_parquet(path).collect()
    assert df.height == 2
    assert df.width == 5
    assert df["position"][0] == 10
    assert df["insertion"][0] == 5


def test_single_deletion_stats_to_dataframe() -> None:
    stats = SingleDeletionStats()
    stats.add("ref1", 5, 10, 8)
    stats.add("ref1", 10, 5, 5)
    df = stats.to_dataframe()
    assert df.height == 2
    assert df.width == 4
    assert df["position"][0] == 5
    assert df["deletion"][0] == 10
    assert df["single_deletion"][0] == 8


def test_single_deletion_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_single_del.parquet"
    stats = SingleDeletionStats()
    stats.add("ref1", 100, 50, 45)
    stats.write_parquet(path)
    assert path.exists()
    df = pl.scan_parquet(path).collect()
    assert df["single_deletion"][0] == 45


def test_mismatch_matrix_stats_to_dataframe() -> None:
    stats = MismatchMatrixStats()
    stats.add("ref1", "A", 0, 5, 2, 1)
    stats.add("ref1", "C", 3, 0, 0, 4)
    df = stats.to_dataframe()
    assert df.height == 2
    assert df.width == 6
    assert df["ref_base"][0] == "A"
    assert df["A"][0] == 0
    assert df["ref_base"][1] == "C"
    assert df["A"][1] == 3


def test_mismatch_matrix_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_mismatch.parquet"
    stats = MismatchMatrixStats()
    stats.add("ref1", "G", 10, 20, 30, 0)
    stats.write_parquet(path)
    assert path.exists()
    df = pl.scan_parquet(path).collect()
    assert df["ref_base"][0] == "G"
    assert df["G"][0] == 0


def test_write_df_to_parquet_creates_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "test_direct.parquet"
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    write_df_to_parquet(df, path)
    assert path.exists()
    read_df = pl.scan_parquet(path).collect()
    assert read_df.height == 3
    assert read_df.width == 2


def test_empty_stats_write_valid_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_empty.parquet"
    stats = SimpleStats("empty")
    stats.write_parquet(path)
    assert path.exists()
    df = pl.scan_parquet(path).collect()
    assert df.height == 0


def test_large_values_in_stats(tmp_path: Path) -> None:
    path = tmp_path / "test_large.parquet"
    stats = SimpleStats("large")
    stats.add("ref1", 2**64 - 1)  # u64 max
    stats.add("ref2", 1_000_000_000)
    stats.write_parquet(path)
    result = _read_simple_parquet(path)
    assert result["ref1"] == 2**64 - 1
    assert result["ref2"] == 1_000_000_000
