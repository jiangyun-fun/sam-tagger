"""Tests for the Parquet writer module, ported from Rust parquet_writer.rs."""

from pathlib import Path

import polars as pl
import pytest

from sam_tagger.parquet_writer import (
    IndelCompositionStats,
    MismatchLongStats,
    PerRefCountsStats,
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
    stats.add("ref1", "both", 10)
    stats.add("ref1", "neither", 5)
    stats.add("ref2", "both", 20)
    df = stats.to_dataframe()
    assert df.height == 3
    assert df.width == 3
    assert df["seq_name"][0] == "ref1"
    assert df["type"][0] == "both"
    assert df["count"][0] == 10


def test_typed_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_typed.parquet"
    stats = TypedStats("AP")
    stats.add("ref1", "both", 100)
    stats.add("ref1", "neither", 50)
    stats.add("ref2", "both", 75)
    stats.write_parquet(path)
    assert path.exists()
    result = _read_typed_parquet(path)
    assert len(result) == 3
    assert ("ref1", "both", 100) in result
    assert ("ref1", "neither", 50) in result


def test_positional_error_stats_to_dataframe() -> None:
    stats = PositionalErrorStats()
    stats.add("ref1", 1, 5, 3, 2 / 3, 2, 0, 2, 0, 0, 6, 5 / 6, 0.5, 1 / 9, 1 / 3, 0, 1 / 3, 0, 0)
    stats.add("ref1", 2, 0, 1, None, 0, 0, 0, 0, 0, 2, 0.0, 0.5, None, 0.0, 0, 0, 0, 0)
    stats.add("ref2", 1, 10, 5, 1.0, 3, 1, 1, 1, 0, 12, 10 / 12, 5 / 12, 1 / 12, 0.25, 1 / 12, 1 / 12, 1 / 12, 0)
    df = stats.to_dataframe()
    assert df.height == 3
    assert df.width == 19
    assert df["position"][0] == 1
    assert df["position"][1] == 2
    assert df["to_c"][0] == 2
    assert df["to_g"][2] == 1
    assert df["smoothed_deletion"][0] == pytest.approx(2 / 3)
    assert df["smoothed_deletion"][1] is None
    assert df["total"][0] == 6
    assert df["insertion_rate"][0] == pytest.approx(5 / 6)
    assert df["smoothed_deletion_rate"][1] is None
    assert df["mismatch_rate"][2] == pytest.approx(0.25)


def test_positional_error_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_positional.parquet"
    stats = PositionalErrorStats()
    stats.add("ref1", 10, 5, 3, 0.5, 2, 1, 1, 0, 0, 8, 0.625, 0.375, 0.0625, 0.25, 0.125, 0.125, 0, 0)
    stats.add("ref1", 20, 0, 0, None, 1, 0, 0, 0, 1, 3, 0.0, 0.0, None, 1 / 3, 0, 0, 0, 1 / 3)
    stats.write_parquet(path)
    assert path.exists()
    df = pl.scan_parquet(path).collect()
    assert df.height == 2
    assert df.width == 19
    assert df["position"][0] == 10
    assert df["insertion"][0] == 5
    assert df["smoothed_deletion"][0] == pytest.approx(0.5)
    assert df["smoothed_deletion"][1] is None
    assert df["total"][1] == 3
    assert df["mismatch_rate"][1] == pytest.approx(1 / 3)


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


def test_indel_composition_stats_to_dataframe() -> None:
    # rates normalize inside the sequence; zero total -> all rates 0.0
    stats = IndelCompositionStats("del")
    stats.add("ref1", 10, 4, 3, 2, 1)
    stats.add("ref2", 0, 0, 0, 0, 0)
    df = stats.to_dataframe()
    assert df.columns == [
        "seq_name", "total",
        "del_a", "del_c", "del_g", "del_t",
        "del_a_rate", "del_c_rate", "del_g_rate", "del_t_rate",
    ]
    assert df["del_a"][0] == 4
    assert df["del_a_rate"][0] == pytest.approx(0.4)
    assert df["del_t_rate"][0] == pytest.approx(0.1)
    assert df["del_a_rate"][1] == 0.0  # zero total, no division error


def test_indel_composition_stats_ins_prefix(tmp_path: Path) -> None:
    path = tmp_path / "ins.parquet"
    stats = IndelCompositionStats("ins")
    stats.add("ref1", 8, 2, 2, 2, 2)
    stats.write_parquet(path)
    df = pl.read_parquet(path)
    assert df.columns == [
        "seq_name", "total",
        "ins_a", "ins_c", "ins_g", "ins_t",
        "ins_a_rate", "ins_c_rate", "ins_g_rate", "ins_t_rate",
    ]
    assert df["ins_a_rate"][0] == pytest.approx(0.25)


def test_per_ref_counts_stats_to_dataframe() -> None:
    stats = PerRefCountsStats()
    stats.add("ref1", 100, 5, 10, 20)
    stats.add("ref2", 0, 0, 0, 0)
    df = stats.to_dataframe()
    assert df.columns == [
        "seq_name", "total", "insertion", "deletion", "mismatch",
        "insertion_rate", "deletion_rate", "mismatch_rate",
    ]
    assert df["insertion_rate"][0] == pytest.approx(0.05)
    assert df["mismatch_rate"][0] == pytest.approx(0.2)
    assert df["deletion_rate"][1] == 0.0  # zero total, no division error


def test_mismatch_long_stats_to_dataframe() -> None:
    stats = MismatchLongStats()
    stats.add("ref1", "A", "C", 3, 1, 0.75, 0.3)
    stats.add("ref1", "G", "T", 0, 1, 0.25, 0.1)
    stats.add("ref1", "A", "A", 40, 20, None, 0.6)  # diagonal: rate null
    df = stats.to_dataframe()
    assert df.columns == ["seq_name", "ref", "alt", "adf", "adr", "rate", "norm_rate"]
    assert df.height == 3
    assert df["ref"][0] == "A" and df["alt"][0] == "C"
    assert (df["adf"][0], df["adr"][0]) == (3, 1)
    assert df["rate"][1] == pytest.approx(0.25)
    assert df["norm_rate"][2] == pytest.approx(0.6)
    assert df["rate"][2] is None


def test_mismatch_long_stats_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test_mismatch.parquet"
    stats = MismatchLongStats()
    stats.add("ref1", "G", "A", 5, 2, 0.7, 0.07)
    stats.write_parquet(path)
    df = pl.read_parquet(path)
    assert (df["adf"][0], df["adr"][0]) == (5, 2)
    assert df["rate"][0] == pytest.approx(0.7)
    assert df["norm_rate"][0] == pytest.approx(0.07)


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
