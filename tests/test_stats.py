"""Unit tests for the stats collectors, rewritten for sam-tagger's decoupled API.

Covers StatsCollector (tag stats), ErrorStatsCollector (cs error tally + path
resolution), and PileupStatsCollector (incl. interval filtering) without going
through the CLI/orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sam_tagger.bed import Interval
from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs
from sam_tagger.config import AllConfigParams, ApParams, ErrorParams, TcParams
from sam_tagger.processing import CS_REGEX, parse_cs_tag
from sam_tagger.stats.error_stats import (
    ErrorStatsCollector,
    ErrorStatsOutputPaths,
    _base_to_idx,
)
from sam_tagger.stats.pileup_stats import PileupStatsCollector
from sam_tagger.stats.tag_stats import StatsCollector

# --------------------------------------------------------------------------
# Tag stats
# --------------------------------------------------------------------------


def _ap() -> ApParams:
    return ApParams(
        primer_start_end_pos=9,
        primer_end_start_pos=90,
        primer_start_seq=b"ACGT",
        primer_end_seq=b"TGCA",
        primer_start_min_match=4,
        primer_end_min_match=4,
    )


def _tag_collector(
    tmp_path: Path,
    *,
    fl: bool = False,
    tc: bool = False,
    ap: bool = False,
    nb: bool = False,
    refs: list[str] | None = None,
) -> StatsCollector:
    args = AnnotateArgs(
        input_bam=Path("in.bam"),
        output_bam=Path("out.bam"),
        fl_stats=tmp_path / "fl.parquet" if fl else None,
        tc_stats=tmp_path / "tc.parquet" if tc else None,
        ap_stats=tmp_path / "ap.parquet" if ap else None,
        nb_stats=tmp_path / "nb.parquet" if nb else None,
    )
    config = AllConfigParams(
        tc={"ref1": TcParams(100, 10, 10), "ref2": TcParams(200, 10, 10)} if tc else None,
        ap={"ref1": _ap(), "ref2": _ap()} if ap else None,
        nb={"ref1": 3, "ref2": 3} if nb else None,
    )
    return StatsCollector(args, config, refs or ["ref1", "ref2"])


def test_tag_collector_fl_zero_fill(tmp_path: Path) -> None:
    # FL zero-fills across ALL header refs (ref3 has no full-length read).
    sc = _tag_collector(tmp_path, fl=True, refs=["ref1", "ref2", "ref3"])
    sc.record_fl("ref1")
    sc.record_fl("ref1")
    sc.record_fl("ref2")
    sc.write_results()

    df = pl.read_parquet(tmp_path / "fl.parquet")
    result = dict(zip(df["seq_name"].to_list(), df["count"].to_list(), strict=True))
    assert result == {"ref1": 2, "ref2": 1, "ref3": 0}


def test_tag_collector_tc(tmp_path: Path) -> None:
    sc = _tag_collector(tmp_path, tc=True)
    sc.record_tc("ref1")
    sc.write_results()
    df = pl.read_parquet(tmp_path / "tc.parquet")
    assert dict(zip(df["seq_name"].to_list(), df["count"].to_list(), strict=True)) == {
        "ref1": 1,
        "ref2": 0,
    }


def test_tag_collector_ap(tmp_path: Path) -> None:
    sc = _tag_collector(tmp_path, ap=True)
    sc.record_ap("ref1", "both")
    sc.record_ap("ref1", "both")
    sc.record_ap("ref1", "start_only")
    sc.record_ap("ref2", "neither")
    sc.write_results()
    df = pl.read_parquet(tmp_path / "ap.parquet")
    rows = {(r["seq_name"], r["type"]): r["count"] for r in df.iter_rows(named=True)}
    # every (ref × 4 statuses) is emitted, zero-filled
    assert rows[("ref1", "both")] == 2
    assert rows[("ref1", "start_only")] == 1
    assert rows[("ref1", "neither")] == 0
    assert rows[("ref2", "neither")] == 1


def test_tag_collector_nb_with_total(tmp_path: Path) -> None:
    sc = _tag_collector(tmp_path, nb=True)
    sc.record_nb("ref1", "ACG")
    sc.record_nb("ref1", "ACG")
    sc.record_nb("ref1", "TGA")
    sc.record_nb("ref2", "ACG")
    sc.write_results()
    df = pl.read_parquet(tmp_path / "nb.parquet")
    rows = {(r["seq_name"], r["type"]): r["count"] for r in df.iter_rows(named=True)}
    assert rows[("ref1", "ACG")] == 2
    assert rows[("ref1", "TGA")] == 1
    assert rows[("total", "ACG")] == 3  # summed across refs


# --------------------------------------------------------------------------
# Error stats
# --------------------------------------------------------------------------


def _err_paths(tmp_path: Path) -> ErrorStatsOutputPaths:
    return ErrorStatsOutputPaths(
        error=tmp_path / "error.parquet",
        mismatch=tmp_path / "mismatch.parquet",
        insertion=tmp_path / "insertion.parquet",
        deletion=tmp_path / "deletion.parquet",
        per_read=tmp_path / "per_read.parquet",
        per_base=tmp_path / "per_base.parquet",
        single_deletion=tmp_path / "single_deletion.parquet",
    )


def test_base_to_idx() -> None:
    assert _base_to_idx(ord("a")) == 0
    assert _base_to_idx(ord("A")) == 0
    assert _base_to_idx(ord("c")) == 1
    assert _base_to_idx(ord("g")) == 2
    assert _base_to_idx(ord("t")) == 3
    assert _base_to_idx(ord("n")) == 4
    assert _base_to_idx(ord("X")) is None


def test_error_stats_tally_mismatch(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":10*ac:79"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.per_read_summary.total_reads == 1
    assert st.per_read_summary.reads_with_mismatch == 1
    assert st.per_base_summary.mismatches == 1
    assert st.positional_errors[11].mismatches == 1  # pos 10 (0-based) -> 11 (1-based)
    assert st.positional_errors[11].to_c == 1  # *ac: read base c -> to_c
    assert st.positional_errors[11].to_a == 0


def test_error_stats_tally_mismatch_n_read_base(tmp_path: Path) -> None:
    # read-base N counts in mismatch but in no to_* column; ref-base N lands in to_*
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":10*an:10*na:78"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.positional_errors[11].mismatches == 1  # *an
    assert st.positional_errors[11].to_a == 0
    assert st.positional_errors[22].mismatches == 1  # *na
    assert st.positional_errors[22].to_a == 1


def test_error_stats_tally_insertion(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":20+at:79"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.per_read_summary.reads_with_insertion == 1
    assert st.per_base_summary.insertions == 2
    assert st.positional_errors[21].insertions == 1


def test_error_stats_tally_deletion(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":20-gt:78"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.per_read_summary.reads_with_deletion == 1
    assert st.per_base_summary.deletions == 2
    assert st.positional_errors[21].deletions == 1
    assert st.positional_errors[22].deletions == 1


def test_error_stats_ignores_outside_region(tmp_path: Path) -> None:
    # mismatch at pos 5 is in the start primer, outside inner [10,90)
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":5*ac:93"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.per_base_summary.mismatches == 0
    assert st.per_read_summary.reads_with_mismatch == 0


def test_error_stats_unknown_ref_skipped(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    col = ErrorStatsCollector(params, _err_paths(tmp_path))
    col.tally_from_parsed(parse_cs_tag(":50*ac:49"), 0, "not-in-config", "r1")
    assert col._stats_by_ref == {}  # nothing tallied


def test_error_stats_write_all_seven(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=100, inner_region_start=10, inner_region_end=90)}
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(params, paths)
    col.tally_from_parsed(parse_cs_tag(":20*ac+g-gt:76"), 0, "ref1", "r1")
    col.write_results()
    for p in (paths.error, paths.mismatch, paths.insertion, paths.deletion,
              paths.per_read, paths.per_base, paths.single_deletion):
        assert p.exists()
    df = pl.read_parquet(paths.error)
    assert df.columns == [
        "seq_name", "position", "insertion", "deletion", "smoothed_deletion", "mismatch",
        "to_a", "to_c", "to_g", "to_t", "total",
        "insertion_rate", "deletion_rate", "smoothed_deletion_rate", "mismatch_rate",
        "to_a_rate", "to_c_rate", "to_g_rate", "to_t_rate",
    ]
    # no get_ref_seq -> the whole smoothed column is null
    assert df["smoothed_deletion"].null_count() == df.height
    row = df.filter(pl.col("position") == 21)  # *ac at 0-based 20 -> 1-based 21
    assert row.height == 1
    assert row["mismatch"].item() == 1
    assert row["to_c"].item() == 1
    assert row["to_a"].item() == 0
    assert row["to_g"].item() == 0
    assert row["to_t"].item() == 0


def test_error_paths_from_prefix(tmp_path: Path) -> None:
    args = ErrorStatsArgs(input_bam=Path("in.bam"), config=Path("c.toml"), prefix=tmp_path / "stats")
    p = ErrorStatsOutputPaths.from_args(args)
    assert p.error == tmp_path / "stats.error.parquet"
    assert p.per_base == tmp_path / "stats.per_base.parquet"
    assert p.single_deletion == tmp_path / "stats.single_deletion.parquet"


def test_error_paths_from_prefix_with_suffix(tmp_path: Path) -> None:
    # prefix that already has a suffix keeps the stem
    args = ErrorStatsArgs(input_bam=Path("in.bam"), config=Path("c.toml"), prefix=tmp_path / "stats.tsv")
    p = ErrorStatsOutputPaths.from_args(args)
    assert p.error == tmp_path / "stats.error.parquet"


def test_error_paths_from_explicit(tmp_path: Path) -> None:
    args = ErrorStatsArgs(
        input_bam=Path("in.bam"), config=Path("c.toml"),
        error_out=tmp_path / "e.parquet", mismatch_out=tmp_path / "m.parquet",
        insertion_out=tmp_path / "i.parquet", deletion_out=tmp_path / "d.parquet",
        per_read_out=tmp_path / "pr.parquet", per_base_out=tmp_path / "pb.parquet",
        single_deletion_out=tmp_path / "sd.parquet",
    )
    p = ErrorStatsOutputPaths.from_args(args)
    assert p.error == tmp_path / "e.parquet" and p.single_deletion == tmp_path / "sd.parquet"


def test_error_paths_from_args_missing_raises(tmp_path: Path) -> None:
    # neither prefix nor all 7 explicit -> ValueError
    args = ErrorStatsArgs(input_bam=Path("in.bam"), config=Path("c.toml"))
    with pytest.raises(ValueError):
        ErrorStatsOutputPaths.from_args(args)


# --------------------------------------------------------------------------
# Pileup stats (collector level, incl. interval filtering)
# --------------------------------------------------------------------------


def test_pileup_snp_no_fasta(tmp_path: Path) -> None:
    # Without a FASTA, ref base for matches is "."; SNP ref base comes from cs.
    col = PileupStatsCollector(
        min_base_qual=20, ref_names=["ref1"], ref_lengths={"ref1": 100},
    )
    col.tally_from_record(
        ":9*at:20", [(0, 30)], 0, (40,) * 30, "ref1", False, CS_REGEX, "r1", b"A" * 30,
    )
    out = tmp_path / "pile.parquet"
    col.write_results(out)
    df = pl.read_parquet(out)
    snp = df.filter((pl.col("pos") == 10) & (pl.col("alt") == "T"))
    assert snp.height == 1
    row = snp.row(0, named=True)
    assert row["ref"] == "A" and row["dp"] == 1 and row["ad"] == 1


def test_pileup_intervals_restrict_positions(tmp_path: Path) -> None:
    # BED [5,15) -> only positions 6..15 (1-based) reported; ref2 read skipped.
    col = PileupStatsCollector(
        min_base_qual=0,
        ref_names=["ref1", "ref2"],
        ref_lengths={"ref1": 30, "ref2": 30},
        intervals={"ref1": [Interval(5, 15)]},
    )
    for ref in ("ref1", "ref2"):
        col.tally_from_record(
            ":30", [(0, 30)], 0, (40,) * 30, ref, False, CS_REGEX, ref, b"A" * 30,
        )
    out = tmp_path / "pile.parquet"
    col.write_results(out)
    df = pl.read_parquet(out).select("seq_name", "pos").sort("pos")
    assert set(df["seq_name"]) == {"ref1"}  # ref2 read filtered out
    assert df["pos"].to_list() == list(range(6, 16))


def test_pileup_quality_filter(tmp_path: Path) -> None:
    # min_base_qual=40 drops the SNP base (qual 30)
    col = PileupStatsCollector(
        min_base_qual=40, ref_names=["ref1"], ref_lengths={"ref1": 30},
    )
    col.tally_from_record(
        ":9*at:20", [(0, 30)], 0, (30,) * 30, "ref1", False, CS_REGEX, "r1", b"A" * 30,
    )
    out = tmp_path / "pile.parquet"
    col.write_results(out)
    assert pl.read_parquet(out).height == 0  # all bases below qual threshold


# --------------------------------------------------------------------------
# smoothed_deletion (homopolymer re-attribution)
# --------------------------------------------------------------------------

_HP_REF = "TTTAAATTT"


def _hp_params(inner_end: int = 9) -> dict[str, ErrorParams]:
    return {"hp": ErrorParams(seq_len=9, inner_region_start=0, inner_region_end=inner_end)}


def _hp_get(seq: str):
    return {"hp": seq}.get


def test_smoothed_deletion_user_example(tmp_path: Path) -> None:
    # Same biological deletion placed at two different positions by two alignments.
    col = ErrorStatsCollector(_hp_params(), _err_paths(tmp_path), get_ref_seq=_hp_get(_HP_REF))
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "p4")
    col.tally_from_parsed(parse_cs_tag(":4-a:4"), 0, "hp", "p5")
    st = col._stats_by_ref["hp"]
    assert st.positional_errors[4].smoothed_deletion == pytest.approx(2 / 3)
    assert st.positional_errors[5].smoothed_deletion == pytest.approx(2 / 3)
    assert st.positional_errors[6].smoothed_deletion == pytest.approx(2 / 3)
    # plain deletion still carries the placement artifact; mass is conserved
    assert st.positional_errors[4].deletions == 1
    assert st.positional_errors[5].deletions == 1
    total_plain = sum(e.deletions for e in st.positional_errors.values())
    total_smooth = sum(e.smoothed_deletion or 0.0 for e in st.positional_errors.values())
    assert total_smooth == pytest.approx(total_plain)


def test_smoothed_deletion_two_bases(tmp_path: Path) -> None:
    # TTTATTT over TTTAAATTT: two As deleted from the AAA run -> 2/3 at each position.
    col = ErrorStatsCollector(_hp_params(), _err_paths(tmp_path), get_ref_seq=_hp_get(_HP_REF))
    col.tally_from_parsed(parse_cs_tag(":3-aa:3"), 0, "hp", "r1")
    st = col._stats_by_ref["hp"]
    for pos in (4, 5, 6):
        assert st.positional_errors[pos].smoothed_deletion == pytest.approx(2 / 3)


def test_smoothed_deletion_non_homopolymer(tmp_path: Path) -> None:
    # Deleting the C of ACGT (run length 1) smooths to itself only.
    col = ErrorStatsCollector(
        {"ref1": ErrorParams(seq_len=100, inner_region_start=0, inner_region_end=100)},
        _err_paths(tmp_path),
        get_ref_seq={"ref1": "A" * 20 + "CGTT" + "A" * 76}.get,
    )
    col.tally_from_parsed(parse_cs_tag(":20-c:79"), 0, "ref1", "r1")
    st = col._stats_by_ref["ref1"]
    assert st.positional_errors[21].smoothed_deletion == pytest.approx(1.0)
    assert st.positional_errors[21].deletions == 1


def test_smoothed_deletion_region_clipping(tmp_path: Path) -> None:
    # Inner region [0, 5): the AAA run (0-based 3-5) loses its last position.
    col = ErrorStatsCollector(_hp_params(inner_end=5), _err_paths(tmp_path),
                              get_ref_seq=_hp_get(_HP_REF))
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "r1")
    st = col._stats_by_ref["hp"]
    assert st.positional_errors[4].smoothed_deletion == pytest.approx(1 / 3)
    assert st.positional_errors[5].smoothed_deletion == pytest.approx(1 / 3)
    assert 6 not in st.positional_errors or not st.positional_errors[6].smoothed_deletion


def test_smoothed_deletion_null_without_ref_seq(tmp_path: Path) -> None:
    # No get_ref_seq -> the written column is null everywhere, others unchanged.
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(_hp_params(), paths)
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "r1")
    col.write_results()
    df = pl.read_parquet(paths.error)
    assert "smoothed_deletion" in df.columns
    assert df["smoothed_deletion"].null_count() == df.height
    assert df["deletion"].sum() == 1


# --------------------------------------------------------------------------
# depth total + rate columns
# --------------------------------------------------------------------------


def _read_error_parquet(paths) -> pl.DataFrame:
    return pl.read_parquet(paths.error)


def test_depth_total_j_semantics(tmp_path: Path) -> None:
    # Both hp reads span the whole reference; deletions count as covered (-J).
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(_hp_params(), paths)
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "p4")
    col.tally_from_parsed(parse_cs_tag(":4-a:4"), 0, "hp", "p5")
    col.write_results()
    df = _read_error_parquet(paths)
    assert df["total"].to_list() == [2] * 9
    # deletion rate: 1 deletion at pos 4 and 5, over total 2
    rates = dict(zip(df["position"].to_list(), df["deletion_rate"].to_list(), strict=True))
    assert rates[4] == pytest.approx(0.5)
    assert rates[5] == pytest.approx(0.5)
    assert rates[6] == pytest.approx(0.0)


def test_depth_total_clean_read_and_insertion(tmp_path: Path) -> None:
    params = {"ref1": ErrorParams(seq_len=20, inner_region_start=0, inner_region_end=20)}
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(params, paths)
    col.tally_from_parsed(parse_cs_tag(":20"), 0, "ref1", "clean")
    col.tally_from_parsed(parse_cs_tag(":10+g:5"), 0, "ref1", "ins")
    col.write_results()
    df = _read_error_parquet(paths)
    total = dict(zip(df["position"].to_list(), df["total"].to_list(), strict=True))
    # clean read covers 1..20; the 15bp insertion read covers 1..15 (insertion adds no ref span)
    assert all(total[p] == 2 for p in range(1, 16))
    assert all(total[p] == 1 for p in range(16, 21))
    assert df["insertion"].sum() == 1
    ins_rate = dict(zip(df["position"].to_list(), df["insertion_rate"].to_list(), strict=True))
    assert ins_rate[11] == pytest.approx(0.5)  # insertion attributed to pos after op


def test_depth_total_not_region_gated(tmp_path: Path) -> None:
    # Region [3, 9): deletion outside the region is not counted, but coverage is.
    paths = _err_paths(tmp_path)
    col2 = ErrorStatsCollector({"hp": ErrorParams(seq_len=9, inner_region_start=3, inner_region_end=9)}, paths)
    col2.tally_from_parsed(parse_cs_tag(":2-a:6"), 0, "hp", "r1")
    col2.write_results()
    df = _read_error_parquet(paths)
    total = dict(zip(df["position"].to_list(), df["total"].to_list(), strict=True))
    assert total[1] == 1 and total[2] == 1  # outside region, still covered
    assert df["deletion"].sum() == 0  # deletion at 0-based 2 is outside [3,9)
    assert df["mismatch_rate"].to_list() == [0.0] * 9  # zero-total guard path
    assert df["total"].to_list() == [1] * 9


def test_depth_rates_with_smoothed(tmp_path: Path) -> None:
    # With fasta: smoothed 2/3 over total 2 -> rate 1/3 at 4,5,6.
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(_hp_params(), paths, get_ref_seq=_hp_get(_HP_REF))
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "p4")
    col.tally_from_parsed(parse_cs_tag(":4-a:4"), 0, "hp", "p5")
    col.write_results()
    df = _read_error_parquet(paths)
    sr = dict(zip(df["position"].to_list(), df["smoothed_deletion_rate"].to_list(), strict=True))
    assert sr[4] == pytest.approx(1 / 3)
    assert sr[5] == pytest.approx(1 / 3)
    assert sr[6] == pytest.approx(1 / 3)
    assert sr[1] == 0.0


def test_depth_rate_null_propagation_without_fasta(tmp_path: Path) -> None:
    paths = _err_paths(tmp_path)
    col = ErrorStatsCollector(_hp_params(), paths)
    col.tally_from_parsed(parse_cs_tag(":3-a:5"), 0, "hp", "p4")
    col.write_results()
    df = _read_error_parquet(paths)
    assert df["smoothed_deletion"].null_count() == df.height
    assert df["smoothed_deletion_rate"].null_count() == df.height
    # other rates are real numbers
    assert df["deletion_rate"].null_count() == 0
