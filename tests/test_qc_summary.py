"""Tests for the ``qc-summary`` subcommand (per-reference QC summary Parquet)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from _helpers import setup_env

from sam_tagger.cli import ErrorStatsArgs, QcSummaryArgs, app
from sam_tagger.lib import run_error_stats, run_qc_summary

Q = "I"  # Phred 40
HDR_3 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28\n@SQ\tSN:seq2\tLN:26\n@SQ\tSN:seq3\tLN:30"
HDR_1_28 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28"


def _read_qc(path: Path) -> dict[str, dict]:
    """seq_name -> full row (as a dict)."""
    df = pl.read_parquet(path)
    return {r["seq_name"]: r for r in df.iter_rows(named=True)}


def _read_per_base(path: Path) -> dict[str, dict[str, int]]:
    """seq_name -> {total/insertion/deletion/substitution: count} from error-stats per_base."""
    out: dict[str, dict[str, int]] = {}
    for r in pl.read_parquet(path).iter_rows(named=True):
        out.setdefault(r["seq_name"], {})[r["type"]] = int(r["count"])
    return out


# ---------------------------------------------------------------------------
# 1. multi-ref, tc + error features, incl. a zero-read ref (seq3)
# ---------------------------------------------------------------------------


def test_qc_summary_multi_ref(tmp_path: Path) -> None:
    cfg = {
        "seq1": {
            "tc_primer_start_len": 8,
            "tc_primer_end_len": 8,
            "error_primer_start_len": 5,
            "error_primer_end_len": 5,
        },
        "seq2": {
            "tc_primer_start_len": 5,
            "tc_primer_end_len": 5,
            "error_primer_start_len": 5,
            "error_primer_end_len": 5,
        },
    }
    sam = (
        f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\tcs:Z::28\n"
        f"r2\t0\tseq1\t1\t60\t24M\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\tcs:Z::10*at:13\n"
        f"r3\t0\tseq2\t1\t60\t26M\t*\t0\t0\t{'AC' * 13}\t{Q * 26}\tcs:Z::26\n"
    )
    setup = setup_env(tmp_path, cfg, HDR_3, sam)
    assert setup.config_path is not None
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out, config=setup.config_path))

    qc = _read_qc(out)
    assert set(qc) == {"seq1", "seq2", "seq3"}  # every header contig, incl. zero-read seq3

    # ref_len mirrors the BAM header LN for every reference, zero-read included
    assert qc["seq1"]["ref_len"] == 28
    assert qc["seq2"]["ref_len"] == 26
    assert qc["seq3"]["ref_len"] == 30

    # seq1: 2 reads, 1 full-length (r1), 1 on-target (r1), 1 mismatch (r2), aligned 28+24.
    assert qc["seq1"]["total_reads"] == 2
    assert qc["seq1"]["fl"] == 1
    assert qc["seq1"]["fl_rate"] == pytest.approx(0.5)
    assert qc["seq1"]["on_target"] == 1
    assert qc["seq1"]["on_target_rate"] == pytest.approx(0.5)
    assert qc["seq1"]["mismatches"] == 1
    assert qc["seq1"]["insertions"] == 0
    assert qc["seq1"]["deletions"] == 0
    assert qc["seq1"]["aligned_bases"] == 52
    assert qc["seq1"]["error_rate"] == pytest.approx(1 / 52)
    # no ap config -> all ap buckets 0
    assert qc["seq1"]["ap_both"] == qc["seq1"]["ap_neither"] == 0

    # seq2: 1 perfect read.
    assert qc["seq2"]["total_reads"] == 1
    assert qc["seq2"]["fl"] == 1
    assert qc["seq2"]["fl_rate"] == pytest.approx(1.0)
    assert qc["seq2"]["on_target"] == 1
    assert qc["seq2"]["aligned_bases"] == 26
    assert qc["seq2"]["error_rate"] == pytest.approx(0.0)

    # seq3: zero reads -> row of zeros, no division-by-zero.
    assert qc["seq3"]["total_reads"] == 0
    assert qc["seq3"]["fl_rate"] == 0.0
    assert qc["seq3"]["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# 2. amplicon-primer (AP) statuses — all four buckets
# ---------------------------------------------------------------------------


def test_qc_summary_ap_all_statuses(tmp_path: Path) -> None:
    cfg = {
        "seq1": {
            "amp_primer_start_len": 8,
            "amp_primer_end_len": 8,
            "amp_primer_start_seq": "ACGTACGT",
            "amp_primer_end_seq": "ACGTACGT",
        }
    }
    sam = (
        f"both\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"neither\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'T' * 28}\t{Q * 28}\n"
        f"start\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGTACGT' + 'T' * 20}\t{Q * 28}\n"
        f"end\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'T' * 20 + 'ACGTACGT'}\t{Q * 28}\n"
    )
    setup = setup_env(tmp_path, cfg, HDR_1_28, sam)
    assert setup.config_path is not None
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out, config=setup.config_path))

    qc = _read_qc(out)
    row = qc["seq1"]
    assert row["ap_both"] == 1
    assert row["ap_start_only"] == 1
    assert row["ap_end_only"] == 1
    assert row["ap_neither"] == 1
    assert row["total_reads"] == 4
    # every read is a full 28M -> FL on all four
    assert row["fl"] == 4


# ---------------------------------------------------------------------------
# 3. no config -> only FL + total reads populated
# ---------------------------------------------------------------------------


def test_qc_summary_no_config(tmp_path: Path) -> None:
    sam = (
        f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\tcs:Z::28\n"
        f"r2\t0\tseq1\t1\t60\t24M\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\tcs:Z::24\n"
    )
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out))  # no --config

    row = _read_qc(out)["seq1"]
    assert row["total_reads"] == 2
    assert row["fl"] == 1  # only the 28M read
    assert row["fl_rate"] == pytest.approx(0.5)
    # cs-dependent / config-dependent columns all zero
    assert row["on_target"] == 0
    assert row["ap_both"] == 0
    assert row["mismatches"] == 0
    assert row["aligned_bases"] == 0


def test_qc_summary_fl_counts_equal_and_diff_cigar(tmp_path: Path) -> None:
    # cs-tagger bams carry =/X ops instead of M; full-length spans written
    # that way must tally as FL (regression: fl was 0 for such bams —
    # found in the wps_1012 refalign workflow, 668/984 full-span reads).
    sam = (
        f"r1\t0\tseq1\t1\t60\t28=\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r2\t0\tseq1\t1\t60\t14=2X12=\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r3\t0\tseq1\t1\t60\t24=\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\n"
    )
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out))

    row = _read_qc(out)["seq1"]
    assert row["total_reads"] == 3
    # 28= and 14=2X12= span the full 28 bp; the 24= read does not
    assert row["fl"] == 2
    assert row["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# 4. error columns cross-check against error-stats per_base on the same BAM
# ---------------------------------------------------------------------------


def test_qc_summary_error_matches_error_stats(tmp_path: Path) -> None:
    cfg = {
        "seq1": {"error_primer_start_len": 5, "error_primer_end_len": 5},
        "seq2": {"error_primer_start_len": 5, "error_primer_end_len": 5},
    }
    sam = (
        f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\tcs:Z::28\n"
        f"r2\t0\tseq1\t1\t60\t24M\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\tcs:Z::10*at:13\n"
        f"r3\t0\tseq2\t1\t60\t26M\t*\t0\t0\t{'AC' * 13}\t{Q * 26}\tcs:Z::26\n"
    )
    setup = setup_env(tmp_path, cfg, HDR_3, sam)
    assert setup.config_path is not None

    qc_out = tmp_path / "qc.parquet"
    run_qc_summary(
        QcSummaryArgs(input_bam=setup.input_bam, output=qc_out, config=setup.config_path)
    )
    qc = _read_qc(qc_out)

    pre = tmp_path / "err"
    run_error_stats(ErrorStatsArgs(input_bam=setup.input_bam, config=setup.config_path, prefix=pre))
    per_base = _read_per_base(tmp_path / "err.per_base.parquet")

    for ref in ("seq1", "seq2"):
        pb = per_base[ref]
        row = qc[ref]
        assert row["aligned_bases"] == pb["total"]
        assert row["mismatches"] == pb["substitution"]
        assert row["insertions"] == pb["insertion"]
        assert row["deletions"] == pb["deletion"]
    # seq3 has no error config and no reads -> absent from error-stats, zeros in qc-summary
    assert qc["seq3"]["aligned_bases"] == 0


# ---------------------------------------------------------------------------
# 5. empty BAM (no records) -> one zero row per header contig
# ---------------------------------------------------------------------------


def test_qc_summary_empty_bam(tmp_path: Path) -> None:
    setup = setup_env(tmp_path, None, HDR_1_28, "")  # no records
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out))

    df = pl.read_parquet(out)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["seq_name"] == "seq1"
    assert row["ref_len"] == 28
    assert row["total_reads"] == 0
    assert row["fl_rate"] == 0.0


# ---------------------------------------------------------------------------
# 6. strand columns — FLAG 0x10 split incl. secondary/supplementary records
# ---------------------------------------------------------------------------


def test_qc_summary_strand_columns(tmp_path: Path) -> None:
    # r1 fwd, r2 rev (16), r3 secondary fwd (0x100), r4 supplementary rev (0x800|16)
    sam = (
        f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r2\t16\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r3\t256\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r4\t2064\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r5\t4\t*\t0\t0\t*\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"  # unmapped: no strand
    )
    setup = setup_env(tmp_path, None, HDR_3, sam)
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out))

    qc = _read_qc(out)
    row = qc["seq1"]
    assert row["total_reads"] == 4  # unmapped r5 excluded
    assert row["forward_reads"] == 2  # r1 + secondary r3
    assert row["reverse_reads"] == 2  # r2 + supplementary r4
    assert row["forward_rate"] == pytest.approx(0.5)
    # identity holds row-wise
    assert row["forward_reads"] + row["reverse_reads"] == row["total_reads"]
    # zero-read ref: zeros, no division error
    assert qc["seq2"]["forward_reads"] == qc["seq2"]["reverse_reads"] == 0
    assert qc["seq2"]["forward_rate"] == 0.0


def test_qc_summary_strand_all_reverse(tmp_path: Path) -> None:
    sam = (
        f"r1\t16\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"r2\t16\tseq1\t1\t60\t24M\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\n"
    )
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out))

    row = _read_qc(out)["seq1"]
    assert row["forward_reads"] == 0
    assert row["reverse_reads"] == 2
    assert row["forward_rate"] == 0.0


# ---------------------------------------------------------------------------
# 7. length-dist side output (read_len / dm), primary alignments only
# ---------------------------------------------------------------------------


def test_qc_summary_length_dist(tmp_path: Path) -> None:
    # primary reads on seq1:
    #   a: 24M4S -> read_len 28 (soft clips included), dm 24
    #   b: 24M2D -> read_len 24, dm 26 (M+D only)
    #   c: 24M4S -> same as a -> read_len 28 x2, dm 24 x2
    # excluded: secondary (0x100) d, supplementary (0x800) e, unmapped f
    sam = (
        f"a\t0\tseq1\t1\t60\t24M4S\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"b\t0\tseq1\t1\t60\t24M2D\t*\t0\t0\t{'ACGT' * 6}\t{Q * 24}\n"
        f"c\t0\tseq1\t1\t60\t24M4S\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"d\t256\tseq1\t1\t60\t24M4S\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"e\t2048\tseq1\t1\t60\t24M4S\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
        f"f\t4\t*\t0\t0\t*\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
    )
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    dist_out = tmp_path / "dist.parquet"
    qc_out = tmp_path / "qc.parquet"
    run_qc_summary(
        QcSummaryArgs(input_bam=setup.input_bam, output=qc_out, length_dist_out=dist_out)
    )

    df = pl.read_parquet(dist_out)
    rows = {(r["seq_name"], r["metric"], r["value"]): r["count"] for r in df.iter_rows(named=True)}
    assert rows == {
        ("seq1", "read_len", 28): 2,  # a + c
        ("seq1", "read_len", 24): 1,  # b
        ("seq1", "dm", 24): 2,  # a + c
        ("seq1", "dm", 26): 1,  # b (24M + 2D)
    }
    # digest unchanged by the side output
    qc = pl.read_parquet(qc_out)
    assert qc["total_reads"].to_list() == [5]  # all mapped, incl. secondary/supplementary


def test_qc_summary_length_dist_absent_flag_writes_nothing(tmp_path: Path) -> None:
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    dist_out = tmp_path / "dist.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=tmp_path / "qc.parquet"))
    assert not dist_out.exists()


# ---------------------------------------------------------------------------
# 8. CLI dispatch
# ---------------------------------------------------------------------------


def test_qc_summary_cli(tmp_path: Path) -> None:
    cfg = {"seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8}}
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\tcs:Z::28\n"
    setup = setup_env(tmp_path, cfg, HDR_1_28, sam)
    assert setup.config_path is not None
    out = tmp_path / "qc.parquet"
    with pytest.raises(SystemExit) as ei:
        app(
            [
                "qc-summary",
                "-i",
                str(setup.input_bam),
                "-o",
                str(out),
                "--config",
                str(setup.config_path),
            ]
        )
    assert ei.value.code in (0, None)
    assert out.exists()
    assert pl.read_parquet(out)["total_reads"].to_list() == [1]
