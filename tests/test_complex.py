"""Complex + edge-case integration tests for sam-tagger.

Beyond the per-feature tests in test_integration, these exercise:
  - a multi-ref, multi-feature run end-to-end (tags + stats + extraction together)
  - --intervals BED pileup on multiple refs with overlapping (merged) intervals
  - an empty / header-only BAM (no records)
  - mapped reads that lack a cs tag (cs-based tags must be skipped, not crash)
  - the NB tag against a real FASTA, incl. the read-at-reference-end boundary
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pysam
from _helpers import (
    H_SEQ1_28,
    H_SEQ1_30,
    H_SEQ1_100_SEQ2_100,
    get_bam_tags_as_int,
    get_bam_tags_as_string,
    read_gzipped_fastq,
    read_simple_stats,
    setup_env,
)

from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs, PileupArgs
from sam_tagger.lib import run_annotate, run_error_stats, run_pileup

HDR_28_26 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28\n@SQ\tSN:seq2\tLN:26"
Q = "I"  # Phred 40


def _args(setup, **kw) -> AnnotateArgs:
    return AnnotateArgs(input_bam=setup.input_bam, output_bam=setup.output_bam, **kw)


# ---------------------------------------------------------------------------
# 1. complex multi-feature, multi-ref
# ---------------------------------------------------------------------------


def test_complex_multi_feature_run(tmp_path: Path) -> None:
    cfg = {
        "seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8},
        "seq2": {
            "tc_primer_start_len": 5,
            "tc_primer_end_len": 5,
            "amp_primer_start_len": 9,
            "amp_primer_end_len": 7,
            "amp_primer_start_seq": "GGGGGGGGG",
            "amp_primer_end_seq": "CCCCCCC",
        },
    }
    s_full = "ACGT" * 7  # 28
    s_off = "A" * 10 + "G" + "C" * 17  # 28 (10M + 1I + 17M)
    s_both = "G" * 9 + "T" * 10 + "C" * 7  # 26 (both primers)
    s_soft = "N" * 5 + "G" * 9 + "T" * 6 + "C" * 6  # 26 (5S + 15M + 6S)
    sam = (
        f"s1_full\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{s_full}\t{Q * 28}\tcs:Z::28\n"
        f"s1_off\t0\tseq1\t1\t60\t10M1I17M\t*\t0\t0\t{s_off}\t{Q * 28}\tcs:Z::10+g:17\n"
        f"s2_both\t0\tseq2\t1\t60\t26M\t*\t0\t0\t{s_both}\t{Q * 26}\tcs:Z::26\n"
        f"s2_soft\t0\tseq2\t3\t60\t5S15M6S\t*\t0\t0\t{s_soft}\t{Q * 26}\tcs:Z::15\n"
        "u1\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*"
    )
    setup = setup_env(tmp_path, cfg, HDR_28_26, sam)
    flp = tmp_path / "fl.parquet"
    tcp = tmp_path / "tc.parquet"
    app_ = tmp_path / "ap.parquet"
    left = tmp_path / "left.fq.gz"
    right = tmp_path / "right.fq.gz"

    run_annotate(
        _args(
            setup,
            add_fl="FL",
            add_ss="SS",
            add_es="ES",
            add_dm="DM",
            add_ee="EE",
            add_tc="TC",
            add_ap="AP",
            fl_stats=flp,
            tc_stats=tcp,
            ap_stats=app_,
            extract_left_softclip_fastq=left,
            extract_right_softclip_fastq=right,
            config=setup.config_path,
        )
    )

    # all 5 records preserved (4 mapped + 1 unmapped)
    assert sum(1 for _ in pysam.AlignmentFile(str(setup.output_bam), "rb")) == 5

    itags = get_bam_tags_as_int(setup.output_bam)
    stags = get_bam_tags_as_string(setup.output_bam)
    assert itags["s1_full"]["FL"] == 1  # full-length
    assert "FL" not in itags["s1_off"]  # insertion -> not full length
    assert itags["s1_full"]["DM"] == 28
    assert stags["s1_full"]["TC"] == "on_target"
    assert "TC" not in stags["s1_off"]  # insertion in inner region -> off target
    assert stags["s2_both"]["AP"] == "both"
    assert itags["s2_soft"]["SS"] == 5 and itags["s2_soft"]["ES"] == 6
    assert itags["s1_off"]["EE"] == 1

    assert read_simple_stats(flp) == {"seq1": 1, "seq2": 1}  # s1_full + s2_both
    assert read_simple_stats(tcp) == {"seq1": 1, "seq2": 1}  # both perfect reads on target
    # soft-clip extraction picked up s2_soft's 5' and 3' clips
    assert len(read_gzipped_fastq(left)) == 1
    assert len(read_gzipped_fastq(right)) == 1


# ---------------------------------------------------------------------------
# 2. --intervals on multiple refs with overlapping (merged) intervals
# ---------------------------------------------------------------------------


def test_pileup_intervals_multiref_merge(tmp_path: Path) -> None:
    s = "ACGT" * 7 + "AC"  # 30
    sam = (
        f"a\t0\tseq1\t1\t60\t30M\t*\t0\t0\t{s}\t{Q * 30}\tcs:Z::30\n"
        f"b\t0\tseq2\t1\t60\t30M\t*\t0\t0\t{'T' * 30}\t{Q * 30}\tcs:Z::30\n"
    )
    setup = setup_env(tmp_path, None, H_SEQ1_100_SEQ2_100, sam)
    bed = tmp_path / "r.bed"
    # seq1: overlapping [5,12)+[10,20) -> merge to [5,20); seq2: [8,15)
    bed.write_text("seq1\t5\t12\nseq1\t10\t20\nseq2\t8\t15\n")
    out = tmp_path / "p.parquet"
    run_pileup(PileupArgs(input_bam=setup.input_bam, output=out, intervals=bed, min_base_qual=0))

    df = pl.read_parquet(out).select("seq_name", "pos").sort(["seq_name", "pos"])
    s1 = df.filter(pl.col("seq_name") == "seq1")["pos"].to_list()
    s2 = df.filter(pl.col("seq_name") == "seq2")["pos"].to_list()
    assert s1 == list(range(6, 21))  # merged [5,20) -> 1-based 6..20
    assert s2 == list(range(9, 16))  # [8,15) -> 1-based 9..15


# ---------------------------------------------------------------------------
# 3. empty / header-only BAM
# ---------------------------------------------------------------------------


def test_empty_bam_annotate_pileup_errorstats(tmp_path: Path) -> None:
    cfg = {"seq1": {"error_primer_start_len": 5, "error_primer_end_len": 5}}
    setup = setup_env(tmp_path, cfg, H_SEQ1_28, "")  # no records

    run_annotate(_args(setup, add_fl="FL"))
    assert sum(1 for _ in pysam.AlignmentFile(str(setup.output_bam), "rb")) == 0

    pileup_out = tmp_path / "p.parquet"
    run_pileup(PileupArgs(input_bam=setup.input_bam, output=pileup_out, min_base_qual=20))
    assert pl.read_parquet(pileup_out).height == 0

    assert setup.config_path is not None
    pre = tmp_path / "err"
    run_error_stats(ErrorStatsArgs(input_bam=setup.input_bam, config=setup.config_path, prefix=pre))
    for ext in (
        "error",
        "mismatch",
        "insertion",
        "deletion",
        "per_read",
        "per_base",
        "single_deletion",
    ):
        assert (tmp_path / f"err.{ext}.parquet").exists()
    # no reads -> per_read counts all zero
    assert pl.read_parquet(tmp_path / "err.per_read.parquet")["count"].sum() == 0


# ---------------------------------------------------------------------------
# 4. mapped reads WITHOUT a cs tag
# ---------------------------------------------------------------------------


def test_reads_without_cs_tag(tmp_path: Path) -> None:
    s = "ACGT" * 7  # 28
    sam = (
        f"r_with\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{s}\t{Q * 28}\tcs:Z::28\n"
        f"r_without\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{s}\t{Q * 28}\n"
    )
    setup = setup_env(tmp_path, None, H_SEQ1_28, sam)
    run_annotate(_args(setup, add_fl="FL", add_ee="EE"))

    tags = get_bam_tags_as_int(setup.output_bam)
    assert tags["r_with"]["FL"] == 1  # CIGAR-based: works
    assert tags["r_without"]["FL"] == 1
    assert tags["r_with"]["EE"] == 0  # cs-based: present
    assert "EE" not in tags["r_without"]  # cs-based: skipped (no cs)


# ---------------------------------------------------------------------------
# 5. NB tag against a real FASTA, incl. reference-end boundary
# ---------------------------------------------------------------------------


def test_nb_tag_with_fasta_and_boundary(tmp_path: Path) -> None:
    seq1 = "ACGTACGTACGTACGTACGTACGTACGTAC"  # 30
    cfg = {"seq1": {"next_n": 5}}
    part = seq1[:20]
    sam = (
        f"r_partial\t0\tseq1\t1\t60\t20M\t*\t0\t0\t{part}\t{Q * 20}\tcs:Z::20\n"
        f"r_full\t0\tseq1\t1\t60\t30M\t*\t0\t0\t{seq1}\t{Q * 30}\tcs:Z::30\n"
    )
    setup = setup_env(tmp_path, cfg, H_SEQ1_30, sam, fasta={"seq1": seq1})
    run_annotate(_args(setup, add_nb="NB", fasta=tmp_path / "ref.fa", config=setup.config_path))

    stags = get_bam_tags_as_string(setup.output_bam)
    assert stags["r_partial"]["NB"] == seq1[20:25]  # ref[20:25]
    assert "NB" not in stags["r_full"]  # ends at ref end -> fetch None -> omitted


def test_nb_stats_with_fasta(tmp_path: Path) -> None:
    seq1 = "ACGTACGTACGTACGTACGTACGTACGTAC"  # 30
    cfg = {"seq1": {"next_n": 3}}
    ten = seq1[:10]
    sam = (
        f"r1\t0\tseq1\t1\t60\t10M\t*\t0\t0\t{ten}\t{Q * 10}\tcs:Z::10\n"
        f"r2\t0\tseq1\t1\t60\t10M\t*\t0\t0\t{ten}\t{Q * 10}\tcs:Z::10\n"
    )
    setup = setup_env(tmp_path, cfg, H_SEQ1_30, sam, fasta={"seq1": seq1})
    nb_stats = tmp_path / "nb.parquet"
    run_annotate(
        _args(
            setup,
            add_nb="NB",
            nb_stats=nb_stats,
            fasta=tmp_path / "ref.fa",
            config=setup.config_path,
        )
    )
    df = pl.read_parquet(nb_stats)
    rows = {(r["seq_name"], r["type"]): r["count"] for r in df.iter_rows(named=True)}
    expected = seq1[10:13]  # both reads fetch the same 3 bases after pos 10
    assert rows[("seq1", expected)] == 2
    assert rows[("total", expected)] == 2
