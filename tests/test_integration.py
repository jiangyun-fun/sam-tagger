"""Integration tests for sam-tagger, ported from py-bam-tagger.

Adapted to the new design:
- config is CSV; seq_len comes from the BAM header, so each test's
  @SQ LN must match the length the test expects (FL / primer inner regions).
- each test builds the subcommand's args dataclass and calls the orchestrator
  directly (run_annotate / run_error_stats / run_pileup).
- HI/HD/SP tags are gone; SP assertions were removed.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pysam
from _helpers import Setup, setup_env

from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs, PileupArgs
from sam_tagger.lib import run_annotate, run_error_stats, run_pileup

# --- Shared headers (seq_len now lives in the BAM header, not config) ---
H_SEQ1_28 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28"
H_SEQ1_100 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:100"
H_SEQ2_26 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq2\tLN:26"
H_SEQ1_28_SEQ2_26 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28\n@SQ\tSN:seq2\tLN:26"
H_SEQ1_100_SEQ2_100 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:100\n@SQ\tSN:seq2\tLN:100"


# --- Helpers ---


def annotate_args(setup: Setup, **kwargs: object) -> AnnotateArgs:
    return AnnotateArgs(
        input_bam=setup.input_bam,
        output_bam=setup.output_bam,
        config=setup.config_path,
        **kwargs,  # type: ignore[arg-type]
    )


def get_bam_tags_as_int(bam_path: Path) -> dict[str, dict[str, int]]:
    """read_name -> tag_name -> int value."""
    result: dict[str, dict[str, int]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as reader:
        for record in reader.fetch(until_eof=True):
            qname = record.query_name or ""
            result[qname] = {
                t: int(record.get_tag(t))
                for t in ("FL", "MS", "ME", "SS", "ES", "DM", "EE", "IL", "DL")
                if record.has_tag(t)
            }
    return result


def get_bam_tags_as_string(bam_path: Path) -> dict[str, dict[str, str]]:
    """read_name -> tag_name -> string value."""
    result: dict[str, dict[str, str]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as reader:
        for record in reader.fetch(until_eof=True):
            qname = record.query_name or ""
            result[qname] = {t: str(record.get_tag(t)) for t in ("TC", "AP") if record.has_tag(t)}
    return result


@dataclass
class FastqRecord:
    head: str
    seq: str
    qual: str


def read_gzipped_fastq(path: Path) -> list[FastqRecord]:
    records: list[FastqRecord] = []
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().split("\n")
    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            records.append(FastqRecord(head=lines[i], seq=lines[i + 1], qual=lines[i + 3]))
    return records


def read_simple_stats(path: Path) -> dict[str, int]:
    return {
        str(r["seq_name"]): int(r["count"]) for r in pl.read_parquet(path).iter_rows(named=True)
    }


def read_pileup(path: Path) -> list[tuple]:
    return [tuple(r) for r in pl.read_parquet(path).sort("seq_name", "pos").iter_rows()]


# ===========================================================================
# annotate — tagging
# ===========================================================================


class TestTagging:
    def test_basic_tags_fl_ss_es_ms_me_dm(self, tmp_path: Path) -> None:
        sam_records = (
            "is_full_len\t0\tseq1\t1\t60\t28M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGT\t*\tcs:Z::28\n"
            "not_full_len\t0\tseq1\t1\t60\t27M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACG\t*\tcs:Z::27\n"
            "soft_clip_read\t0\tseq1\t3\t60\t5S20M3S\t*\t0\t0\tNNNNNACGTACGTACGTACGTACGTNNN\t*\tcs:Z::20"
        )
        setup = setup_env(tmp_path, None, H_SEQ1_28, sam_records)

        run_annotate(
            annotate_args(
                setup, add_fl="FL", add_ms="MS", add_me="ME", add_ss="SS", add_es="ES", add_dm="DM"
            )
        )

        tags = get_bam_tags_as_int(setup.output_bam)
        fl = tags["is_full_len"]
        assert fl["FL"] == 1
        assert fl["MS"] == 1
        assert fl["ME"] == 28
        assert fl["SS"] == 0
        assert fl["ES"] == 0
        assert fl["DM"] == 28

        assert "FL" not in tags["not_full_len"]

        sc = tags["soft_clip_read"]
        assert "FL" not in sc
        assert sc["MS"] == 6
        assert sc["ME"] == 25
        assert sc["SS"] == 5
        assert sc["ES"] == 3
        assert sc["DM"] == 20

    def test_on_target_tc_tagging(self, tmp_path: Path) -> None:
        cfg = {"seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8}}
        sam_records = (
            "on_target_perfect\t0\tseq1\t1\t60\t28M\t*\t0\t0\tAAAAAAAACCCCCCCCCCCCBBBBBBBB\t*\tcs:Z::28\n"
            "on_target_variant_in_primer\t0\tseq1\t1\t60\t2M1D25M\t*\t0\t0\tAAAAAAAAAAAAAAAACCCCCBBBBBB\t*\tcs:Z::2-a:25\n"
            "on_target_variant_no_start_primer\t0\tseq1\t9\t60\t20M\t*\t0\t0\tNNNNNNNNNNNNNNNNNNNN\t*\tcs:Z::20\n"
            "off_target_variant_in_body\t0\tseq1\t1\t60\t10M1I17M\t*\t0\t0\tAAAAAAAAACCGCCCCCCCCBBBBBBBB\t*\tcs:Z::10+g:17\n"
            "off_target_del_span_start\t0\tseq1\t1\t60\t7M2D19M\t*\t0\t0\tAAAAAAACCCCCCCCCCCBBBBBBBB\t*\tcs:Z::7-aa:19"
        )
        setup = setup_env(tmp_path, cfg, H_SEQ1_28, sam_records)
        run_annotate(annotate_args(setup, add_tc="TC"))

        tags = get_bam_tags_as_string(setup.output_bam)
        assert tags["on_target_perfect"]["TC"] == "OnTarget"
        assert tags["on_target_variant_in_primer"]["TC"] == "OnTarget"
        assert tags["on_target_variant_no_start_primer"]["TC"] == "OnTarget"
        assert "TC" not in tags["off_target_variant_in_body"]
        assert "TC" not in tags["off_target_del_span_start"]

    def test_amplicon_primer_ap_tagging(self, tmp_path: Path) -> None:
        cfg = {
            "seq2": {
                "amp_primer_start_len": 9,
                "amp_primer_end_len": 7,
                "amp_primer_start_seq": "GGGGGGGGG",
                "amp_primer_end_seq": "CCCCCCC",
                "amp_primer_start_min_match": 9,
                "amp_primer_end_min_match": 7,
            }
        }
        sam_records = (
            "has_both\t0\tseq2\t1\t60\t26M\t*\t0\t0\tGGGGGGGGGTTTTTTTTTTCCCCCCC\t*\tcs:Z::26\n"
            "has_start_only\t0\tseq2\t1\t60\t15M\t*\t0\t0\tGGGGGGGGGTTTTTT\t*\tcs:Z::15\n"
            "has_end_only\t0\tseq2\t12\t60\t15M\t*\t0\t0\tTTTTTTTTCCCCCCC\t*\tcs:Z::15\n"
            "has_neither\t0\tseq2\t10\t60\t8M\t*\t0\t0\tTTTTTTTT\t*\tcs:Z::8\n"
            "start_fails_min_match\t0\tseq2\t1\t60\t26M\t*\t0\t0\tGGGGGGGGXTTTTTTTTTTCCCCCCC\t*\tcs:Z:=8*ga:17\n"
            "end_in_softclip_fail\t0\tseq2\t1\t60\t19M6S\t*\t0\t0\tGGGGGGGGGTTTTTTTTCCCCCCCC\t*\tcs:Z::19"
        )
        setup = setup_env(tmp_path, cfg, H_SEQ2_26, sam_records)
        run_annotate(annotate_args(setup, add_ap="AP"))

        tags = get_bam_tags_as_string(setup.output_bam)
        assert tags["has_both"]["AP"] == "Both"
        assert tags["has_start_only"]["AP"] == "StartOnly"
        assert tags["has_end_only"]["AP"] == "EndOnly"
        assert tags["has_neither"]["AP"] == "Neither"
        assert tags["start_fails_min_match"]["AP"] == "EndOnly"
        assert tags["end_in_softclip_fail"]["AP"] == "StartOnly"

    def test_new_tags_ee_il_dl(self, tmp_path: Path) -> None:
        sam_records = (
            "read1\t0\tseq1\t6\t60\t7M1I4M1D15M\t*\t0\t0\tAAAAACCTXCCCGCCCCCCCCCCCCCC\t*\tcs:Z::7*ct+g:4-c:15\n"
            "read2\t0\tseq1\t10\t60\t5M3I10M2D5M\t*\t0\t0\tAAAAAGGGAAAAAAAAAAAAACC\t*\tcs:Z::5+ggg:10-cc:5"
        )
        setup = setup_env(tmp_path, None, H_SEQ1_100, sam_records)
        run_annotate(annotate_args(setup, add_ee="EE", add_il="IL", add_dl="DL"))

        tags = get_bam_tags_as_int(setup.output_bam)
        assert tags["read1"]["EE"] == 3
        assert tags["read1"]["IL"] == 1
        assert tags["read1"]["DL"] == 1
        assert tags["read2"]["EE"] == 2
        assert tags["read2"]["IL"] == 3
        assert tags["read2"]["DL"] == 2


# ===========================================================================
# annotate — tag statistics
# ===========================================================================


class TestTagStats:
    def test_fl_and_tc_stats_generation(self, tmp_path: Path) -> None:
        cfg = {
            "seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8},
            "seq2": {"tc_primer_start_len": 5, "tc_primer_end_len": 5},
        }
        sam_records = (
            "s1_fl_tc\t0\tseq1\t1\t60\t28M\t*\t0\t0\tAAAAAAAACCCCCCCCCCCCBBBBBBBB\t*\tcs:Z::28\n"
            "s1_fl_notc\t0\tseq1\t1\t60\t28M\t*\t0\t0\tAAAAAAAACCGCCCCCCCCCBBBBBBBB\t*\tcs:Z::10+g:17\n"
            "s2_nofl_tc\t0\tseq2\t1\t60\t26M\t*\t0\t0\tAAAAACCCCCCCCCCCCCCCCBBBBB\t*\tcs:Z::26\n"
            "s1_unmapped\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*"
        )
        setup = setup_env(tmp_path, cfg, H_SEQ1_28_SEQ2_26, sam_records)
        fl_stats = tmp_path / "fl.parquet"
        tc_stats = tmp_path / "tc.parquet"
        run_annotate(annotate_args(setup, fl_stats=fl_stats, tc_stats=tc_stats))

        assert read_simple_stats(fl_stats) == {"seq1": 2, "seq2": 1}
        assert read_simple_stats(tc_stats) == {"seq1": 1, "seq2": 1}


# ===========================================================================
# error-stats
# ===========================================================================


class TestErrorStats:
    def test_error_stats_generation(self, tmp_path: Path) -> None:
        cfg = {"seq1": {"error_primer_start_len": 5, "error_primer_end_len": 5}}
        sam_records = (
            "err_read1\t0\tseq1\t1\t60\t7M1I4M1D15M\t*\t0\t0\tAAAAACCTXCCCGCCCCCCCCCCCCCC\t*\tcs:Z::7*ct+g:4-c:15\n"
            "err_read2\t0\tseq1\t1\t60\t28M\t*\t0\t0\tAAAAACACTCCCCCCCCCCCCCCCCCCC\t*\tcs:Z::7*ct:20"
        )
        setup = setup_env(tmp_path, cfg, H_SEQ1_28, sam_records)
        assert setup.config_path is not None
        prefix = tmp_path / "stats"
        run_error_stats(
            ErrorStatsArgs(input_bam=setup.input_bam, config=setup.config_path, prefix=prefix)
        )

        per_base = tmp_path / "stats.per_base.parquet"
        assert per_base.exists()
        stats: dict[str, int] = {
            str(r["type"]): int(r["count"])
            for r in pl.read_parquet(per_base)
            .filter(pl.col("seq_name") == "seq1")
            .iter_rows(named=True)
        }
        assert stats["total"] == 56
        assert stats["insertion"] == 1
        assert stats["deletion"] == 1
        assert stats["substitution"] == 2


# ===========================================================================
# pileup
# ===========================================================================


class TestPileup:
    def _run(
        self,
        tmp_path: Path,
        sam_records: str,
        *,
        min_base_qual: int = 20,
        header: str = H_SEQ1_100,
        fasta: Path | None = None,
    ) -> Path:
        setup = setup_env(tmp_path, None, header, sam_records)
        out = tmp_path / "pileup.parquet"
        run_pileup(
            PileupArgs(
                input_bam=setup.input_bam,
                output=out,
                min_base_qual=min_base_qual,
                fasta=fasta,
            )
        )
        return out

    def test_pileup_snp_detection(self, tmp_path: Path) -> None:
        sam_records = (
            "snp1_forward\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n"
            "snp2_forward\t0\tseq1\t15\t60\t30M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTAC\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::5*cg:24\n"
            "ref_forward\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTAC\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::30"
        )
        stats = read_pileup(self._run(tmp_path, sam_records))

        pos10_t = [r for r in stats if r[0] == "seq1" and r[1] == 10 and r[3] == "T"]
        assert len(pos10_t) == 1
        _, _, ref_base, alt, dp, adf, adr, ad, af = pos10_t[0]
        assert ref_base == "A"
        assert alt == "T"
        assert dp == 2 and adf == 1 and adr == 0 and ad == 1 and abs(af - 0.5) < 0.001

        pos20_g = [r for r in stats if r[0] == "seq1" and r[1] == 20 and r[3] == "G"]
        assert len(pos20_g) == 1
        _, _, ref_base, alt, dp, adf, adr, ad, af = pos20_g[0]
        assert ref_base == "C" and alt == "G"
        assert dp == 3 and ad == 1 and abs(af - 0.333) < 0.01

    def test_pileup_strand_tracking(self, tmp_path: Path) -> None:
        sam_records = (
            "snp_forward\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n"
            "snp_reverse\t16\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20"
        )
        stats = read_pileup(self._run(tmp_path, sam_records))
        pos10 = [r for r in stats if r[0] == "seq1" and r[1] == 10]
        assert len(pos10) == 1
        _, _, ref_base, alt, dp, adf, adr, ad, af = pos10[0]
        assert (
            ref_base == "A"
            and alt == "T"
            and dp == 2
            and adf == 1
            and adr == 1
            and ad == 2
            and abs(af - 1.0) < 0.001
        )

    def test_pileup_quality_filtering(self, tmp_path: Path) -> None:
        sam_records = (
            "high_quality_snp\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n"
            "low_quality_snp\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\t!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\tcs:Z::9*at:20"
        )
        stats = read_pileup(self._run(tmp_path, sam_records))
        pos10_t = [r for r in stats if r[0] == "seq1" and r[1] == 10 and r[3] == "T"]
        assert len(pos10_t) >= 1
        _, _, ref_base, alt, _dp, _adf, _adr, ad, _af = pos10_t[0]
        assert ref_base == "A" and alt == "T" and ad == 1

    def test_pileup_insertion_detection(self, tmp_path: Path) -> None:
        sam_records = "ins_forward\t0\tseq1\t1\t60\t10M2I18M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::10+ac:18\n"
        stats = read_pileup(self._run(tmp_path, sam_records))
        ins = [r for r in stats if r[0] == "seq1" and r[1] == 10 and r[3].startswith("+")]
        assert len(ins) >= 1
        _, _, _ref, alt, dp, adf, adr, ad, af = ins[0]
        assert (
            alt == "+AC" and dp == 1 and adf == 1 and adr == 0 and ad == 1 and abs(af - 1.0) < 0.001
        )

    def test_pileup_deletion_detection(self, tmp_path: Path) -> None:
        sam_records = "del_forward\t0\tseq1\t1\t60\t14M2D14M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTAC\tIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::14-ac:14\n"
        stats = read_pileup(self._run(tmp_path, sam_records))
        dels = [r for r in stats if r[0] == "seq1" and r[1] == 14 and r[3].startswith("-")]
        assert len(dels) >= 1
        _, _, _ref, alt, dp, adf, adr, ad, af = dels[0]
        assert (
            alt == "-AC" and dp == 1 and adf == 1 and adr == 0 and ad == 1 and abs(af - 1.0) < 0.001
        )

    def test_pileup_mixed_alleles(self, tmp_path: Path) -> None:
        sam_records = (
            "allele_A\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::30\n"
            "allele_T\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n"
            "allele_C\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*ac:20"
        )
        stats = read_pileup(self._run(tmp_path, sam_records))
        pos10 = [r for r in stats if r[0] == "seq1" and r[1] == 10]
        assert len(pos10) == 3
        alleles = {r[3] for r in pos10}
        assert "." in alleles and "T" in alleles and "C" in alleles
        assert all(r[4] == 3 for r in pos10)

    def test_pileup_multiple_references(self, tmp_path: Path) -> None:
        sam_records = (
            "seq1_read\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n"
            "seq2_read\t0\tseq2\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*cg:20"
        )
        stats = read_pileup(self._run(tmp_path, sam_records, header=H_SEQ1_100_SEQ2_100))
        seq_names = {r[0] for r in stats}
        assert "seq1" in seq_names and "seq2" in seq_names

    def test_pileup_degenerate_bases(self, tmp_path: Path) -> None:
        sam_records = "degenerate_N\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*an:20\n"
        stats = read_pileup(self._run(tmp_path, sam_records))
        pos10 = [r for r in stats if r[0] == "seq1" and r[1] == 10]
        assert len(pos10) >= 1
        _, _, ref_base, alt, *_ = pos10[0]
        assert ref_base == "A" and alt == "N"

    def test_pileup_real_data_example(self, tmp_path: Path) -> None:
        header = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:150-1-P\tLN:76"
        sam_records = (
            "A00202:2042:HFHKWDSXF:3:1101:11369:1235\t0\t150-1-P\t1\t60\t37S4M1I72M37S\t*\t0\t0\tGAGATTCCTCGGGCTGTATTCTAAAGGGATGTACCGTAGGGAAACTGTCATAAATAGCCCGAAAACTACGGTCGTTCTAAGAATGGTGCGGAAAGTTGGTCCTATGTCGTGGACAATGGCGAGCACATTAGTCTCTGCTTTATAGGCGACC\tFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\tcs:Z::4+a:72\n"
            "A00202:2042:HFHKWDSXF:3:1101:11532:1579\t0\t150-1-P\t1\t60\t37S76M37S\t*\t0\t0\tGAGATTCCTCGGGCTGTATTCTAAAGGGATGTACCGTAGGGAACTGTCATAAATAGCCCGAAAACTACGGTCGTTCTAAGAATGGTGCGGAAAGTTGGTCCTATGTCGTGGACAATGGCGAGCACATTAGTCTCTGCTTTATAGGCGACC\tFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\tcs:Z::69*ca:6\n"
            "A00202:2042:HFHKWDSXF:3:1101:30789:3051\t0\t150-1-P\t1\t60\t37S43M1D32M37S\t*\t0\t0\tGAGATTCCTCGGGCTGTATTCTAAAGGGATGTACCGTAGGGAACTGTCATAAATAGCCCGAAAACTACGGTCGTTCTAAGATGGTGCGGAAAGTTGGTCCTATGTCGTGGACAATGGCGAGCACATTAGTCTCTGCTTTATAGGCGACC\tFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\tcs:Z::43-a:32"
        )
        stats = read_pileup(self._run(tmp_path, sam_records, header=header))
        assert len(stats) > 0

        pos4_ins = [r for r in stats if r[0] == "150-1-P" and r[1] == 4 and r[3].startswith("+")]
        assert len(pos4_ins) >= 1
        _, _, _, alt, dp, adf, adr, ad, af = pos4_ins[0]
        assert alt == "+A" and dp == 3 and adf == 1 and ad == 1 and abs(af - 0.333) < 0.01

        pos4_ref = [r for r in stats if r[0] == "150-1-P" and r[1] == 4 and r[3] == "."]
        assert len(pos4_ref) >= 1
        _, _, _, _, dp, adf, adr, ad, _ = pos4_ref[0]
        assert dp == 3 and adf == 2 and ad == 2

        pos43_del = [r for r in stats if r[0] == "150-1-P" and r[1] == 43 and r[3].startswith("-")]
        assert len(pos43_del) >= 1
        _, _, _, alt, dp, adf, adr, ad, af = pos43_del[0]
        assert alt == "-A" and dp == 3 and ad == 1 and abs(af - 0.333) < 0.01

        pos70_snp = [r for r in stats if r[0] == "150-1-P" and r[1] == 70 and r[3] == "A"]
        assert len(pos70_snp) >= 1
        _, _, ref_base, alt, dp, _, _, ad, af = pos70_snp[0]
        assert ref_base == "C" and alt == "A" and dp == 3 and ad == 1 and abs(af - 0.333) < 0.01


# ===========================================================================
# annotate — extraction
# ===========================================================================


class TestExtraction:
    def test_softclip_extraction(self, tmp_path: Path) -> None:
        sam_records = (
            "sc_read\t0\tseq1\t10\t60\t5S10M5S\t*\t0\t0\tAAAAATTTTTTTTTTGGGGG\t@@@@@IIIIIIIIII?????"
        )
        setup = setup_env(tmp_path, None, H_SEQ1_100, sam_records)

        left = tmp_path / "left.fastq.gz"
        right = tmp_path / "right.fastq.gz"
        run_annotate(
            annotate_args(
                setup,
                extract_left_softclip_fastq=left,
                extract_right_softclip_fastq=right,
            )
        )

        lr = read_gzipped_fastq(left)
        assert (
            len(lr) == 1
            and lr[0].head == "@sc_read"
            and lr[0].seq == "AAAAA"
            and lr[0].qual == "@@@@@"
        )
        rr = read_gzipped_fastq(right)
        assert (
            len(rr) == 1
            and rr[0].head == "@sc_read"
            and rr[0].seq == "GGGGG"
            and rr[0].qual == "?????"
        )

    def test_sub_read_extraction(self, tmp_path: Path) -> None:
        cfg = {"seq1": {"subread_start": 12, "subread_end": 21}}
        sam_records = "subread_test\t0\tseq1\t10\t60\t5M2I8M3D5M\t*\t0\t0\tAAAAAIICCCCCCCCMMMMM\t@@@@@##IIIIIIIIJJJJJ"
        setup = setup_env(tmp_path, cfg, H_SEQ1_100, sam_records)

        sub = tmp_path / "sub.fastq.gz"
        run_annotate(annotate_args(setup, extract_sub_reads_fastq=sub))

        records = read_gzipped_fastq(sub)
        assert len(records) == 1
        assert records[0].head == "@subread_test"
        assert records[0].seq == "AANNCCCCCCC"
        assert records[0].qual == "@@##IIIIIII"
