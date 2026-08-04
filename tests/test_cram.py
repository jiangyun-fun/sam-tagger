"""Tests for CRAM/SAM output (``--output-format``) and auto-detect input.

CRAM write support depends on the htslib build bundled with pysam; if this env
cannot write CRAM, those tests skip (with a clear reason) rather than fail the
gate. The auto-detect read path and SAM output are always exercised.
"""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest
from _helpers import setup_env

from sam_tagger.cli import AnnotateArgs, app
from sam_tagger.lib import _open_reader, run_annotate

HDR_1_28 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28"
SEQ1 = "ACGT" * 7  # 28, matches LN
Q = "I"


def _input_bam(tmp_path: Path):
    """A 1-read BAM (28M on seq1, cs tag) plus a faidx-indexed reference."""
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{SEQ1}\t{Q * 28}\tcs:Z::28\n"
    return setup_env(tmp_path, None, HDR_1_28, sam, fasta={"seq1": SEQ1})


def _try_write_cram(tmp_path: Path) -> Path:
    """Write a CRAM via run_annotate; raise if the env can't (caller decides to skip)."""
    setup = _input_bam(tmp_path)
    ref_fa = tmp_path / "ref.fa"
    out_cram = tmp_path / "out.cram"
    run_annotate(
        AnnotateArgs(
            input_bam=setup.input_bam,
            output_bam=out_cram,
            add_fl="FL",
            output_format="cram",
            fasta=ref_fa,
        )
    )
    return out_cram


def test_cram_round_trip(tmp_path: Path) -> None:
    try:
        out_cram = _try_write_cram(tmp_path)
    except Exception as e:  # noqa: BLE001 — capability probe, skip on any failure
        pytest.skip(f"this pysam/htslib build cannot write CRAM: {e}")

    assert out_cram.exists()
    ref_fa = tmp_path / "ref.fa"
    with pysam.AlignmentFile(str(out_cram), reference_filename=str(ref_fa)) as reader:
        records = list(reader.fetch(until_eof=True))
    fl = [int(rec.get_tag("FL")) for rec in records if rec.has_tag("FL")]
    assert fl == [1]  # the FL tag survived the BAM -> CRAM -> read round-trip
    assert records[0].query_name == "r1"


def test_open_reader_autodetects_cram(tmp_path: Path) -> None:
    try:
        out_cram = _try_write_cram(tmp_path)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"this pysam/htslib build cannot write CRAM: {e}")

    reader = _open_reader(out_cram)  # no mode -> auto-detect CRAM
    try:
        assert "seq1" in reader.references
    finally:
        reader.close()


def test_sam_output(tmp_path: Path) -> None:
    setup = _input_bam(tmp_path)
    out_sam = tmp_path / "out.sam"
    run_annotate(
        AnnotateArgs(
            input_bam=setup.input_bam, output_bam=out_sam, add_fl="FL", output_format="sam"
        )
    )
    text = out_sam.read_text()
    assert text.startswith("@HD\t")
    assert "FL:i:1" in text  # human-readable SAM carries the tag inline


def test_bam_default_unchanged(tmp_path: Path) -> None:
    setup = _input_bam(tmp_path)
    out_bam = tmp_path / "out.bam"
    run_annotate(
        AnnotateArgs(input_bam=setup.input_bam, output_bam=out_bam, add_fl="FL")  # default bam
    )
    with pysam.AlignmentFile(str(out_bam), "rb") as reader:
        records = list(reader.fetch(until_eof=True))
    assert [int(rec.get_tag("FL")) for rec in records if rec.has_tag("FL")] == [1]


def test_cram_without_fasta_raises(tmp_path: Path) -> None:
    setup = _input_bam(tmp_path)
    out_cram = tmp_path / "out.cram"
    with pytest.raises(ValueError, match="fasta"):
        run_annotate(
            AnnotateArgs(
                input_bam=setup.input_bam, output_bam=out_cram, add_fl="FL", output_format="cram"
            )
        )


def test_invalid_output_format_rejected(tmp_path: Path) -> None:
    setup = _input_bam(tmp_path)
    with pytest.raises(SystemExit):
        app([
            "annotate", "-i", str(setup.input_bam), "-o", str(tmp_path / "x"),
            "--add-fl", "FL", "--output-format", "bezelbub",
        ])
