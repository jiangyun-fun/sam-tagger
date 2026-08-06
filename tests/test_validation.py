"""Fail-fast validation tests for the orchestrators.

sam-tagger added explicit pre-flight checks (the port's robustness improvement
over bam_tagger); these pin them down so a missing config / fasta surfaces as a
clear ValueError BEFORE any output file is written.
"""

from __future__ import annotations

import pytest
from _helpers import H_SEQ1_28, setup_env

from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs
from sam_tagger.lib import run_annotate, run_error_stats

# A minimal valid mapped record (28M on seq1 LN=28) so run_annotate can open the
# BAM and reach validation; the record content itself is irrelevant here.
_REC = "r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGT\t*\tcs:Z::28\n"


def _args(setup, **kw) -> AnnotateArgs:
    return AnnotateArgs(input_bam=setup.input_bam, output_bam=setup.output_bam, **kw)


def test_annotate_no_action_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="no action"):
        run_annotate(_args(setup))  # no tags, no stats, no extraction


def test_annotate_tc_without_config_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="tc"):
        run_annotate(_args(setup, add_tc="TC"))  # no --config


def test_annotate_tc_stats_without_config_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="tc"):
        run_annotate(_args(setup, tc_stats=tmp_path / "tc.parquet"))


def test_annotate_ap_without_config_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="amp"):
        run_annotate(_args(setup, add_ap="AP"))


def test_annotate_subread_without_config_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="subread"):
        run_annotate(_args(setup, extract_sub_reads_fastq=tmp_path / "sub.fq.gz"))


def test_annotate_nb_without_config_raises(tmp_path) -> None:
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="next_n"):
        run_annotate(_args(setup, add_nb="NB"))


def test_annotate_nb_without_fasta_raises(tmp_path) -> None:
    # config supplies next_n, but no --fasta -> still requires fasta
    cfg = {"seq1": {"next_n": 6}}
    setup = setup_env(tmp_path, cfg, H_SEQ1_28, _REC)
    with pytest.raises(ValueError, match="fasta"):
        run_annotate(_args(setup, add_nb="NB", config=setup.config_path))


def test_annotate_validation_creates_no_output(tmp_path) -> None:
    # A failed validate must not leave a partial output BAM behind.
    setup = setup_env(tmp_path, None, H_SEQ1_28, _REC)
    out = tmp_path / "out.bam"
    args = AnnotateArgs(input_bam=setup.input_bam, output_bam=out, add_tc="TC")
    with pytest.raises(ValueError):
        run_annotate(args)
    assert not out.exists()


def test_error_stats_missing_error_config_raises(tmp_path) -> None:
    # config declares tc params but no error_primer_*_len -> cfg.error is None
    cfg = {"seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8}}
    setup = setup_env(tmp_path, cfg, H_SEQ1_28, _REC)
    assert setup.config_path is not None
    with pytest.raises(ValueError, match="error_primer"):
        run_error_stats(
            ErrorStatsArgs(
                input_bam=setup.input_bam, config=setup.config_path, prefix=tmp_path / "err"
            )
        )


def test_error_stats_neither_prefix_nor_explicit_raises(tmp_path) -> None:
    cfg = {"seq1": {"error_primer_start_len": 5, "error_primer_end_len": 5}}
    setup = setup_env(tmp_path, cfg, H_SEQ1_28, _REC)
    assert setup.config_path is not None
    # error config is fine, but no prefix and no explicit out paths -> from_args raises
    with pytest.raises(ValueError, match="prefix|seven"):  # message mentions prefix / seven paths
        run_error_stats(ErrorStatsArgs(input_bam=setup.input_bam, config=setup.config_path))
