"""CLI dispatch tests — exercise the cyclopts app(argv) -> orchestrator path.

These cover the layer that the orchestrator-level tests skip entirely: argv
parsing, the dataclass flatten binding (Parameter(name="")), -i/-o/--add-*/
--config/--intervals, and that each subcommand actually drives its orchestrator
and emits output.
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import H_SEQ1_28, H_SEQ1_30, get_bam_tags_as_int, setup_env

from sam_tagger.cli import app


def _run(argv: list[str]) -> object:
    """Invoke the app in-process; return the SystemExit code."""
    with pytest.raises(SystemExit) as ei:
        app(argv)
    return ei.value.code


# ---------------------------------------------------------------------------


def test_cli_annotate_writes_tags(tmp_path) -> None:
    setup = setup_env(
        tmp_path,
        None,
        H_SEQ1_28,
        "r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGT\t*\tcs:Z::28\n",
    )
    code = _run(
        [
            "annotate",
            "-i",
            str(setup.input_bam),
            "-o",
            str(setup.output_bam),
            "--add-fl",
            "FL",
            "--add-dm",
            "DM",
        ]
    )
    assert code in (0, None)
    tags = get_bam_tags_as_int(setup.output_bam)
    assert tags["r1"]["FL"] == 1
    assert tags["r1"]["DM"] == 28


def test_cli_annotate_long_form_flags_and_config(tmp_path) -> None:
    cfg = {"seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8}}
    setup = setup_env(
        tmp_path,
        cfg,
        H_SEQ1_28,
        "r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\tAAAAAAAACCCCCCCCCCCCBBBBBBBB\t*\tcs:Z::28\n",
    )
    flp = tmp_path / "fl.parquet"
    # long-form --input-bam/--output-bam (not -i/-o) + --config + stats
    code = _run(
        [
            "annotate",
            "--input-bam",
            str(setup.input_bam),
            "--output-bam",
            str(setup.output_bam),
            "--config",
            str(setup.config_path),
            "--add-tc",
            "TC",
            "--fl-stats",
            str(flp),
        ]
    )
    assert code in (0, None)
    assert flp.exists()


def test_cli_error_stats_prefix(tmp_path) -> None:
    cfg = {"seq1": {"error_primer_start_len": 5, "error_primer_end_len": 5}}
    setup = setup_env(
        tmp_path,
        cfg,
        H_SEQ1_28,
        "r1\t0\tseq1\t1\t60\t7M1I4M1D15M\t*\t0\t0\tAAAAACCTXCCCGCCCCCCCCCCCCCC\t*\tcs:Z::7*ct+g:4-c:15\n",
    )
    pre = tmp_path / "err"
    code = _run(
        [
            "error-stats",
            "-i",
            str(setup.input_bam),
            "--config",
            str(setup.config_path),
            "--prefix",
            str(pre),
        ]
    )
    assert code in (0, None)
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


def test_cli_pileup_emits_rows(tmp_path) -> None:
    setup = setup_env(
        tmp_path,
        None,
        H_SEQ1_30,
        "r1\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACACGTACGTACGTACGTACGT\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::9*at:20\n",
    )
    out = tmp_path / "p.parquet"
    code = _run(["pileup", "-i", str(setup.input_bam), "-o", str(out), "--min-base-qual", "20"])
    assert code in (0, None)
    assert pl.read_parquet(out).height > 0


def test_cli_pileup_intervals(tmp_path) -> None:
    setup = setup_env(
        tmp_path,
        None,
        H_SEQ1_30,
        "r1\t0\tseq1\t1\t60\t30M\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTAC\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tcs:Z::30\n",
    )
    bed = tmp_path / "r.bed"
    bed.write_text("seq1\t5\t15\n")
    out = tmp_path / "p.parquet"
    code = _run(
        [
            "pileup",
            "-i",
            str(setup.input_bam),
            "-o",
            str(out),
            "--intervals",
            str(bed),
            "--min-base-qual",
            "0",
        ]
    )
    assert code in (0, None)
    assert pl.read_parquet(out)["pos"].sort().to_list() == list(range(6, 16))


def test_cli_help_exits_zero() -> None:
    assert _run(["--help"]) in (0, None)


def test_cli_version_exits_zero() -> None:
    assert _run(["--version"]) in (0, None)


def test_cli_unknown_command_errors() -> None:
    with pytest.raises(SystemExit) as ei:
        app(["bogus-subcommand"])
    assert ei.value.code not in (0, None)


def test_cli_missing_required_arg_errors(tmp_path) -> None:
    # annotate with neither -i nor -o -> usage error
    with pytest.raises(SystemExit) as ei:
        app(["annotate", "--add-fl", "FL"])
    assert ei.value.code not in (0, None)
