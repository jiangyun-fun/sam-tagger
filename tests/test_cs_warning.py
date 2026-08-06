"""Tests for the cs-tag pre-flight warning (``_CsWatch``).

``_CsWatch`` is unit-tested directly (it owns all the warning logic), then the
wiring into the orchestrators is exercised through ``run_annotate`` /
``run_qc_summary``. Warnings are captured via a temporary loguru sink.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from _helpers import setup_env
from loguru import logger

from sam_tagger.cli import AnnotateArgs, QcSummaryArgs
from sam_tagger.lib import _CsWatch, run_annotate, run_qc_summary

Q = "I"
HDR_1_28 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28"


class _FakeRec:
    """Minimal stand-in for ``pysam.AlignedSegment`` (only ``has_tag`` needed)."""

    def __init__(self, has_cs: bool) -> None:
        self._has_cs = has_cs

    def has_tag(self, tag: str) -> bool:
        return self._has_cs if tag == "cs" else False


@pytest.fixture
def warn_messages() -> Iterator[list[str]]:
    """Capture loguru WARNING messages into a list."""
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


# ---------------------------------------------------------------------------
# unit: _CsWatch logic
# ---------------------------------------------------------------------------


def test_cswatch_disabled_is_silent(warn_messages: list[str]) -> None:
    watch = _CsWatch(needs_cs=False)
    for _ in range(2000):
        watch.observe(_FakeRec(has_cs=False))
    watch.summarize()
    assert warn_messages == []


def test_cswatch_below_threshold_only_summarizes(warn_messages: list[str]) -> None:
    watch = _CsWatch(needs_cs=True)
    for _ in range(500):  # below the 1000 early threshold
        watch.observe(_FakeRec(has_cs=False))
    watch.summarize()
    assert len(warn_messages) == 1
    assert "500" in warn_messages[0]


def test_cswatch_early_warning_fires_once_then_summary(warn_messages: list[str]) -> None:
    watch = _CsWatch(needs_cs=True)
    for _ in range(600):
        watch.observe(_FakeRec(has_cs=False))  # missing
    for _ in range(400):
        watch.observe(_FakeRec(has_cs=True))  # has cs -> at scan 1000, 600/1000 = 0.6 > 0.5
    for _ in range(500):
        watch.observe(_FakeRec(has_cs=False))  # more reads -> no second early warning
    watch.summarize()
    early = [m for m in warn_messages if "first" in m]
    summary = [m for m in warn_messages if "lacked" in m]
    assert len(early) == 1
    assert len(summary) == 1


def test_cswatch_all_have_cs_is_silent(warn_messages: list[str]) -> None:
    watch = _CsWatch(needs_cs=True)
    for _ in range(2000):
        watch.observe(_FakeRec(has_cs=True))
    watch.summarize()
    assert warn_messages == []


# ---------------------------------------------------------------------------
# integration: wired into the orchestrators
# ---------------------------------------------------------------------------


def test_qc_summary_warns_when_cs_missing(warn_messages: list[str], tmp_path: Path) -> None:
    cfg = {"seq1": {"tc_primer_start_len": 8, "tc_primer_end_len": 8}}
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"  # no cs tag
    setup = setup_env(tmp_path, cfg, HDR_1_28, sam)
    assert setup.config_path is not None
    out = tmp_path / "qc.parquet"
    run_qc_summary(QcSummaryArgs(input_bam=setup.input_bam, output=out, config=setup.config_path))
    assert any("lacked a 'cs' tag" in m for m in warn_messages)
    assert pl.read_parquet(out)["fl"].to_list() == [1]  # FL still computes without cs


def test_annotate_silent_when_only_fl(warn_messages: list[str], tmp_path: Path) -> None:
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"  # no cs tag
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    run_annotate(AnnotateArgs(input_bam=setup.input_bam, output_bam=setup.output_bam, add_fl="FL"))
    assert warn_messages == []  # needs_cs False -> silent


def test_annotate_warns_when_ee_requested_without_cs(
    warn_messages: list[str], tmp_path: Path
) -> None:
    sam = f"r1\t0\tseq1\t1\t60\t28M\t*\t0\t0\t{'ACGT' * 7}\t{Q * 28}\n"  # no cs tag
    setup = setup_env(tmp_path, None, HDR_1_28, sam)
    run_annotate(
        AnnotateArgs(
            input_bam=setup.input_bam, output_bam=setup.output_bam, add_fl="FL", add_ee="EE"
        )
    )
    assert any("lacked a 'cs' tag" in m for m in warn_messages)
