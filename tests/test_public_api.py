"""Smoke test for the public library API (``sam_tagger.__all__``)."""

from __future__ import annotations

import sam_tagger


def test_all_names_are_exported() -> None:
    for name in sam_tagger.__all__:
        assert hasattr(sam_tagger, name), f"{name!r} listed in __all__ but not importable"


def test_public_imports_callable() -> None:
    from sam_tagger import (
        AnnotateArgs,
        ErrorStatsArgs,
        ErrorStatsCollector,
        PileupArgs,
        PileupStatsCollector,
        QcSummaryArgs,
        QcSummaryCollector,
        StatsCollector,
        load_config,
        run_annotate,
        run_error_stats,
        run_pileup,
        run_qc_summary,
    )

    # the four orchestrators are callable; the rest are constructable types
    assert callable(run_annotate)
    assert callable(run_error_stats)
    assert callable(run_pileup)
    assert callable(run_qc_summary)
    assert callable(load_config)
    for cls in (AnnotateArgs, ErrorStatsArgs, PileupArgs, QcSummaryArgs,
                StatsCollector, ErrorStatsCollector, PileupStatsCollector, QcSummaryCollector):
        assert isinstance(cls, type)
