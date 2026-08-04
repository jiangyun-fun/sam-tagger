"""sam-tagger — SAM/BAM tagger and amplicon QC statistics generator.

A clean redesign of bam_tagger / py-bam-tagger, split into four focused
subcommands: ``annotate`` (add tags + FASTQ extraction + tag stats),
``error-stats`` (7-file cs-tag error analysis), ``pileup`` (bcftools
mpileup-like per-position statistics), and ``qc-summary`` (one per-reference QC
row).

The orchestrators, args dataclasses, collectors, and config loader are all
importable directly from the top-level package (see ``__all__``). ``__version__``
is defined before the re-exports so ``sam_tagger.cli`` (which does
``from sam_tagger import __version__``) resolves during import.
"""

__version__ = "3.0.0"

from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs, PileupArgs, QcSummaryArgs
from sam_tagger.config import load_config
from sam_tagger.lib import run_annotate, run_error_stats, run_pileup, run_qc_summary
from sam_tagger.stats.error_stats import ErrorStatsCollector
from sam_tagger.stats.pileup_stats import PileupStatsCollector
from sam_tagger.stats.qc_summary import QcSummaryCollector
from sam_tagger.stats.tag_stats import StatsCollector

__all__ = [
    "__version__",
    "AnnotateArgs",
    "ErrorStatsArgs",
    "PileupArgs",
    "QcSummaryArgs",
    "StatsCollector",
    "ErrorStatsCollector",
    "PileupStatsCollector",
    "QcSummaryCollector",
    "load_config",
    "run_annotate",
    "run_error_stats",
    "run_pileup",
    "run_qc_summary",
]
