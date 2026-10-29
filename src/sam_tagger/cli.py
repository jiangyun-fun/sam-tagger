"""CLI for sam-tagger: four focused subcommands.

``annotate``     — BAM → tagged BAM (+ optional FASTQ, optional tag stats).
``error-stats``  — BAM → 8-file cs-tag error analysis (read-only).
``pileup``       — BAM → per-position pileup statistics (read-only).
``qc-summary``   — BAM → one per-reference QC summary Parquet (FL/TC/AP/error).

Each subcommand binds directly to a dataclass (``AnnotateArgs`` /
``ErrorStatsArgs`` / ``PileupArgs``) via ``Annotated[Args, Parameter(name="")]``,
which flattens the dataclass fields into clean top-level flags (``--config``
rather than ``--args.config``). The dataclass is therefore the single source of
truth for both the CLI surface and the orchestrator input — no duplicated field
lists.

The removed tags ``HI``/``HD`` (=IL>0/=DL>0) and ``SP`` (=SS+1) are gone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
from cyclopts import Parameter

from sam_tagger import __version__

app = cyclopts.App(
    name="sam-tagger",
    version=__version__,
    help="SAM/BAM tagger and amplicon QC statistics generator.",
)


def _configure_logging(verbose: int) -> None:
    """Map a -v count to a loguru stderr level."""
    from loguru import logger

    level = {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(verbose, "TRACE")
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")


# ===========================================================================
# annotate
# ===========================================================================


@dataclass
class AnnotateArgs:
    """Arguments for ``sam-tagger annotate``."""

    input_bam: Annotated[Path, Parameter(alias="-i", help="Input BAM. Use '-' for stdin.")]
    output_bam: Annotated[
        Path,
        Parameter(alias="-o", help="Output file; format via --output-format. '-' = stdout."),
    ]
    config: Annotated[
        Path | None,
        Parameter(
            help="CSV config of per-ref primer/subread/NB params. Required for TC/AP/NB/subread."
        ),
    ] = None
    fasta: Annotated[
        Path | None,
        Parameter(help="Indexed reference FASTA (samtools faidx). Required for --add-nb."),
    ] = None
    output_format: Annotated[
        Literal["bam", "cram", "sam"],
        Parameter(group="Misc", help="Output format (default bam). 'cram' requires --fasta."),
    ] = "bam"

    # --- Tags (value = the BAM tag name to write) ---
    add_fl: Annotated[
        str | None, Parameter(group="Tags", help="Full-length: '1' if read spans whole reference.")
    ] = None
    add_ms: Annotated[
        str | None, Parameter(group="Tags", help="Match start (1-based read position).")
    ] = None
    add_me: Annotated[
        str | None, Parameter(group="Tags", help="Match end (1-based read position).")
    ] = None
    add_tc: Annotated[
        str | None,
        Parameter(
            group="Tags", help="On-target ('on_target') via cs tag, no errors in primer regions."
        ),
    ] = None
    add_ap: Annotated[
        str | None,
        Parameter(group="Tags", help="Amplicon primer: 'both'/'start_only'/'end_only'/'neither'."),
    ] = None
    add_ss: Annotated[str | None, Parameter(group="Tags", help="Start (5') soft-clip length.")] = (
        None
    )
    add_es: Annotated[str | None, Parameter(group="Tags", help="End (3') soft-clip length.")] = None
    add_dm: Annotated[
        str | None,
        Parameter(group="Tags", help="Sum of M+D CIGAR ops (aligned ref length, excl. introns)."),
    ] = None
    add_ee: Annotated[
        str | None, Parameter(group="Tags", help="Error-event count from cs tag.")
    ] = None
    add_il: Annotated[
        str | None, Parameter(group="Tags", help="Longest insertion length in CIGAR.")
    ] = None
    add_dl: Annotated[
        str | None, Parameter(group="Tags", help="Longest deletion length in CIGAR.")
    ] = None
    add_nb: Annotated[
        str | None,
        Parameter(
            group="Tags", help="Next N bases from reference after alignment end (needs --fasta)."
        ),
    ] = None

    # --- Extraction ---
    extract_sub_reads_fastq: Annotated[
        Path | None,
        Parameter(group="Extraction", help="Extract per-ref sub-reads to gzipped FASTQ."),
    ] = None
    extract_left_softclip_fastq: Annotated[
        Path | None, Parameter(group="Extraction", help="Extract 5' soft-clips to gzipped FASTQ.")
    ] = None
    extract_right_softclip_fastq: Annotated[
        Path | None, Parameter(group="Extraction", help="Extract 3' soft-clips to gzipped FASTQ.")
    ] = None

    # --- Tag statistics ---
    fl_stats: Annotated[
        Path | None, Parameter(group="Tag statistics", help="FL tag statistics Parquet.")
    ] = None
    tc_stats: Annotated[
        Path | None, Parameter(group="Tag statistics", help="TC tag statistics Parquet.")
    ] = None
    ap_stats: Annotated[
        Path | None, Parameter(group="Tag statistics", help="AP tag statistics Parquet.")
    ] = None
    nb_stats: Annotated[
        Path | None, Parameter(group="Tag statistics", help="NB tag statistics Parquet.")
    ] = None

    verbose: Annotated[
        int,
        Parameter(
            alias="-v",
            count=True,
            group="Misc",
            help="Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE).",
        ),
    ] = 0


@app.command
def annotate(args: Annotated[AnnotateArgs, Parameter(name="")]) -> None:
    """Add BAM tags (optionally extract FASTQ + emit tag statistics) in a single pass."""
    _configure_logging(args.verbose)
    from sam_tagger.lib import run_annotate

    run_annotate(args)


# ===========================================================================
# error-stats
# ===========================================================================


@dataclass
class ErrorStatsArgs:
    """Arguments for ``sam-tagger error-stats``."""

    input_bam: Annotated[Path, Parameter(alias="-i", help="Input BAM (must carry cs tags).")]
    config: Annotated[Path, Parameter(help="CSV config with per-ref error_primer_*_len.")]
    fasta: Annotated[
        Path | None,
        Parameter(
            help="Indexed reference FASTA (samtools faidx). Enables the smoothed_deletion "
            "column; without it the column is null.",
        ),
    ] = None
    prefix: Annotated[
        Path | None,
        Parameter(
            group="Output",
            help="Prefix for all 8 outputs ({p}.error.parquet, …). Exclusive with --*-out flags.",
        ),
    ] = None
    error_out: Annotated[
        Path | None, Parameter(group="Output", help="Positional error counts Parquet.")
    ] = None
    mismatch_out: Annotated[
        Path | None, Parameter(group="Output", help="Mismatch matrix Parquet.")
    ] = None
    insertion_out: Annotated[
        Path | None, Parameter(group="Output", help="Insertion summary Parquet.")
    ] = None
    deletion_out: Annotated[
        Path | None, Parameter(group="Output", help="Deletion summary Parquet.")
    ] = None
    per_read_out: Annotated[
        Path | None, Parameter(group="Output", help="Per-read error Parquet.")
    ] = None
    per_base_out: Annotated[
        Path | None, Parameter(group="Output", help="Per-base error Parquet.")
    ] = None
    single_deletion_out: Annotated[
        Path | None, Parameter(group="Output", help="Single-deletion counts Parquet.")
    ] = None
    ref_summary_out: Annotated[
        Path | None,
        Parameter(
            group="Output",
            help="Per-reference summary Parquet (max_min_rate, max_error_rate).",
        ),
    ] = None

    verbose: Annotated[
        int, Parameter(alias="-v", count=True, group="Misc", help="Increase verbosity.")
    ] = 0


@app.command
def error_stats(args: Annotated[ErrorStatsArgs, Parameter(name="")]) -> None:
    """Generate 8 cs-tag error-statistics Parquet files (read-only, no BAM output)."""
    _configure_logging(args.verbose)
    from sam_tagger.lib import run_error_stats

    run_error_stats(args)


# ===========================================================================
# pileup
# ===========================================================================


@dataclass
class PileupArgs:
    """Arguments for ``sam-tagger pileup``."""

    input_bam: Annotated[Path, Parameter(alias="-i", help="Input BAM (must carry cs tags).")]
    output: Annotated[
        Path,
        Parameter(
            alias="-o", help="Output pileup Parquet (seq_name,pos,ref,alt,dp,adf,adr,ad,af)."
        ),
    ]
    fasta: Annotated[
        Path | None, Parameter(help="Indexed reference FASTA for ref bases (else ref='.').")
    ] = None
    intervals: Annotated[
        Path | None, Parameter(help="BED restricting which positions are reported.")
    ] = None
    min_base_qual: Annotated[
        int, Parameter(group="Misc", help="Min base quality for pileup (Phred). Default 20.")
    ] = 20

    verbose: Annotated[
        int, Parameter(alias="-v", count=True, group="Misc", help="Increase verbosity.")
    ] = 0


@app.command
def pileup(args: Annotated[PileupArgs, Parameter(name="")]) -> None:
    """Generate bcftools-mpileup-like per-position statistics (read-only)."""
    _configure_logging(args.verbose)
    from sam_tagger.lib import run_pileup

    run_pileup(args)


# ===========================================================================
# qc-summary
# ===========================================================================


@dataclass
class QcSummaryArgs:
    """Arguments for ``sam-tagger qc-summary``."""

    input_bam: Annotated[
        Path, Parameter(alias="-i", help="Input BAM (must carry cs tags for TC/error).")
    ]
    output: Annotated[
        Path,
        Parameter(alias="-o", help="Output per-reference QC summary Parquet."),
    ]
    config: Annotated[
        Path | None,
        Parameter(
            help="CSV config (tc/amp/error primer params). Omit to compute FL + read counts only.",
        ),
    ] = None
    length_dist_out: Annotated[
        Path | None,
        Parameter(
            help="Optional long-format length distributions (seq_name, metric, value, count); "
            "read_len + dm, primary alignments only.",
        ),
    ] = None
    verbose: Annotated[
        int, Parameter(alias="-v", count=True, group="Misc", help="Increase verbosity.")
    ] = 0


@app.command
def qc_summary(args: Annotated[QcSummaryArgs, Parameter(name="")]) -> None:
    """Emit one Parquet row per reference: FL, TC, AP, and per-base error rate (read-only)."""
    _configure_logging(args.verbose)
    from sam_tagger.lib import run_qc_summary

    run_qc_summary(args)
