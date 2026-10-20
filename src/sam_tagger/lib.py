"""Per-subcommand orchestrators for sam-tagger.

Four independent entry points, each a focused single pass over the BAM:

- ``run_annotate``    — BAM -> tagged BAM (+ optional FASTQ + tag stats). Writes BAM.
- ``run_error_stats`` — BAM -> 7 cs-tag error Parquet files. Read-only.
- ``run_pileup``      — BAM -> pileup Parquet. Read-only.
- ``run_qc_summary``  — BAM -> one per-reference QC summary Parquet. Read-only.

Splitting the work across four commands means the BAM is re-read when a user
runs more than one, but each command has a fixed I/O contract (one output
shape) — making them clean, composable pipeline steps. The ``cs`` tag is
already on the BAM, so the re-reads are cheap.
"""

from __future__ import annotations

import gzip
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Protocol

import pysam
from loguru import logger

from sam_tagger.bed import parse_bed
from sam_tagger.cli import AnnotateArgs, ErrorStatsArgs, PileupArgs, QcSummaryArgs
from sam_tagger.config import AllConfigParams, load_config
from sam_tagger.processing import (
    CS_REGEX,
    calculate_match_start_end,
    calculate_max_indel_lengths,
    calculate_ref_consuming_length_before_end_soft,
    calculate_soft_clip_lengths,
    check_amplicon_primers,
    count_errors_from_parsed,
    extract_sub_read_for_ref_region,
    fetch_next_n_bases,
    is_on_target_parsed,
    parse_cs_tag,
    resolve_subread_coords,
    sum_match_del_skip_cigar,
    write_fastq_entry,
)
from sam_tagger.stats.error_stats import ErrorStatsCollector, ErrorStatsOutputPaths
from sam_tagger.stats.length_dist import LengthDistCollector
from sam_tagger.stats.pileup_stats import PileupStatsCollector
from sam_tagger.stats.qc_summary import QcSummaryCollector
from sam_tagger.stats.tag_stats import StatsCollector

# --- shared helpers ---


def _open_reader(input_bam: Path) -> pysam.AlignmentFile:
    """Open an alignment reader; '-' means stdin (BAM). Files auto-detect BAM/CRAM/SAM."""
    path = str(input_bam)
    if path == "-":
        return pysam.AlignmentFile("-", "rb")  # stdin stays BAM
    return pysam.AlignmentFile(path)  # auto-detect BAM/CRAM/SAM by content


def _open_writer(
    output: Path,
    header: pysam.AlignmentHeader,
    fmt: str = "bam",
    fasta: Path | None = None,
) -> pysam.AlignmentFile:
    """Open a BAM/CRAM/SAM writer preserving the input header; '-' means stdout.

    ``fmt`` selects the container (``bam``/``cram``/``sam``); CRAM requires a
    ``fasta`` reference to encode against.
    """
    dest = str(output)
    mode = {"bam": "wb", "cram": "wc", "sam": "w"}[fmt]
    if fmt == "cram":
        if fasta is None:
            raise ValueError("CRAM output requires --fasta (a reference to encode against).")
        return pysam.AlignmentFile(dest, mode, header=header, reference_filename=str(fasta))
    return pysam.AlignmentFile(dest, mode, header=header)


def _header_refs(reader: pysam.AlignmentFile) -> list[tuple[str, int]]:
    """Return [(ref_name, seq_len), ...] for every reference in the BAM header."""
    out: list[tuple[str, int]] = []
    for tid in range(reader.nreferences):
        name = reader.get_reference_name(tid)
        out.append((name, reader.get_reference_length(name)))
    return out


class _HasCsTag(Protocol):
    """Structural type for the cs-tag check — accepts pysam records and test fakes."""

    def has_tag(self, tag: str) -> bool: ...


class _CsWatch:
    """Counts mapped reads lacking a ``cs`` tag; warns early + summarizes.

    ``needs_cs`` gates the whole thing (``False`` -> ``observe``/``summarize`` are
    no-ops), so a run that only uses CIGAR-based features (FL) stays silent.
    Warn-only — never aborts (FL still works without cs).
    """

    _EARLY_THRESHOLD = 1000
    _EARLY_FRAC = 0.5

    def __init__(self, *, needs_cs: bool) -> None:
        self._needs = needs_cs
        self._scanned = 0
        self._missing = 0
        self._warned = False

    def observe(self, record: _HasCsTag) -> None:
        """Call once per mapped record, before any cs-based work."""
        if not self._needs:
            return
        self._scanned += 1
        if not record.has_tag("cs"):
            self._missing += 1
        if (
            not self._warned
            and self._scanned >= self._EARLY_THRESHOLD
            and self._missing / self._scanned > self._EARLY_FRAC
        ):
            logger.warning(
                "{} of the first {} mapped reads lack a 'cs' tag — TC/EE/AP/error/pileup "
                "will be empty for them. Align with minimap2 --cs so reads carry the cs tag?",
                self._missing,
                self._scanned,
            )
            self._warned = True

    def summarize(self) -> None:
        """Call after the read loop."""
        if self._needs and self._missing > 0:
            logger.warning(
                "{} mapped read(s) lacked a 'cs' tag and were skipped for cs-based features.",
                self._missing,
            )


# ===========================================================================
# annotate
# ===========================================================================


def _validate_annotate(args: AnnotateArgs, cfg: AllConfigParams) -> None:
    """Fail fast on missing inputs before any output file is created."""
    logger.info("Validating annotate arguments...")

    has_action = any([
        args.add_fl, args.add_ms, args.add_me, args.add_tc, args.add_ap,
        args.add_ss, args.add_es, args.add_dm, args.add_ee,
        args.add_il, args.add_dl, args.add_nb,
        args.fl_stats, args.tc_stats, args.ap_stats, args.nb_stats,
        args.extract_sub_reads_fastq,
        args.extract_left_softclip_fastq,
        args.extract_right_softclip_fastq,
    ])
    if not has_action:
        raise ValueError(
            "annotate: no action requested. Pass at least one --add-* tag, "
            "extraction, or --*-stats flag."
        )

    if args.output_format == "cram" and args.fasta is None:
        raise ValueError("annotate: --output-format cram requires --fasta.")

    if (args.add_tc is not None or args.tc_stats is not None) and cfg.tc is None:
        raise ValueError(
            "annotate: --add-tc/--tc-stats require a config with tc_primer_start_len "
            "and tc_primer_end_len."
        )
    if (args.add_ap is not None or args.ap_stats is not None) and cfg.ap is None:
        raise ValueError(
            "annotate: --add-ap/--ap-stats require a config with amp_primer_*_len "
            "and amp_primer_*_seq."
        )
    if args.extract_sub_reads_fastq is not None and cfg.sub_read is None:
        raise ValueError(
            "annotate: --extract-sub-reads-fastq requires a config with subread_start "
            "and subread_end."
        )
    if (args.add_nb is not None or args.nb_stats is not None) and cfg.nb is None:
        raise ValueError("annotate: --add-nb/--nb-stats require a config with next_n.")
    if (args.add_nb is not None or args.nb_stats is not None) and args.fasta is None:
        raise ValueError("annotate: --add-nb/--nb-stats require --fasta.")

    logger.info("Argument validation successful.")


def run_annotate(args: AnnotateArgs) -> None:
    """Single pass: add tags (+ extract FASTQ + tag stats), write output BAM."""
    with ExitStack() as stack:
        bam_reader = stack.enter_context(closing(_open_reader(args.input_bam)))
        header_refs = _header_refs(bam_reader)
        ref_lengths = dict(header_refs)
        cfg = load_config(args.config, ref_lengths)
        _validate_annotate(args, cfg)  # raises before any output file is opened

        logger.info("--- annotate: initialization ---")
        need_fasta = args.fasta is not None and (
            args.add_nb is not None or args.nb_stats is not None
        )
        faidx_reader = (
            stack.enter_context(closing(pysam.FastaFile(str(args.fasta))))
            if need_fasta else None
        )

        def _maybe_gzip(path: Path | None):
            return stack.enter_context(closing(gzip.open(path, "wt"))) if path else None

        fastq_writer = _maybe_gzip(args.extract_sub_reads_fastq)
        left_sc_writer = _maybe_gzip(args.extract_left_softclip_fastq)
        right_sc_writer = _maybe_gzip(args.extract_right_softclip_fastq)

        bam_writer = stack.enter_context(
            closing(
                _open_writer(
                    args.output_bam, bam_reader.header, args.output_format, args.fasta
                )
            )
        )
        stats = StatsCollector(args, cfg, [name for name, _ in header_refs])
        _annotate_loop(
            args, cfg, ref_lengths, bam_reader, bam_writer,
            faidx_reader, fastq_writer, left_sc_writer, right_sc_writer, stats,
        )
        logger.info("--- writing tag statistics ---")
        stats.write_results()

    logger.info("annotate complete. Output written to {}", args.output_bam)


def _annotate_loop(
    args: AnnotateArgs,
    cfg: AllConfigParams,
    ref_lengths: dict[str, int],
    bam_reader: pysam.AlignmentFile,
    bam_writer: pysam.AlignmentFile,
    faidx_reader,
    fastq_writer,
    left_sc_writer,
    right_sc_writer,
    stats: StatsCollector,
) -> None:
    fl_active = args.add_fl is not None or args.fl_stats is not None
    # cs is only needed for TC/EE (AP uses the read seq; FL/NB don't need cs).
    watch = _CsWatch(needs_cs=any([args.add_tc, args.tc_stats, args.add_ee]))
    record_count = 0

    logger.info("Processing BAM records...")
    for record in bam_reader.fetch(until_eof=True):
        record_count += 1
        if record_count % 50000 == 0:
            logger.info("Processed {} records...", record_count)

        if record.is_unmapped:
            bam_writer.write(record)
            continue

        watch.observe(record)

        qname = record.query_name or ""
        ref_name = bam_reader.get_reference_name(record.reference_id)
        header_ref_len = ref_lengths[ref_name]
        cigar = record.cigartuples or []

        # --- FL tag (header ref length; needs no config) ---
        if fl_active and sum_match_del_skip_cigar(cigar) == header_ref_len:
            stats.record_fl(ref_name)
            if args.add_fl is not None:
                record.set_tag(args.add_fl, 1, "i")

        # --- cs-derived tags (TC, EE) ---
        cs_tag = record.get_tag("cs") if record.has_tag("cs") else None
        parsed_ops = None
        if cs_tag is not None:
            parsed_ops = parse_cs_tag(str(cs_tag))

            if cfg.tc is not None:
                tc_params = cfg.tc.get(ref_name)
                if tc_params is not None and is_on_target_parsed(
                    record.reference_start, parsed_ops,
                    tc_params.primer_start_len, tc_params.primer_end_len, tc_params.seq_len,
                ):
                    stats.record_tc(ref_name)
                    if args.add_tc is not None:
                        record.set_tag(args.add_tc, "OnTarget", "Z")

            if args.add_ee is not None:
                record.set_tag(args.add_ee, count_errors_from_parsed(parsed_ops), "i")

        # --- AP tag ---
        if cfg.ap is not None:
            ap_params = cfg.ap.get(ref_name)
            if ap_params is not None:
                read_seq = record.query_sequence
                if read_seq:
                    status = check_amplicon_primers(
                        cigar, read_seq.encode(), record.reference_start, ap_params
                    )
                    stats.record_ap(ref_name, status)
                    if args.add_ap is not None:
                        record.set_tag(args.add_ap, status, "Z")

        # --- NB tag ---
        if cfg.nb is not None and faidx_reader is not None:
            nb_n = cfg.nb.get(ref_name)
            if nb_n is not None and record.reference_end is not None:
                fetched = fetch_next_n_bases(
                    record.reference_end, header_ref_len, ref_name, nb_n, faidx_reader
                )
                stats.record_nb(ref_name, fetched if fetched is not None else "")
                if args.add_nb is not None and fetched is not None:
                    record.set_tag(args.add_nb, fetched, "Z")

        # --- IL / DL tags (HI/HD removed: =IL>0 / =DL>0) ---
        if args.add_il is not None or args.add_dl is not None:
            max_ins, max_del = calculate_max_indel_lengths(cigar)
            if args.add_il is not None:
                record.set_tag(args.add_il, max_ins, "i")
            if args.add_dl is not None:
                record.set_tag(args.add_dl, max_del, "i")

        # --- MS / ME tags (1-based read positions) ---
        if args.add_ms is not None or args.add_me is not None:
            match_start, match_end = calculate_match_start_end(cigar)
            if match_start > 0:
                if args.add_ms is not None:
                    record.set_tag(args.add_ms, match_start, "i")
                if args.add_me is not None:
                    record.set_tag(args.add_me, match_end, "i")

        # --- SS / ES tags (SP removed: =SS+1) ---
        start_soft_len, end_soft_len = calculate_soft_clip_lengths(cigar)
        if args.add_ss is not None:
            record.set_tag(args.add_ss, start_soft_len, "i")
        if args.add_es is not None:
            record.set_tag(args.add_es, end_soft_len, "i")

        # --- DM tag ---
        if args.add_dm is not None:
            record.set_tag(
                args.add_dm, calculate_ref_consuming_length_before_end_soft(cigar), "i"
            )

        # --- soft-clip FASTQ extraction ---
        if left_sc_writer is not None and start_soft_len > 0:
            seq = record.query_sequence
            qual = record.query_qualities
            if seq and qual:
                write_fastq_entry(
                    left_sc_writer, qname,
                    seq[:start_soft_len].encode(), list(qual[:start_soft_len]),
                )
        if right_sc_writer is not None and end_soft_len > 0:
            seq = record.query_sequence
            qual = record.query_qualities
            if seq and qual:
                start_idx = len(seq) - end_soft_len
                write_fastq_entry(
                    right_sc_writer, qname,
                    seq[start_idx:].encode(), list(qual[start_idx:]),
                )

        # --- sub-read FASTQ extraction ---
        if fastq_writer is not None and cfg.sub_read is not None:
            unresolved = cfg.sub_read.get(ref_name)
            if unresolved is not None:
                resolved = resolve_subread_coords(unresolved.start, unresolved.end, header_ref_len)
                if resolved is not None:
                    start_0, end_0 = resolved
                    seq = record.query_sequence
                    qual = record.query_qualities
                    if seq and qual:
                        result = extract_sub_read_for_ref_region(
                            cigar, seq.encode(), list(qual), record.reference_start, start_0, end_0
                        )
                        if result is not None:
                            sub_seq, sub_qual = result
                            if sub_seq:
                                write_fastq_entry(fastq_writer, qname, sub_seq, sub_qual)

        bam_writer.write(record)

    logger.info("Processed {} records.", record_count)
    watch.summarize()


# ===========================================================================
# error-stats
# ===========================================================================


def run_error_stats(args: ErrorStatsArgs) -> None:
    """Single pass: tally cs-tag errors and write 7 Parquet files (read-only)."""
    bam_reader = _open_reader(args.input_bam)
    try:
        header_refs = _header_refs(bam_reader)
        cfg = load_config(args.config, dict(header_refs))
        if cfg.error is None:
            raise ValueError(
                "error-stats: config has no error_primer_start_len / error_primer_end_len entries."
            )
        paths = ErrorStatsOutputPaths.from_args(args)
        collector = ErrorStatsCollector(cfg.error, paths)

        logger.info("Processing BAM records for error statistics...")
        watch = _CsWatch(needs_cs=True)
        record_count = 0
        for record in bam_reader.fetch(until_eof=True):
            record_count += 1
            if record.is_unmapped:
                continue
            watch.observe(record)
            if not record.has_tag("cs"):
                continue
            parsed = parse_cs_tag(str(record.get_tag("cs")))
            ref_name = bam_reader.get_reference_name(record.reference_id)
            collector.tally_from_parsed(
                parsed, record.reference_start, ref_name, record.query_name or ""
            )

        logger.info("Processed {} records.", record_count)
        watch.summarize()
        collector.write_results()
    finally:
        bam_reader.close()

    logger.info("error-stats complete. Wrote 7 Parquet files (prefix/paths as given).")


# ===========================================================================
# pileup
# ===========================================================================


def run_pileup(args: PileupArgs) -> None:
    """Single pass: tally per-position pileup and write one Parquet (read-only)."""
    intervals = None
    if args.intervals is not None:
        logger.info("Parsing BED intervals from: {}", args.intervals)
        intervals = parse_bed(args.intervals)
        total = sum(len(v) for v in intervals.values())
        logger.info("Loaded {} intervals across {} chromosomes", total, len(intervals))

    faidx_reader = pysam.FastaFile(str(args.fasta)) if args.fasta is not None else None

    bam_reader = _open_reader(args.input_bam)
    try:
        header_refs = _header_refs(bam_reader)
        ref_names = [name for name, _ in header_refs]
        ref_lengths = dict(header_refs)

        logger.info("Initializing pileup statistics with min_base_qual={}", args.min_base_qual)
        collector = PileupStatsCollector(
            min_base_qual=args.min_base_qual,
            ref_names=ref_names,
            ref_lengths=ref_lengths,
            faidx_reader=faidx_reader,
            intervals=intervals,
        )

        logger.info("Processing BAM records for pileup statistics...")
        watch = _CsWatch(needs_cs=True)
        record_count = 0
        for record in bam_reader.fetch(until_eof=True):
            record_count += 1
            if record.is_unmapped:
                continue
            watch.observe(record)
            cigar = record.cigartuples
            if cigar is None or not record.has_tag("cs"):
                continue
            ref_name = bam_reader.get_reference_name(record.reference_id)
            collector.tally_from_record(
                str(record.get_tag("cs")),
                cigar,
                record.reference_start,
                record.query_qualities or (),
                ref_name,
                record.is_reverse,
                CS_REGEX,
                record.query_name or "",
                (record.query_sequence or "").encode(),
            )

        logger.info("Processed {} records.", record_count)
        watch.summarize()
        collector.write_results(args.output)
    finally:
        bam_reader.close()
        if faidx_reader is not None:
            faidx_reader.close()

    logger.info("pileup complete. Output written to {}", args.output)


# ===========================================================================
# qc-summary
# ===========================================================================


def run_qc_summary(args: QcSummaryArgs) -> None:
    """Single pass: emit one per-reference QC summary Parquet row (read-only).

    Joins FL (always, header length), TC + per-base error (need cs + config), and
    AP (need config) into one row per BAM-header reference. With no ``--config``,
    only ``total_reads``/``fl``/``fl_pct`` are populated.
    """
    bam_reader = _open_reader(args.input_bam)
    try:
        header_refs = _header_refs(bam_reader)
        ref_lengths = dict(header_refs)
        ref_names = [name for name, _ in header_refs]
        cfg = load_config(args.config, ref_lengths)
        error_collector = ErrorStatsCollector(cfg.error) if cfg.error is not None else None
        qc = QcSummaryCollector(ref_names)
        length_dist = (
            LengthDistCollector(args.length_dist_out) if args.length_dist_out is not None else None
        )

        logger.info("Processing BAM records for QC summary...")
        watch = _CsWatch(needs_cs=(cfg.tc is not None or cfg.error is not None))
        record_count = 0
        for record in bam_reader.fetch(until_eof=True):
            if record.is_unmapped:
                continue
            record_count += 1
            watch.observe(record)
            ref_name = bam_reader.get_reference_name(record.reference_id)
            cigar = record.cigartuples or []
            qc.tally_total(ref_name)
            qc.tally_strand(ref_name, record.is_reverse)
            if length_dist is not None and not record.is_secondary and not record.is_supplementary:
                length_dist.tally(
                    ref_name,
                    record.query_length,
                    calculate_ref_consuming_length_before_end_soft(cigar),
                )
            if sum_match_del_skip_cigar(cigar) == ref_lengths[ref_name]:
                qc.tally_fl(ref_name)

            cs_tag = record.get_tag("cs") if record.has_tag("cs") else None
            if cs_tag is not None:
                parsed_ops = parse_cs_tag(str(cs_tag))
                if cfg.tc is not None:
                    tc_params = cfg.tc.get(ref_name)
                    if tc_params is not None and is_on_target_parsed(
                        record.reference_start,
                        parsed_ops,
                        tc_params.primer_start_len,
                        tc_params.primer_end_len,
                        tc_params.seq_len,
                    ):
                        qc.tally_on_target(ref_name)
                if error_collector is not None:
                    error_collector.tally_from_parsed(
                        parsed_ops, record.reference_start, ref_name, record.query_name or ""
                    )

            if cfg.ap is not None:
                ap_params = cfg.ap.get(ref_name)
                if ap_params is not None:
                    read_seq = record.query_sequence
                    if read_seq:
                        status = check_amplicon_primers(
                            cigar, read_seq.encode(), record.reference_start, ap_params
                        )
                        qc.tally_ap(ref_name, status)

        logger.info("Processed {} records.", record_count)
        watch.summarize()
        if length_dist is not None:
            length_dist.write_results()
        qc.write_results(args.output, error_collector)
    finally:
        bam_reader.close()

    logger.info("qc-summary complete. Output written to {}", args.output)
