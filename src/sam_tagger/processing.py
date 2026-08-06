"""Core BAM record processing functions.

CIGAR + cs-tag helpers. None of the removed tags (HI/HD/SP) lived here — those
were orchestrator-level, handled in ``lib.py``.
"""

from __future__ import annotations

import re
from typing import IO

from loguru import logger

from sam_tagger.config import ApParams

# pysam CIGAR operation codes
BAM_CMATCH = 0
BAM_CINS = 1
BAM_CDEL = 2
BAM_CREF_SKIP = 3
BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5
BAM_CPAD = 6
BAM_CEQUAL = 7
BAM_CDIFF = 8

# CS tag regex — single compiled instance shared across all modules
CS_REGEX = re.compile(r"([:*+\-])([a-zA-Z0-9]+)")

# Ref-consuming CIGAR ops: M, D, N, =, X
_REF_OPS = frozenset({BAM_CMATCH, BAM_CDEL, BAM_CREF_SKIP, BAM_CEQUAL, BAM_CDIFF})
# Read-consuming CIGAR ops: M, I, S, =, X
_READ_OPS = frozenset({BAM_CMATCH, BAM_CINS, BAM_CSOFT_CLIP, BAM_CEQUAL, BAM_CDIFF})


def sum_match_del_skip_cigar(cigartuples: list[tuple[int, int]]) -> int:
    """Sum of M/D/N CIGAR operation lengths (reference-consuming for FL check)."""
    total = 0
    for op, length in cigartuples:
        if op in (BAM_CMATCH, BAM_CDEL, BAM_CREF_SKIP):
            total += length
    return total


def calculate_soft_clip_lengths(cigartuples: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (start_soft_len, end_soft_len) from CIGAR."""
    start_soft = 0
    end_soft = 0
    if cigartuples:
        first_op, first_len = cigartuples[0]
        if first_op == BAM_CSOFT_CLIP:
            start_soft = first_len
        if len(cigartuples) > 1:
            last_op, last_len = cigartuples[-1]
            if last_op == BAM_CSOFT_CLIP:
                end_soft = last_len
    return start_soft, end_soft


def calculate_max_indel_lengths(cigartuples: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (max_insertion, max_deletion) lengths from CIGAR."""
    max_ins = 0
    max_del = 0
    for op, length in cigartuples:
        if op == BAM_CINS:
            max_ins = max(max_ins, length)
        elif op in (BAM_CDEL, BAM_CREF_SKIP):
            max_del = max(max_del, length)
    return max_ins, max_del


def calculate_match_start_end(cigartuples: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (match_start, match_end) — 1-based read positions.

    match_start is the 1-based position of the first match/mismatch base.
    match_end is the 1-based position of the last match/mismatch base.
    """
    read_pos = 0
    match_start = 0
    match_end = 0
    for op, length in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF):
            if match_start == 0:
                match_start = read_pos + 1
            match_end = read_pos + length
            read_pos += length
        elif op in (BAM_CINS, BAM_CSOFT_CLIP):
            read_pos += length
    return match_start, match_end


def calculate_ref_consuming_length_before_end_soft(cigartuples: list[tuple[int, int]]) -> int:
    """Sum M+D ops excluding trailing soft clip."""
    end = len(cigartuples)
    if cigartuples and cigartuples[-1][0] == BAM_CSOFT_CLIP:
        end -= 1
    total = 0
    for i in range(end):
        op, length = cigartuples[i]
        if op in (BAM_CMATCH, BAM_CDEL):
            total += length
    return total


def count_cs_error_events(cs_tag: str) -> int:
    """Count error events (*, +, -) in a cs tag string."""
    clean = cs_tag.removeprefix("cs:Z:")
    count = 0
    for caps in CS_REGEX.finditer(clean):
        op = caps.group(1)
        if op in ("*", "+", "-"):
            count += 1
    return count


def parse_cs_tag(cs_tag: str) -> list[tuple[str, str]]:
    """Parse CS tag into list of (op_type, value) tuples."""
    clean = cs_tag.removeprefix("cs:Z:")
    return [(m.group(1), m.group(2)) for m in CS_REGEX.finditer(clean)]


def count_errors_from_parsed(parsed_ops: list[tuple[str, str]]) -> int:
    """Count error events (*, +, -) from pre-parsed CS operations."""
    count = 0
    for op_type, _ in parsed_ops:
        if op_type in ("*", "+", "-"):
            count += 1
    return count


def is_on_target(
    pos: int,
    cs_tag: str,
    primer_start_len: int,
    primer_end_len: int,
    seq_len: int,
) -> bool:
    """Check if a read is 'on target' — no errors in the inner region.

    The inner region is [primer_start_len, seq_len - primer_end_len).
    """
    current_pos = pos
    clean = cs_tag.removeprefix("cs:Z:")
    inner_start = primer_start_len
    inner_end = seq_len - primer_end_len

    if current_pos > inner_start:
        return False

    for caps in CS_REGEX.finditer(clean):
        op_type = caps.group(1)
        val = caps.group(2)
        op_pos = current_pos

        if op_type == ":":
            current_pos += int(val)
        elif op_type == "*":
            current_pos += 1
        elif op_type == "-":
            current_pos += len(val)
        elif op_type == "+":
            pass  # no ref advance
        else:
            raise ValueError(f"Unexpected cs operation '{op_type}'")

        if op_type in ("*", "+", "-"):
            if inner_start < op_pos < inner_end:
                return False
            if op_type == "-" and current_pos > inner_start and op_pos <= inner_start:
                return False

    return current_pos >= inner_end


def is_on_target_parsed(
    pos: int,
    parsed_ops: list[tuple[str, str]],
    primer_start_len: int,
    primer_end_len: int,
    seq_len: int,
) -> bool:
    """Check if a read is 'on target' from pre-parsed CS operations."""
    current_pos = pos
    inner_start = primer_start_len
    inner_end = seq_len - primer_end_len

    if current_pos > inner_start:
        return False

    for op_type, val in parsed_ops:
        op_pos = current_pos

        if op_type == ":":
            current_pos += int(val)
        elif op_type == "*":
            current_pos += 1
        elif op_type == "-":
            current_pos += len(val)
        elif op_type == "+":
            pass

        if op_type in ("*", "+", "-"):
            if inner_start < op_pos < inner_end:
                return False
            if op_type == "-" and current_pos > inner_start and op_pos <= inner_start:
                return False

    return current_pos >= inner_end


def contains_primer(
    sequence_to_check: bytes,
    primer: bytes,
    min_match_len: int,
) -> bool:
    """Check if any min_match_len-length subsequence of sequence_to_check appears in primer."""
    if len(primer) < min_match_len or len(sequence_to_check) < min_match_len:
        return False
    return any(
        sequence_to_check[i : i + min_match_len] in primer
        for i in range(len(sequence_to_check) - min_match_len + 1)
    )


def check_amplicon_primers(
    cigartuples: list[tuple[int, int]],
    read_seq: bytes,
    pos: int,
    params: ApParams,
) -> str:
    """Return 'Both', 'StartOnly', 'EndOnly', or 'Neither'."""
    if not read_seq:
        return "Neither"

    start_soft_len, end_soft_len = calculate_soft_clip_lengths(cigartuples)
    has_start = _check_start_primer(cigartuples, read_seq, pos, params, start_soft_len)
    has_end = _check_end_primer(cigartuples, read_seq, pos, params, end_soft_len)

    match (has_start, has_end):
        case (True, True):
            return "Both"
        case (True, False):
            return "StartOnly"
        case (False, True):
            return "EndOnly"
        case _:
            return "Neither"


def _check_start_primer(
    cigartuples: list[tuple[int, int]],
    read_seq: bytes,
    pos: int,
    params: ApParams,
    start_soft_len: int,
) -> bool:
    read_pos = 0
    ref_pos = pos
    for op, length in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF, BAM_CDEL):
            if ref_pos <= params.primer_start_end_pos < ref_pos + length:
                offset = params.primer_start_end_pos - ref_pos + 1
                slice_end = read_pos + offset if op != BAM_CDEL else read_pos
                if slice_end > start_soft_len:
                    seq_to_check = read_seq[start_soft_len:slice_end]
                    return contains_primer(
                        seq_to_check, params.primer_start_seq, params.primer_start_min_match
                    )
            ref_pos += length
            if op != BAM_CDEL:
                read_pos += length
        elif op in (BAM_CINS, BAM_CSOFT_CLIP):
            read_pos += length
        else:
            raise ValueError(f"Unexpected CIGAR op in check_start_primer: {op}")
    return False


def _check_end_primer(
    cigartuples: list[tuple[int, int]],
    read_seq: bytes,
    pos: int,
    params: ApParams,
    end_soft_len: int,
) -> bool:
    read_pos = 0
    ref_pos = pos
    read_len = len(read_seq)
    for op, length in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF, BAM_CDEL):
            if ref_pos < params.primer_end_start_pos <= ref_pos + length:
                offset = params.primer_end_start_pos - ref_pos
                slice_start = read_pos + offset if op != BAM_CDEL else read_pos
                slice_end = read_len - end_soft_len
                if slice_end > slice_start:
                    seq_to_check = read_seq[slice_start:slice_end]
                    return contains_primer(
                        seq_to_check, params.primer_end_seq, params.primer_end_min_match
                    )
            ref_pos += length
            if op != BAM_CDEL:
                read_pos += length
        elif op in (BAM_CINS, BAM_CSOFT_CLIP):
            read_pos += length
    return False


def extract_sub_read_for_ref_region(
    cigartuples: list[tuple[int, int]],
    read_seq: bytes,
    read_qual: list[int],
    record_pos: int,
    target_ref_start: int,
    target_ref_end: int,
) -> tuple[bytes, list[int]] | None:
    """Extract sub-read seq/qual for a reference region using CIGAR."""
    current_ref_pos = record_pos
    current_read_pos = 0
    sub_seq = bytearray()
    sub_qual: list[int] = []

    # Calculate reference end from CIGAR
    record_ref_end = record_pos + sum(length for op, length in cigartuples if op in _REF_OPS)

    if current_ref_pos >= target_ref_end or record_ref_end <= target_ref_start:
        return None

    for op, op_len in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF):
            op_ref_end = current_ref_pos + op_len
            overlap_start = max(current_ref_pos, target_ref_start)
            overlap_end = min(op_ref_end, target_ref_end)
            if overlap_start < overlap_end:
                copy_len = overlap_end - overlap_start
                read_offset = overlap_start - current_ref_pos
                rs = current_read_pos + read_offset
                sub_seq.extend(read_seq[rs : rs + copy_len])
                sub_qual.extend(read_qual[rs : rs + copy_len])
            current_ref_pos += op_len
            current_read_pos += op_len
        elif op == BAM_CINS:
            if current_ref_pos > target_ref_start and current_ref_pos < target_ref_end:
                sub_seq.extend(read_seq[current_read_pos : current_read_pos + op_len])
                sub_qual.extend(read_qual[current_read_pos : current_read_pos + op_len])
            current_read_pos += op_len
        elif op in (BAM_CDEL, BAM_CREF_SKIP):
            current_ref_pos += op_len
        elif op == BAM_CSOFT_CLIP:
            current_read_pos += op_len
        elif op in (BAM_CHARD_CLIP, BAM_CPAD):
            pass

        if current_ref_pos >= target_ref_end:
            break

    return bytes(sub_seq), sub_qual


def resolve_subread_coords(
    unresolved_start: int, unresolved_end: int, ref_len: int
) -> tuple[int, int] | None:
    """Resolve potentially negative coordinates.

    Negative values count from end: -1 → ref_len.
    Returns (start_0based, end_0based) if valid, else None.
    """
    start_0based = unresolved_start if unresolved_start >= 0 else ref_len + unresolved_start + 1
    end_0based = unresolved_end if unresolved_end >= 0 else ref_len + unresolved_end + 1
    if end_0based >= start_0based and end_0based <= ref_len:
        return (start_0based, end_0based)
    return None


def write_fastq_entry(writer: IO[str], qname: str, seq: bytes, qual: list[int]) -> None:
    """Write a 4-line FASTQ entry. Convert Phred scores to Phred+33."""
    seq_str = seq.decode("ascii", errors="replace")
    qual_str = "".join(chr(q + 33) for q in qual)
    writer.write(f"@{qname}\n{seq_str}\n+\n{qual_str}\n")


def fetch_next_n_bases(
    reference_end: int,
    ref_len: int,
    ref_name: str,
    n_bases: int,
    faidx_reader,  # pysam.FastaFile
) -> str | None:
    """Fetch N bases from reference after alignment end."""
    if reference_end >= ref_len:
        return None
    fetch_len = min(n_bases, ref_len - reference_end)
    if fetch_len <= 0:
        return None
    try:
        seq = faidx_reader.fetch(ref_name, reference_end, reference_end + fetch_len).upper()
        return seq
    except Exception as e:
        logger.warning(
            "Failed to fetch sequence for NB tag from FASTA for ref '{}': {}", ref_name, e
        )
        return None
