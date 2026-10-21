"""Tests for the processing module, ported from Rust processing.rs and lib.rs tests."""

from __future__ import annotations

from sam_tagger.config import ApParams
from sam_tagger.processing import (
    BAM_CDEL,
    BAM_CINS,
    BAM_CMATCH,
    BAM_CREF_SKIP,
    BAM_CSOFT_CLIP,
    calculate_match_start_end,
    calculate_max_indel_lengths,
    calculate_ref_consuming_length_before_end_soft,
    calculate_soft_clip_lengths,
    check_amplicon_primers,
    contains_primer,
    count_cs_error_events,
    count_errors_from_parsed,
    extract_sub_read_for_ref_region,
    homopolymer_run_span,
    is_on_target,
    is_on_target_parsed,
    parse_cs_tag,
    resolve_subread_coords,
    sum_match_del_skip_cigar,
    write_fastq_entry,
)

# --- CIGAR helper tests ---


def test_calculate_soft_clip_lengths() -> None:
    assert calculate_soft_clip_lengths([(BAM_CSOFT_CLIP, 5), (BAM_CMATCH, 20), (BAM_CSOFT_CLIP, 10)]) == (5, 10)
    assert calculate_soft_clip_lengths([(BAM_CSOFT_CLIP, 7), (BAM_CMATCH, 30)]) == (7, 0)
    assert calculate_soft_clip_lengths([(BAM_CMATCH, 40), (BAM_CSOFT_CLIP, 8)]) == (0, 8)
    assert calculate_soft_clip_lengths([(BAM_CMATCH, 50)]) == (0, 0)
    # Soft clip in middle doesn't count
    assert calculate_soft_clip_lengths([(BAM_CMATCH, 10), (BAM_CSOFT_CLIP, 5), (BAM_CMATCH, 10)]) == (0, 0)
    # Empty CIGAR
    assert calculate_soft_clip_lengths([]) == (0, 0)


def test_sum_match_del_skip_cigar() -> None:
    cigartuples = [(BAM_CMATCH, 10), (BAM_CINS, 5), (BAM_CDEL, 3), (BAM_CREF_SKIP, 2), (BAM_CSOFT_CLIP, 8)]
    assert sum_match_del_skip_cigar(cigartuples) == 15  # 10 + 3 + 2


def test_calculate_max_indel_lengths() -> None:
    cigartuples = [(BAM_CMATCH, 10), (BAM_CINS, 3), (BAM_CMATCH, 5), (BAM_CINS, 7), (BAM_CDEL, 4)]
    max_ins, max_del = calculate_max_indel_lengths(cigartuples)
    assert max_ins == 7
    assert max_del == 4
    # No indels
    assert calculate_max_indel_lengths([(BAM_CMATCH, 10)]) == (0, 0)


def test_calculate_match_start_end() -> None:
    # 5S + 10M + 2I + 8M + 4S
    cigartuples = [
        (BAM_CSOFT_CLIP, 5), (BAM_CMATCH, 10), (BAM_CINS, 2),
        (BAM_CMATCH, 8), (BAM_CSOFT_CLIP, 4),
    ]
    start, end = calculate_match_start_end(cigartuples)
    assert start == 6   # after 5S, 1-based pos of first match base
    assert end == 25    # 5S + 10M + 2I + 8M = 25
    # No match ops
    assert calculate_match_start_end([(BAM_CSOFT_CLIP, 5)]) == (0, 0)


def test_calculate_ref_consuming_length_before_end_soft() -> None:
    # 10M + 2D + 5I + 8M + 4S
    cigartuples = [
        (BAM_CMATCH, 10), (BAM_CDEL, 2), (BAM_CINS, 5),
        (BAM_CMATCH, 8), (BAM_CSOFT_CLIP, 4),
    ]
    assert calculate_ref_consuming_length_before_end_soft(cigartuples) == 20  # 10+2+8


# --- CS tag tests ---


def test_count_cs_error_events() -> None:
    assert count_cs_error_events(":10*ac:5-gt:3+at:2") == 3  # *, -, +
    assert count_cs_error_events(":100") == 0
    assert count_cs_error_events("cs:Z::10*ac:5") == 1  # cs:Z: prefix stripped


def test_parse_cs_tag_and_count() -> None:
    ops = parse_cs_tag("cs:Z::10*ac+g:5-tt:3")
    assert ops == [(":", "10"), ("*", "ac"), ("+", "g"), (":", "5"), ("-", "tt"), (":", "3")]
    assert count_errors_from_parsed(ops) == 3
    assert parse_cs_tag(":50") == [(":", "50")]


def test_is_on_target() -> None:
    # seq_len=100, primer_start_len=10, primer_end_len=10
    # inner region: [10, 90)

    # Perfect 100bp match → on target
    assert is_on_target(0, ":100", 10, 10, 100) is True

    # Mismatch at pos 20 (inside inner region) → off target
    assert is_on_target(0, ":20*at:79", 10, 10, 100) is False

    # Mismatch at pos 5 (in start primer) → on target
    assert is_on_target(0, ":5*at:94", 10, 10, 100) is True

    # Deletion spanning into inner region → off target
    assert is_on_target(0, ":9-gg:90", 10, 10, 100) is False

    # Start position past inner region start → off target
    assert is_on_target(11, ":89", 10, 10, 100) is False


def test_is_on_target_parsed_matches() -> None:
    # The parsed variant must agree with the string variant.
    for cs in (":100", ":20*at:79", ":5*at:94", ":9-gg:90"):
        ops = parse_cs_tag(cs)
        assert is_on_target(0, cs, 10, 10, 100) == is_on_target_parsed(0, ops, 10, 10, 100)


# --- Primer detection tests ---


def test_contains_primer() -> None:
    primer = b"GATTACAGATTACA"
    assert contains_primer(b"XXGATTACAGAYY", primer, 8) is True
    assert contains_primer(b"XXGATTAGAGAYY", primer, 8) is False
    assert contains_primer(b"XXGATTACYYY", primer, 8) is False
    assert contains_primer(b"GATTACA", primer, 8) is False  # sequence shorter than min_match


def test_check_amplicon_primers() -> None:
    params = ApParams(
        primer_start_end_pos=9,   # primer region ends at 0-based pos 9
        primer_end_start_pos=90,  # end primer starts at 0-based pos 90
        primer_start_seq=b"ACGTACGT",
        primer_end_seq=b"TGCATGCA",
        primer_start_min_match=4,
        primer_end_min_match=4,
    )
    # Read at pos 0 with 100M and matching seq → Both
    seq = b"ACGTACGTAC" + b"A" * 80 + b"TGCATGCA" + b"A" * 10
    result = check_amplicon_primers(
        [(BAM_CMATCH, 100)], seq, 0, params,
    )
    assert result == "both"

    # Empty read → Neither
    assert check_amplicon_primers([(BAM_CMATCH, 10)], b"", 0, params) == "neither"


# --- Sub-read extraction test (from Rust) ---


def test_extract_sub_read_for_ref_region() -> None:
    # CIGAR: 10S + 2M + 2I + 2M + 3D + 4M + 10S at pos 100
    seq = b"S" * 10 + b"AA" + b"II" + b"CC" + b"GG" + b"TT" + b"S" * 10
    qual = [10] * 10 + [20, 20] + [25, 25] + [30, 30] + [35, 35] + [15, 15] + [10] * 10
    cigartuples = [
        (BAM_CSOFT_CLIP, 10),
        (BAM_CMATCH, 2),
        (BAM_CINS, 2),
        (BAM_CMATCH, 2),
        (BAM_CDEL, 3),
        (BAM_CMATCH, 4),
        (BAM_CSOFT_CLIP, 10),
    ]

    # Case 1: Target region within first match (pos 100-101)
    result = extract_sub_read_for_ref_region(cigartuples, seq, qual, 100, 100, 101)
    assert result is not None
    sub_seq, sub_qual = result
    assert sub_seq == b"A"
    assert sub_qual == [20]

    # Case 2: Target spans first match + insertion + second match (pos 100-103)
    result = extract_sub_read_for_ref_region(cigartuples, seq, qual, 100, 100, 103)
    assert result is not None
    sub_seq, sub_qual = result
    assert sub_seq == b"AAIIC"  # AA (match) + II (ins) + C (partial match)
    assert sub_qual == [20, 20, 25, 25, 30]

    # Case 3: target entirely outside the read span → None
    assert extract_sub_read_for_ref_region(cigartuples, seq, qual, 100, 500, 600) is None


def test_resolve_subread_coords() -> None:
    # Positive coords
    assert resolve_subread_coords(10, 90, 100) == (10, 90)

    # Negative coords: -1 means last position
    assert resolve_subread_coords(-1, -1, 100) == (100, 100)

    # Mixed
    assert resolve_subread_coords(0, -1, 100) == (0, 100)

    # Invalid: end before start
    assert resolve_subread_coords(90, 10, 100) is None

    # Invalid: end exceeds ref_len
    assert resolve_subread_coords(0, 200, 100) is None


def test_write_fastq_entry(tmp_path) -> None:
    path = tmp_path / "test.fq"
    with open(path, "w") as f:
        write_fastq_entry(f, "read1", b"ACGT", [30, 31, 32, 33])
    content = path.read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "@read1"
    assert lines[1] == "ACGT"
    assert lines[2] == "+"
    assert lines[3] == "?@AB"  # Phred+33: 30+33=63='?', 31+33=64='@', etc.


def test_homopolymer_run_span() -> None:
    ref = "TTTAAATTT"
    # AAA run: 0-based [3, 6)
    assert homopolymer_run_span(ref, 3) == (3, 6)
    assert homopolymer_run_span(ref, 4) == (3, 6)
    assert homopolymer_run_span(ref, 5) == (3, 6)
    # T runs at both sequence edges
    assert homopolymer_run_span(ref, 0) == (0, 3)
    assert homopolymer_run_span(ref, 2) == (0, 3)
    assert homopolymer_run_span(ref, 6) == (6, 9)
    assert homopolymer_run_span(ref, 8) == (6, 9)
    # single-base run (no neighbors equal)
    assert homopolymer_run_span("ACGT", 1) == (1, 2)
    assert homopolymer_run_span("ACGT", 2) == (2, 3)
    # whole-sequence homopolymer
    assert homopolymer_run_span("AAAA", 2) == (0, 4)
