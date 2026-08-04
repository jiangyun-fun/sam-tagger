"""Tests for the BED interval module, ported from Rust bed.rs."""

import pytest

from sam_tagger.bed import (
    Interval,
    _merge_intervals_sorted,
    position_in_intervals,
    read_overlaps_intervals,
)


# ---------------------------------------------------------------------------
# 1. test_interval_creation
# ---------------------------------------------------------------------------
def test_interval_creation() -> None:
    interval = Interval(100, 200)
    assert interval.start == 100
    assert interval.end == 200


# ---------------------------------------------------------------------------
# 2. test_interval_invalid
# ---------------------------------------------------------------------------
def test_interval_invalid() -> None:
    with pytest.raises(ValueError):
        Interval(200, 100)  # start > end

    with pytest.raises(ValueError):
        Interval(100, 100)  # start == end


# ---------------------------------------------------------------------------
# 3. test_interval_overlaps
# ---------------------------------------------------------------------------
def test_interval_overlaps() -> None:
    a = Interval(100, 200)
    b = Interval(150, 250)
    c = Interval(200, 300)  # abutting
    d = Interval(300, 400)

    assert a.overlaps(b) is True
    assert a.overlaps(c) is False  # abutting does NOT overlap
    assert a.overlaps(d) is False


# ---------------------------------------------------------------------------
# 4. test_read_overlaps
# ---------------------------------------------------------------------------
def test_read_overlaps() -> None:
    interval = Interval(100, 200)

    assert interval.read_overlaps(50, 150) is True  # partial overlap
    assert interval.read_overlaps(100, 200) is True  # exact match
    assert interval.read_overlaps(150, 250) is True  # partial overlap
    assert interval.read_overlaps(50, 100) is False  # adjacent left
    assert interval.read_overlaps(200, 250) is False  # adjacent right
    assert interval.read_overlaps(250, 300) is False  # outside


# ---------------------------------------------------------------------------
# 5. test_interval_merge
# ---------------------------------------------------------------------------
def test_interval_merge() -> None:
    a = Interval(100, 200)
    b = Interval(150, 250)
    merged = a.merge(b)
    assert merged.start == 100
    assert merged.end == 250


# ---------------------------------------------------------------------------
# 6. test_merge_intervals_sorted
# ---------------------------------------------------------------------------
def test_merge_intervals_sorted() -> None:
    # 4 intervals -> 2 after merge
    intervals = [
        Interval(100, 200),
        Interval(150, 250),
        Interval(300, 400),
        Interval(400, 500),
    ]
    _merge_intervals_sorted(intervals)
    assert len(intervals) == 2
    assert intervals[0] == Interval(100, 250)
    assert intervals[1] == Interval(300, 500)


# ---------------------------------------------------------------------------
# 7. test_read_overlaps_intervals
# ---------------------------------------------------------------------------
def test_read_overlaps_intervals() -> None:
    intervals = [Interval(100, 200), Interval(300, 400), Interval(500, 600)]

    assert read_overlaps_intervals(150, 180, intervals) is True  # inside first
    assert read_overlaps_intervals(350, 380, intervals) is True  # inside second
    assert read_overlaps_intervals(150, 350, intervals) is True  # spans across
    assert read_overlaps_intervals(250, 280, intervals) is False  # gap
    assert read_overlaps_intervals(50, 90, intervals) is False  # before all
    assert read_overlaps_intervals(650, 700, intervals) is False  # after all


# ---------------------------------------------------------------------------
# 8. test_position_in_intervals
# ---------------------------------------------------------------------------
def test_position_in_intervals() -> None:
    intervals = [Interval(100, 200), Interval(300, 400)]

    # Inside intervals
    assert position_in_intervals(150, intervals) is True  # inside first
    assert position_in_intervals(350, intervals) is True  # inside second

    # Inclusive start boundaries
    assert position_in_intervals(100, intervals) is True  # start of first
    assert position_in_intervals(300, intervals) is True  # start of second

    # Exclusive end boundaries
    assert position_in_intervals(200, intervals) is False  # end of first
    assert position_in_intervals(400, intervals) is False  # end of second

    # Gap and outside
    assert position_in_intervals(250, intervals) is False  # gap
    assert position_in_intervals(50, intervals) is False  # before
    assert position_in_intervals(450, intervals) is False  # after


# ---------------------------------------------------------------------------
# 9. parse_bed merges + sorts across chromosomes
# ---------------------------------------------------------------------------
def test_parse_bed_merges_and_sorts(tmp_path):
    from sam_tagger.bed import parse_bed

    bed = tmp_path / "r.bed"
    bed.write_text(
        "seq1\t300\t400\n"   # out of order
        "seq1\t100\t250\n"   # overlaps next
        "seq1\t150\t200\n"
        "seq2\t10\t20\n"
        "# comment line\n"
        "seq2\t30\t40\n"
    )
    m = parse_bed(bed)
    # seq1: 100-250 absorbs 150-200; 300-400 stays separate (gap 250->300)
    assert m["seq1"] == [Interval(100, 250), Interval(300, 400)]
    assert m["seq2"] == [Interval(10, 20), Interval(30, 40)]


def test_parse_bed_rejects_bad(tmp_path):
    from sam_tagger.bed import parse_bed

    bed = tmp_path / "bad.bed"
    bed.write_text("seq1\t200\t100\n")  # start >= end
    with pytest.raises(ValueError):
        parse_bed(bed)
