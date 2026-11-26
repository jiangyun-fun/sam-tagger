"""The dense pileup against the frozen per-base reference (`pileup_oracle.py`) on random BAMs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pysam
import pytest
from _helpers import write_fasta
from pileup_oracle import run_reference_pileup
from polars.testing import assert_frame_equal

from sam_tagger.cli import PileupArgs
from sam_tagger.lib import run_pileup
from sam_tagger.stats import pileup_dense, pileup_stats

_BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass
class _Reference:
    name: str
    seq: str
    header_len: int


def _random_references(rng: random.Random) -> list[_Reference]:
    refs = []
    for i in range(rng.randint(1, 4)):
        n = rng.randint(20, 140)
        seq = "".join(
            rng.choice(_BASES) if rng.random() > 0.05 else rng.choice("NnRYSWU-") for _ in range(n)
        )
        # The BAM header is sometimes longer than the FASTA sequence.
        header_len = n + (rng.randint(1, 8) if rng.random() < 0.3 else 0)
        refs.append(_Reference(f"ref{i}", seq, header_len))
    return refs


def _random_alignment(
    rng: random.Random,
    ref: _Reference,
    pos: int,
    *,
    error_rate: float = 0.18,
    stop_rate: float = 0.02,
) -> tuple[list[str], list, list[str]]:
    """cs ops, CIGAR ops and the aligned read bases, all in reference orientation."""
    # Past the FASTA end the reference base is unknown; the cs tag calls it N.
    seq = ref.seq.ljust(ref.header_len, "N")
    cs: list[str] = []
    cigar: list[tuple[int, int]] = []
    bases: list[str] = []
    for _ in range(400):
        if pos >= ref.header_len or rng.random() < stop_rate:
            break
        roll = rng.random()
        if roll < error_rate * 8 / 18:
            ref_char = seq[pos]
            alt = rng.choice([b for b in "ACGTN" if b != ref_char.upper()])
            if rng.random() < 0.1:
                alt = ref_char.upper()  # a substitution that names the reference base
            cs.append(f"*{ref_char}{alt.lower() if rng.random() < 0.5 else alt}")
            bases.append(alt)
            cigar.append((8, 1))
            pos += 1
        elif roll < error_rate * 13 / 18:
            inserted = "".join(rng.choice(_BASES) for _ in range(rng.randint(1, 4)))
            cs.append("+" + inserted)
            bases.extend(inserted)
            cigar.append((1, len(inserted)))
        elif roll < error_rate and ref.header_len - pos >= 2:
            deleted = rng.randint(1, min(4, ref.header_len - pos - 1))
            cs.append("-" + seq[pos : pos + deleted])
            cigar.append((2, deleted))
            pos += deleted
        else:
            matched = min(rng.randint(1, 8), ref.header_len - pos)
            cs.append(f":{matched}")
            bases.extend(seq[pos : pos + matched].upper())
            cigar.append((7 if rng.random() < 0.3 else 0, matched))
            pos += matched
    return cs, cigar, bases


def _random_read(
    rng: random.Random,
    header: pysam.AlignmentHeader,
    refs: list[_Reference],
    index: int,
    min_q: int,
    *,
    error_rate: float = 0.18,
    stop_rate: float = 0.02,
    kinds: tuple[str, ...] = (
        "plain",
        "secondary",
        "supplementary",
        "unmapped",
        "no_cs",
        "no_qual",
        "no_cigar",
        "long_cs",
    ),
) -> pysam.AlignedSegment:
    ref = refs[rng.randrange(len(refs))]
    kind = rng.choices(kinds, weights=[70] + [5] * (len(kinds) - 1))[0]
    pos = rng.randrange(ref.header_len)
    cs, cigar, bases = _random_alignment(rng, ref, pos, error_rate=error_rate, stop_rate=stop_rate)
    if kind == "long_cs":
        cs.append(f":{rng.randint(1, 5)}")  # cs ops past the end of the read
    clips = [
        rng.randint(1, n) if rng.random() < p else 0
        for n, p in ((6, 0.5), (3, 0.15), (6, 0.5), (3, 0.15))
    ]
    lead_s, lead_h, trail_s, trail_h = clips
    swapped = rng.random() < 0.3  # the hard clip after the soft clip
    ops = [(4, lead_s), (5, lead_h)] if swapped else [(5, lead_h), (4, lead_s)]
    ops = [op for op in ops if op[1]]
    ops += cigar
    ops += [(5, trail_h), (4, trail_s)] if swapped else [(4, trail_s), (5, trail_h)]
    ops = [op for op in ops if op[1]]

    seq = "".join(rng.choice(_BASES) for _ in range(lead_s)) + "".join(bases)
    seq += "".join(rng.choice(_BASES) for _ in range(trail_s))
    quals = [rng.choice([40] * 5 + [max(min_q - 1, 0), min_q, min_q + 1, 2, 30]) for _ in seq]
    reverse = rng.random() < 0.5
    if reverse:
        seq = seq.translate(_COMPLEMENT)[::-1]
        quals = quals[::-1]

    record = pysam.AlignedSegment(header)
    record.query_name = f"r{index}"
    record.flag = 0x10 * reverse + {
        "secondary": 0x100,
        "supplementary": 0x800,
        "unmapped": 0x4,
    }.get(kind, 0)
    record.reference_id = header.get_tid(ref.name)
    record.reference_start = pos
    record.mapping_quality = 60
    if kind != "no_cigar":
        if ops:
            record.cigartuples = ops
        record.query_sequence = seq
        if kind != "no_qual":
            record.query_qualities = pysam.qualitystring_to_array(
                "".join(chr(q + 33) for q in quals)
            )
    if kind != "no_cs":
        record.set_tag("cs", "cs:Z:" + "".join(cs))
    return record


def _header(refs: list[_Reference]) -> pysam.AlignmentHeader:
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": ref.name, "LN": ref.header_len} for ref in refs],
        }
    )


def _write_bam(
    path: Path,
    refs: list[_Reference],
    rng: random.Random,
    min_q: int,
    grouped: bool,
    *,
    n_reads: tuple[int, int] = (40, 140),
    error_rate: float = 0.18,
    stop_rate: float = 0.02,
    kinds: tuple[str, ...] = (
        "plain",
        "secondary",
        "supplementary",
        "unmapped",
        "no_cs",
        "no_qual",
        "no_cigar",
    ),
) -> None:
    header = _header(refs)
    records = [
        _random_read(
            rng, header, refs, i, min_q, error_rate=error_rate, stop_rate=stop_rate, kinds=kinds
        )
        for i in range(rng.randint(*n_reads))
    ]
    if grouped:
        records.sort(key=lambda record: (record.reference_id, record.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=header) as writer:
        for record in records:
            writer.write(record)


def _write_fasta(path: Path, refs: list[_Reference], rng: random.Random) -> Path:
    sequences = {ref.name: ref.seq for ref in refs}
    if len(refs) > 1 and rng.random() < 0.3:
        del sequences[refs[-1].name]  # a reference that the FASTA does not hold
    elif rng.random() < 0.3:
        name = refs[0].name
        sequences[name] = sequences[name][: -rng.randint(1, 5)]  # shorter than the header
    write_fasta(path, sequences)
    pysam.faidx(str(path))
    return path


def _write_bed(path: Path, refs: list[_Reference], rng: random.Random) -> Path:
    lines = []
    for ref in refs:
        for _ in range(rng.randint(1, 3)):
            start = rng.randint(0, ref.header_len)
            lines.append(
                f"{ref.name}\t{start}\t{start + rng.randint(1, ref.header_len // 2 + 4)}\n"
            )
    path.write_text("".join(lines))
    return path


def _assert_matches_reference(
    bam: Path, fasta: Path | None, bed: Path | None, min_q: int, tmp_path: Path
) -> pl.DataFrame:
    dense = tmp_path / "dense.parquet"
    run_pileup(
        PileupArgs(input_bam=bam, output=dense, min_base_qual=min_q, fasta=fasta, intervals=bed)
    )
    reference = tmp_path / "reference.parquet"
    run_reference_pileup(bam, reference, fasta=fasta, bed=bed, min_base_qual=min_q)
    expected = pl.read_parquet(reference)
    assert expected.height > 0
    assert_frame_equal(pl.read_parquet(dense), expected, check_exact=True)
    return expected


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("use_fasta", [False, True])
@pytest.mark.parametrize("use_bed", [False, True])
def test_dense_pileup_matches_reference(
    tmp_path: Path, seed: int, use_fasta: bool, use_bed: bool
) -> None:
    rng = random.Random(seed * 1000 + 7 * use_fasta + 131 * use_bed)
    refs = _random_references(rng)
    min_q = rng.choice([0, 10, 20, 30])
    bam = tmp_path / "reads.bam"
    _write_bam(bam, refs, rng, min_q, grouped=seed % 2 == 0)
    fasta = _write_fasta(tmp_path / "ref.fa", refs, rng) if use_fasta else None
    bed = _write_bed(tmp_path / "targets.bed", refs, rng) if use_bed else None
    _assert_matches_reference(bam, fasta, bed, min_q, tmp_path)


@pytest.mark.parametrize("seed", range(4))
def test_dense_pileup_matches_reference_with_small_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    # Tiny buffers send every read through the flush and the single-read path.
    monkeypatch.setattr(pileup_stats, "_MAX_BATCH", 2)
    monkeypatch.setattr(pileup_stats, "_MAX_PLAN_BYTES", 1)
    rng = random.Random(seed)
    refs = _random_references(rng)
    bam = tmp_path / "reads.bam"
    _write_bam(bam, refs, rng, min_q=20, grouped=False)
    fasta = _write_fasta(tmp_path / "ref.fa", refs, rng)
    bed = _write_bed(tmp_path / "targets.bed", refs, rng)
    _assert_matches_reference(bam, fasta, bed, 20, tmp_path)


@pytest.mark.parametrize("batch", [3, 17])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("use_bed", [False, True])
def test_dense_pileup_matches_reference_on_deep_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int, use_bed: bool, batch: int
) -> None:
    # Hundreds of reads per position: an indel read applies at once, with earlier
    # reads still waiting in their small batches.
    monkeypatch.setattr(pileup_stats, "_MAX_BATCH", batch)
    calls = {"indel_without_flush": 0, "indel_with_flush": 0}
    original_pending_cannot_matter = pileup_dense._RefCounts._pending_cannot_matter

    def counting(self: pileup_dense._RefCounts, anchor: int, slot: int, is_forward: bool) -> bool:
        settled = original_pending_cannot_matter(self, anchor, slot, is_forward)
        calls["indel_without_flush" if settled else "indel_with_flush"] += 1
        return settled

    monkeypatch.setattr(pileup_dense._RefCounts, "_pending_cannot_matter", counting)
    rng = random.Random(500 + seed)
    refs = [
        _Reference(f"ref{i}", "".join(rng.choice(_BASES) for _ in range(60)), 60) for i in range(2)
    ]
    bam = tmp_path / "reads.bam"
    _write_bam(
        bam,
        refs,
        rng,
        min_q=20,
        grouped=bool(seed % 2),
        n_reads=(1500, 1500),
        error_rate=0.3,
        stop_rate=0.0,
        kinds=("plain", "supplementary"),
    )
    fasta = _write_fasta(tmp_path / "ref.fa", refs, rng)
    bed = _write_bed(tmp_path / "targets.bed", refs, rng) if use_bed else None
    _assert_matches_reference(bam, fasta, bed, 20, tmp_path)
    assert calls["indel_without_flush"] > 100


def _write_records(
    bam: Path, cs_tags: Sequence[str], cigars: Sequence[list[tuple[int, int]]]
) -> None:
    """Write plain forward reads of Q40 at POS 1 of one 30 bp reference."""
    header = _header([_Reference("ref0", "A" * 30, 30)])
    with pysam.AlignmentFile(str(bam), "wb", header=header) as writer:
        for index, (cs, cigar) in enumerate(zip(cs_tags, cigars, strict=True)):
            read_len = sum(length for op, length in cigar if op in (0, 1, 4, 7, 8))
            record = pysam.AlignedSegment(header)
            record.query_name = f"r{index}"
            record.reference_id = 0
            record.reference_start = 0
            record.cigartuples = cigar
            record.query_sequence = "A" * read_len
            record.query_qualities = pysam.qualitystring_to_array("I" * read_len)
            record.set_tag("cs", "cs:Z:" + cs)
            writer.write(record)


def test_pileup_indel_before_the_first_base(tmp_path: Path) -> None:
    # An insertion at POS 1 has no anchor base; the row keeps position 0.
    bam = tmp_path / "reads.bam"
    _write_records(bam, ["+ac:10", ":10"], [[(1, 2), (0, 10)], [(0, 10)]])
    dense = _assert_matches_reference(bam, None, None, 20, tmp_path)
    assert dense.filter(pl.col("alt") == "+AC")["pos"].to_list() == [0]


@pytest.mark.parametrize(
    ("cs_tags", "expected_ref"),
    [
        (["*tg", "*ag", "*tg"], "T"),  # the last passing substitution read in file order wins
        (["*ag", "*tg", "*ag"], "A"),
        (["*tg", ":1", "*ag"], "A"),
        (["*tg", ":1", ":1"], "T"),
    ],
)
def test_pileup_ref_base_of_conflicting_substitutions(
    tmp_path: Path, cs_tags: list[str], expected_ref: str
) -> None:
    # The cs tags disagree on the reference base; only a BAM of two reference builds has it.
    bam = tmp_path / "reads.bam"
    _write_records(bam, cs_tags, [[(0, 1)]] * len(cs_tags))
    dense = _assert_matches_reference(bam, None, None, 20, tmp_path)
    assert dense.filter(pl.col("pos") == 1)["ref"].unique().to_list() == [expected_ref]
