"""Shared test helpers for sam-tagger tests (BAM fixtures + output readers)."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pysam

# Common headers. seq_len lives in the BAM header now, so LN must match what a
# test expects (FL length / primer inner regions).
H_SEQ1_28 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:28"
H_SEQ1_30 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:30"
H_SEQ1_100 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:100"
H_SEQ2_26 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq2\tLN:26"
H_SEQ1_100_SEQ2_100 = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:seq1\tLN:100\n@SQ\tSN:seq2\tLN:100"


@dataclass
class Setup:
    tmp_dir: Path
    config_path: Path | None
    input_bam: Path
    output_bam: Path


def write_fasta(path: Path, refs: dict[str, str]) -> Path:
    lines: list[str] = []
    for name, seq in refs.items():
        lines += [f">{name}", seq]
    path.write_text("\n".join(lines) + "\n")
    return path


def setup_env(
    tmp_path: Path,
    config_toml: str | None,
    sam_header_sq: str,
    sam_records: str,
    *,
    fasta: dict[str, str] | None = None,
) -> Setup:
    """Write an optional TOML config, an optional FASTA (+faidx), and a BAM from SAM."""
    config_path: Path | None = None
    if config_toml is not None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(config_toml)

    if fasta is not None:
        write_fasta(tmp_path / "ref.fa", fasta)
        pysam.faidx(str(tmp_path / "ref.fa"))  # type: ignore[attr-defined]

    sam_path = tmp_path / "test.sam"
    input_bam = tmp_path / "test.bam"
    output_bam = tmp_path / "output.bam"
    sam_path.write_text(f"{sam_header_sq}\n{sam_records}")

    with pysam.AlignmentFile(str(sam_path), "r") as sam_reader, \
         pysam.AlignmentFile(str(input_bam), "wb", header=sam_reader.header) as bam_writer:
        for record in sam_reader:
            bam_writer.write(record)

    return Setup(tmp_dir=tmp_path, config_path=config_path, input_bam=input_bam, output_bam=output_bam)


def get_bam_tags_as_int(bam_path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as reader:
        for record in reader.fetch(until_eof=True):
            qname = record.query_name or ""
            result[qname] = {
                t: int(record.get_tag(t))
                for t in ("FL", "MS", "ME", "SS", "ES", "DM", "EE", "IL", "DL")
                if record.has_tag(t)
            }
    return result


def get_bam_tags_as_string(bam_path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as reader:
        for record in reader.fetch(until_eof=True):
            qname = record.query_name or ""
            result[qname] = {
                t: str(record.get_tag(t)) for t in ("TC", "AP", "NB") if record.has_tag(t)
            }
    return result


@dataclass
class FastqRecord:
    head: str
    seq: str
    qual: str


def read_gzipped_fastq(path: Path) -> list[FastqRecord]:
    records: list[FastqRecord] = []
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().split("\n")
    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            records.append(FastqRecord(head=lines[i], seq=lines[i + 1], qual=lines[i + 3]))
    return records


def read_simple_stats(path: Path) -> dict[str, int]:
    return {str(r["seq_name"]): int(r["count"]) for r in pl.read_parquet(path).iter_rows(named=True)}


def read_pileup(path: Path) -> list[tuple]:
    return [tuple(r) for r in pl.read_parquet(path).sort("seq_name", "pos").iter_rows()]
