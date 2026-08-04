#!/usr/bin/env python
"""Differential test: sam-tagger (Python) vs bam_tagger (Rust 2.0.6, ground truth).

Pipeline:
  1. Generate a multi-amplicon reference FASTA.
  2. Simulate reads (both strands, SNPs/indels, varied qualities, some unmapped).
  3. Align with minimap2 -ax sr --cs  -> BAM with REAL cs tags.
  4. Run bam_tagger (one mega-command) and sam-tagger (3 subcommands) with
     IDENTICAL per-reference params (CSV vs TOML).
  5. Compare every shared output surface: BAM tags (FL/MS/ME/TC/AP/SS/ES/DM/EE/
     IL/DL/NB), the 4 tag-stats Parquet, the 7 error-stats Parquet, the pileup
     Parquet, and the 3 extracted FASTQ.gz.

sam-tagger dropped HI/HD/SP and override flags and splits stats into separate
subcommands; the comparison is restricted to the shared surface, with identical
inputs. Anything that differs is a real regression.
"""

from __future__ import annotations

import gzip
import random
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pysam

ROOT = Path("/tmp/st-difftest")
BAM_TAGGER = shutil.which("bam_tagger") or "/bioproj/tools/share/.pixi/bin/bam_tagger"
SAM_TAGGER = shutil.which("sam-tagger")
MINIMAP2 = shutil.which("minimap2")
SAMTOOLS = shutil.which("samtools")

RNG = random.Random(20260804)
INT_TAGS = ("FL", "MS", "ME", "SS", "ES", "DM", "EE", "IL", "DL")
STR_TAGS = ("TC", "AP", "NB")
ALL_TAGS = INT_TAGS + STR_TAGS

# Per-amplicon params: (name, length, tc_s, tc_e, amp_s, amp_e, amp_min, err_s, err_e, sub_s, sub_e, next_n)
AMPS = [
    ("amp1", 250, 20, 20, 20, 20, 7, 15, 15, 40, 210, 12),
    ("amp2", 380, 25, 25, 25, 25, 8, 20, 20, 60, 320, 10),
    ("amp3", 500, 22, 22, 22, 22, 7, 18, 18, None, None, None),
    ("amp4", 150, 18, 18, 18, 18, 7, 10, 10, None, None, 8),
]


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


# ---------------------------------------------------------------------------
# 1. reference
# ---------------------------------------------------------------------------


def gen_reference() -> tuple[Path, dict[str, tuple]]:
    bases = "ACGT"
    refs: dict[str, tuple] = {}
    lines = []
    for (name, ln, *_rest) in AMPS:
        seq = "".join(RNG.choice(bases) for _ in range(ln))
        refs[name] = (seq, ln)
        lines.append(f">{name}")
        lines.append(seq)
    fa = ROOT / "ref.fa"
    fa.write_text("\n".join(lines) + "\n")
    return fa, refs


# ---------------------------------------------------------------------------
# 2. simulated reads
# ---------------------------------------------------------------------------


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def mutate(seq: str, rate: float) -> str:
    """Apply random SNP/insertion/deletion; minimap2 will recompute cs from this."""
    bases = "ACGT"
    out = []
    i = 0
    while i < len(seq):
        r = RNG.random()
        if r < rate * 0.5:  # SNP
            b = RNG.choice([x for x in bases if x != seq[i]])
            out.append(b)
            i += 1
        elif r < rate * 0.5 + rate * 0.25:  # insertion
            out.append(seq[i])
            out.append(RNG.choice(bases))
            i += 1
        elif r < rate * 0.5 + rate * 0.5:  # deletion
            i += 1  # skip base
        else:
            out.append(seq[i])
            i += 1
    return "".join(out)


def simulate_reads(refs: dict[str, tuple]) -> Path:
    fq = ROOT / "reads.fq"
    n_read = 0
    with open(fq, "w") as fh:
        for (name, ln, _tc_s, _tc_e, amp_s, _amp_e, _amp_min, *_rest) in AMPS:
            seq, _ = refs[name]
            for _ in range(220):
                # varied sub-region: some full-amplicon (primers at both ends),
                # some partial, some tiny -> exercises Both/StartOnly/EndOnly/Neither
                lo = RNG.randint(0, max(0, ln - 41))
                hi = RNG.randint(lo + 40, ln)
                sub = seq[lo:hi]
                read = mutate(sub, rate=RNG.choice([0.0, 0.005, 0.01, 0.03, 0.06]))
                if RNG.random() < 0.5:  # half reverse-strand
                    read = revcomp(read)
                # varied qualities incl. some low (Phred+33, printable '!'..'I')
                quals = "".join(chr(33 + RNG.randint(0, 40)) for _ in read) if RNG.random() < 0.7 \
                    else "I" * len(read)
                fh.write(f"@{name}__r{n_read}\n{read}\n+\n{quals}\n")
                n_read += 1
        # a few unmappable reads (poly-N)
        for k in range(20):
            fh.write(f"@unmapped__{k}\n{'N' * 150}\n+\n{'I' * 150}\n")
    print(f"simulated {n_read} mappable + 20 unmapped reads")
    return fq


# ---------------------------------------------------------------------------
# 3. align
# ---------------------------------------------------------------------------


def align(fa: Path, fq: Path) -> Path:
    bam = ROOT / "in.bam"
    sam_path = ROOT / "in.sam"
    with open(sam_path, "w") as sfh:
        subprocess.run(
            [MINIMAP2, "-ax", "sr", "--cs", "-t", "8", str(fa), str(fq)],
            stdout=sfh, check=True,
        )
    subprocess.run([SAMTOOLS, "sort", "-o", str(bam), str(sam_path)], check=True)
    subprocess.run([SAMTOOLS, "index", str(bam)], check=True)
    subprocess.run([SAMTOOLS, "faidx", str(fa)], check=True)
    n = 0
    n_cs = 0
    with pysam.AlignmentFile(str(bam), "rb") as r:
        for rec in r.fetch(until_eof=True):
            n += 1
            if rec.has_tag("cs"):
                n_cs += 1
    print(f"aligned: {n} records, {n_cs} carry cs tags")
    return bam


# ---------------------------------------------------------------------------
# 4. configs + run both tools
# ---------------------------------------------------------------------------


def filter_rust_compatible(bam_in: Path) -> Path:
    """Keep only records bam_tagger 2.0.6 can parse.

    bam_tagger rejects mapped reads whose ``cs`` tag does not start with a match
    op (``:``). sam-tagger accepts them (cs validation was removed in the port).
    Both tools are run on THIS filtered BAM so the comparison is apples-to-apples;
    sam-tagger is separately shown to process the full BAM (see ``robustness``).
    """
    out = ROOT / "in_cmp.bam"
    kept = dropped = 0
    with pysam.AlignmentFile(str(bam_in), "rb") as r, \
         pysam.AlignmentFile(str(out), "wb", header=r.header) as w:
        for rec in r.fetch(until_eof=True):
            cs = rec.get_tag("cs") if rec.has_tag("cs") else None
            if cs is not None and not str(cs).startswith(":"):
                dropped += 1
                continue
            w.write(rec)
            kept += 1
    subprocess.run([SAMTOOLS, "index", str(out)], check=True)
    print(f"Rust-compatible BAM: kept {kept}, dropped {dropped} (cs not starting with ':')")
    return out


def robustness_full_bam(bam: Path, fa: Path, toml: Path) -> None:
    """Prove sam-tagger processes the FULL BAM that bam_tagger crashes on."""
    banner("robustness: sam-tagger annotate on FULL bam (bam_tagger rejects this)")
    r = subprocess.run(
        [SAM_TAGGER, "annotate", "-i", str(bam), "-o", str(ROOT / "full_py.bam"),
         "--config", str(toml), "--fasta", str(fa),
         "--add-fl", "FL", "--add-tc", "TC", "--add-ee", "EE"],
        capture_output=True, text=True,
    )
    n = sum(1 for _ in pysam.AlignmentFile(str(ROOT / "full_py.bam"), "rb")) if r.returncode == 0 else 0
    print(f"  sam-tagger exit={r.returncode}, wrote {n} records  (bam_tagger exits 1 on this input)")


def write_configs(refs: dict[str, tuple]) -> tuple[Path, Path]:
    csv_cols = [
        "seq_name", "seq_len", "tc_tag_primer_start_len", "tc_tag_primer_end_len",
        "amplification_primer_start_len", "amplification_primer_end_len",
        "amplification_primer_start_seq", "amplification_primer_end_seq",
        "amplification_primer_start_min_match", "amplification_primer_end_min_match",
        "error_primer_start_len", "error_primer_end_len",
        "subread_start", "subread_end", "next_n_cnt",
    ]
    csv_rows = [",".join(csv_cols)]
    toml_lines: list[str] = []
    for (name, ln, tc_s, tc_e, amp_s, amp_e, amp_min, err_s, err_e, sub_s, sub_e, next_n) in AMPS:
        seq, _ = refs[name]
        start_seq = seq[:amp_s].upper()
        end_seq = seq[-amp_e:].upper() if amp_e else ""
        row = {
            "seq_name": name, "seq_len": ln,
            "tc_tag_primer_start_len": tc_s, "tc_tag_primer_end_len": tc_e,
            "amplification_primer_start_len": amp_s, "amplification_primer_end_len": amp_e,
            "amplification_primer_start_seq": start_seq, "amplification_primer_end_seq": end_seq,
            "amplification_primer_start_min_match": amp_min, "amplification_primer_end_min_match": amp_min,
            "error_primer_start_len": err_s, "error_primer_end_len": err_e,
            "subread_start": sub_s if sub_s is not None else "",
            "subread_end": sub_e if sub_e is not None else "",
            "next_n_cnt": next_n if next_n is not None else "",
        }
        csv_rows.append(",".join(str(row[c]) for c in csv_cols))
        # TOML equivalent
        toml_lines.append(f"[{name}]")
        toml_lines.append(f"tc_primer_start_len = {tc_s}")
        toml_lines.append(f"tc_primer_end_len = {tc_e}")
        toml_lines.append(f"amp_primer_start_len = {amp_s}")
        toml_lines.append(f"amp_primer_end_len = {amp_e}")
        toml_lines.append(f'amp_primer_start_seq = "{start_seq}"')
        toml_lines.append(f'amp_primer_end_seq = "{end_seq}"')
        toml_lines.append(f"amp_primer_start_min_match = {amp_min}")
        toml_lines.append(f"amp_primer_end_min_match = {amp_min}")
        toml_lines.append(f"error_primer_start_len = {err_s}")
        toml_lines.append(f"error_primer_end_len = {err_e}")
        if sub_s is not None:
            toml_lines.append(f"subread_start = {sub_s}")
            toml_lines.append(f"subread_end = {sub_e}")
        if next_n is not None:
            toml_lines.append(f"next_n = {next_n}")
        toml_lines.append("")
    csv = ROOT / "rust.csv"
    csv.write_text("\n".join(csv_rows) + "\n")
    toml = ROOT / "py.toml"
    toml.write_text("\n".join(toml_lines) + "\n")
    return csv, toml


def run_both(bam: Path, fa: Path, csv: Path, toml: Path) -> tuple[Path, Path]:
    rust = ROOT / "rust"; py = ROOT / "py"
    rust.mkdir(exist_ok=True); py.mkdir(exist_ok=True)
    rust_bam = rust / "out.bam"; py_bam = py / "out.bam"

    add_flags = ["--add-fl", "FL", "--add-ms", "MS", "--add-me", "ME", "--add-tc", "TC",
                 "--add-ap", "AP", "--add-ss", "SS", "--add-es", "ES", "--add-dm", "DM",
                 "--add-ee", "EE", "--add-il", "IL", "--add-dl", "DL", "--add-nb", "NB"]

    banner("running bam_tagger (Rust, one mega-command)")
    rust_cmd = [
        BAM_TAGGER, "-i", str(bam), "-o", str(rust_bam),
        "--config", str(csv), "--fasta", str(fa),
        *add_flags,
        "--fl-stats", str(rust / "fl.parquet"), "--tc-stats", str(rust / "tc.parquet"),
        "--ap-stats", str(rust / "ap.parquet"), "--nb-stats", str(rust / "nb.parquet"),
        "--pileup-stats", str(rust / "pileup.parquet"), "--min-base-qual", "20",
        "--stats-errors", "--stats-errors-prefix", str(rust / "err"),
        "--extract-sub-reads-fastq", str(rust / "sub.fastq.gz"),
        "--extract-left-softclip-fastq", str(rust / "left.fastq.gz"),
        "--extract-right-softclip-fastq", str(rust / "right.fastq.gz"),
    ]
    r = subprocess.run(rust_cmd, capture_output=True, text=True)
    print(f"  exit={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-2000:]); print("STDERR:", r.stderr[-3000:]); sys.exit(1)

    banner("running sam-tagger annotate")
    ann_cmd = [
        SAM_TAGGER, "annotate", "-i", str(bam), "-o", str(py_bam),
        "--config", str(toml), "--fasta", str(fa),
        *add_flags,
        "--fl-stats", str(py / "fl.parquet"), "--tc-stats", str(py / "tc.parquet"),
        "--ap-stats", str(py / "ap.parquet"), "--nb-stats", str(py / "nb.parquet"),
        "--extract-sub-reads-fastq", str(py / "sub.fastq.gz"),
        "--extract-left-softclip-fastq", str(py / "left.fastq.gz"),
        "--extract-right-softclip-fastq", str(py / "right.fastq.gz"),
    ]
    _run_st(ann_cmd)

    banner("running sam-tagger error-stats")
    _run_st([SAM_TAGGER, "error-stats", "-i", str(bam), "--config", str(toml), "--prefix", str(py / "err")])

    banner("running sam-tagger pileup")
    _run_st([SAM_TAGGER, "pileup", "-i", str(bam), "-o", str(py / "pileup.parquet"),
             "--fasta", str(fa), "--min-base-qual", "20"])

    return rust, py


def _run_st(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  {' '.join(cmd[:3])}… exit={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-2000:]); print("STDERR:", r.stderr[-3000:]); sys.exit(1)


# ---------------------------------------------------------------------------
# 5. compare
# ---------------------------------------------------------------------------


def read_bam_tags(bam: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    with pysam.AlignmentFile(str(bam), "rb") as r:
        for rec in r.fetch(until_eof=True):
            qname = rec.query_name or ""
            out[qname] = {t: rec.get_tag(t) for t in ALL_TAGS if rec.has_tag(t)}
    return out


def cmp_bam_tags(rust_bam: Path, py_bam: Path) -> int:
    banner("compare BAM tags (FL/MS/ME/TC/AP/SS/ES/DM/EE/IL/DL/NB)")
    a = read_bam_tags(rust_bam); b = read_bam_tags(py_bam)
    if set(a) != set(b):
        print(f"  READ-NAME MISMATCH: rust={len(a)} py={len(b)} "
              f"only-rust={len(set(a)-set(b))} only-py={len(set(b)-set(a))}")
    mism = 0
    for q in sorted(set(a) | set(b), key=lambda x: (len(x), x)):
        ta, tb = a.get(q, {}), b.get(q, {})
        keys = set(ta) | set(tb)
        diff = {k: (ta.get(k, "<absent>"), tb.get(k, "<absent>")) for k in keys if ta.get(k, "<absent>") != tb.get(k, "<absent>")}
        if diff:
            mism += 1
            if mism <= 10:
                print(f"  DIFF {q}: {diff}")
    print(f"  reads compared: {len(set(a)|set(b))}, reads with >=1 tag diff: {mism}")
    return mism


def cmp_parquet(rust_p: Path, py_p: Path, label: str, tol: float = 1e-9) -> int:
    if not rust_p.exists() or not py_p.exists():
        print(f"  [{label}] MISSING file: rust={rust_p.exists()} py={py_p.exists()}")
        return 1
    dr = pl.read_parquet(rust_p)
    dp = pl.read_parquet(py_p)
    if dr.shape != dp.shape:
        print(f"  [{label}] SHAPE mismatch rust={dr.shape} py={dp.shape}")
        return 1
    # sort both by all columns for order-independent compare
    cols = dr.columns
    dr = dr.sort(cols); dp = dp.sort(cols)
    n_diff = 0
    for col in cols:
        cr = dr[col]; cp = dp[col]
        if cr.dtype.is_float() or cp.dtype.is_float():
            close = (cr.cast(pl.Float64) - cp.cast(pl.Float64)).abs() < tol
            n_diff += int((~close).sum())
        else:
            n_diff += int((cr != cp).sum())
    if n_diff:
        print(f"  [{label}] {n_diff} cell diffs; sample:")
        print(dr.join(dp, on=cols, how="full", suffix="_py").head(5))
    else:
        print(f"  [{label}] identical ({dr.shape[0]} rows)")
    return n_diff


def cmp_fastq(rust_gz: Path, py_gz: Path, label: str) -> int:
    if not rust_gz.exists() or not py_gz.exists():
        print(f"  [{label}] MISSING: rust={rust_gz.exists()} py={py_gz.exists()}")
        return 1
    with gzip.open(rust_gz, "rt") as f:
        a = f.read()
    with gzip.open(py_gz, "rt") as f:
        b = f.read()
    if a == b:
        ra = len(a.splitlines()) // 4
        print(f"  [{label}] identical ({ra} records)")
        return 0
    la, lb = a.splitlines(), b.splitlines()
    print(f"  [{label}] DIFFER: rust_lines={len(la)} py_lines={len(lb)}")
    for i in range(min(len(la), len(lb))):
        if la[i] != lb[i]:
            print(f"    first diff at line {i}:\n      rust: {la[i][:120]}\n      py  : {lb[i][:120]}")
            break
    return 1


def compare(rust: Path, py: Path) -> None:
    total = 0
    total += cmp_bam_tags(rust / "out.bam", py / "out.bam")
    banner("compare tag-stats Parquet")
    for s in ("fl", "tc", "ap", "nb"):
        total += cmp_parquet(rust / f"{s}.parquet", py / f"{s}.parquet", f"{s}-stats")
    banner("compare error-stats Parquet (7 files)")
    for ext in ("error", "mismatch", "insertion", "deletion", "per_read", "per_base", "single_deletion"):
        total += cmp_parquet(rust / f"err.{ext}.parquet", py / f"err.{ext}.parquet", f"err-{ext}")
    banner("compare pileup Parquet")
    total += cmp_parquet(rust / "pileup.parquet", py / "pileup.parquet", "pileup")
    banner("compare extracted FASTQ.gz")
    total += cmp_fastq(rust / "sub.fastq.gz", py / "sub.fastq.gz", "subreads")
    total += cmp_fastq(rust / "left.fastq.gz", py / "left.fastq.gz", "left-softclip")
    total += cmp_fastq(rust / "right.fastq.gz", py / "right.fastq.gz", "right-softclip")

    banner("RESULT")
    print(f"  total diffs: {total}")
    print("  *** ALL OUTPUTS IDENTICAL ***" if total == 0 else "  *** DIFFERENCES FOUND ***")


def main() -> None:
    global RNG, ROOT
    if SHUTIL_MISSING := [n for n, v in [("bam_tagger", BAM_TAGGER), ("sam-tagger", SAM_TAGGER),
                                          ("minimap2", MINIMAP2), ("samtools", SAMTOOLS)] if not v]:
        print("missing tools:", SHUTIL_MISSING); sys.exit(1)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260804
    RNG = random.Random(seed)
    ROOT = Path(f"/tmp/st-difftest-{seed}")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    print(f"### SEED {seed} ###")
    fa, refs = gen_reference()
    fq = simulate_reads(refs)
    bam = align(fa, fq)
    csv, toml = write_configs(refs)
    in_cmp = filter_rust_compatible(bam)
    robustness_full_bam(bam, fa, toml)
    rust, py = run_both(in_cmp, fa, csv, toml)
    compare(rust, py)


if __name__ == "__main__":
    main()
