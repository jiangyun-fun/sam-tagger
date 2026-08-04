#!/usr/bin/env python
"""Performance: sam-tagger (Python) vs bam_tagger (Rust 2.0.6).

Generates a large amplicon BAM (~120k reads, minimap2 --cs aligned), then times
two fair workloads:

  A. annotate-only  — tags + extraction + tag-stats. ONE pass in each tool
     (apples-to-apples on the shared hot loop).
  B. full pipeline  — tags + error-stats + pileup. bam_tagger does it in ONE
     pass; sam-tagger runs 3 subcommands (3 BAM reads). This is where the
     subcommand split shows its cost.

Reports wall time (min of 3 reps), throughput, Rust:Python ratio, and peak RSS.
Both tools single-threaded, default (WARN) verbosity.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pysam

ROOT = Path("/tmp/st-bench")
BAM_TAGGER = shutil.which("bam_tagger") or "/bioproj/tools/share/.pixi/bin/bam_tagger"
SAM_TAGGER = shutil.which("sam-tagger")
MINIMAP2 = shutil.which("minimap2")
SAMTOOLS = shutil.which("samtools")
GNU_TIME = "/usr/bin/time"

RNG = random.Random(99)
READS_PER_AMP = 30000  # ~120k reads total
# (name, length, tc_s/e, amp_s/e, amp_min, err_s/e, sub_s/e, next_n)
AMPS = [
    ("amp1", 250, 20, 20, 20, 20, 7, 15, 15, 40, 210, 12),
    ("amp2", 380, 25, 25, 25, 25, 8, 20, 20, 60, 320, 10),
    ("amp3", 500, 22, 22, 22, 22, 7, 18, 18, None, None, None),
    ("amp4", 150, 18, 18, 18, 18, 7, 10, 10, None, None, 8),
]


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def gen_inputs() -> tuple[Path, Path, Path]:
    """Return (filtered_bam, csv, toml) for ~READS_PER_AMP*4 reads."""
    bases = "ACGT"
    # reference
    ref_lines = []
    refs: dict[str, str] = {}
    for (name, ln, *_rest) in AMPS:
        seq = "".join(RNG.choice(bases) for _ in range(ln))
        refs[name] = seq
        ref_lines += [f">{name}", seq]
    fa = ROOT / "ref.fa"
    fa.write_text("\n".join(ref_lines) + "\n")

    # reads
    fq = ROOT / "reads.fq"
    n = 0
    with open(fq, "w") as fh:
        for (name, ln, _t1, _t2, _a1, _a2, _am, *_rest) in AMPS:
            seq = refs[name]
            for _ in range(READS_PER_AMP):
                lo = RNG.randint(0, max(0, ln - 41))
                hi = RNG.randint(lo + 40, ln)
                read = seq[lo:hi]
                # light mutation so cs tags carry real diffs
                r = RNG.random()
                if r < 0.01:
                    pos = RNG.randint(0, len(read) - 1)
                    read = read[:pos] + RNG.choice([b for b in bases if b != read[pos]]) + read[pos + 1:]
                if RNG.random() < 0.5:
                    read = revcomp(read)
                quals = "I" * len(read)
                fh.write(f"@{name}__r{n}\n{read}\n+\n{quals}\n")
                n += 1
    print(f"generated {n} reads")

    # align with minimap2 --cs -> sorted indexed BAM
    sam = ROOT / "in.sam"
    with open(sam, "w") as sfh:
        subprocess.run([MINIMAP2, "-ax", "sr", "--cs", "-t", "16", str(fa), str(fq)],
                       stdout=sfh, check=True, stderr=subprocess.DEVNULL)
    bam_full = ROOT / "in.bam"
    subprocess.run([SAMTOOLS, "sort", "-o", str(bam_full), str(sam)], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([SAMTOOLS, "index", str(bam_full)], check=True, stderr=subprocess.DEVNULL)
    subprocess.run([SAMTOOLS, "faidx", str(fa)], check=True, stderr=subprocess.DEVNULL)

    # filter to bam_tagger-compatible (cs starts with ':')
    bam = ROOT / "in_cmp.bam"
    kept = 0
    with pysam.AlignmentFile(str(bam_full), "rb") as r, \
         pysam.AlignmentFile(str(bam), "wb", header=r.header) as w:
        for rec in r.fetch(until_eof=True):
            cs = rec.get_tag("cs") if rec.has_tag("cs") else None
            if cs is not None and not str(cs).startswith(":"):
                continue
            w.write(rec)
            kept += 1
    subprocess.run([SAMTOOLS, "index", str(bam)], check=True, stderr=subprocess.DEVNULL)
    print(f"Rust-compatible BAM: {kept} records")

    # configs
    csv_cols = ["seq_name", "seq_len", "tc_tag_primer_start_len", "tc_tag_primer_end_len",
                "amplification_primer_start_len", "amplification_primer_end_len",
                "amplification_primer_start_seq", "amplification_primer_end_seq",
                "amplification_primer_start_min_match", "amplification_primer_end_min_match",
                "error_primer_start_len", "error_primer_end_len", "subread_start", "subread_end", "next_n_cnt"]
    csv_rows = [",".join(csv_cols)]
    toml_lines: list[str] = []
    for (name, ln, tc_s, tc_e, amp_s, amp_e, amp_min, err_s, err_e, sub_s, sub_e, next_n) in AMPS:
        seq = refs[name]
        s_seq, e_seq = seq[:amp_s].upper(), seq[-amp_e:].upper()
        row = {"seq_name": name, "seq_len": ln, "tc_tag_primer_start_len": tc_s, "tc_tag_primer_end_len": tc_e,
               "amplification_primer_start_len": amp_s, "amplification_primer_end_len": amp_e,
               "amplification_primer_start_seq": s_seq, "amplification_primer_end_seq": e_seq,
               "amplification_primer_start_min_match": amp_min, "amplification_primer_end_min_match": amp_min,
               "error_primer_start_len": err_s, "error_primer_end_len": err_e,
               "subread_start": sub_s if sub_s is not None else "", "subread_end": sub_e if sub_e is not None else "",
               "next_n_cnt": next_n if next_n is not None else ""}
        csv_rows.append(",".join(str(row[c]) for c in csv_cols))
        toml_lines.append(f"[{name}]\ntc_primer_start_len = {tc_s}\ntc_primer_end_len = {tc_e}\n"
                          f"amp_primer_start_len = {amp_s}\namp_primer_end_len = {amp_e}\n"
                          f'amp_primer_start_seq = "{s_seq}"\namp_primer_end_seq = "{e_seq}"\n'
                          f"amp_primer_start_min_match = {amp_min}\namp_primer_end_min_match = {amp_min}\n"
                          f"error_primer_start_len = {err_s}\nerror_primer_end_len = {err_e}")
        if sub_s is not None:
            toml_lines[-1] += f"\nsubread_start = {sub_s}\nsubread_end = {sub_e}"
        if next_n is not None:
            toml_lines[-1] += f"\nnext_n = {next_n}"
        toml_lines.append("")
    csv = ROOT / "rust.csv"; csv.write_text("\n".join(csv_rows) + "\n")
    toml = ROOT / "py.toml"; toml.write_text("\n".join(toml_lines) + "\n")
    return bam, csv, toml, fa, kept


TAGS = (["--add-fl", "FL", "--add-ms", "MS", "--add-me", "ME", "--add-tc", "TC", "--add-ap", "AP",
         "--add-ss", "SS", "--add-es", "ES", "--add-dm", "DM", "--add-ee", "EE", "--add-il", "IL",
         "--add-dl", "DL", "--add-nb", "NB"])


def time_cmd(cmd: list[str], reps: int = 3) -> tuple[float, int]:
    """Return (min_wall_s, peak_rss_kb)."""
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()
        if r.returncode != 0:
            print("CMD FAILED:", " ".join(cmd[:6]))
            print(r.stderr[-2000:])
            sys.exit(1)
        times.append(t1 - t0)
    # peak RSS via GNU time on one more run
    rv = subprocess.run([GNU_TIME, "-v", *cmd], capture_output=True, text=True)
    rss = 0
    for line in rv.stderr.splitlines():
        if "Maximum resident set size" in line:
            rss = int(line.split(":")[-1].strip())
            break
    return min(times), rss


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    bam, csv, toml, fa, n_reads = gen_inputs()
    rust_dir = ROOT / "rust"; rust_dir.mkdir()
    py_dir = ROOT / "py"; py_dir.mkdir()

    print(f"\ninput: {n_reads} records\n")

    # ---- Workload A: annotate-only (single pass each) ----
    print("timing workload A (annotate-only: 12 tags + 3 FASTQ + 4 tag-stats)...")
    rust_a = ([BAM_TAGGER, "-i", str(bam), "-o", str(rust_dir / "a.bam"), "--config", str(csv), "--fasta", str(fa),
               *TAGS, "--fl-stats", str(rust_dir / "fl.parquet"), "--tc-stats", str(rust_dir / "tc.parquet"),
               "--ap-stats", str(rust_dir / "ap.parquet"), "--nb-stats", str(rust_dir / "nb.parquet"),
               "--extract-sub-reads-fastq", str(rust_dir / "sub.fq.gz"),
               "--extract-left-softclip-fastq", str(rust_dir / "left.fq.gz"),
               "--extract-right-softclip-fastq", str(rust_dir / "right.fq.gz")])
    py_a = ([SAM_TAGGER, "annotate", "-i", str(bam), "-o", str(py_dir / "a.bam"), "--config", str(toml), "--fasta", str(fa),
             *TAGS, "--fl-stats", str(py_dir / "fl.parquet"), "--tc-stats", str(py_dir / "tc.parquet"),
             "--ap-stats", str(py_dir / "ap.parquet"), "--nb-stats", str(py_dir / "nb.parquet"),
             "--extract-sub-reads-fastq", str(py_dir / "sub.fq.gz"),
             "--extract-left-softclip-fastq", str(py_dir / "left.fq.gz"),
             "--extract-right-softclip-fastq", str(py_dir / "right.fq.gz")])
    rt_a, rm_a = time_cmd(rust_a)
    pt_a, pm_a = time_cmd(py_a)

    # ---- Workload B: full pipeline (bam_tagger 1 pass; sam-tagger 3 subcommands) ----
    print("timing workload B (full: tags + error-stats + pileup)...")
    rust_b = (rust_a + ["--pileup-stats", str(rust_dir / "pileup.parquet"), "--min-base-qual", "20",
                        "--stats-errors", "--stats-errors-prefix", str(rust_dir / "err")])
    rt_b, rm_b = time_cmd(rust_b)

    # sam-tagger full = annotate (already timed as A) + error-stats + pileup
    py_err = [SAM_TAGGER, "error-stats", "-i", str(bam), "--config", str(toml), "--prefix", str(py_dir / "err")]
    py_pile = [SAM_TAGGER, "pileup", "-i", str(bam), "-o", str(py_dir / "pileup.parquet"), "--fasta", str(fa), "--min-base-qual", "20"]
    pt_err, pm_err = time_cmd(py_err)
    pt_pile, pm_pile = time_cmd(py_pile)
    pt_b = pt_a + pt_err + pt_pile
    pm_b = max(pm_a, pm_err, pm_pile)

    # ---- report ----
    def kreads(n: float) -> float:
        return n_reads / n / 1000.0

    print("\n" + "=" * 78)
    print(f"{'workload':<34}{'bam_tagger':>14}{'sam-tagger':>14}{'py/rust':>9}")
    print("-" * 78)
    print(f"{'A. annotate (1 pass)':<34}{rt_a:>11.2f} s{pt_a:>11.2f} s{pt_a/rt_a:>9.1f}x")
    print(f"{'   throughput (kreads/s)':<34}{kreads(rt_a):>11.1f}  {kreads(pt_a):>11.1f} ")
    print(f"{'B. full pipeline (tags+err+pile)':<34}{rt_b:>11.2f} s{pt_b:>11.2f} s{pt_b/rt_b:>9.1f}x")
    print(f"{'   throughput (kreads/s)':<34}{kreads(rt_b):>11.1f}  {kreads(pt_b):>11.1f} ")
    print("-" * 78)
    print("sam-tagger per-subcommand (workload B):")
    print(f"   annotate     {pt_a:>8.2f} s   (1 BAM pass, writes BAM + FASTQ + tag-stats)")
    print(f"   error-stats  {pt_err:>8.2f} s   (1 BAM pass, 7 Parquet)")
    print(f"   pileup       {pt_pile:>8.2f} s   (1 BAM pass, 1 Parquet)")
    print(f"   total        {pt_b:>8.2f} s   (3 BAM passes)")
    print("-" * 78)
    print("peak RSS (MB):")
    print(f"   workload A   rust={rm_a/1024:.0f}   py={pm_a/1024:.0f}")
    print(f"   workload B   rust={rm_b/1024:.0f}   py={pm_b/1024:.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
