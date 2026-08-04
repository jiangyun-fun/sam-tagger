#!/usr/bin/env python
"""Real-data performance: sam-tagger vs bam_tagger on a user-provided BAM.

Usage: python scripts/bench_real.py <input.bam> <ref.fa>

Compares the two config-independent hot paths on the FULL file:
  A. annotate  — core tags (FL/MS/ME/SS/ES/DM/EE/IL/DL), no config needed.
  B. pileup    — per-position stats (needs cs + fasta), no config needed.

bam_tagger always writes an output BAM (-o is required), so its pileup run also
copies the BAM; sam-tagger pileup is read-only. That asymmetry (noted in
output) slightly favors sam-tagger on pileup. Both single-threaded, default
verbosity, min of --reps runs, outputs to tmpfs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pysam

BAM_TAGGER = shutil.which("bam_tagger") or "/bioproj/tools/share/.pixi/bin/bam_tagger"
SAM_TAGGER = shutil.which("sam-tagger")
GNU_TIME = "/usr/bin/time"
OUT = Path("/dev/shm/st-realbench")

CORE_TAGS = (["--add-fl", "FL", "--add-ms", "MS", "--add-me", "ME", "--add-ss", "SS",
              "--add-es", "ES", "--add-dm", "DM", "--add-ee", "EE", "--add-il", "IL", "--add-dl", "DL"])


def time_cmd(cmd: list[str], reps: int) -> tuple[float, int]:
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        times.append(time.perf_counter() - t0)
        if r.returncode != 0:
            print("CMD FAILED:", " ".join(cmd[:5]))
            print(r.stderr[-2000:])
            sys.exit(1)
    rv = subprocess.run([GNU_TIME, "-v", *cmd], capture_output=True, text=True)
    rss = next((int(line.split(":")[-1]) for line in rv.stderr.splitlines()
                if "Maximum resident set size" in line), 0)
    return min(times), rss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bam"); ap.add_argument("fa")
    ap.add_argument("-n", "--reps", type=int, default=2)
    args = ap.parse_args()
    bam = Path(args.bam); fa = Path(args.fa)
    if not bam.is_file() or not fa.is_file():
        sys.exit("missing input file(s)")
    if SAM_TAGGER is None:
        sys.exit("sam-tagger not on PATH (run via pixi dev env)")

    OUT.mkdir(exist_ok=True)
    n = sum(1 for _ in pysam.AlignmentFile(str(bam), "rb"))
    bam_mb = bam.stat().st_size / 1e6
    print(f"\ninput: {bam.name}  ({n:,} records, {bam_mb:.1f} MB), reps={args.reps}\n")

    # ---- A. annotate (core tags) ----
    print("timing A: annotate (core tags, single pass each)...")
    rust_o = OUT / "rust_ann.bam"; py_o = OUT / "py_ann.bam"
    rt_a, rm_a = time_cmd([BAM_TAGGER, "-i", str(bam), "-o", str(rust_o), *CORE_TAGS], args.reps)
    pt_a, pm_a = time_cmd([SAM_TAGGER, "annotate", "-i", str(bam), "-o", str(py_o), *CORE_TAGS], args.reps)

    # ---- B. pileup ----
    print("timing B: pileup (bam_tagger also writes BAM; sam-tagger read-only)...")
    rust_pile = OUT / "rust_pileup.parquet"; py_pile = OUT / "py_pileup.parquet"
    rt_b, rm_b = time_cmd(
        [BAM_TAGGER, "-i", str(bam), "-o", str(OUT / "rust_pile.bam"),
         "--pileup-stats", str(rust_pile), "--min-base-qual", "20", "--fasta", str(fa)], args.reps)
    pt_b, pm_b = time_cmd(
        [SAM_TAGGER, "pileup", "-i", str(bam), "-o", str(py_pile),
         "--fasta", str(fa), "--min-base-qual", "20"], args.reps)

    def kreads(n_: float) -> float:
        return n / n_ / 1000.0

    print("\n" + "=" * 80)
    print(f"{'workload':<40}{'bam_tagger':>13}{'sam-tagger':>13}{'py/rust':>9}")
    print("-" * 80)
    print(f"{'A. annotate (core tags, 1 pass)':<40}{rt_a:>10.1f} s{pt_a:>10.1f} s{pt_a/rt_a:>9.1f}x")
    print(f"{'   kreads/s':<40}{kreads(rt_a):>10.1f}  {kreads(pt_a):>10.1f} ")
    print(f"{'B. pileup':<40}{rt_b:>10.1f} s{pt_b:>10.1f} s{pt_b/rt_b:>9.1f}x")
    print(f"{'   kreads/s':<40}{kreads(rt_b):>10.1f}  {kreads(pt_b):>10.1f} ")
    print("-" * 80)
    print("peak RSS (MB):")
    print(f"   A  rust={rm_a/1024:.0f}   py={pm_a/1024:.0f}")
    print(f"   B  rust={rm_b/1024:.0f}   py={pm_b/1024:.0f}")
    print("=" * 80)
    print("note: bam_tagger B writes a full BAM copy (required -o); sam-tagger B is read-only.")
    print("note: TC/AP/NB/error-stats omitted — need per-oligo primer config (not provided).")


if __name__ == "__main__":
    main()
