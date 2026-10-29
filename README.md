# sam-tagger

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![release](../../actions/workflows/release.yml/badge.svg)](../../actions/workflows/release.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

A SAM/BAM tagger and **amplicon-sequencing QC** statistics generator. Four
focused subcommands, each with a fixed I/O contract (one well-defined output), so
they're equally usable standalone or as steps in a larger pipeline.

📖 For the full flag reference and copy-paste examples, see
[`docs/usage.qmd`](docs/usage.qmd).

## Subcommands

| Command       | Reads                          | Writes                                              |
|---------------|--------------------------------|-----------------------------------------------------|
| `annotate`    | BAM (+ optional FASTA, config) | BAM, optional `.fastq.gz`, optional tag `.parquet`  |
| `error-stats` | BAM + FASTA (+ config)         | 8 error `.parquet` files                            |
| `pileup`      | BAM (+ optional FASTA, BED)    | 1 pileup `.parquet`                                 |
| `qc-summary`  | BAM (+ optional config)        | 1 per-reference QC `.parquet`                       |

`qc-summary` emits one row per reference joining FL, TC, AP and per-base error
counts/rate into a single table — a one-pass digest of the per-feature stats.
With no `--config`, only `total_reads`/`fl`/`fl_rate` are computed.

`annotate` keeps the tags: `FL MS ME TC AP SS ES DM EE IL DL NB`.
The redundant `HI`/`HD` (`=IL>0`/`=DL>0`) and `SP` (`=SS+1`) tags were removed.

## Output formats

`annotate` writes **BAM** by default. Pass `--output-format cram|sam` to write
**CRAM** or **SAM** instead (CRAM requires `--fasta` to encode against). Input
auto-detects BAM/CRAM/SAM. If reads lack a `cs` tag that a cs-dependent feature
(TC/EE/AP/error/pileup) needs, a warning fires early in the run.

## Configuration

Per-reference primer/subread/NB parameters live in a **wide CSV** file (one row
per reference, validated by pydantic). `seq_name` must match a `@SQ SN` in the
BAM header; `seq_len` comes from the header, so the config only carries what the
header cannot. Empty cells mean "unset"; unknown columns and duplicate/empty
`seq_name` are hard errors. See [`config.example.csv`](config.example.csv).

```csv
seq_name,tc_primer_start_len,tc_primer_end_len,amp_primer_start_seq,amp_primer_end_seq,error_primer_start_len,error_primer_end_len,subread_start,subread_end,next_n
seq1,8,8,GGGGGGGGG,CCCCCCC,5,5,12,21,6
```

`FL` and the CIGAR-only tags need **no config**. Primer-dependent features
(`TC`, `AP`, `NB`, sub-read extraction, `error-stats`) require a config.

## Pipeline placement

`sam-tagger` reads the `cs` (alignment comparison) tag, which `minimap2`
produces with `--cs`. In an amplicon pipeline it runs right after alignment
(and after any cs-tagging step your pipeline may use):

```
minimap2 (--cs) → sam-tagger
```

## Python API

Every subcommand is also callable in-process — the orchestrators, args
dataclasses, collectors, and `load_config` are re-exported from the top-level
package (`__all__`).

```python
from pathlib import Path
from sam_tagger import AnnotateArgs, run_annotate

run_annotate(
    AnnotateArgs(
        input_bam=Path("in.bam"),
        output_bam=Path("out.bam"),
        add_fl="FL",
        config=Path("config.csv"),
    )
)
```

## Development

```bash
pixi run -e dev test       # pytest
pixi run -e dev lint       # ruff
pixi run -e dev typecheck  # pyright
```

`pytest`, `ruff`, and `pyright` live in the `dev` environment, so dev tasks take
`-e dev`.

> `pysam` is provided via conda (reliable htslib build), not PyPI.
