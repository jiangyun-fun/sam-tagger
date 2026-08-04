# sam-tagger

A SAM/BAM tagger and amplicon-sequencing QC statistics generator. Clean redesign
of `bam_tagger` / `py-bam-tagger`, split into four focused subcommands with
fixed I/O contracts (so each maps cleanly onto a `snakeforge-templates` rule).

## Subcommands

| Command       | Reads                          | Writes                                              |
|---------------|--------------------------------|-----------------------------------------------------|
| `annotate`    | BAM (+ optional FASTA, config) | BAM, optional `.fastq.gz`, optional tag `.parquet`  |
| `error-stats` | BAM (+ config)                 | 7 error `.parquet` files                            |
| `pileup`      | BAM (+ optional FASTA, BED)    | 1 pileup `.parquet`                                 |
| `qc-summary`  | BAM (+ optional config)        | 1 per-reference QC `.parquet`                       |

`qc-summary` emits one row per reference joining FL, TC, AP and per-base error
counts/rate into a single table — a one-pass digest of the per-feature stats.
With no `--config`, only `total_reads`/`fl`/`fl_pct` are computed.

`annotate` keeps the tags: `FL MS ME TC AP SS ES DM EE IL DL NB`.
The redundant `HI`/`HD` (`=IL>0`/`=DL>0`) and `SP` (`=SS+1`) tags were removed.

## Configuration

Per-reference primer/subread/NB parameters live in a **TOML** file (validated by
pydantic). `seq_name`/`seq_len` come from the BAM header, so the config only
carries what the header cannot. See [`config.example.toml`](config.example.toml).

```toml
[seq1]
tc_primer_start_len  = 8
tc_primer_end_len    = 8
amp_primer_start_seq = "GGGGGGGGG"
amp_primer_end_seq   = "CCCCCCC"
error_primer_start_len = 5
error_primer_end_len   = 5
subread_start = 12
subread_end   = 21
next_n        = 6
```

`FL` and the CIGAR-only tags need **no config**. Primer-dependent features
(`TC`, `AP`, `NB`, sub-read extraction, `error-stats`) require a config.

## Pipeline placement

`sam-tagger` reads the `cs` tag, so it runs **after** `cs-tagger`:

```
minimap2 → cs-tagger → sam-tagger
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
