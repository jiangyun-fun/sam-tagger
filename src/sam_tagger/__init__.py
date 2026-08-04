"""sam-tagger — SAM/BAM tagger and amplicon QC stat generator.

A clean redesign of bam_tagger / py-bam-tagger, split into three focused
subcommands: ``annotate`` (add tags + FASTQ extraction + tag stats),
``error-stats`` (7-file cs-tag error analysis), and ``pileup`` (bcftools
mpileup-like per-position statistics).
"""

__version__ = "3.0.0"
