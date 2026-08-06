"""Configuration loading for sam-tagger.

Single source of truth: an optional **CSV** file validated by pydantic. The 13
CLI override flags and the "generic" CSV columns of bam_tagger are intentionally
gone — there is exactly one way to supply each parameter.

``seq_name``/``seq_len`` come from the BAM header, so the config only carries
per-reference primer / sub-read / NB parameters the header cannot provide.

The CSV is **wide** — one row per reference, a ``seq_name`` column plus any
subset of the :class:`RefConfig` field columns. Empty cells mean "unset" (the
field's model default applies). Example ``config.csv``::

    seq_name,tc_primer_start_len,tc_primer_end_len,amp_primer_start_seq,amp_primer_end_seq,error_primer_start_len,error_primer_end_len,subread_start,subread_end,next_n
    seq1,8,8,GGGGGGGGG,CCCCCCC,5,5,12,21,6
    150-1-P,,,,,,,,10,10,,
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger
from pydantic import BaseModel, ConfigDict

DEFAULT_MIN_MATCH_LEN: int = 7

# Columns the config CSV may carry besides the seq_name key. Any other column
# name is rejected (the extra="forbid" equivalent — catches typos).
_SEQ_NAME_COL: str = "seq_name"
_REF_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "tc_primer_start_len",
        "tc_primer_end_len",
        "amp_primer_start_len",
        "amp_primer_end_len",
        "amp_primer_start_seq",
        "amp_primer_end_seq",
        "amp_primer_start_min_match",
        "amp_primer_end_min_match",
        "error_primer_start_len",
        "error_primer_end_len",
        "subread_start",
        "subread_end",
        "next_n",
    }
)


# --- Resolved, immutable parameter structs consumed by `processing` ---


@dataclass(frozen=True)
class TcParams:
    seq_len: int
    primer_start_len: int
    primer_end_len: int


@dataclass(frozen=True)
class ApParams:
    primer_start_end_pos: int
    primer_end_start_pos: int
    primer_start_seq: bytes
    primer_end_seq: bytes
    primer_start_min_match: int
    primer_end_min_match: int


@dataclass(frozen=True)
class ErrorParams:
    seq_len: int
    inner_region_start: int
    inner_region_end: int


@dataclass(frozen=True)
class UnresolvedSubReadParams:
    start: int
    end: int


@dataclass(frozen=True)
class AllConfigParams:
    """Resolved per-reference parameter maps.

    A map is ``None`` when no config file was provided, or empty-then-``None``
    when the config declared no entries for that feature.
    """

    tc: dict[str, TcParams] | None = None
    ap: dict[str, ApParams] | None = None
    error: dict[str, ErrorParams] | None = None
    sub_read: dict[str, UnresolvedSubReadParams] | None = None
    nb: dict[str, int] | None = None


# --- pydantic input model ---


class RefConfig(BaseModel):
    """Per-reference parameters as declared in the CSV file.

    Every field is optional; a reference only declares what its features need.
    ``extra="forbid"`` turns typos (e.g. ``tc_primer_star_len``) into hard errors
    instead of silently-ignored rows.
    """

    model_config = ConfigDict(extra="forbid")

    tc_primer_start_len: int | None = None
    tc_primer_end_len: int | None = None
    amp_primer_start_len: int | None = None
    amp_primer_end_len: int | None = None
    amp_primer_start_seq: str | None = None
    amp_primer_end_seq: str | None = None
    amp_primer_start_min_match: int = DEFAULT_MIN_MATCH_LEN
    amp_primer_end_min_match: int = DEFAULT_MIN_MATCH_LEN
    error_primer_start_len: int | None = None
    error_primer_end_len: int | None = None
    subread_start: int | None = None
    subread_end: int | None = None
    next_n: int | None = None


def parse_config(path: Path) -> dict[str, RefConfig]:
    """Parse + validate the CSV config into per-reference ``RefConfig`` objects.

    The CSV is wide: a ``seq_name`` column (must match a BAM ``@SQ SN``) plus any
    subset of ``RefConfig`` field columns. Each row is one reference. Empty cells
    mean "unset" (the field's model default applies — a cell is simply omitted
    from the ``RefConfig`` kwargs). ``seq_len`` is not in the CSV; it comes from
    the BAM header.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        On a missing ``seq_name`` column, an unknown column, or a
        duplicate/empty ``seq_name``.
    pydantic.ValidationError
        On a wrong-typed value (e.g. non-int in an int column).
    """
    if not path.exists():
        raise FileNotFoundError(f"Config CSV not found: {path}")
    df = pl.read_csv(path)

    columns = df.columns
    if _SEQ_NAME_COL not in columns:
        raise ValueError(f"Config CSV must have a {_SEQ_NAME_COL!r} column; got {columns!r}")
    unknown = [c for c in columns if c != _SEQ_NAME_COL and c not in _REF_CONFIG_FIELDS]
    if unknown:
        raise ValueError(
            f"Unknown config column(s) {unknown!r}; allowed: {sorted(_REF_CONFIG_FIELDS)}"
        )

    refs: dict[str, RefConfig] = {}
    for row in df.iter_rows(named=True):
        raw_name = row[_SEQ_NAME_COL]
        if raw_name is None or str(raw_name).strip() == "":
            raise ValueError(f"Empty {_SEQ_NAME_COL!r} in config row: {row!r}")
        name = str(raw_name)
        if name in refs:
            raise ValueError(f"Duplicate {_SEQ_NAME_COL!r} {name!r} in config CSV")
        fields = {k: v for k, v in row.items() if k != _SEQ_NAME_COL and v is not None}
        refs[name] = RefConfig(**fields)
    return refs


def build_params(
    refs: dict[str, RefConfig],
    header_ref_lengths: dict[str, int],
) -> AllConfigParams:
    """Join per-ref CSV config with BAM-header ``seq_len`` into resolved params.

    A reference present in the config but absent from the BAM header (no
    ``seq_len``) is skipped for every feature that needs ``seq_len`` — and now
    warned about, since at thousands of refs that mismatch is likely. Sub-read
    and NB do not require ``seq_len`` here (sub-read resolves against the
    per-read header length later; NB fetches against the FASTA).
    """
    missing = [name for name in refs if name not in header_ref_lengths]
    if missing:
        preview = ", ".join(missing[:5])
        logger.warning(
            f"{len(missing)} config reference(s) not in the BAM header "
            f"(dropped from seq_len-dependent features TC/AP/error); first: {preview}"
        )

    tc: dict[str, TcParams] = {}
    ap: dict[str, ApParams] = {}
    error: dict[str, ErrorParams] = {}
    sub_read: dict[str, UnresolvedSubReadParams] = {}
    nb: dict[str, int] = {}

    for name, ref in refs.items():
        seq_len = header_ref_lengths.get(name)

        # TC
        if (
            seq_len is not None
            and ref.tc_primer_start_len is not None
            and ref.tc_primer_end_len is not None
        ):
            tc[name] = TcParams(
                seq_len=seq_len,
                primer_start_len=ref.tc_primer_start_len,
                primer_end_len=ref.tc_primer_end_len,
            )

        # AP
        if (
            seq_len is not None
            and ref.amp_primer_start_len is not None
            and ref.amp_primer_end_len is not None
            and ref.amp_primer_start_seq is not None
            and ref.amp_primer_end_seq is not None
        ):
            ap[name] = ApParams(
                primer_start_end_pos=ref.amp_primer_start_len - 1,
                primer_end_start_pos=seq_len - ref.amp_primer_end_len,
                primer_start_seq=ref.amp_primer_start_seq.upper().encode(),
                primer_end_seq=ref.amp_primer_end_seq.upper().encode(),
                primer_start_min_match=ref.amp_primer_start_min_match,
                primer_end_min_match=ref.amp_primer_end_min_match,
            )

        # Error
        if (
            seq_len is not None
            and ref.error_primer_start_len is not None
            and ref.error_primer_end_len is not None
        ):
            error[name] = ErrorParams(
                seq_len=seq_len,
                inner_region_start=ref.error_primer_start_len,
                inner_region_end=seq_len - ref.error_primer_end_len,
            )

        # SubRead (stored unresolved; resolved per-read against header length)
        if ref.subread_start is not None and ref.subread_end is not None:
            sub_read[name] = UnresolvedSubReadParams(start=ref.subread_start, end=ref.subread_end)

        # NB
        if ref.next_n is not None:
            nb[name] = ref.next_n

    return AllConfigParams(
        tc=tc or None,
        ap=ap or None,
        error=error or None,
        sub_read=sub_read or None,
        nb=nb or None,
    )


def load_config(
    config_path: Path | None,
    header_ref_lengths: dict[str, int],
) -> AllConfigParams:
    """Load config from CSV (if given) and resolve against BAM-header lengths.

    Returns an empty ``AllConfigParams`` (all maps ``None``) when no config file
    is provided. Callers are responsible for validating that the maps they need
    are present (fail fast with a clear message).
    """
    if config_path is None:
        return AllConfigParams()
    refs = parse_config(config_path)
    return build_params(refs, header_ref_lengths)
