"""Configuration loading for sam-tagger.

Single source of truth: an optional **TOML** file validated by pydantic. The 13
CLI override flags and the "generic" CSV columns of bam_tagger are intentionally
gone — there is exactly one way to supply each parameter.

``seq_name``/``seq_len`` come from the BAM header, so the config only carries
per-reference primer / sub-read / NB parameters the header cannot provide.

Example ``config.toml``::

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
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_MIN_MATCH_LEN: int = 7


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
    """Per-reference parameters as declared in the TOML file.

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
    """Parse + validate the TOML config into per-reference ``RefConfig`` objects.

    The TOML is a flat map of ``[ref-name]`` tables. Ref names containing
    characters outside ``[A-Za-z0-9_-]`` must be quoted (``["150-1-P"]``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    tomllib.TOMLDecodeError
        On malformed TOML.
    pydantic.ValidationError
        On unknown keys or wrong types.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return {str(name): RefConfig(**fields) for name, fields in data.items()}


def build_params(
    refs: dict[str, RefConfig],
    header_ref_lengths: dict[str, int],
) -> AllConfigParams:
    """Join per-ref TOML config with BAM-header ``seq_len`` into resolved params.

    A reference present in the config but absent from the BAM header (no
    ``seq_len``) is skipped for every feature that needs ``seq_len``. Sub-read
    and NB do not require ``seq_len`` here (sub-read resolves against the
    per-read header length later; NB fetches against the FASTA).
    """
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
    """Load config from TOML (if given) and resolve against BAM-header lengths.

    Returns an empty ``AllConfigParams`` (all maps ``None``) when no config file
    is provided. Callers are responsible for validating that the maps they need
    are present (fail fast with a clear message).
    """
    if config_path is None:
        return AllConfigParams()
    refs = parse_config(config_path)
    return build_params(refs, header_ref_lengths)
