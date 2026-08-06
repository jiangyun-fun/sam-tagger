"""Unit tests for the pydantic CSV config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sam_tagger.config import (
    AllConfigParams,
    ApParams,
    ErrorParams,
    RefConfig,
    TcParams,
    UnresolvedSubReadParams,
    build_params,
    load_config,
    parse_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_parse_config_basic(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "c.csv",
        "seq_name,tc_primer_start_len,tc_primer_end_len,amp_primer_start_seq,amp_primer_end_seq,next_n\n"
        "seq1,8,8,GGGGGGGGG,CCCCCCC,6\n",
    )
    refs = parse_config(cfg)
    assert set(refs) == {"seq1"}
    assert refs["seq1"].tc_primer_start_len == 8
    assert refs["seq1"].next_n == 6
    # omitted min_match column -> model default applies
    assert refs["seq1"].amp_primer_start_min_match == 7


def test_parse_config_empty_cell_is_unset(tmp_path: Path) -> None:
    # A row with only seq_name populated; every other cell empty.
    cfg = _write(tmp_path / "c.csv", "seq_name,next_n,subread_start\n150-1-P,,\n")
    refs = parse_config(cfg)
    assert set(refs) == {"150-1-P"}
    assert refs["150-1-P"].next_n is None
    assert refs["150-1-P"].subread_start is None


def test_parse_config_rejects_unknown_column(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "seq_name,tc_primer_star_len\nx,8\n")  # typo
    with pytest.raises(ValueError, match="Unknown config column"):
        parse_config(cfg)


def test_parse_config_rejects_missing_seq_name_column(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "ref,next_n\nx,6\n")
    with pytest.raises(ValueError, match="seq_name"):
        parse_config(cfg)


def test_parse_config_rejects_duplicate_seq_name(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "seq_name,next_n\nx,1\nx,2\n")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_config(cfg)


def test_parse_config_rejects_empty_seq_name(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "seq_name,next_n\n,6\n")
    with pytest.raises(ValueError, match="Empty"):
        parse_config(cfg)


def test_parse_config_rejects_bad_type(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "seq_name,tc_primer_start_len\nx,abc\n")
    with pytest.raises(ValidationError):
        parse_config(cfg)


def test_parse_config_header_only_is_empty(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.csv", "seq_name,next_n\n")
    assert parse_config(cfg) == {}


def test_parse_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_config(tmp_path / "nope.csv")


def test_build_params_full(tmp_path: Path) -> None:
    refs = {
        "seq1": RefConfig(
            tc_primer_start_len=8,
            tc_primer_end_len=8,
            amp_primer_start_len=9,
            amp_primer_end_len=7,
            amp_primer_start_seq="ggggggggg",
            amp_primer_end_seq="ccccccc",
            error_primer_start_len=5,
            error_primer_end_len=5,
            subread_start=12,
            subread_end=21,
            next_n=6,
        )
    }
    params = build_params(refs, {"seq1": 100})
    assert params.tc is not None and params.tc["seq1"] == TcParams(100, 8, 8)

    assert params.ap is not None
    ap = params.ap["seq1"]
    assert isinstance(ap, ApParams)
    assert ap.primer_start_end_pos == 8  # 9 - 1
    assert ap.primer_end_start_pos == 93  # 100 - 7
    assert ap.primer_start_seq == b"GGGGGGGGG"  # uppercased + encoded
    assert ap.primer_end_seq == b"CCCCCCC"

    assert params.error is not None
    assert params.error["seq1"] == ErrorParams(100, 5, 95)  # inner = 100-5

    assert params.sub_read is not None
    assert params.sub_read["seq1"] == UnresolvedSubReadParams(12, 21)

    assert params.nb is not None
    assert params.nb["seq1"] == 6


def test_build_params_skips_ref_missing_from_header() -> None:
    # ref declared in config but not in BAM header -> seq_len-dependent features skipped
    refs = {"seq1": RefConfig(tc_primer_start_len=8, tc_primer_end_len=8, next_n=6)}
    params = build_params(refs, header_ref_lengths={})  # no seq1 in header
    assert params.tc == {} or params.tc is None
    # NB does not need seq_len
    assert params.nb == {"seq1": 6}


def test_build_params_partial_features() -> None:
    refs = {"seq1": RefConfig(tc_primer_start_len=8, tc_primer_end_len=8)}
    params = build_params(refs, {"seq1": 50})
    assert params.tc is not None
    assert params.ap is None
    assert params.error is None
    assert params.sub_read is None
    assert params.nb is None


def test_load_config_none_returns_empty() -> None:
    params = load_config(None, {"seq1": 100})
    assert params == AllConfigParams()


def test_load_config_roundtrip(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "c.csv",
        "seq_name,error_primer_start_len,error_primer_end_len\n150-1-P,10,10\n",
    )
    params = load_config(cfg, {"150-1-P": 76})
    assert params.error is not None
    assert params.error["150-1-P"].seq_len == 76
    assert params.error["150-1-P"].inner_region_end == 66  # 76 - 10
