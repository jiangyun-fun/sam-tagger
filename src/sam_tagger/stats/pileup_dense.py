"""Dense per-reference allele counts and cs-tag plans for the pileup."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from sam_tagger.bed import Interval, position_in_intervals
from sam_tagger.processing import BAM_CHARD_CLIP, BAM_CSOFT_CLIP

SCHEMA = {
    "seq_name": pl.Utf8,
    "pos": pl.UInt32,
    "ref": pl.Utf8,
    "alt": pl.Utf8,
    "dp": pl.UInt64,
    "adf": pl.UInt64,
    "adr": pl.UInt64,
    "ad": pl.UInt64,
    "af": pl.Float64,
}
_DOT = ord(".")
_INITIAL_SLOTS = 8
_CHAR_OF_BYTE = np.array([chr(b) for b in range(256)], dtype="<U1")
NO_INTERVALS = np.empty(0, dtype=np.intp)
NO_LAST = np.empty(0, dtype=np.int64)


@dataclass(slots=True)
class _MatchOp:
    """A `:N` run: its positions inside the BED intervals and their slots."""

    rp: int
    length: int
    rows: np.ndarray
    codes: np.ndarray
    ref_bytes: np.ndarray
    idx: np.ndarray | None
    ref_set: bool = False
    flat: np.ndarray | None = None
    flat_slots: int = 0

    def est_bytes(self) -> int:
        return self.rows.size * (33 if self.idx is not None else 25)


@dataclass(slots=True)
class _SnpOp:
    """A `*xy` op."""

    rp: int
    row: int
    ref_byte: int
    alt_slot: int


@dataclass(slots=True)
class _IndelOp:
    """A `+seq` or `-seq` op, anchored at the base before it."""

    rp: int
    anchor: int
    key: str
    ref_byte: int


_Op = _MatchOp | _SnpOp | _IndelOp


@dataclass(slots=True)
class _Plan:
    """The ops of one `(record_pos, cs tag)` pair on one reference."""

    ops: list[_Op]
    read_len: int
    has_snp: bool
    has_indel: bool
    est_bytes: int


@dataclass
class _IndelRows:
    pos: list[int] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    ref_code: list[int] = field(default_factory=list)
    dp: list[int] = field(default_factory=list)
    fwd: list[int] = field(default_factory=list)
    rev: list[int] = field(default_factory=list)
    tot: list[int] = field(default_factory=list)


def leading_soft_clip(cigartuples: Sequence[tuple[int, int]]) -> int:
    """Soft-clipped bases before the alignment; hard clips are not in QUAL."""
    lead = 0
    for op, length in cigartuples:
        if op == BAM_CSOFT_CLIP:
            lead += length
        elif op != BAM_CHARD_CLIP:
            break
    return lead


def aligned_qualities(
    qual: np.ndarray, cigartuples: Sequence[tuple[int, int]], is_reverse: bool
) -> np.ndarray:
    """QUAL in cs order: index 0 is the quality of the first aligned base."""
    # SAM stores QUAL reverse-complemented for 0x10 records.
    if is_reverse:
        qual = qual[::-1]
    return qual[leading_soft_clip(cigartuples) :]


class _RefCounts:
    """Allele counts of one reference: one array row per position, one column per base allele."""

    def __init__(
        self, name: str, seq: bytes, extent: int, intervals: list[Interval] | None
    ) -> None:
        self.name = name
        self.seq: np.ndarray = np.frombuffer(seq, dtype=np.uint8)
        self.intervals = intervals
        if intervals is None:
            self.starts = self.ends = NO_INTERVALS
        else:
            self.starts = np.fromiter((iv.start for iv in intervals), dtype=np.intp)
            self.ends = np.fromiter((iv.end for iv in intervals), dtype=np.intp)
        self.indels: dict[int, dict[str, list[int]]] = {}
        self.high_water = 0
        self.extent = max(extent, 1)
        self._slot_of = np.full(256, -1, dtype=np.intp)
        self.slot_bytes: list[int] = []
        shape = (self.extent, _INITIAL_SLOTS)
        self.fwd = np.zeros(shape, dtype=np.int32)
        self.rev = np.zeros(shape, dtype=np.int32)
        self.tot = np.zeros(shape, dtype=np.int32)
        self.exists = np.zeros(shape, dtype=bool)
        self.dp = np.zeros(self.extent, dtype=np.int32)
        self.ref_code = np.full(self.extent, -1, dtype=np.int16)
        # Sequence number of the last substitution read that set `ref_code`.
        self.snp_last = np.zeros(self.extent, dtype=np.int64)
        self._bind_flat()

    def slot(self, byte: int) -> int:
        slot = int(self._slot_of[byte])
        if slot < 0:
            slot = len(self.slot_bytes)
            self._slot_of[byte] = slot
            self.slot_bytes.append(byte)
            if slot == self.fwd.shape[1]:
                self._widen()
        return slot

    def _bind_flat(self) -> None:
        """Call after every reallocation of the count arrays."""
        self.fwd_flat = self.fwd.reshape(-1)
        self.rev_flat = self.rev.reshape(-1)
        self.tot_flat = self.tot.reshape(-1)
        self.exists_flat = self.exists.reshape(-1)

    def _widen(self) -> None:
        extra = (self.extent, self.fwd.shape[1])
        self.fwd = np.concatenate((self.fwd, np.zeros(extra, np.int32)), axis=1)
        self.rev = np.concatenate((self.rev, np.zeros(extra, np.int32)), axis=1)
        self.tot = np.concatenate((self.tot, np.zeros(extra, np.int32)), axis=1)
        self.exists = np.concatenate((self.exists, np.zeros(extra, bool)), axis=1)
        self._bind_flat()

    def _ensure(self, end: int) -> None:
        if end <= self.extent:
            return
        new_extent = max(end, self.extent * 2)
        extra = (new_extent - self.extent, self.fwd.shape[1])
        self.fwd = np.concatenate((self.fwd, np.zeros(extra, np.int32)))
        self.rev = np.concatenate((self.rev, np.zeros(extra, np.int32)))
        self.tot = np.concatenate((self.tot, np.zeros(extra, np.int32)))
        self.exists = np.concatenate((self.exists, np.zeros(extra, bool)))
        self.dp = np.concatenate((self.dp, np.zeros(extra[0], np.int32)))
        self.ref_code = np.concatenate((self.ref_code, np.full(extra[0], -1, np.int16)))
        self.snp_last = np.concatenate((self.snp_last, np.zeros(extra[0], np.int64)))
        self.extent = new_extent
        self._bind_flat()

    def ref_byte_at(self, pos: int) -> int:
        return int(self.seq[pos]) if 0 <= pos < self.seq.shape[0] else _DOT

    def ref_bytes_at(self, rows: np.ndarray) -> np.ndarray:
        if rows[0] >= 0 and rows[-1] < self.seq.shape[0]:
            return self.seq[rows]
        out = np.full(rows.shape, _DOT, dtype=np.uint8)
        inside = (rows >= 0) & (rows < self.seq.shape[0])
        if inside.any():
            out[inside] = self.seq[rows[inside]]
        return out

    def inside(self, positions: np.ndarray) -> np.ndarray:
        k = np.searchsorted(self.starts, positions, side="right") - 1
        return (k >= 0) & (positions < self.ends[np.maximum(k, 0)])

    def in_scope(self, pos: int) -> bool:
        return self.intervals is None or position_in_intervals(pos, self.intervals)

    def build_plan(self, cs_tag: str, record_pos: int, cs_regex: re.Pattern[str]) -> _Plan:
        ops: list[_Op] = []
        has_indel = False
        has_snp = False
        ref_idx = record_pos + 1
        read_pos = 0
        for caps in cs_regex.finditer(cs_tag.removeprefix("cs:Z:")):
            op_type, val = caps.group(1), caps.group(2)
            if op_type == ":":
                length = int(val)
                op: _Op | None = self._match_op(ref_idx - 1, read_pos, length)
                ref_idx += length
                read_pos += length
            elif op_type == "*":
                op = self._snp_op(ref_idx - 1, read_pos, val)
                has_snp |= op is not None
                ref_idx += 1
                read_pos += 1
            else:
                op = self._indel_op(ref_idx - 2, read_pos, op_type + val.upper())
                if op_type == "+":
                    read_pos += len(val)
                else:
                    ref_idx += len(val)
            if op is not None:
                ops.append(op)
                has_indel |= isinstance(op, _IndelOp)
        est_bytes = 64 * len(ops) + sum(op.est_bytes() for op in ops if isinstance(op, _MatchOp))
        return _Plan(ops, read_pos, has_snp, has_indel, est_bytes)

    def _match_op(self, r0: int, read_pos: int, length: int) -> _MatchOp | None:
        self._ensure(r0 + length)
        rows = np.arange(r0, r0 + length, dtype=np.intp)
        idx = None
        if self.intervals is not None:
            idx = np.nonzero(self.inside(rows))[0]
            rows = rows[idx]
        if rows.size == 0:
            return None
        ref_bytes = self.ref_bytes_at(rows)
        return _MatchOp(read_pos, length, rows, self._codes(ref_bytes), ref_bytes, idx)

    def _snp_op(self, row: int, read_pos: int, val: str) -> _SnpOp | None:
        if len(val) < 2 or not self.in_scope(row):
            return None
        self._ensure(row + 1)
        return _SnpOp(read_pos, row, ord(val[0].upper()), self.slot(ord(val[1].upper())))

    def _indel_op(self, anchor: int, read_pos: int, key: str) -> _IndelOp | None:
        if not self.in_scope(anchor):
            return None
        return _IndelOp(read_pos, anchor, key, self.ref_byte_at(anchor))

    def _codes(self, ref_bytes: np.ndarray) -> np.ndarray:
        codes = self._slot_of[ref_bytes]
        for byte in sorted(set(ref_bytes[codes < 0].tolist())):
            self.slot(byte)
        return self._slot_of[ref_bytes]

    def apply(
        self,
        plan: _Plan,
        pass_fwd: np.ndarray,
        pass_rev: np.ndarray,
        flush: Callable[[], None] | None = None,
        pass_last: np.ndarray = NO_LAST,
    ) -> None:
        """`pass_fwd`/`pass_rev` count passing reads; `pass_last` is their highest read number."""
        for op in plan.ops:
            if isinstance(op, _MatchOp):
                fwd = pass_fwd[op.rp : op.rp + op.length]
                rev = pass_rev[op.rp : op.rp + op.length]
                if op.idx is not None:
                    fwd = fwd[op.idx]
                    rev = rev[op.idx]
                self._add_match(op, fwd, rev)
            elif isinstance(op, _SnpOp):
                fwd, rev = int(pass_fwd[op.rp]), int(pass_rev[op.rp])
                if fwd or rev:
                    self._add_snp(op, fwd, rev, int(pass_last[op.rp]))
            elif pass_fwd[op.rp] or pass_rev[op.rp]:
                self._add_indel(op, pass_fwd[op.rp] > 0, flush)

    def apply_single(
        self,
        plan: _Plan,
        mask: np.ndarray,
        is_forward: bool,
        flush: Callable[[], None] | None = None,
        pass_last: np.ndarray = NO_LAST,
    ) -> None:
        """Apply one read; `mask` is True at the cs read offsets that pass the quality filter."""
        zeros = np.zeros(mask.shape[0], dtype=np.int32)
        if is_forward:
            self.apply(plan, mask, zeros, flush, pass_last)
        else:
            self.apply(plan, zeros, mask, flush, pass_last)

    def _add_match(self, op: _MatchOp, fwd: np.ndarray, rev: np.ndarray) -> None:
        total = fwd + rev
        hit = total > 0
        flat = op.flat
        if flat is None or op.flat_slots != self.fwd.shape[1]:
            # Flat indexes of (position, slot); a wider slot dimension changes them.
            flat = op.flat = op.rows * self.fwd.shape[1] + op.codes
            op.flat_slots = self.fwd.shape[1]
        rows, ref_bytes = op.rows, op.ref_bytes
        if not hit.all():
            if not hit.any():
                return
            flat, rows, ref_bytes = flat[hit], rows[hit], ref_bytes[hit]
            fwd, rev, total = fwd[hit], rev[hit], total[hit]
        self.fwd_flat[flat] += fwd
        self.rev_flat[flat] += rev
        self.tot_flat[flat] += total
        self.exists_flat[flat] = True
        self.dp[rows] += total
        self.high_water = max(self.high_water, int(rows[-1]) + 1)
        if not op.ref_set:
            current = self.ref_code[rows]
            unset = current < 0
            if unset.any():
                self.ref_code[rows[unset]] = ref_bytes[unset]
            op.ref_set = bool((self.ref_code[op.rows] >= 0).all())

    def _add_snp(self, op: _SnpOp, fwd: int, rev: int, last: int) -> None:
        row, slot = op.row, op.alt_slot
        self.fwd[row, slot] += fwd
        self.rev[row, slot] += rev
        self.tot[row, slot] += fwd + rev
        self.exists[row, slot] = True
        self.dp[row] += fwd + rev
        if last > self.snp_last[row]:
            # The last passing substitution read in file order sets the reference base.
            self.snp_last[row] = last
            self.ref_code[row] = op.ref_byte
        self.high_water = max(self.high_water, row + 1)

    def _add_indel(self, op: _IndelOp, is_forward: bool, flush: Callable[[], None] | None) -> None:
        anchor = op.anchor
        if anchor >= 0:
            self._ensure(anchor + 1)
            slot = int(self._slot_of[op.ref_byte])
            # Earlier reads may still wait in a batch, so a 0 counter needs them applied first.
            if flush is not None and not self._pending_cannot_matter(anchor, slot, is_forward):
                flush()
            # The retro-fix decrement needs the reference allele at the anchor.
            if slot >= 0 and self.exists[anchor, slot]:
                side = self.fwd if is_forward else self.rev
                if side[anchor, slot] > 0:
                    side[anchor, slot] -= 1
                if self.tot[anchor, slot] > 0:
                    self.tot[anchor, slot] -= 1
                if self.dp[anchor] > 0:
                    self.dp[anchor] -= 1
            self.high_water = max(self.high_water, anchor + 1)
        counts = self.indels.setdefault(anchor, {}).setdefault(op.key, [0, 0, 0])
        counts[0 if is_forward else 1] += 1
        counts[2] += 1
        if anchor >= 0:
            self.dp[anchor] += 1

    def _pending_cannot_matter(self, anchor: int, slot: int, is_forward: bool) -> bool:
        """True when every counter of the decrement is positive: waiting reads cannot change it."""
        if slot < 0 or not self.exists[anchor, slot]:
            return False
        side = self.fwd if is_forward else self.rev
        return bool(side[anchor, slot] > 0 and self.tot[anchor, slot] > 0 and self.dp[anchor] > 0)

    def frame(self) -> pl.DataFrame | None:
        """Rows in the output order: positions ascending, alleles by name."""
        pos_d, slot_d = np.nonzero(self.exists[: self.high_water])
        ind = self._indel_rows()
        if pos_d.size == 0 and not ind.pos:
            return None

        slot_chars = _CHAR_OF_BYTE[np.array(self.slot_bytes, dtype=np.uint8)]
        labels = sorted([str(c) for c in slot_chars] + ind.key)
        rank_of = {label: i for i, label in enumerate(labels)}
        rank = _join(
            np.array([rank_of[str(c)] for c in slot_chars], dtype=np.intp)[slot_d],
            np.array([rank_of[k] for k in ind.key], dtype=np.intp),
        )
        ref_code = _join(self.ref_code[pos_d], np.array(ind.ref_code, dtype=np.int16))
        pos = _join(pos_d, np.array(ind.pos, dtype=np.intp))
        columns = {
            "pos": pos + 1,
            "ref": np.where(ref_code >= 0, _CHAR_OF_BYTE[np.maximum(ref_code, 0)], "N"),
            "alt": _join(slot_chars[slot_d], np.array(ind.key)),
            "dp": _join(self.dp[pos_d], np.array(ind.dp, dtype=np.int64)),
            "adf": _join(self.fwd[pos_d, slot_d], np.array(ind.fwd, dtype=np.int64)),
            "adr": _join(self.rev[pos_d, slot_d], np.array(ind.rev, dtype=np.int64)),
            "ad": _join(self.tot[pos_d, slot_d], np.array(ind.tot, dtype=np.int64)),
        }
        order = np.lexsort((rank, pos))
        data = {
            name: pl.Series(name, values[order], dtype=SCHEMA[name])
            for name, values in columns.items()
        }
        data["seq_name"] = pl.Series("seq_name", [self.name] * order.size, dtype=pl.Utf8)
        dp = columns["dp"][order].astype(np.float64)
        af = np.zeros(order.size, dtype=np.float64)
        np.divide(columns["ad"][order].astype(np.float64), dp, out=af, where=dp > 0)
        data["af"] = pl.Series("af", af, dtype=pl.Float64)
        return pl.DataFrame(data, schema=SCHEMA)

    def _indel_rows(self) -> _IndelRows:
        rows = _IndelRows()
        for anchor in sorted(self.indels):
            alleles = self.indels[anchor]
            for key in sorted(alleles):
                counts = alleles[key]
                rows.pos.append(anchor)
                rows.key.append(key)
                rows.ref_code.append(int(self.ref_code[anchor]) if anchor >= 0 else -1)
                rows.dp.append(
                    int(self.dp[anchor]) if anchor >= 0 else sum(c[2] for c in alleles.values())
                )
                rows.fwd.append(counts[0])
                rows.rev.append(counts[1])
                rows.tot.append(counts[2])
        return rows


def _join(dense: np.ndarray, indel: np.ndarray) -> np.ndarray:
    if dense.size == 0:
        return indel
    if indel.size == 0:
        return dense
    return np.concatenate((dense, indel))
