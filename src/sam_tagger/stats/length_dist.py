"""Length-distribution collector for the ``qc-summary`` ``--length-dist-out`` side output.

Counts primary alignments only (secondary 0x100 / supplementary 0x800 excluded
by the caller): a length distribution is a property of reads, and secondary
records double-count while supplementary records are hard-clip fragments.

Two metrics per reference: ``read_len`` (``query_length`` — full observed
read, soft clips included) and ``dm`` (M+D CIGAR sum, the ``--add-dm`` tag
computation). Output is long format ``(seq_name, metric, value, count)`` —
the same relational distribution shape as ``TypedStats``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sam_tagger.parquet_writer import DistributionStats


class LengthDistCollector:
    """Accumulates per-reference read_len/dm histograms; writes one long-format Parquet.

    Owns its output path from construction (the ``ErrorStatsCollector`` pattern),
    so the orchestrator only gates on the collector's existence.
    """

    def __init__(self, output: Path) -> None:
        self._output = output
        # {seq_name: {metric: {value: count}}}
        self._dists: dict[str, dict[str, dict[int, int]]] = {}

    def tally(self, seq_name: str, read_len: int | None, dm: int) -> None:
        """Add one primary record. ``read_len`` is ``None`` for sequence-less records."""
        metrics = self._dists.setdefault(seq_name, {"read_len": {}, "dm": {}})
        if read_len is not None:
            metrics["read_len"][read_len] = metrics["read_len"].get(read_len, 0) + 1
        metrics["dm"][dm] = metrics["dm"].get(dm, 0) + 1

    def write_results(self) -> None:
        """Write all (seq_name, metric, value, count) rows sorted by seq_name, metric, value."""
        logger.info("Writing length distributions to {}", self._output)
        dist = DistributionStats()
        for seq_name in sorted(self._dists):
            for metric in sorted(self._dists[seq_name]):
                for value in sorted(self._dists[seq_name][metric]):
                    dist.add(seq_name, metric, value, self._dists[seq_name][metric][value])
        dist.write_parquet(self._output)
        logger.info("Length distributions successfully written.")
