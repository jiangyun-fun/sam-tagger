"""Tag statistics collection (FL, TC, AP, NB).

Decoupled from the CLI: takes an ``AnnotateArgs`` (for the 4 output paths) plus
the resolved ``AllConfigParams`` and the full list of reference names (FL stats
zero-fill across every header reference, since FL needs no config).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from loguru import logger

from sam_tagger.cli import AnnotateArgs
from sam_tagger.config import AllConfigParams
from sam_tagger.parquet_writer import SimpleStats, TypedStats


class StatsCollector:
    """Collects and writes tagging-related statistics (FL, TC, AP, NB)."""

    def __init__(
        self,
        args: AnnotateArgs,
        config: AllConfigParams,
        all_ref_names: list[str],
    ) -> None:
        def _ref_keys(need: Path | None, m: Mapping[str, object] | None) -> list[str] | None:
            return sorted(m.keys()) if (need is not None and m is not None) else None

        # Sorted reference lists for each stat type
        self._all_ap_refs = _ref_keys(args.ap_stats, config.ap)
        self._all_nb_refs = _ref_keys(args.nb_stats, config.nb)
        self._all_tc_refs = _ref_keys(args.tc_stats, config.tc)
        # FL stats span every header reference (FL needs no per-ref config).
        self._all_fl_refs = sorted(all_ref_names) if args.fl_stats is not None else None

        # Data collection maps
        self._ap_stats: dict[str, dict[str, int]] | None = {} if args.ap_stats is not None else None
        self._nb_stats: dict[str, dict[str, int]] | None = {} if args.nb_stats is not None else None
        self._fl_stats: dict[str, int] | None = {} if args.fl_stats is not None else None
        self._tc_stats: dict[str, int] | None = {} if args.tc_stats is not None else None

        # Output paths
        self._ap_stats_path = args.ap_stats
        self._nb_stats_path = args.nb_stats
        self._fl_stats_path = args.fl_stats
        self._tc_stats_path = args.tc_stats

    def record_ap(self, seq_name: str, status: str) -> None:
        if self._ap_stats is not None:
            seq_counts = self._ap_stats.setdefault(seq_name, {})
            seq_counts[status] = seq_counts.get(status, 0) + 1

    def record_nb(self, seq_name: str, status: str) -> None:
        if self._nb_stats is not None:
            seq_counts = self._nb_stats.setdefault(seq_name, {})
            seq_counts[status] = seq_counts.get(status, 0) + 1

    def record_fl(self, seq_name: str) -> None:
        if self._fl_stats is not None:
            self._fl_stats[seq_name] = self._fl_stats.get(seq_name, 0) + 1

    def record_tc(self, seq_name: str) -> None:
        if self._tc_stats is not None:
            self._tc_stats[seq_name] = self._tc_stats.get(seq_name, 0) + 1

    def write_results(self) -> None:
        self._write_ap_stats()
        self._write_nb_stats()
        self._write_fl_stats()
        self._write_tc_stats()

    def _write_ap_stats(self) -> None:
        if self._ap_stats_path is None or self._ap_stats is None or self._all_ap_refs is None:
            return
        logger.info("Writing AP tag statistics to {}", self._ap_stats_path)
        ap_statuses = ["both", "end_only", "neither", "start_only"]
        typed = TypedStats("AP")
        for seq_name in self._all_ap_refs:
            type_counts = self._ap_stats.get(seq_name, {})
            for ap_type in ap_statuses:
                count = type_counts.get(ap_type, 0)
                typed.add(seq_name, ap_type, count)
        typed.write_parquet(self._ap_stats_path)
        logger.info("AP statistics successfully written.")

    def _write_nb_stats(self) -> None:
        if self._nb_stats_path is None or self._nb_stats is None or self._all_nb_refs is None:
            return
        # Discover all NB types dynamically
        all_nb_types: set[str] = set()
        for type_counts in self._nb_stats.values():
            all_nb_types.update(type_counts.keys())
        sorted_nb_types = sorted(all_nb_types)

        logger.info("Writing NB tag statistics to {}", self._nb_stats_path)
        typed = TypedStats("NB")
        for seq_name in self._all_nb_refs:
            type_counts = self._nb_stats.get(seq_name, {})
            for nb_type in sorted_nb_types:
                count = type_counts.get(nb_type, 0)
                typed.add(seq_name, nb_type, count)
        # Total row
        for nb_type in sorted_nb_types:
            total = sum(
                self._nb_stats.get(seq_name, {}).get(nb_type, 0) for seq_name in self._all_nb_refs
            )
            typed.add("total", nb_type, total)
        typed.write_parquet(self._nb_stats_path)
        logger.info("NB statistics successfully written.")

    def _write_fl_stats(self) -> None:
        if self._fl_stats_path is None or self._fl_stats is None or self._all_fl_refs is None:
            return
        self._write_simple_stats(self._fl_stats_path, self._fl_stats, self._all_fl_refs, "FL")

    def _write_tc_stats(self) -> None:
        if self._tc_stats_path is None or self._tc_stats is None or self._all_tc_refs is None:
            return
        self._write_simple_stats(self._tc_stats_path, self._tc_stats, self._all_tc_refs, "TC")

    def _write_simple_stats(
        self,
        path: Path,
        stats: dict[str, int],
        all_refs: list[str],
        stat_name: str,
    ) -> None:
        logger.info("Writing {} tag statistics to {}", stat_name, path)
        simple = SimpleStats(stat_name)
        for seq_name in all_refs:
            count = stats.get(seq_name, 0)
            simple.add(seq_name, count)
        simple.write_parquet(path)
        logger.info("{} statistics successfully written.", stat_name)
