"""Bounded metadata compaction engine.

Prunes CRDT cell versions that are causally stable — i.e., acknowledged
by all known peers. This ensures metadata doesn't grow unboundedly over
years of operation.

Compaction bound:
- For each (table, row_id, col_name), we retain at most max(1, |peers|) versions
- In steady state (all peers syncing regularly): exactly 1 version per cell
- This gives O(P * C) metadata where P = peers, C = cells
- Without compaction: O(O * C) where O = total operations (unbounded)

The compaction rule: a cell version can be dropped when:
1. It is NOT the current winner (is_winner = 0)
2. Every known peer's vector clock shows they have received a newer version
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.clock_manager import ClockManager


@dataclass
class CompactionResult:
    """Result of a compaction pass."""
    cell_versions_pruned: int = 0
    tombstones_purged: int = 0
    total_cells_before: int = 0
    total_cells_after: int = 0

    @property
    def savings_pct(self) -> float:
        if self.total_cells_before == 0:
            return 0.0
        return (self.cell_versions_pruned / self.total_cells_before) * 100


class CompactionEngine:
    """Prunes CRDT cell versions that are causally stable.

    Works with the ClockManager to determine which cell versions
    have been acknowledged by all peers and can safely be removed.
    """

    def __init__(self, conn: sqlite3.Connection, clock_manager: ClockManager):
        self.conn = conn
        self.clock_mgr = clock_manager

    def compact(self) -> CompactionResult:
        """Run one compaction pass.

        Steps:
        1. For each non-winning cell version:
           a. Check if all peers have acknowledged a newer version for this cell
           b. If yes, delete this version
        2. For resolved tombstones (is_resolved = 1):
           a. Check if all peers have the tombstone
           b. If yes, physically delete the tombstone
        3. Return statistics

        Returns:
            CompactionResult with pruning statistics.
        """
        result = CompactionResult()

        # Count cells before
        result.total_cells_before = self._count_cells()

        # Get all known peers
        peers = self.clock_mgr.get_all_peers()
        if not peers:
            result.total_cells_after = result.total_cells_before
            return result

        # Step 1: Prune non-winning cell versions that are causally stable
        result.cell_versions_pruned = self._prune_stable_cells(peers)

        # Step 2: Purge resolved tombstones that are known by all peers
        result.tombstones_purged = self._purge_stable_tombstones(peers)

        # Count cells after
        result.total_cells_after = self._count_cells()

        self.conn.commit()
        return result

    def estimate_savings(self) -> dict:
        """Calculate potential compaction savings without executing.

        Returns:
            Dict with 'pruneable_cells', 'purgeable_tombstones', 'total_cells'.
        """
        peers = self.clock_mgr.get_all_peers()
        total_cells = self._count_cells()

        if not peers:
            return {
                "pruneable_cells": 0,
                "purgeable_tombstones": 0,
                "total_cells": total_cells,
            }

        # Count non-winning cells that are globally acknowledged
        pruneable = 0
        non_winners = self.conn.execute(
            """SELECT table_name, row_id, col_name, writer_id, hlc_ts
               FROM _crdt_cells WHERE is_winner = 0"""
        ).fetchall()

        for table, row_id, col_name, writer_id, hlc_ts in non_winners:
            # Check if a newer winning version exists for this cell
            winner = self.conn.execute(
                """SELECT hlc_ts FROM _crdt_cells 
                   WHERE table_name = ? AND row_id = ? AND col_name = ? AND is_winner = 1""",
                (table, row_id, col_name),
            ).fetchone()

            if winner and self.clock_mgr.is_globally_acknowledged(writer_id, hlc_ts):
                pruneable += 1

        # Count resolved tombstones
        purgeable_ts = self.conn.execute(
            "SELECT COUNT(*) FROM _tombstones WHERE is_resolved = 1"
        ).fetchone()[0]

        return {
            "pruneable_cells": pruneable,
            "purgeable_tombstones": purgeable_ts,
            "total_cells": total_cells,
        }

    def _prune_stable_cells(self, peers: list[str]) -> int:
        """Prune non-winning cell versions that all peers have acknowledged."""
        pruned = 0

        # Get all non-winning cell versions
        non_winners = self.conn.execute(
            """SELECT rowid, table_name, row_id, col_name, writer_id, hlc_ts
               FROM _crdt_cells WHERE is_winner = 0"""
        ).fetchall()

        rowids_to_delete = []
        for rowid, table, row_id, col_name, writer_id, hlc_ts in non_winners:
            # Check if this version is globally acknowledged
            if self.clock_mgr.is_globally_acknowledged(writer_id, hlc_ts):
                # Verify that a newer winning version exists
                winner = self.conn.execute(
                    """SELECT hlc_ts FROM _crdt_cells 
                       WHERE table_name = ? AND row_id = ? AND col_name = ? 
                       AND is_winner = 1 AND hlc_ts > ?""",
                    (table, row_id, col_name, hlc_ts),
                ).fetchone()

                if winner:
                    rowids_to_delete.append(rowid)
                    pruned += 1

        # Batch delete
        if rowids_to_delete:
            placeholders = ",".join("?" * len(rowids_to_delete))
            self.conn.execute(
                f"DELETE FROM _crdt_cells WHERE rowid IN ({placeholders})",
                rowids_to_delete,
            )

        return pruned

    def _purge_stable_tombstones(self, peers: list[str]) -> int:
        """Purge resolved tombstones that all peers have acknowledged."""
        purged = 0

        resolved = self.conn.execute(
            "SELECT table_name, row_id, deleted_at_hlc, deleted_by FROM _tombstones WHERE is_resolved = 1"
        ).fetchall()

        for table, row_id, hlc_ts, writer_id in resolved:
            if self.clock_mgr.is_globally_acknowledged(writer_id, hlc_ts):
                self.conn.execute(
                    "DELETE FROM _tombstones WHERE table_name = ? AND row_id = ?",
                    (table, row_id),
                )
                purged += 1

        return purged

    def _count_cells(self) -> int:
        """Count total rows in _crdt_cells."""
        row = self.conn.execute("SELECT COUNT(*) FROM _crdt_cells").fetchone()
        return row[0] if row else 0
