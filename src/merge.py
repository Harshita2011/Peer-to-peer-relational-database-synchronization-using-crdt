"""Cell-level merge engine using Last-HLC-Wins semantics.

Resolves conflicts at the individual cell level (table, row_id, col_name),
not at the row level. This means two devices editing different columns of
the same row will NEVER conflict — both changes are preserved.

When two devices edit the SAME cell:
1. The cell with the higher HLC timestamp wins
2. If HLC timestamps are equal, the lower writer_id wins (deterministic tiebreak)
3. The losing cell version is stored but marked is_winner=0

This is strictly better than row-level Last-Write-Wins used by most systems.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.hlc import HLC, HLCTimestamp
from src.utils import serialize_value, deserialize_value


class MergeAction(Enum):
    """Outcome of merging a single cell."""
    ACCEPTED = "accepted"           # Incoming value wins
    REJECTED = "rejected"           # Local value wins
    NO_CONFLICT = "no_conflict"     # No local version existed
    IDENTICAL = "identical"         # Values are the same


@dataclass
class MergeCellResult:
    """Result of merging a single cell."""
    action: MergeAction
    table: str
    row_id: str
    col_name: str
    winning_value: Any
    winning_hlc: str
    winning_writer: str
    is_new_row: bool = False


@dataclass
class MergeRowResult:
    """Result of merging all cells of a row."""
    table: str
    row_id: str
    cell_results: list[MergeCellResult] = field(default_factory=list)
    is_new_row: bool = False

    @property
    def had_conflicts(self) -> bool:
        """True if any cell had a conflict (ACCEPTED or REJECTED)."""
        return any(
            r.action in (MergeAction.ACCEPTED, MergeAction.REJECTED)
            for r in self.cell_results
        )

    @property
    def accepted_count(self) -> int:
        """Number of cells where incoming value won."""
        return sum(1 for r in self.cell_results if r.action == MergeAction.ACCEPTED)

    @property
    def rejected_count(self) -> int:
        """Number of cells where local value won."""
        return sum(1 for r in self.cell_results if r.action == MergeAction.REJECTED)


@dataclass
class MergeResult:
    """Result of a full delta merge operation."""
    row_results: list[MergeRowResult] = field(default_factory=list)
    total_cells_merged: int = 0
    conflicts_resolved: int = 0
    new_rows_created: int = 0

    @property
    def rows_affected(self) -> int:
        return len(self.row_results)


class CellMerger:
    """Resolves cell-level conflicts using Last-HLC-Wins semantics.

    Each cell (table, row_id, col_name) is tracked independently in
    the _crdt_cells shadow table. When an incoming cell arrives during
    sync, it is compared against the local winning version.

    The merge rules ensure:
    - Commutativity: merge(A, B) == merge(B, A)
    - Idempotency: merge(A, A) == A
    - Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def merge_cell(
        self,
        table: str,
        row_id: str,
        col_name: str,
        incoming_value: Any,
        incoming_hlc: str | HLCTimestamp,
        incoming_writer: str,
    ) -> MergeCellResult:
        """Merge a single incoming cell value against the local state.

        Args:
            table: Table name.
            row_id: Serialized primary key.
            col_name: Column name.
            incoming_value: The incoming cell value (Python native type).
            incoming_hlc: HLC timestamp of the incoming write.
            incoming_writer: Writer/device ID of the incoming write.

        Returns:
            MergeCellResult describing the outcome.
        """
        incoming_hlc_str = str(incoming_hlc) if not isinstance(incoming_hlc, str) else incoming_hlc
        serialized_value = serialize_value(incoming_value)

        # Find the current winning version for this cell
        local_winner = self.conn.execute(
            """SELECT value, hlc_ts, writer_id 
               FROM _crdt_cells 
               WHERE table_name = ? AND row_id = ? AND col_name = ? AND is_winner = 1""",
            (table, row_id, col_name),
        ).fetchone()

        if local_winner is None:
            # No local version — accept incoming unconditionally
            self._upsert_cell(
                table, row_id, col_name, incoming_writer,
                serialized_value, incoming_hlc_str, is_winner=1,
            )
            return MergeCellResult(
                action=MergeAction.NO_CONFLICT,
                table=table,
                row_id=row_id,
                col_name=col_name,
                winning_value=incoming_value,
                winning_hlc=incoming_hlc_str,
                winning_writer=incoming_writer,
                is_new_row=True,
            )

        local_value, local_hlc, local_writer = local_winner

        # Check if values are identical (including same writer)
        if (serialized_value == local_value 
                and incoming_hlc_str == local_hlc 
                and incoming_writer == local_writer):
            return MergeCellResult(
                action=MergeAction.IDENTICAL,
                table=table,
                row_id=row_id,
                col_name=col_name,
                winning_value=deserialize_value(local_value),
                winning_hlc=local_hlc,
                winning_writer=local_writer,
            )

        # Compare HLC timestamps
        incoming_wins = self._incoming_wins(
            incoming_hlc_str, incoming_writer, local_hlc, local_writer
        )

        if incoming_wins:
            # Incoming wins — demote local, promote incoming
            # Mark old winner as non-winner
            self.conn.execute(
                """UPDATE _crdt_cells SET is_winner = 0
                   WHERE table_name = ? AND row_id = ? AND col_name = ? AND is_winner = 1""",
                (table, row_id, col_name),
            )
            # Insert or update incoming as winner
            self._upsert_cell(
                table, row_id, col_name, incoming_writer,
                serialized_value, incoming_hlc_str, is_winner=1,
            )
            return MergeCellResult(
                action=MergeAction.ACCEPTED,
                table=table,
                row_id=row_id,
                col_name=col_name,
                winning_value=incoming_value,
                winning_hlc=incoming_hlc_str,
                winning_writer=incoming_writer,
            )
        else:
            # Local wins — store incoming as non-winner
            self._upsert_cell(
                table, row_id, col_name, incoming_writer,
                serialized_value, incoming_hlc_str, is_winner=0,
            )
            return MergeCellResult(
                action=MergeAction.REJECTED,
                table=table,
                row_id=row_id,
                col_name=col_name,
                winning_value=deserialize_value(local_value),
                winning_hlc=local_hlc,
                winning_writer=local_writer,
            )

    def merge_row(
        self,
        table: str,
        row_id: str,
        incoming_cells: dict[str, Any],
        incoming_hlc: str | HLCTimestamp,
        incoming_writer: str,
    ) -> MergeRowResult:
        """Merge all cells of an incoming row independently.

        Each cell is resolved independently, so edits to different columns
        from different devices never conflict.

        Args:
            table: Table name.
            row_id: Serialized primary key.
            incoming_cells: Dict mapping column names to values.
            incoming_hlc: HLC timestamp (same for all cells in this write).
            incoming_writer: Writer/device ID.

        Returns:
            MergeRowResult with per-cell outcomes.
        """
        result = MergeRowResult(table=table, row_id=row_id)
        is_any_new = False

        for col_name, value in incoming_cells.items():
            cell_result = self.merge_cell(
                table, row_id, col_name, value, incoming_hlc, incoming_writer
            )
            result.cell_results.append(cell_result)
            if cell_result.is_new_row:
                is_any_new = True

        result.is_new_row = is_any_new
        return result

    def get_winning_cells(self, table: str, row_id: str) -> dict[str, Any]:
        """Get the current winning value for all cells of a row.

        Args:
            table: Table name.
            row_id: Serialized primary key.

        Returns:
            Dict mapping column names to their winning values.
        """
        cursor = self.conn.execute(
            """SELECT col_name, value 
               FROM _crdt_cells 
               WHERE table_name = ? AND row_id = ? AND is_winner = 1""",
            (table, row_id),
        )
        return {
            row[0]: deserialize_value(row[1])
            for row in cursor.fetchall()
        }

    def get_all_cell_versions(
        self, table: str, row_id: str, col_name: str
    ) -> list[dict]:
        """Get all versions (including losers) for a specific cell.

        Useful for debugging and audit trails.

        Args:
            table: Table name.
            row_id: Serialized primary key.
            col_name: Column name.

        Returns:
            List of dicts with value, hlc_ts, writer_id, is_winner.
        """
        cursor = self.conn.execute(
            """SELECT value, hlc_ts, writer_id, is_winner
               FROM _crdt_cells
               WHERE table_name = ? AND row_id = ? AND col_name = ?
               ORDER BY hlc_ts DESC""",
            (table, row_id, col_name),
        )
        return [
            {
                "value": deserialize_value(row[0]),
                "hlc_ts": row[1],
                "writer_id": row[2],
                "is_winner": bool(row[3]),
            }
            for row in cursor.fetchall()
        ]

    def _incoming_wins(
        self,
        incoming_hlc: str,
        incoming_writer: str,
        local_hlc: str,
        local_writer: str,
    ) -> bool:
        """Determine if the incoming cell value should win over the local one.

        Resolution order:
        1. Higher HLC timestamp wins (physical time, then logical counter)
        2. If HLC (physical, logical) are equal, LOWER writer_id wins

        IMPORTANT: We compare only (physical, logical) from the HLC — NOT the
        node_id suffix. The node_id in the HLC string is part of the timestamp
        format but tiebreaking is done exclusively via writer_id. This ensures
        deterministic, symmetric resolution regardless of which device
        performs the merge.
        """
        incoming_ts = HLCTimestamp.from_string(incoming_hlc)
        local_ts = HLCTimestamp.from_string(local_hlc)

        # Compare physical time first
        if incoming_ts.physical != local_ts.physical:
            return incoming_ts.physical > local_ts.physical
        # Then logical counter
        if incoming_ts.logical != local_ts.logical:
            return incoming_ts.logical > local_ts.logical
        # Physical and logical are equal — tiebreak by writer_id (lower wins)
        return incoming_writer < local_writer

    def _upsert_cell(
        self,
        table: str,
        row_id: str,
        col_name: str,
        writer_id: str,
        value: Optional[str],
        hlc_ts: str,
        is_winner: int,
    ) -> None:
        """Insert or update a cell version in _crdt_cells."""
        self.conn.execute(
            """INSERT INTO _crdt_cells 
               (table_name, row_id, col_name, writer_id, value, hlc_ts, is_winner)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(table_name, row_id, col_name, writer_id)
               DO UPDATE SET value = ?, hlc_ts = ?, is_winner = ?""",
            (table, row_id, col_name, writer_id, value, hlc_ts, is_winner,
             value, hlc_ts, is_winner),
        )
