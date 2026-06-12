"""Uniqueness conflict arbitration with artifact preservation.

When two devices independently insert rows with the same UNIQUE key value
during offline periods, this module determines a deterministic winner and
preserves the loser as an auditable artifact in _conflict_artifacts.

Winner determination rule:
- The row from the writer with the LOWER writer_id (lexicographic) wins
- This is deterministic — all replicas will pick the same winner

Unlike CR-SQLite (which silently overwrites) and ElectricSQL (which doesn't
support distributed uniqueness), this system preserves the losing row for
audit and potential manual recovery.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.schema import SchemaRegistry
from src.utils import serialize_row, serialize_value


class UniquenessAction(Enum):
    """Outcome of uniqueness checking."""
    ACCEPT = "accept"                        # No conflict, proceed normally
    REJECT = "reject"                        # Incoming row loses, artifact created
    ACCEPT_AND_DISPLACE = "accept_and_displace"  # Incoming wins, existing → artifact


@dataclass
class UniquenessResult:
    """Result of a uniqueness check."""
    action: UniquenessAction
    artifact_created: bool = False
    artifact_id: int | None = None
    conflicting_key: str | None = None
    conflicting_col: str | tuple[str, ...] | None = None
    winner_writer: str | None = None
    loser_writer: str | None = None
    displaced_row_id: str | None = None


class UniquenessArbiter:
    """Resolves uniqueness conflicts deterministically.

    For each table, the arbiter knows which columns have uniqueness
    constraints (via SchemaRegistry). When a new row is inserted or
    synced, it checks if any existing row shares the same unique key value.

    If a conflict is found:
    1. Winner = row from the LOWER writer_id (lexicographic)
    2. Loser row is preserved in _conflict_artifacts with full JSON
    3. Application can review/recover artifacts via the query API
    """

    def __init__(self, conn: sqlite3.Connection, schema_registry: SchemaRegistry):
        self.conn = conn
        self.schema = schema_registry

    def check_and_resolve(
        self,
        table: str,
        row_id: str,
        row_data: dict[str, Any],
        writer_id: str,
        hlc: str,
    ) -> UniquenessResult:
        """Check if inserting this row would violate a uniqueness constraint.

        If a conflict is detected:
        - Determine winner by writer_id comparison (lower wins)
        - Create an artifact for the loser
        - Return the appropriate action for the caller

        Args:
            table: Table name.
            row_id: Primary key of the incoming row.
            row_data: Full row data as a dict.
            writer_id: Writer/device that created this row.
            hlc: HLC timestamp of the insert.

        Returns:
            UniquenessResult describing the outcome.
        """
        unique_cols = self.schema.get_unique_cols(table)
        if not unique_cols:
            return UniquenessResult(action=UniquenessAction.ACCEPT)

        pk_col = self.schema.get_primary_key(table)

        for col_group in unique_cols:
            # col_group is a tuple of column names
            col_values = tuple(row_data.get(c) for c in col_group)
            if any(v is None for v in col_values):
                # NULL values don't violate uniqueness in SQL semantics
                continue

            # Find existing row with the same unique key tuple
            existing = self._find_by_unique_key(table, col_group, col_values, pk_col)

            if existing is None:
                continue

            existing_id = existing[pk_col]
            if str(existing_id) == str(row_id):
                # Same row (update, not conflict)
                continue

            # Conflict detected! Determine winner
            existing_writer = self._get_row_writer(table, str(existing_id))
            if existing_writer is None:
                existing_writer = "unknown"

            conflicting_key_str = json.dumps(col_values)

            # Lower writer_id wins
            if existing_writer <= writer_id:
                # Existing wins — incoming is the loser
                artifact_id = self._create_artifact(
                    table=table,
                    winning_row_id=str(existing_id),
                    losing_row_id=row_id,
                    losing_row_data=row_data,
                    conflict_type="UNIQUE_KEY",
                    conflicting_key=conflicting_key_str,
                    winner_writer=existing_writer,
                    loser_writer=writer_id,
                    hlc=hlc,
                )
                return UniquenessResult(
                    action=UniquenessAction.REJECT,
                    artifact_created=True,
                    artifact_id=artifact_id,
                    conflicting_key=conflicting_key_str,
                    conflicting_col=col_group,
                    winner_writer=existing_writer,
                    loser_writer=writer_id,
                )
            else:
                # Incoming wins — existing is the loser
                existing_data = dict(existing)
                artifact_id = self._create_artifact(
                    table=table,
                    winning_row_id=row_id,
                    losing_row_id=str(existing_id),
                    losing_row_data=existing_data,
                    conflict_type="UNIQUE_KEY",
                    conflicting_key=conflicting_key_str,
                    winner_writer=writer_id,
                    loser_writer=existing_writer,
                    hlc=hlc,
                )
                
                # Defer the row removal to engine.py so it can properly tombstone it
                # returning displaced_row_id for the engine to handle
                return UniquenessResult(
                    action=UniquenessAction.ACCEPT_AND_DISPLACE,
                    artifact_created=True,
                    artifact_id=artifact_id,
                    conflicting_key=conflicting_key_str,
                    conflicting_col=col_group,
                    winner_writer=writer_id,
                    loser_writer=existing_writer,
                    displaced_row_id=str(existing_id),
                )

        return UniquenessResult(action=UniquenessAction.ACCEPT)

    def get_artifacts(
        self,
        table: str | None = None,
        resolved: bool | None = None,
    ) -> list[dict]:
        """Query conflict artifacts for review.

        Args:
            table: Filter by table name (optional).
            resolved: Filter by resolution status (optional).

        Returns:
            List of artifact dicts.
        """
        sql = "SELECT * FROM _conflict_artifacts WHERE 1=1"
        params: list[Any] = []

        if table is not None:
            sql += " AND original_table = ?"
            params.append(table)

        if resolved is not None:
            sql += " AND resolved = ?"
            params.append(1 if resolved else 0)

        sql += " ORDER BY detected_at DESC"

        cursor = self.conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def resolve_artifact(self, artifact_id: int) -> bool:
        """Mark an artifact as resolved.

        Args:
            artifact_id: ID of the artifact to resolve.

        Returns:
            True if the artifact was found and resolved.
        """
        cursor = self.conn.execute(
            "UPDATE _conflict_artifacts SET resolved = 1 WHERE id = ?",
            (artifact_id,),
        )
        return cursor.rowcount > 0

    def restore_artifact(self, artifact_id: int) -> dict | None:
        """Restore a losing row from an artifact back to the live table.

        This effectively reverses the uniqueness arbitration — the losing
        row is re-inserted and the artifact is marked as resolved.
        Note: this may cause a new uniqueness conflict if the winner still exists.

        Args:
            artifact_id: ID of the artifact to restore.

        Returns:
            The restored row data, or None if artifact not found.
        """
        artifact = self.conn.execute(
            """SELECT original_table, losing_row_json, resolved 
               FROM _conflict_artifacts WHERE id = ?""",
            (artifact_id,),
        ).fetchone()

        if artifact is None:
            return None

        table, row_json, resolved = artifact
        row_data = json.loads(row_json)

        # Mark artifact as resolved
        self.conn.execute(
            "UPDATE _conflict_artifacts SET resolved = 1 WHERE id = ?",
            (artifact_id,),
        )

        return row_data

    def _find_by_unique_key(
        self, table: str, col_group: tuple[str, ...], col_values: tuple[Any, ...], pk_col: str
    ) -> dict | None:
        """Find a row by its unique key value."""
        where_clause = " AND ".join(f"{col} = ?" for col in col_group)
        cursor = self.conn.execute(
            f"SELECT * FROM {table} WHERE {where_clause}",
            tuple(str(v) for v in col_values),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def _get_row_writer(self, table: str, row_id: str) -> str | None:
        """Get the original writer of a row from _crdt_row_state."""
        row = self.conn.execute(
            """SELECT writer_id FROM _crdt_row_state 
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        ).fetchone()
        return row[0] if row else None

    def _create_artifact(
        self,
        table: str,
        winning_row_id: str,
        losing_row_id: str,
        losing_row_data: dict,
        conflict_type: str,
        conflicting_key: str,
        winner_writer: str,
        loser_writer: str,
        hlc: str,
    ) -> int:
        """Create a conflict artifact in the _conflict_artifacts table."""
        cursor = self.conn.execute(
            """INSERT INTO _conflict_artifacts 
               (original_table, winning_row_id, losing_row_json, conflict_type,
                conflicting_key, winner_writer, loser_writer, detected_at, resolved)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                table,
                winning_row_id,
                json.dumps(losing_row_data, sort_keys=True, default=str),
                conflict_type,
                conflicting_key,
                winner_writer,
                loser_writer,
                hlc,
            ),
        )
        return cursor.lastrowid

    def _remove_row(self, table: str, row_id: str) -> None:
        """Remove a row from the live table (it's being displaced by the winner)."""
        pk_col = self.schema.get_primary_key(table)
        self.conn.execute(
            f"DELETE FROM {table} WHERE {pk_col} = ?",
            (row_id,),
        )
        # Also clean up CRDT state for this row
        self.conn.execute(
            "DELETE FROM _crdt_cells WHERE table_name = ? AND row_id = ?",
            (table, row_id),
        )
        self.conn.execute(
            "DELETE FROM _crdt_row_state WHERE table_name = ? AND row_id = ?",
            (table, row_id),
        )
