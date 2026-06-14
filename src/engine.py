"""CRDTEngine — Main orchestrator for the offline-first CRDT sync engine.

This is the primary entry point for all database operations. All writes
go through the engine (not raw SQL), ensuring that every INSERT, UPDATE,
and DELETE is properly tracked in the CRDT shadow tables.

Usage:
    engine = CRDTEngine(":memory:", "device_A")
    engine.register_table("doctors", primary_key="id",
                          columns=["id", "name", "specialty"])
    engine.register_table("patients", primary_key="id",
                          columns=["id", "name", "nhs_number", "doctor_id"],
                          foreign_keys=[("doctor_id", "doctors", "id")],
                          unique_cols=["nhs_number"],
                          on_tombstone_policy="preserve")
    
    # Insert data
    engine.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
    engine.insert("patients", {"id": "p1", "name": "Alice", "doctor_id": "d1"})
    
    # Sync with another engine
    delta = engine_a.get_delta(since_hlc="0", for_peer="device_B")
    result = engine_b.apply_delta(delta, from_peer="device_A")
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.hlc import HLC, HLCTimestamp
from src.merge import CellMerger, MergeResult, MergeRowResult
from src.schema import SchemaRegistry
from src.tombstone import TombstoneResolver, TombstoneResult
from src.uniqueness import UniquenessArbiter, UniquenessAction, UniquenessResult
from src.utils import (
    generate_row_id,
    serialize_value,
    deserialize_value,
    serialize_row,
    build_row_from_cells,
)
from src.vector_clock import VectorClock
import uuid


@dataclass
class DeltaEntry:
    """A single entry in a delta exchange, representing a causal operation."""
    op_id: str
    peer_id: str
    table_name: str
    row_id: str
    column_name: str | None = None
    operation_type: str = "insert"
    value: str | None = None
    vector_clock_json: str = "{}"
    hlc_ts: str = ""


@dataclass
class Delta:
    """A collection of changes to be synced between peers."""
    entries: list[DeltaEntry] = field(default_factory=list)
    source_peer: str = ""
    source_hlc: str = ""
    last_seq: int = 0

    @property
    def size(self) -> int:
        return len(self.entries)


@dataclass
class SyncResult:
    """Result of applying a delta from a peer."""
    merge_result: MergeResult
    tombstone_results: list[TombstoneResult] = field(default_factory=list)
    uniqueness_results: list[UniquenessResult] = field(default_factory=list)
    rows_created: int = 0
    rows_updated: int = 0
    local_hash: str = ""


class CRDTEngine:
    """Main orchestrator for the CRDT sync engine.

    Intercepts all database writes to maintain CRDT shadow tables,
    tombstone tracking, and uniqueness arbitration. Provides delta
    exchange for peer-to-peer sync.
    """

    def __init__(self, db_path: str, node_id: str):
        """Initialize a CRDTEngine instance.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
            node_id: Unique identifier for this device/replica.
        """
        self.db_path = db_path
        self.node_id = node_id
        self.hlc = HLC(node_id)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=OFF")  # We handle FK ourselves
        
        self._apply_schema()
        
        self.vclock = self._load_vector_clock()
        self.schema = SchemaRegistry(self.conn)
        self.merger = CellMerger(self.conn)
        self.tombstone_resolver = TombstoneResolver(self.conn, self.schema, update_callback=self.update)
        self.uniqueness_arbiter = UniquenessArbiter(self.conn, self.schema)

    def _load_vector_clock(self) -> VectorClock:
        cursor = self.conn.execute("SELECT writer_id, max_hlc_ts FROM _vector_clocks WHERE peer_id = ?", (self.node_id,))
        state = {}
        for row in cursor.fetchall():
            try:
                state[row[0]] = int(row[1])
            except ValueError:
                pass
        return VectorClock(state)

    def _save_vector_clock(self) -> None:
        for writer_id, seq in self.vclock.state.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts)
                   VALUES (?, ?, ?)""",
                (self.node_id, writer_id, str(seq))
            )

    def _log_operation(self, table: str, row_id: str, col_name: str | None, op_type: str, value: str | None, vc_json: str, hlc_ts: str) -> None:
        op_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO _operations 
               (op_id, peer_id, table_name, row_id, column_name, operation_type, value, vector_clock_json, hlc_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (op_id, self.node_id, table, row_id, col_name, op_type, value, vc_json, hlc_ts)
        )

    def _apply_schema(self) -> None:
        """Apply the CRDT shadow table schema."""
        schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
        if schema_path.exists():
            sql = schema_path.read_text()
            self.conn.executescript(sql)
        else:
            # Inline fallback for in-memory databases or test scenarios
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS _crdt_schema (
                    table_name TEXT PRIMARY KEY,
                    primary_key_col TEXT NOT NULL DEFAULT 'id',
                    columns_json TEXT NOT NULL,
                    foreign_keys_json TEXT DEFAULT '[]',
                    unique_cols_json TEXT DEFAULT '[]',
                    on_tombstone_policy TEXT NOT NULL DEFAULT 'preserve',
                    registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS _crdt_cells (
                    table_name TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    col_name TEXT NOT NULL,
                    writer_id TEXT NOT NULL,
                    value TEXT,
                    vector_clock_json TEXT NOT NULL,
                    hlc_ts TEXT NOT NULL,
                    is_winner INTEGER DEFAULT 1,
                    PRIMARY KEY (table_name, row_id, col_name, writer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_crdt_cells_lookup
                    ON _crdt_cells(table_name, row_id, col_name, is_winner);
                CREATE TABLE IF NOT EXISTS _tombstones (
                    table_name TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    vector_clock_json TEXT NOT NULL,
                    deleted_by TEXT NOT NULL,
                    ref_count INTEGER DEFAULT 0,
                    PRIMARY KEY (table_name, row_id)
                );
                CREATE TABLE IF NOT EXISTS _operations (
                    local_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    op_id TEXT NOT NULL,
                    peer_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    column_name TEXT,
                    operation_type TEXT NOT NULL,
                    value TEXT,
                    vector_clock_json TEXT NOT NULL,
                    hlc_ts TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS _conflict_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_table TEXT NOT NULL,
                    winning_row_id TEXT,
                    losing_row_json TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    conflicting_key TEXT NOT NULL,
                    winner_writer TEXT NOT NULL,
                    loser_writer TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS _vector_clocks (
                    peer_id TEXT NOT NULL,
                    writer_id TEXT NOT NULL,
                    max_hlc_ts TEXT NOT NULL,
                    PRIMARY KEY (peer_id, writer_id)
                );
                CREATE TABLE IF NOT EXISTS _crdt_row_state (
                    table_name TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    is_deleted INTEGER DEFAULT 0,
                    created_hlc TEXT NOT NULL,
                    writer_id TEXT NOT NULL,
                    PRIMARY KEY (table_name, row_id)
                );
            """)

    def register_table(
        self,
        table_name: str,
        primary_key: str = "id",
        columns: list[str] | None = None,
        foreign_keys: list[tuple[str, str, str]] | None = None,
        unique_cols: list[str] | None = None,
        on_tombstone_policy: str = "preserve",
        on_tombstone_callback: Callable | None = None,
    ) -> None:
        """Register a table for CRDT tracking.

        Creates the application table and records its schema metadata.

        Args:
            table_name: Name of the table.
            primary_key: Primary key column name.
            columns: List of column names (if None, uses [primary_key]).
            foreign_keys: List of (child_col, parent_table, parent_col) tuples.
            unique_cols: Columns with uniqueness constraints.
            on_tombstone_policy: Resolution policy: 'cascade', 'nullify', 'preserve'.
            on_tombstone_callback: Optional custom resolution callback.
        """
        if columns is None:
            columns = [primary_key]
        
        ts = self.hlc.now()
        self.schema.register_table(
            table_name=table_name,
            primary_key=primary_key,
            columns=columns,
            foreign_keys=foreign_keys,
            unique_cols=unique_cols,
            on_tombstone_policy=on_tombstone_policy,
            on_tombstone_callback=on_tombstone_callback,
            hlc_ts=str(ts),
        )

    def insert(self, table: str, row: dict[str, Any]) -> str:
        """Insert a new row into a CRDT-managed table.

        The row is inserted into both the application table and the
        _crdt_cells shadow table. FK references to tombstoned parents
        are detected and handled.

        Args:
            table: Table name.
            row: Dict of column values. If PK is missing, one is generated.

        Returns:
            The row ID (primary key value).

        Raises:
            KeyError: If the table is not registered.
        """
        schema = self.schema.get_schema(table)
        pk_col = schema.primary_key
        
        # Generate row ID if not provided
        if pk_col not in row or row[pk_col] is None:
            row[pk_col] = generate_row_id()
        
        row_id = str(row[pk_col])
        ts = self.hlc.now()
        ts_str = str(ts)

        # Check uniqueness constraints
        uniqueness_result = self.uniqueness_arbiter.check_and_resolve(
            table, row_id, row, self.node_id, ts_str
        )
        if uniqueness_result.action == UniquenessAction.REJECT:
            # This row is the loser — it's already been preserved as an artifact
            return row_id
        elif uniqueness_result.action == UniquenessAction.ACCEPT_AND_DISPLACE:
            if uniqueness_result.displaced_row_id:
                # Tombstone the displaced row
                self.delete(table, uniqueness_result.displaced_row_id)

        # Insert into application table
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [str(v) if v is not None else None for v in row.values()]
        
        updates = ", ".join(f"{c} = ?" for c in cols if c != pk_col)
        update_values = [str(v) if v is not None else None for k, v in row.items() if k != pk_col]

        if updates:
            self.conn.execute(
                f"""INSERT INTO {table} ({col_names}) VALUES ({placeholders})
                    ON CONFLICT({pk_col}) DO UPDATE SET {updates}""",
                values + update_values,
            )
        else:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})",
                values,
            )

        # Record in _crdt_row_state
        self.conn.execute(
            """INSERT INTO _crdt_row_state 
               (table_name, row_id, is_deleted, created_hlc, writer_id)
               VALUES (?, ?, 0, ?, ?)
               ON CONFLICT(table_name, row_id) DO UPDATE SET is_deleted = 0""",
            (table, row_id, ts_str, self.node_id),
        )

        # Increment vector clock for this write transaction
        self.vclock = self.vclock.increment(self.node_id)
        self._save_vector_clock()
        vc_json = str(self.vclock)

        # Write cell entries to _crdt_cells and log operations
        for col_name, value in row.items():
            serialized = serialize_value(value)
            self.merger.merge_cell(
                table=table,
                row_id=row_id,
                col_name=col_name,
                incoming_value=value,
                incoming_vc=vc_json,
                incoming_hlc=ts_str,
                incoming_writer=self.node_id,
            )
            self._log_operation(table, row_id, col_name, "insert", serialized, vc_json, ts_str)

        # Check if any FK columns reference tombstoned parents
        self.tombstone_resolver.on_child_insert(table, row)

        self.conn.commit()
        return row_id

    def update(self, table: str, row_id: str, changes: dict[str, Any]) -> None:
        """Update specific columns of a row.

        Only the changed columns are written to _crdt_cells, enabling
        cell-level merge without overwriting other columns.

        Args:
            table: Table name.
            row_id: Primary key of the row to update.
            changes: Dict of {column_name: new_value} for changed columns.

        Raises:
            KeyError: If the table is not registered.
            ValueError: If the row doesn't exist.
        """
        schema = self.schema.get_schema(table)
        pk_col = schema.primary_key
        
        # Verify row exists
        existing = self.conn.execute(
            f"SELECT 1 FROM {table} WHERE {pk_col} = ?",
            (row_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Row '{row_id}' not found in table '{table}'.")

        ts = self.hlc.now()
        ts_str = str(ts)

        # Update application table
        set_clauses = ", ".join(f"{col} = ?" for col in changes.keys())
        values = [str(v) if v is not None else None for v in changes.values()]
        values.append(row_id)
        
        self.conn.execute(
            f"UPDATE {table} SET {set_clauses} WHERE {pk_col} = ?",
            values,
        )

        # Increment vector clock
        self.vclock = self.vclock.increment(self.node_id)
        self._save_vector_clock()
        vc_json = str(self.vclock)

        # Write cell entries for changed columns only
        for col_name, value in changes.items():
            serialized = serialize_value(value)
            self.merger.merge_cell(
                table=table,
                row_id=row_id,
                col_name=col_name,
                incoming_value=value,
                incoming_vc=vc_json,
                incoming_hlc=ts_str,
                incoming_writer=self.node_id,
            )
            self._log_operation(table, row_id, col_name, "update", serialized, vc_json, ts_str)

        self.conn.commit()

    def delete(self, table: str, row_id: str) -> TombstoneResult:
        """Delete a row via tombstone.

        The row is not physically deleted if it has child rows referencing it.
        Instead, a tombstone is created with a ref_count of the live children.

        Args:
            table: Table name.
            row_id: Primary key of the row to delete.

        Returns:
            TombstoneResult describing what happened.
        """
        ts = self.hlc.now()
        ts_str = str(ts)

        # Get current row data before tombstoning (for child delete tracking)
        schema = self.schema.get_schema(table)
        pk_col = schema.primary_key
        row = self.conn.execute(
            f"SELECT * FROM {table} WHERE {pk_col} = ?",
            (row_id,),
        ).fetchone()
        
        if row is not None:
            cols = [desc[0] for desc in self.conn.execute(
                f"SELECT * FROM {table} LIMIT 0"
            ).description]
            row_data = dict(zip(cols, row))
        else:
            row_data = {}

        # Increment vector clock
        self.vclock = self.vclock.increment(self.node_id)
        self._save_vector_clock()
        vc_json = str(self.vclock)

        # Notify tombstone resolver (handles ref_count and potential children)
        result = self.tombstone_resolver.on_delete(table, row_id, vc_json, ts_str, self.node_id)

        # Log delete operation
        self._log_operation(table, row_id, None, "delete", None, vc_json, ts_str)

        # If this row is itself a child, notify parent tombstone resolvers
        if row_data:
            self.tombstone_resolver.on_child_delete(table, row_data)

        self.conn.commit()
        return result

    def query(self, table: str, where: dict[str, Any] | None = None) -> list[dict]:
        """Query rows from a table, excluding tombstoned rows.

        Args:
            table: Table name.
            where: Optional dict of {column: value} for filtering.

        Returns:
            List of row dicts.
        """
        schema = self.schema.get_schema(table)
        
        sql = f"SELECT * FROM {table}"
        params: list[Any] = []
        
        conditions = []
        if where:
            for col, val in where.items():
                conditions.append(f"{col} = ?")
                params.append(str(val) if val is not None else None)

        # Exclude tombstoned rows
        pk_col = schema.primary_key
        conditions.append(f"""
            NOT EXISTS (
                SELECT 1 FROM _tombstones 
                WHERE table_name = ? AND row_id = {table}.{pk_col}
            )
        """)
        params.append(table)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        cursor = self.conn.execute(sql, params)
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def query_raw(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a raw SQL query.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            List of row dicts.
        """
        cursor = self.conn.execute(sql, params)
        if cursor.description is None:
            return []
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_delta(self, since_seq: int = 0, for_peer: str | None = None) -> Delta:
        """Get all changes made since the given local sequence number.

        Extracts state for replication, including:
        1. Operations log entries (which cover cells)
        2. All _tombstones
        3. All _crdt_row_state entries

        Args:
            since_seq: Only include changes after this local_seq.
            for_peer: The peer this delta is intended for.

        Returns:
            A Delta object with all entries to sync.
        """
        delta = Delta(
            source_peer=self.node_id,
            source_hlc=str(self.hlc.current),
        )

        cursor = self.conn.execute(
            """SELECT op_id, peer_id, table_name, row_id, column_name, operation_type, value, vector_clock_json, hlc_ts, local_seq
               FROM _operations
               WHERE local_seq > ?
               ORDER BY local_seq""",
            (since_seq,),
        )
        max_seq = since_seq
        for row in cursor.fetchall():
            delta.entries.append(DeltaEntry(
                op_id=row[0],
                peer_id=row[1],
                table_name=row[2],
                row_id=row[3],
                column_name=row[4],
                operation_type=row[5],
                value=row[6],
                vector_clock_json=row[7],
                hlc_ts=row[8],
            ))
            max_seq = max(max_seq, row[9])
        delta.last_seq = max_seq
        return delta

    def apply_delta(self, delta: Delta, from_peer: str) -> SyncResult:
        """Apply an incoming delta from a peer."""
        sync_result = SyncResult(merge_result=MergeResult())

        if delta.source_hlc:
            self.hlc.receive(delta.source_hlc)

        # Pre-process rows for uniqueness checking (Constraint-Preserving CRDT)
        # We group inserts/updates by row_id to build the incoming state
        incoming_rows: dict[tuple[str, str], dict[str, Any]] = {}
        row_hlcs: dict[tuple[str, str], str] = {}
        row_writers: dict[tuple[str, str], str] = {}
        
        for op in delta.entries:
            if not self.schema.is_registered(op.table_name):
                continue
            if op.peer_id == self.node_id:
                continue

            existing_op = self.conn.execute(
                "SELECT 1 FROM _operations WHERE op_id = ? AND column_name IS ?",
                (op.op_id, op.column_name)
            ).fetchone()
            if existing_op:
                continue

            key = (op.table_name, op.row_id)
            if op.operation_type in ("insert", "update") and op.column_name:
                if key not in incoming_rows:
                    incoming_rows[key] = {}
                incoming_rows[key][op.column_name] = deserialize_value(op.value)
                row_hlcs[key] = op.hlc_ts
                row_writers[key] = op.peer_id

        # Run uniqueness checks for all incoming rows
        rejected_rows = set()
        for (table, row_id), row_data in incoming_rows.items():
            u_result = self.uniqueness_arbiter.check_and_resolve(
                table, row_id, row_data, row_writers[(table, row_id)], row_hlcs[(table, row_id)]
            )
            if u_result.action == UniquenessAction.REJECT:
                # Incoming row loses uniqueness conflict - skip its cell operations
                rejected_rows.add((table, row_id))
            elif u_result.action == UniquenessAction.ACCEPT_AND_DISPLACE:
                # Incoming row wins, existing row must be tombstoned
                if u_result.displaced_row_id:
                    # Tombstone the displaced row to replicate its deletion
                    self.delete(table, u_result.displaced_row_id)

        # Now process all operations log
        affected_rows = set()
        for op in delta.entries:
            if not self.schema.is_registered(op.table_name):
                continue
            if op.peer_id == self.node_id:
                continue
                
            existing_op = self.conn.execute(
                "SELECT 1 FROM _operations WHERE op_id = ? AND column_name IS ?",
                (op.op_id, op.column_name)
            ).fetchone()
            if existing_op:
                continue
                
            key = (op.table_name, op.row_id)
                
            # Log the operation locally to maintain causality
            self.conn.execute(
                """INSERT OR IGNORE INTO _operations 
                   (op_id, peer_id, table_name, row_id, column_name, operation_type, value, vector_clock_json, hlc_ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (op.op_id, op.peer_id, op.table_name, op.row_id, op.column_name, op.operation_type, op.value, op.vector_clock_json, op.hlc_ts)
            )

            if key in rejected_rows:
                # The operation's row lost a uniqueness conflict, do not merge its cells
                continue

            # Update HLC
            self.hlc.receive(op.hlc_ts)
            affected_rows.add((op.table_name, op.row_id))

            if op.operation_type in ("insert", "update") and op.column_name:
                cell_result = self.merger.merge_cell(
                    table=op.table_name,
                    row_id=op.row_id,
                    col_name=op.column_name,
                    incoming_value=deserialize_value(op.value),
                    incoming_vc=op.vector_clock_json,
                    incoming_hlc=op.hlc_ts,
                    incoming_writer=op.peer_id,
                )
                sync_result.merge_result.total_cells_merged += 1
            elif op.operation_type == "delete":
                # Handle tombstone via Remove-Wins semantics
                result = self.tombstone_resolver.on_delete(
                    op.table_name, op.row_id, op.vector_clock_json, op.hlc_ts, op.peer_id
                )
                sync_result.tombstone_results.append(result)
                
            self.vclock = self.vclock.merge(VectorClock.from_string(op.vector_clock_json))
        
        self._save_vector_clock()

        # After merging operations, rebuild application table rows
        for table, row_id in affected_rows:
            self._rebuild_app_row(table, row_id)

        # Re-evaluate all tombstones to ensure constraints (multi-level FKs etc)
        unresolved = self.conn.execute(
            "SELECT table_name, row_id FROM _tombstones"
        ).fetchall()
        for table, row_id in unresolved:
            if self.schema.is_registered(table):
                self.tombstone_resolver.recalculate_ref_count(table, row_id)

        # Update vector clock
        self.conn.execute(
            """INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts)
               VALUES (?, ?, ?)""",
            (from_peer, from_peer, delta.source_hlc or str(self.hlc.current)),
        )

        self.conn.commit()
        return sync_result





    def _rebuild_app_row(self, table: str, row_id: str) -> None:
        """Rebuild an application table row from winning CRDT cells.

        After merging cells, the application table must reflect the
        current winning values. This method reconstructs the row.
        """
        schema = self.schema.get_schema(table)
        pk_col = schema.primary_key
        
        # Get all winning cells for this row
        winning_cells = self.merger.get_winning_cells(table, row_id)
        if not winning_cells:
            return

        # Check if row is tombstoned
        if self.tombstone_resolver.is_tombstoned(table, row_id):
            return

        # Build the full row
        row_data = {pk_col: row_id}
        for col, value in winning_cells.items():
            if col != pk_col:
                row_data[col] = value

        # Upsert into application table
        cols = list(row_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        updates = ", ".join(f"{c} = ?" for c in cols if c != pk_col)
        values = [str(v) if v is not None else None for v in row_data.values()]
        update_values = [str(v) if v is not None else None 
                        for k, v in row_data.items() if k != pk_col]

        if updates:
            self.conn.execute(
                f"""INSERT INTO {table} ({col_names}) VALUES ({placeholders})
                    ON CONFLICT({pk_col}) DO UPDATE SET {updates}""",
                values + update_values,
            )
        else:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})",
                values,
            )
            
        # Check if any FK columns reference tombstoned parents
        self.tombstone_resolver.on_child_insert(table, row_data)

    def get_unresolved_tombstones(self) -> list:
        """Return all unresolved tombstones for application review."""
        return self.tombstone_resolver.get_unresolved_tombstones()

    def get_conflict_artifacts(self, table: str | None = None) -> list[dict]:
        """Return all conflict artifacts for review."""
        return self.uniqueness_arbiter.get_artifacts(table=table)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        tables = self.schema.get_all_tables()
        return f"CRDTEngine(node='{self.node_id}', tables={tables})"
