"""Dynamic schema registry for CRDT-managed tables.

Allows arbitrary user-defined tables to be registered for CRDT tracking.
Schema metadata is stored in the _crdt_schema table and used by the
merge engine, tombstone resolver, and uniqueness arbiter to handle
relational constraints correctly.

Usage:
    registry = SchemaRegistry(conn)
    registry.register_table(
        table_name="patients",
        primary_key="id",
        columns=["id", "name", "nhs_number", "doctor_id"],
        foreign_keys=[("doctor_id", "doctors", "id")],
        unique_cols=["nhs_number"],
        on_tombstone_policy="preserve"
    )
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ForeignKeyDef:
    """Defines a foreign key relationship."""
    child_col: str
    parent_table: str
    parent_col: str


@dataclass
class TableSchema:
    """Complete schema definition for a CRDT-managed table."""
    table_name: str
    primary_key: str
    columns: list[str]
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)
    unique_cols: list[tuple[str, ...]] = field(default_factory=list)
    on_tombstone_policy: str = "preserve"  # cascade | nullify | preserve
    on_tombstone_callback: Optional[Callable] = None


class SchemaRegistry:
    """Manages registration and lookup of CRDT-managed table schemas.
    
    All schema metadata is persisted in the _crdt_schema SQLite table,
    enabling the engine to reconstruct its state after restart.
    """
    
    VALID_POLICIES = {"cascade", "nullify", "preserve"}
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._schemas: dict[str, TableSchema] = {}
        self._callbacks: dict[str, Callable] = {}  # not persisted
        self._ensure_schema_table()
        self._load_from_db()
    
    def _ensure_schema_table(self) -> None:
        """Create the _crdt_schema table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _crdt_schema (
                table_name        TEXT PRIMARY KEY,
                primary_key_col   TEXT NOT NULL DEFAULT 'id',
                columns_json      TEXT NOT NULL,
                foreign_keys_json TEXT DEFAULT '[]',
                unique_cols_json  TEXT DEFAULT '[]',
                on_tombstone_policy TEXT NOT NULL DEFAULT 'preserve',
                registered_at     TEXT NOT NULL
            )
        """)
        self.conn.commit()
    
    def _load_from_db(self) -> None:
        """Load all registered schemas from the database."""
        cursor = self.conn.execute("SELECT * FROM _crdt_schema")
        for row in cursor.fetchall():
            table_name = row[0]
            primary_key = row[1]
            columns = json.loads(row[2])
            foreign_keys_raw = json.loads(row[3])
            unique_cols_raw = json.loads(row[4])
            unique_cols = [tuple(uc) if isinstance(uc, list) else (uc,) for uc in unique_cols_raw]
            policy = row[5]
            
            fk_defs = [
                ForeignKeyDef(
                    child_col=fk[0],
                    parent_table=fk[1],
                    parent_col=fk[2]
                )
                for fk in foreign_keys_raw
            ]
            
            self._schemas[table_name] = TableSchema(
                table_name=table_name,
                primary_key=primary_key,
                columns=columns,
                foreign_keys=fk_defs,
                unique_cols=unique_cols,
                on_tombstone_policy=policy,
            )
    
    def register_table(
        self,
        table_name: str,
        primary_key: str,
        columns: list[str],
        foreign_keys: list[tuple[str, str, str]] | None = None,
        unique_cols: list[str | tuple[str, ...]] | None = None,
        on_tombstone_policy: str = "preserve",
        on_tombstone_callback: Callable | None = None,
        hlc_ts: str = "0000000000000:00000:system",
    ) -> TableSchema:
        """Register a table for CRDT tracking.
        
        Args:
            table_name: Name of the table to register.
            primary_key: Name of the primary key column.
            columns: List of all column names (including PK).
            foreign_keys: List of (child_col, parent_table, parent_col) tuples.
            unique_cols: List of column names or tuples of column names with uniqueness constraints.
            on_tombstone_policy: One of 'cascade', 'nullify', 'preserve'.
            on_tombstone_callback: Optional callable for custom resolution.
            hlc_ts: HLC timestamp for the registration event.
            
        Returns:
            The created TableSchema.
            
        Raises:
            ValueError: If the policy is invalid or table is already registered.
        """
        if on_tombstone_policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Invalid tombstone policy '{on_tombstone_policy}'. "
                f"Must be one of: {self.VALID_POLICIES}"
            )
        
        if table_name in self._schemas:
            raise ValueError(f"Table '{table_name}' is already registered.")
        
        if primary_key not in columns:
            raise ValueError(
                f"Primary key '{primary_key}' must be in the columns list."
            )
        
        fk_defs = []
        for fk in (foreign_keys or []):
            if len(fk) != 3:
                raise ValueError(
                    f"Foreign key must be (child_col, parent_table, parent_col), got {fk}"
                )
            child_col, parent_table, parent_col = fk
            if child_col not in columns:
                raise ValueError(
                    f"FK column '{child_col}' not in columns list for table '{table_name}'."
                )
            fk_defs.append(ForeignKeyDef(child_col, parent_table, parent_col))
        
        parsed_unique_cols = []
        for uc in (unique_cols or []):
            if isinstance(uc, str):
                cols = tuple(c.strip() for c in uc.split(",") if c.strip())
            else:
                cols = tuple(uc)
            
            for c in cols:
                if c not in columns:
                    raise ValueError(
                        f"Unique column '{c}' not in columns list for table '{table_name}'."
                    )
            parsed_unique_cols.append(cols)
        
        schema = TableSchema(
            table_name=table_name,
            primary_key=primary_key,
            columns=columns,
            foreign_keys=fk_defs,
            unique_cols=parsed_unique_cols,
            on_tombstone_policy=on_tombstone_policy,
            on_tombstone_callback=on_tombstone_callback,
        )
        
        # Persist to database
        fk_serialized = json.dumps(
            [(fk.child_col, fk.parent_table, fk.parent_col) for fk in fk_defs]
        )
        self.conn.execute(
            """INSERT INTO _crdt_schema 
               (table_name, primary_key_col, columns_json, foreign_keys_json, 
                unique_cols_json, on_tombstone_policy, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                table_name,
                primary_key,
                json.dumps(columns),
                fk_serialized,
                json.dumps(parsed_unique_cols),
                on_tombstone_policy,
                hlc_ts,
            ),
        )
        self.conn.commit()
        
        # Store in memory
        self._schemas[table_name] = schema
        if on_tombstone_callback:
            self._callbacks[table_name] = on_tombstone_callback
        
        # Create the actual application table in SQLite
        self._create_app_table(schema)
        
        return schema
    
    def _create_app_table(self, schema: TableSchema) -> None:
        """Create the actual application table in SQLite."""
        non_pk_cols = [c for c in schema.columns if c != schema.primary_key]
        col_defs = [f"{schema.primary_key} TEXT PRIMARY KEY"]
        col_defs.extend(f"{col} TEXT" for col in non_pk_cols)
        
        sql = f"CREATE TABLE IF NOT EXISTS {schema.table_name} ({', '.join(col_defs)})"
        self.conn.execute(sql)
        self.conn.commit()
    
    def get_schema(self, table_name: str) -> TableSchema:
        """Get the schema for a registered table.
        
        Raises:
            KeyError: If the table is not registered.
        """
        if table_name not in self._schemas:
            raise KeyError(f"Table '{table_name}' is not registered for CRDT tracking.")
        return self._schemas[table_name]
    
    def is_registered(self, table_name: str) -> bool:
        """Check if a table is registered for CRDT tracking."""
        return table_name in self._schemas
    
    def get_all_tables(self) -> list[str]:
        """Return all registered table names, sorted."""
        return sorted(self._schemas.keys())
    
    def get_children_of(self, parent_table: str) -> list[tuple[str, str, str]]:
        """Return all (child_table, child_col, parent_col) that reference parent_table."""
        children = []
        for table_name, schema in self._schemas.items():
            for fk in schema.foreign_keys:
                if fk.parent_table == parent_table:
                    children.append((table_name, fk.child_col, fk.parent_col))
        return children
    
    def get_parents_of(self, child_table: str) -> list[tuple[str, str, str]]:
        """Return all (parent_table, parent_col, child_col) that child_table references."""
        if child_table not in self._schemas:
            return []
        schema = self._schemas[child_table]
        return [
            (fk.parent_table, fk.parent_col, fk.child_col)
            for fk in schema.foreign_keys
        ]
    
    def get_unique_cols(self, table_name: str) -> list[tuple[str, ...]]:
        """Return unique column groups for a table."""
        return self._schemas[table_name].unique_cols if table_name in self._schemas else []
    
    def get_tombstone_policy(self, table_name: str) -> str:
        """Return the tombstone resolution policy for a table."""
        return self._schemas[table_name].on_tombstone_policy if table_name in self._schemas else "preserve"
    
    def get_tombstone_callback(self, table_name: str) -> Callable | None:
        """Return the tombstone callback for a table, if any."""
        return self._callbacks.get(table_name)
    
    def get_primary_key(self, table_name: str) -> str:
        """Return the primary key column name for a table."""
        return self._schemas[table_name].primary_key
    
    def get_columns(self, table_name: str) -> list[str]:
        """Return all column names for a table."""
        return self._schemas[table_name].columns
    
    def get_non_pk_columns(self, table_name: str) -> list[str]:
        """Return all non-primary-key column names for a table."""
        schema = self._schemas[table_name]
        return [c for c in schema.columns if c != schema.primary_key]
    
    def get_fk_columns(self, table_name: str) -> list[str]:
        """Return the FK column names for a table."""
        if table_name not in self._schemas:
            return []
        return [fk.child_col for fk in self._schemas[table_name].foreign_keys]
    
    def get_fk_target(self, child_table: str, child_col: str) -> tuple[str, str] | None:
        """Return (parent_table, parent_col) for a given FK column."""
        if child_table not in self._schemas:
            return None
        for fk in self._schemas[child_table].foreign_keys:
            if fk.child_col == child_col:
                return (fk.parent_table, fk.parent_col)
        return None
