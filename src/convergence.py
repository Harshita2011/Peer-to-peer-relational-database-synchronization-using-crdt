"""Canonical convergence hashing using SHA-256.

Computes a deterministic hash of the entire database state. Two replicas
that have converged to identical state will produce the same hash,
regardless of the order in which operations were applied.

This is especially valuable in IoT/edge deployments where you cannot
transfer full datasets to verify state.

Algorithm:
1. For each registered table (sorted by table name):
   a. SELECT all non-tombstoned rows ORDER BY primary key
   b. For each row: serialize all columns deterministically (sorted by col name)
   c. Feed each serialized row into the SHA-256 hasher
2. Return the hex digest

No existing CRDT system offers this capability.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from src.schema import SchemaRegistry


@dataclass
class ConvergenceResult:
    """Result of a convergence verification."""
    converged: bool
    local_hash: str
    peer_hash: str
    mismatched_tables: list[str] | None = None


class ConvergenceHasher:
    """Computes deterministic SHA-256 hashes for convergence verification.

    The hash covers all live (non-tombstoned) rows in all registered tables.
    Tombstoned rows, CRDT metadata, and conflict artifacts are excluded
    from the hash — only the application-visible state is hashed.
    """

    def __init__(self, conn: sqlite3.Connection, schema: SchemaRegistry):
        self.conn = conn
        self.schema = schema

    def compute_hash(self, tables: list[str] | None = None) -> str:
        """Compute canonical hash of current database state.

        Args:
            tables: Optional list of tables to hash. If None, hashes all
                    registered tables.

        Returns:
            Hex string of the SHA-256 hash.
        """
        hasher = hashlib.sha256()

        target_tables = sorted(tables or self.schema.get_all_tables())

        for table in target_tables:
            if not self.schema.is_registered(table):
                continue

            # Hash the table name itself (so empty tables still contribute)
            hasher.update(f"TABLE:{table}\n".encode("utf-8"))

            rows = self._get_sorted_rows(table)
            for row in rows:
                serialized = self._deterministic_serialize(row)
                hasher.update(serialized.encode("utf-8"))
                hasher.update(b"\n")  # Row separator

        return hasher.hexdigest()

    def compute_table_hash(self, table: str) -> str:
        """Compute hash for a single table.

        Useful for identifying which table has diverged.

        Args:
            table: Table name.

        Returns:
            Hex string of the SHA-256 hash for this table.
        """
        hasher = hashlib.sha256()
        hasher.update(f"TABLE:{table}\n".encode("utf-8"))

        rows = self._get_sorted_rows(table)
        for row in rows:
            serialized = self._deterministic_serialize(row)
            hasher.update(serialized.encode("utf-8"))
            hasher.update(b"\n")

        return hasher.hexdigest()

    def verify_with_peer(self, peer_hash: str) -> ConvergenceResult:
        """Compare local hash with a peer's hash.

        Args:
            peer_hash: The peer's canonical hash string.

        Returns:
            ConvergenceResult indicating whether states match.
        """
        local_hash = self.compute_hash()
        return ConvergenceResult(
            converged=(local_hash == peer_hash),
            local_hash=local_hash,
            peer_hash=peer_hash,
        )

    def find_divergent_tables(self, peer_table_hashes: dict[str, str]) -> list[str]:
        """Find which tables have diverged from a peer.

        Args:
            peer_table_hashes: Dict mapping table name to peer's hash.

        Returns:
            List of table names where hashes don't match.
        """
        divergent = []
        for table in self.schema.get_all_tables():
            local = self.compute_table_hash(table)
            peer = peer_table_hashes.get(table, "")
            if local != peer:
                divergent.append(table)
        return divergent

    def _get_sorted_rows(self, table: str) -> list[dict]:
        """Get all non-tombstoned rows from a table, sorted by PK.

        Excludes rows that have unresolved tombstones.
        """
        pk_col = self.schema.get_primary_key(table)

        cursor = self.conn.execute(
            f"""SELECT * FROM {table}
                WHERE NOT EXISTS (
                    SELECT 1 FROM _tombstones 
                    WHERE table_name = ? 
                    AND row_id = {table}.{pk_col}
                    AND is_resolved = 0
                )
                ORDER BY {pk_col}""",
            (table,),
        )

        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    @staticmethod
    def _deterministic_serialize(row: dict) -> str:
        """JSON serialize with sorted keys and consistent type handling.

        Ensures that two identical rows always produce the same string,
        regardless of dict insertion order or Python version.
        """
        return json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
            separators=(",", ":"),  # Compact format, no extra whitespace
        )
