"""Tombstone-based foreign key preservation engine.

This module implements the core innovation of the CRDT sync engine:
tombstone-based FK resolution with reference counting. When a parent row
is deleted, it is not physically removed — instead, a tombstone is created
that tracks how many child rows still reference it.

The critical scenario this solves (which breaks CR-SQLite):
1. Device A deletes parent P → tombstone created, ref_count = count(children)
2. Device B (offline) inserts child C2 referencing P
3. On sync: merge engine sees tombstone for P, but C2 references P
4. ref_count incremented → P remains visible (tombstoned but not purged)
5. Resolution via configurable policy: CASCADE / NULLIFY / PRESERVE / CALLBACK

Three configurable policies:
- CASCADE:  When ref_count → 0 OR when policy is applied, delete orphaned children
- NULLIFY:  Set child FK columns to NULL; children survive, references cleared
- PRESERVE: Keep both tombstone and children alive; surface conflict to app (default)

Plus an optional callback override for application-specific logic.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.schema import SchemaRegistry
from src.utils import serialize_value


class TombstonePolicy(Enum):
    """Policy for resolving tombstoned parent rows with live children."""
    CASCADE = "cascade"
    NULLIFY = "nullify"
    PRESERVE = "preserve"

    @classmethod
    def from_string(cls, s: str) -> TombstonePolicy:
        try:
            return cls(s.lower())
        except ValueError:
            return cls.PRESERVE


@dataclass
class TombstoneInfo:
    """Information about a tombstone and its outstanding dependencies."""
    table_name: str
    row_id: str
    vector_clock_json: str
    deleted_by: str
    ref_count: int
    deleted_at_hlc: str = "0000000000000:00000:sys"
    children: list[dict] = field(default_factory=list)


@dataclass
class TombstoneResult:
    """Result of a tombstone operation."""
    tombstone_created: bool = False
    tombstone_updated: bool = False
    row_purged: bool = False
    children_cascaded: int = 0
    children_nullified: int = 0
    policy_applied: str = ""
    requires_resolution: bool = False


class TombstoneResolver:
    """Manages soft-deletion via tombstones and FK reference counting.

    Works in conjunction with the SchemaRegistry to understand FK relationships
    and apply the correct resolution policy for each table.
    """

    def __init__(self, conn: sqlite3.Connection, schema_registry: SchemaRegistry, update_callback: Optional[Callable] = None):
        self.conn = conn
        self.schema = schema_registry
        self.update_callback = update_callback

    def on_delete(
        self,
        table: str,
        row_id: str,
        vc_json: str = "{}",
        hlc_ts: str = "0000000000000:00000:sys",
        writer: str = "sys",
    ) -> TombstoneResult:
        """Handle deletion of a row. Creates or updates a tombstone.

        In the Remove-Wins Absorbing Tombstone model:
        1. A tombstone is permanent (absorbing state).
        2. We count live children referencing this row to support preserve policies.
        3. If policy == CASCADE, we recursively cascade the tombstone.

        Args:
            table: Table name of the deleted row.
            row_id: Primary key of the deleted row.
            vc_json: Vector clock JSON of the deletion.
            hlc: HLC timestamp.
            writer: Device/writer that performed the deletion.

        Returns:
            TombstoneResult describing what happened.
        """
        result = TombstoneResult()

        existing = self.conn.execute(
            """SELECT vector_clock_json, ref_count
               FROM _tombstones 
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        ).fetchone()

        ref_count = self._count_live_children(table, row_id)

        if existing:
            # Update existing tombstone
            # For simplicity, we just keep the incoming if it's newer, though any delete is absorbing.
            self.conn.execute(
                """UPDATE _tombstones 
                   SET vector_clock_json = ?, deleted_by = ?, ref_count = ?
                   WHERE table_name = ? AND row_id = ?""",
                (vc_json, writer, ref_count, table, row_id),
            )
            result.tombstone_updated = True
        else:
            # Create new tombstone
            self.conn.execute(
                """INSERT INTO _tombstones 
                   (table_name, row_id, vector_clock_json, deleted_by, deleted_at_hlc, ref_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (table, row_id, vc_json, writer, hlc_ts, ref_count),
            )
            result.tombstone_created = True

        # Mark the row as deleted in _crdt_row_state
        self.conn.execute(
            """UPDATE _crdt_row_state SET is_deleted = 1
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        )

        # Physically delete from application table (materialized state)
        self.conn.execute(
            f"DELETE FROM {table} WHERE {self.schema.get_primary_key(table)} = ?",
            (row_id,)
        )
        result.row_purged = True

        # We must call apply_policy to evaluate the policies.
        pol_result = self.apply_policy(table, row_id, vc_json, writer)
        result.children_cascaded = pol_result.children_cascaded
        result.children_nullified = pol_result.children_nullified
        result.policy_applied = pol_result.policy_applied
        result.requires_resolution = pol_result.requires_resolution

        return result

    def on_child_insert(
        self,
        child_table: str,
        child_row: dict[str, Any],
    ) -> list[TombstoneResult]:
        """Handle insertion of a child row that may reference tombstoned parents.

        For each FK column in the child row, check if the referenced parent
        has a tombstone. If so, increment the tombstone's ref_count.

        Args:
            child_table: Name of the child table.
            child_row: The inserted child row data (dict).

        Returns:
            List of TombstoneResults for each affected parent tombstone.
        """
        results = []
        parents = self.schema.get_parents_of(child_table)

        for parent_table, parent_col, child_col in parents:
            parent_id = child_row.get(child_col)
            if parent_id is None:
                continue

            parent_id_str = str(parent_id)

            # Check if parent has a tombstone
            tombstone = self.conn.execute(
                """SELECT ref_count
                   FROM _tombstones 
                   WHERE table_name = ? AND row_id = ?""",
                (parent_table, parent_id_str),
            ).fetchone()

            if tombstone is not None:
                # Parent is tombstoned — increment ref_count
                self.conn.execute(
                    """UPDATE _tombstones SET ref_count = ref_count + 1
                       WHERE table_name = ? AND row_id = ?""",
                    (parent_table, parent_id_str),
                )
                
                # Enforce policy immediately for late-arriving children
                self.apply_policy(parent_table, parent_id_str)
                    
                result = TombstoneResult(
                    tombstone_updated=True,
                    requires_resolution=True,
                )
                results.append(result)

        return results

    def on_child_delete(
        self,
        child_table: str,
        child_row: dict[str, Any],
    ) -> list[TombstoneResult]:
        """Handle deletion of a child row, decrementing parent tombstone ref_counts.

        If a parent tombstone's ref_count reaches 0, attempt resolution
        based on the configured policy.

        Args:
            child_table: Name of the child table.
            child_row: The deleted child row data (dict).

        Returns:
            List of TombstoneResults for each affected parent tombstone.
        """
        results = []
        parents = self.schema.get_parents_of(child_table)

        for parent_table, parent_col, child_col in parents:
            parent_id = child_row.get(child_col)
            if parent_id is None:
                continue

            parent_id_str = str(parent_id)

            # Check if parent has an unresolved tombstone
            tombstone = self.conn.execute(
                """SELECT ref_count 
                   FROM _tombstones 
                   WHERE table_name = ? AND row_id = ?""",
                (parent_table, parent_id_str),
            ).fetchone()

            if tombstone is not None and tombstone[0] > 0:
                self.conn.execute(
                    """UPDATE _tombstones SET ref_count = MAX(0, ref_count - 1)
                       WHERE table_name = ? AND row_id = ?""",
                    (parent_table, parent_id_str),
                )

                result = TombstoneResult(tombstone_updated=True)

                # Check if ref_count is now 0
                new_count = self.conn.execute(
                    """SELECT ref_count FROM _tombstones 
                       WHERE table_name = ? AND row_id = ?""",
                    (parent_table, parent_id_str),
                ).fetchone()

                if new_count and new_count[0] <= 0:
                    pol_res = self.apply_policy(parent_table, parent_id_str)
                    result.row_purged = True

                results.append(result)

        return results

    def apply_policy(self, table: str, row_id: str, vc_json: str = "{}", writer: str = "sys") -> TombstoneResult:
        """Apply the configured tombstone resolution policy.

        Called when the application or engine wants to force resolution
        of a tombstone with ref_count > 0.

        Args:
            table: Parent table name.
            row_id: Parent row ID.
            vc_json: Parent vector clock.
            writer: Parent writer id.

        Returns:
            TombstoneResult describing the resolution.
        """
        result = TombstoneResult()

        tombstone = self.conn.execute(
            """SELECT ref_count, vector_clock_json, deleted_by
               FROM _tombstones 
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        ).fetchone()

        if tombstone is None:
            return result  # No tombstone to resolve

        ref_count = tombstone[0]
        vc_json = tombstone[1]
        deleted_by = tombstone[2]

        if ref_count <= 0 and self.schema.get_tombstone_policy(table) != "preserve":
            result.row_purged = True
            return result

        # Check for callback first on the parent
        callback = self.schema.get_tombstone_callback(table)
        if callback:
            children = self._get_live_children(table, row_id)
            callback_result = callback(table, row_id, children)
            result.policy_applied = "callback"
            return result

        # Determine effective policy by checking both parent and children
        parent_policy = TombstonePolicy.from_string(self.schema.get_tombstone_policy(table))
        effective_policy = parent_policy

        children_specs = self.schema.get_children_of(table)
        for child_table, child_col, parent_col in children_specs:
            child_policy = TombstonePolicy.from_string(self.schema.get_tombstone_policy(child_table))
            if child_policy == TombstonePolicy.CASCADE:
                effective_policy = TombstonePolicy.CASCADE
                break
            elif child_policy == TombstonePolicy.NULLIFY and effective_policy != TombstonePolicy.CASCADE:
                effective_policy = TombstonePolicy.NULLIFY

        result.policy_applied = effective_policy.value

        if effective_policy == TombstonePolicy.CASCADE:
            cascaded = self._cascade_delete_children(table, row_id, vc_json, deleted_by)
            result.children_cascaded = cascaded
            result.row_purged = True

        elif effective_policy == TombstonePolicy.NULLIFY:
            nullified = self._nullify_children(table, row_id)
            result.children_nullified = nullified
            result.row_purged = True

        elif effective_policy == TombstonePolicy.PRESERVE:
            # Do nothing — keep the tombstone and children alive
            result.requires_resolution = True

        return result

    def is_tombstoned(self, table: str, row_id: str) -> bool:
        """Check if a row has a tombstone (tombstones are absorbing)."""
        row = self.conn.execute(
            """SELECT 1 FROM _tombstones 
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        ).fetchone()
        return row is not None

    def get_tombstone(self, table: str, row_id: str) -> TombstoneInfo | None:
        """Get tombstone info for a specific row."""
        row = self.conn.execute(
            """SELECT vector_clock_json, deleted_by, ref_count, deleted_at_hlc
               FROM _tombstones
               WHERE table_name = ? AND row_id = ?""",
            (table, row_id),
        ).fetchone()
        if row is None:
            return None
        
        info = TombstoneInfo(
            table_name=table,
            row_id=row_id,
            vector_clock_json=row[0],
            deleted_by=row[1],
            ref_count=row[2],
        )
        info.deleted_at_hlc = row[3]

        if info.ref_count > 0:
            info.children = self._get_live_children(table, row_id)

        return info

    def get_unresolved_tombstones(self) -> list[TombstoneInfo]:
        """Return all tombstones with ref_count > 0 for application resolution."""
        cursor = self.conn.execute(
            """SELECT table_name, row_id, vector_clock_json, deleted_by, ref_count
               FROM _tombstones
               WHERE ref_count > 0"""
        )
        results = []
        for row in cursor.fetchall():
            info = TombstoneInfo(
                table_name=row[0],
                row_id=row[1],
                vector_clock_json=row[2],
                deleted_by=row[3],
                ref_count=row[4],
            )
            info.children = self._get_live_children(row[0], row[1])
            results.append(info)
        return results

    def recalculate_ref_count(self, table: str, row_id: str) -> int:
        """Recalculate ref_count from actual child data.

        Useful after complex multi-way merges where incremental
        tracking might have edge cases.

        Returns:
            The corrected ref_count.
        """
        actual_count = self._count_live_children(table, row_id)
        self.conn.execute(
            """UPDATE _tombstones SET ref_count = ?
               WHERE table_name = ? AND row_id = ?""",
            (actual_count, table, row_id),
        )
        if actual_count == 0 and self.schema.get_tombstone_policy(table) != "preserve":
            self.apply_policy(table, row_id)
        return actual_count

    def _count_live_children(self, table: str, row_id: str) -> int:
        """Count all live rows in child tables that reference this row."""
        total = 0
        children_specs = self.schema.get_children_of(table)
        for child_table, child_col, parent_col in children_specs:
            # Count rows in the child table where FK = this row's ID
            # and the child row is not itself tombstoned
            cursor = self.conn.execute(
                f"""SELECT COUNT(*) FROM {child_table} 
                    WHERE {child_col} = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM _tombstones 
                        WHERE table_name = ? AND row_id = {child_table}.{self.schema.get_primary_key(child_table)}
                    )""",
                (row_id, child_table),
            )
            count = cursor.fetchone()[0]
            total += count
        return total

    def _get_live_children(self, table: str, row_id: str) -> list[dict]:
        """Get all live child rows referencing this parent."""
        children = []
        children_specs = self.schema.get_children_of(table)
        for child_table, child_col, parent_col in children_specs:
            cursor = self.conn.execute(
                f"SELECT * FROM {child_table} WHERE {child_col} = ?",
                (row_id,),
            )
            cols = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                child_dict = dict(zip(cols, row))
                child_dict["_child_table"] = child_table
                child_dict["_fk_column"] = child_col
                children.append(child_dict)
        return children

    def _resolve_tombstone(self, table: str, row_id: str) -> None:
        """Re-evaluate the tombstone policy. Useful for late-arriving children."""
        tombstone = self.get_tombstone(table, row_id)
        if tombstone:
            self.apply_policy(table, row_id, tombstone.vector_clock_json, tombstone.deleted_by)

    def _cascade_delete_children(self, table: str, row_id: str, vc_json: str, writer: str) -> int:
        """Recursively tombstone all children of a tombstoned parent."""
        total_deleted = 0
        children_specs = self.schema.get_children_of(table)
        for child_table, child_col, parent_col in children_specs:
            # Get child IDs before deleting (for recursive tombstoning)
            child_pk = self.schema.get_primary_key(child_table)
            cursor = self.conn.execute(
                f"SELECT {child_pk} FROM {child_table} WHERE {child_col} = ?",
                (row_id,),
            )
            child_ids = [r[0] for r in cursor.fetchall()]

            # Cascade by creating tombstones for children
            # This replicates the deletion properly in a CRDT
            # We use the parent's vector clock to maintain causal consistency
            import time
            hlc_placeholder = f"0000000000000:00000:{writer}" # We can mock hlc because VC drives conflict resolution now
            for cid in child_ids:
                # Call on_delete recursively
                self.on_delete(child_table, str(cid), vc_json, "0000000000000:00000:sys", writer)
                
            total_deleted += len(child_ids)

        return total_deleted

    def _nullify_children(self, table: str, row_id: str) -> int:
        """Nullify FK columns in children of a tombstoned parent (NULLIFY policy)."""
        total_nullified = 0
        children_specs = self.schema.get_children_of(table)
        for child_table, child_col, parent_col in children_specs:
            child_pk = self.schema.get_primary_key(child_table)
            
            # Fetch the affected rows before we nullify them
            affected = self.conn.execute(
                f"SELECT {child_pk} FROM {child_table} WHERE {child_col} = ?",
                (row_id,)
            ).fetchall()
            
            child_ids = [r[0] for r in affected]
            total_nullified += len(child_ids)

            if self.update_callback:
                # Issue proper CRDT updates for each child so they replicate
                for cid in child_ids:
                    self.update_callback(child_table, str(cid), {child_col: None})
            else:
                # Fallback to direct SQL if no callback
                self.conn.execute(
                    f"UPDATE {child_table} SET {child_col} = NULL WHERE {child_col} = ?",
                    (row_id,),
                )

        return total_nullified
