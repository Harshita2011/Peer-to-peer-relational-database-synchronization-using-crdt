from typing import Any, Dict, List
import json
import sqlite3

from src.engine import CRDTEngine
from src.convergence import ConvergenceHasher


class Adapter:
    def __init__(self, peer_id: str, db_path: str = ":memory:"):
        self.peer_id = peer_id
        self.engine = CRDTEngine(db_path, peer_id)
        self.hasher = ConvergenceHasher(self.engine.conn, self.engine.schema)

    def apply_schema(self, schema_def: Dict[str, Any]) -> None:
        """
        schema_def example:
        {
            "table_name": "users",
            "primary_key": "id",
            "columns": ["id", "name", "email"],
            "foreign_keys": [],
            "unique_cols": ["email"],
            "on_tombstone_policy": "cascade"
        }
        """
        # Handling list of schemas or single schema dict
        if isinstance(schema_def, list):
            schemas = schema_def
        else:
            schemas = [schema_def]

        for s in schemas:
            self.engine.register_table(
                table_name=s.get("table_name"),
                primary_key=s.get("primary_key", "id"),
                columns=s.get("columns", []),
                foreign_keys=s.get("foreign_keys", []),
                unique_cols=s.get("unique_cols", []),
                on_tombstone_policy=s.get("on_tombstone_policy", "preserve")
            )

    def execute(self, sql: str, params: tuple = ()) -> None:
        """
        Parses simple INSERT, UPDATE, DELETE to route through engine.
        Note: This is a simplistic parser for benchmark purposes.
        """
        sql_upper = sql.upper().strip()
        if sql_upper.startswith("INSERT"):
            # Minimal parsing for INSERT INTO table (col1, col2) VALUES (?, ?)
            parts = sql.split("(")
            table_part = parts[0].split()
            table_name = table_part[-1]
            cols_part = parts[1].split(")")[0]
            cols = [c.strip() for c in cols_part.split(",")]
            
            row = dict(zip(cols, params))
            self.engine.insert(table_name, row)

        elif sql_upper.startswith("UPDATE"):
            # Minimal parsing for UPDATE table SET col1 = ? WHERE pk = ?
            parts = sql.split(" SET ")
            table_name = parts[0].split()[-1]
            set_and_where = parts[1].split(" WHERE ")
            set_part = set_and_where[0]
            where_part = set_and_where[1]
            
            set_cols = [c.split("=")[0].strip() for c in set_part.split(",")]
            changes = dict(zip(set_cols, params[:-1]))
            row_id = params[-1]
            self.engine.update(table_name, str(row_id), changes)

        elif sql_upper.startswith("DELETE"):
            # Minimal parsing for DELETE FROM table WHERE pk = ?
            parts = sql.split(" WHERE ")
            table_name = parts[0].split()[-1]
            row_id = params[0]
            self.engine.delete(table_name, str(row_id))
        else:
            self.engine.query_raw(sql, params)

    def sync(self, peer_b: 'Adapter') -> None:
        """Bidirectional sync between self and peer_b."""
        delta_a = self.engine.get_delta(since_seq=0, for_peer=peer_b.peer_id)
        delta_b = peer_b.engine.get_delta(since_seq=0, for_peer=self.peer_id)
        
        peer_b.engine.apply_delta(delta_a, from_peer=self.peer_id)
        self.engine.apply_delta(delta_b, from_peer=peer_b.peer_id)

    def snapshot_hash(self) -> str:
        return self.hasher.compute_hash()

    def snapshot_state(self) -> Dict[str, List[Dict[str, Any]]]:
        state = {}
        for table in self.engine.schema.get_all_tables():
            rows = self.engine.query(table)
            # Ensure deterministic sorting
            pk = self.engine.schema.get_primary_key(table)
            rows.sort(key=lambda x: str(x[pk]))
            state[table] = rows
        return state

    def close(self) -> None:
        self.engine.close()


def open_peer(peer_id: str, db_path: str = ":memory:") -> Adapter:
    return Adapter(peer_id, db_path)
