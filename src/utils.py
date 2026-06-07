"""Utility functions for the CRDT sync engine.

Provides deterministic serialization, row sorting, and helper functions
used across the merge engine, convergence hasher, and sync protocol.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def generate_row_id() -> str:
    """Generate a globally unique row ID.

    Uses UUID4 for collision resistance across disconnected devices.

    Returns:
        A string UUID suitable for use as a primary key.
    """
    return str(uuid.uuid4())


def serialize_value(value: Any) -> str | None:
    """Serialize a Python value to a JSON string for storage in _crdt_cells.

    Args:
        value: Any JSON-serializable Python value.

    Returns:
        JSON string representation, or None if value is None.
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def deserialize_value(json_str: str | None) -> Any:
    """Deserialize a JSON string from _crdt_cells back to a Python value.

    Args:
        json_str: JSON string, or None.

    Returns:
        The deserialized Python value, or None.
    """
    if json_str is None:
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return json_str


def serialize_row(row: dict[str, Any]) -> str:
    """Deterministically serialize a row dict for hashing or transport.

    Keys are sorted alphabetically. Values are converted to their
    JSON representations. This ensures two identical rows always
    produce the same serialized string regardless of dict insertion order.

    Args:
        row: A dictionary representing a database row.

    Returns:
        A deterministic JSON string.
    """
    return json.dumps(row, sort_keys=True, ensure_ascii=True, default=str)


def deserialize_row(json_str: str) -> dict[str, Any]:
    """Deserialize a JSON string back to a row dict.

    Args:
        json_str: JSON string produced by serialize_row().

    Returns:
        The row as a dictionary.
    """
    return json.loads(json_str)


def serialize_pk(pk_value: Any) -> str:
    """Serialize a primary key value to a string for use in _crdt_cells.

    Handles both simple PKs (single value) and composite PKs (tuple/list).

    Args:
        pk_value: The primary key value (string, int, or tuple for composite).

    Returns:
        String representation suitable for storage.
    """
    if isinstance(pk_value, (list, tuple)):
        return json.dumps(list(pk_value), sort_keys=True, ensure_ascii=True)
    return str(pk_value)


def deserialize_pk(pk_str: str) -> str | list:
    """Deserialize a primary key string.

    Args:
        pk_str: String from _crdt_cells.row_id.

    Returns:
        The primary key value (string or list for composite).
    """
    try:
        parsed = json.loads(pk_str)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return pk_str


def build_row_from_cells(cells: list[dict]) -> dict[str, Any]:
    """Reconstruct a row dict from a list of winning cell records.

    Args:
        cells: List of dicts with keys 'col_name' and 'value'.
               Only winning cells (is_winner=1) should be passed.

    Returns:
        A dictionary mapping column names to their deserialized values.
    """
    row = {}
    for cell in cells:
        col = cell["col_name"]
        value = cell.get("value")
        row[col] = deserialize_value(value)
    return row


def rows_to_sorted_list(rows: list[dict], sort_key: str = "id") -> list[dict]:
    """Sort a list of row dicts by a given key for deterministic ordering.

    Args:
        rows: List of row dictionaries.
        sort_key: Column name to sort by.

    Returns:
        Sorted list of rows.
    """
    return sorted(rows, key=lambda r: str(r.get(sort_key, "")))


def dict_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compute the changed columns between two row dicts.

    Args:
        old: The previous row state.
        new: The current row state.

    Returns:
        A dict containing only the columns whose values changed.
    """
    changed = {}
    all_keys = set(old.keys()) | set(new.keys())
    for key in all_keys:
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            changed[key] = new_val
    return changed
