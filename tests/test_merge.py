"""Tests for the cell-level merge engine."""

import sqlite3
import pytest

from src.hlc import HLC, HLCTimestamp
from src.merge import CellMerger, MergeAction


@pytest.fixture
def merge_db():
    """Create an in-memory database with CRDT schema for merge testing."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
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
    """)
    yield conn
    conn.close()


@pytest.fixture
def merger(merge_db):
    """Create a CellMerger instance."""
    return CellMerger(merge_db)


class TestMergeCell:
    """Tests for individual cell merging."""

    def test_no_local_version(self, merger):
        """First write to a cell should always be accepted."""
        result = merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        assert result.action == MergeAction.NO_CONFLICT
        assert result.winning_value == "Alice"
        assert result.is_new_row is True

    def test_incoming_wins_higher_hlc(self, merger):
        """Incoming value with higher HLC should win."""
        merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        result = merger.merge_cell(
            "patients", "p1", "name",
            "Bob", "{}", "0000000002000:00000:device_B", "device_B"
        )
        assert result.action == MergeAction.ACCEPTED
        assert result.winning_value == "Bob"

    def test_local_wins_higher_hlc(self, merger):
        """Local value with higher HLC should win."""
        merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000002000:00000:device_A", "device_A"
        )
        result = merger.merge_cell(
            "patients", "p1", "name",
            "Bob", "{}", "0000000001000:00000:device_B", "device_B"
        )
        assert result.action == MergeAction.REJECTED
        assert result.winning_value == "Alice"

    def test_tiebreak_by_writer_id(self, merger):
        """Equal HLCs should be broken by lower writer_id."""
        merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_B", "device_B"
        )
        result = merger.merge_cell(
            "patients", "p1", "name",
            "Bob", "{}", "0000000001000:00000:device_A", "device_A"
        )
        # device_A < device_B, so incoming (device_A) wins
        assert result.action == MergeAction.ACCEPTED
        assert result.winning_value == "Bob"

    def test_identical_values(self, merger):
        """Identical cell writes should be detected."""
        merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        result = merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        assert result.action == MergeAction.IDENTICAL

    def test_different_columns_no_conflict(self, merger):
        """Edits to different columns of the same row should never conflict."""
        result_a = merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        result_b = merger.merge_cell(
            "patients", "p1", "age",
            "30", "{}", "0000000001000:00000:device_B", "device_B"
        )
        assert result_a.action == MergeAction.NO_CONFLICT
        assert result_b.action == MergeAction.NO_CONFLICT

        # Both values should be preserved
        cells = merger.get_winning_cells("patients", "p1")
        assert cells["name"] == "Alice"
        assert cells["age"] == "30"

    def test_null_value(self, merger):
        """NULL values should be handled correctly."""
        result = merger.merge_cell(
            "patients", "p1", "middle_name",
            None, "{}", "0000000001000:00000:device_A", "device_A"
        )
        assert result.action == MergeAction.NO_CONFLICT
        cells = merger.get_winning_cells("patients", "p1")
        assert cells["middle_name"] is None


class TestMergeRow:
    """Tests for row-level merging (all cells at once)."""

    def test_merge_new_row(self, merger):
        """Merging a completely new row."""
        result = merger.merge_row(
            "patients", "p1",
            {"name": "Alice", "age": "30", "city": "London"},
            "{}", "0000000001000:00000:device_A", "device_A"
        )
        assert result.is_new_row is True
        assert len(result.cell_results) == 3

    def test_merge_partial_update(self, merger):
        """Merging updates to some columns of an existing row."""
        merger.merge_row(
            "patients", "p1",
            {"name": "Alice", "age": "30", "city": "London"},
            "{}", "0000000001000:00000:device_A", "device_A"
        )
        result = merger.merge_row(
            "patients", "p1",
            {"age": "31"},
            "{}", "0000000002000:00000:device_B", "device_B"
        )
        assert result.accepted_count == 1
        cells = merger.get_winning_cells("patients", "p1")
        assert cells["name"] == "Alice"  # Unchanged
        assert cells["age"] == "31"  # Updated
        assert cells["city"] == "London"  # Unchanged


class TestIdempotency:
    """Tests for merge idempotency."""

    def test_applying_same_delta_twice(self, merger):
        """Applying the same merge twice should produce no change."""
        merger.merge_cell(
            "patients", "p1", "name",
            "Alice", "{}", "0000000001000:00000:device_A", "device_A"
        )
        result1 = merger.merge_cell(
            "patients", "p1", "name",
            "Bob", "{}", "0000000002000:00000:device_B", "device_B"
        )
        result2 = merger.merge_cell(
            "patients", "p1", "name",
            "Bob", "{}", "0000000002000:00000:device_B", "device_B"
        )
        assert result1.action == MergeAction.ACCEPTED
        assert result2.action == MergeAction.IDENTICAL


class TestCellVersionHistory:
    """Tests for version tracking."""

    def test_all_versions_preserved(self, merger):
        """All cell versions should be stored, not just the winner."""
        merger.merge_cell("t", "r1", "c1", "v1", "{}", "0000000001000:00000:A", "A")
        merger.merge_cell("t", "r1", "c1", "v2", "{}", "0000000002000:00000:B", "B")
        merger.merge_cell("t", "r1", "c1", "v3", "{}", "0000000000500:00000:C", "C")

        versions = merger.get_all_cell_versions("t", "r1", "c1")
        assert len(versions) == 3
        
        winners = [v for v in versions if v["is_winner"]]
        assert len(winners) == 1
        assert winners[0]["value"] == "v2"  # Highest HLC

    def test_commutativity(self, merge_db):
        """merge(A, B) should produce the same result as merge(B, A)."""
        # Forward order
        merger1 = CellMerger(merge_db)
        merger1.merge_cell("t", "r1", "c1", "v1", "{}", "0000000001000:00000:device_A", "device_A")
        merger1.merge_cell("t", "r1", "c1", "v2", "{}", "0000000002000:00000:device_B", "device_B")
        forward = merger1.get_winning_cells("t", "r1")

        # Reverse order - new db
        conn2 = sqlite3.connect(":memory:")
        conn2.executescript("""
            CREATE TABLE IF NOT EXISTS _crdt_cells (
                table_name TEXT NOT NULL, row_id TEXT NOT NULL,
                col_name TEXT NOT NULL, writer_id TEXT NOT NULL,
                value TEXT, vector_clock_json TEXT NOT NULL, hlc_ts TEXT NOT NULL,
                is_winner INTEGER DEFAULT 1,
                PRIMARY KEY (table_name, row_id, col_name, writer_id)
            );
            CREATE INDEX IF NOT EXISTS idx_crdt_cells_lookup
                ON _crdt_cells(table_name, row_id, col_name, is_winner);
        """)
        merger2 = CellMerger(conn2)
        merger2.merge_cell("t", "r1", "c1", "v2", "{}", "0000000002000:00000:device_B", "device_B")
        merger2.merge_cell("t", "r1", "c1", "v1", "{}", "0000000001000:00000:device_A", "device_A")
        reverse = merger2.get_winning_cells("t", "r1")
        conn2.close()

        assert forward == reverse
