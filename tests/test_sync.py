"""Tests for sync protocol and multi-device convergence."""

import pytest

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.convergence import ConvergenceHasher


@pytest.fixture
def two_engines():
    """Two engines with matching schema."""
    engines = []
    for node_id in ("device_A", "device_B"):
        e = CRDTEngine(":memory:", node_id)
        e.register_table("doctors", primary_key="id",
                         columns=["id", "name", "specialty"])
        e.register_table("patients", primary_key="id",
                         columns=["id", "name", "nhs_number", "doctor_id"],
                         foreign_keys=[("doctor_id", "doctors", "id")],
                         unique_cols=["nhs_number"],
                         on_tombstone_policy="preserve")
        engines.append(e)
    yield engines
    for e in engines:
        e.close()


@pytest.fixture
def five_engines():
    """Five engines for stress-like sync tests."""
    engines = []
    for i in range(5):
        e = CRDTEngine(":memory:", f"device_{i}")
        e.register_table("items", primary_key="id",
                         columns=["id", "name", "value"])
        engines.append(e)
    yield engines
    for e in engines:
        e.close()


class TestLocalSyncBridge:
    """Tests for the in-process sync bridge."""

    def test_basic_one_way_sync(self, two_engines):
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        rows = b.query("doctors")
        assert len(rows) == 1
        assert rows[0]["name"] == "Dr. Smith"

    def test_bidirectional_sync(self, two_engines):
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        b.insert("doctors", {"id": "d2", "name": "Dr. Jones", "specialty": "Cardio"})

        bridge.sync_bidirectional(a, b)

        assert len(a.query("doctors")) == 2
        assert len(b.query("doctors")) == 2

    def test_sync_preserves_cell_level_edits(self, two_engines):
        """Two devices editing different columns of the same row."""
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        # A edits name, B edits specialty — offline
        a.update("doctors", "d1", {"name": "Dr. Smithson"})
        b.update("doctors", "d1", {"specialty": "Cardiology"})

        bridge.sync_bidirectional(a, b)

        # Both changes should be preserved (cell-level merge)
        for engine in (a, b):
            rows = engine.query("doctors")
            assert len(rows) == 1
            assert rows[0]["name"] == "Dr. Smithson"
            assert rows[0]["specialty"] == "Cardiology"

    def test_sync_all_full_mesh(self, five_engines):
        """Full mesh sync converges all 5 devices."""
        bridge = LocalSyncBridge()
        for e in five_engines:
            bridge.register_peer(e)

        # Each device inserts a unique item
        for i, e in enumerate(five_engines):
            e.insert("items", {"id": f"item_{i}", "name": f"Item {i}", "value": str(i)})

        # Full mesh sync (may need 2 rounds for transitive propagation)
        bridge.sync_all()
        bridge.sync_all()

        # All devices should have all 5 items
        for e in five_engines:
            rows = e.query("items")
            assert len(rows) == 5, f"Device {e.node_id} has {len(rows)} items, expected 5"

    def test_sync_stats_tracking(self, two_engines):
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        assert bridge.stats.total_syncs == 1
        assert bridge.stats.total_cells_synced > 0

    def test_incremental_delta(self, two_engines):
        """Second sync should only send new changes."""
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        # Insert another doctor
        a.insert("doctors", {"id": "d2", "name": "Dr. Jones", "specialty": "Cardio"})
        bridge.sync(a, b)

        assert len(b.query("doctors")) == 2


class TestConvergence:
    """Tests for convergence hash verification."""

    def test_identical_state_same_hash(self, two_engines):
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        hasher_a = ConvergenceHasher(a.conn, a.schema)
        hasher_b = ConvergenceHasher(b.conn, b.schema)

        assert hasher_a.compute_hash() == hasher_b.compute_hash()

    def test_divergent_state_different_hash(self, two_engines):
        a, b = two_engines

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        # Don't sync — they should diverge

        hasher_a = ConvergenceHasher(a.conn, a.schema)
        hasher_b = ConvergenceHasher(b.conn, b.schema)

        assert hasher_a.compute_hash() != hasher_b.compute_hash()

    def test_convergence_after_complex_merge(self, two_engines):
        """Both devices do concurrent inserts+updates, then sync."""
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        # Seed shared data
        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        # Concurrent operations
        a.update("doctors", "d1", {"name": "Dr. Smithson"})
        a.insert("doctors", {"id": "d2", "name": "Dr. A", "specialty": "X"})
        b.update("doctors", "d1", {"specialty": "Cardiology"})
        b.insert("doctors", {"id": "d3", "name": "Dr. B", "specialty": "Y"})

        # Sync and verify convergence
        bridge.sync_bidirectional(a, b)
        bridge.sync_bidirectional(a, b)

        hasher_a = ConvergenceHasher(a.conn, a.schema)
        hasher_b = ConvergenceHasher(b.conn, b.schema)
        assert hasher_a.compute_hash() == hasher_b.compute_hash()

    def test_five_device_convergence(self, five_engines):
        """All 5 devices converge after full mesh sync."""
        bridge = LocalSyncBridge()
        for e in five_engines:
            bridge.register_peer(e)

        for i, e in enumerate(five_engines):
            e.insert("items", {"id": f"item_{i}", "name": f"Item {i}", "value": str(i * 10)})

        bridge.sync_all()
        bridge.sync_all()

        hashes = set()
        for e in five_engines:
            hasher = ConvergenceHasher(e.conn, e.schema)
            hashes.add(hasher.compute_hash())

        assert len(hashes) == 1, f"Expected 1 unique hash, got {len(hashes)}"

    def test_table_level_hash(self, two_engines):
        a, b = two_engines
        bridge = LocalSyncBridge()
        bridge.register_peer(a)
        bridge.register_peer(b)

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync(a, b)

        hasher_a = ConvergenceHasher(a.conn, a.schema)
        hasher_b = ConvergenceHasher(b.conn, b.schema)

        assert hasher_a.compute_table_hash("doctors") == hasher_b.compute_table_hash("doctors")

    def test_find_divergent_tables(self, two_engines):
        a, b = two_engines

        a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        # Don't sync — doctors table should diverge

        hasher_a = ConvergenceHasher(a.conn, a.schema)
        hasher_b = ConvergenceHasher(b.conn, b.schema)

        peer_hashes = {t: hasher_a.compute_table_hash(t) for t in a.schema.get_all_tables()}
        divergent = hasher_b.find_divergent_tables(peer_hashes)

        assert "doctors" in divergent
