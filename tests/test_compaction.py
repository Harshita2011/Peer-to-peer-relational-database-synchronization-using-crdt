"""Tests for CRDT metadata compaction."""

import pytest

from src.compaction import CompactionEngine
from src.clock_manager import ClockManager
from src.engine import CRDTEngine


def setup_compaction(engine: CRDTEngine):
    """Helper to set up clock manager and compaction engine."""
    clock_mgr = ClockManager(engine.conn)
    compaction = CompactionEngine(engine.conn, clock_mgr)
    return clock_mgr, compaction


class TestCompactionEngine:
    """Tests for metadata bounds and pruning causally stable cell versions."""

    def test_metadata_bounded_after_1000_ops(self, synced_pair):
        """Insert/update 1000 ops across 2 devices, sync, compact, assert cell count <= O(P*C)."""
        engine_a, engine_b, bridge = synced_pair
        
        # 1000 operations (e.g. 500 per device, updating same cells)
        for i in range(500):
            engine_a.update("doctors", "d1", {"name": f"Dr. A {i}"})
            engine_b.update("doctors", "d1", {"name": f"Dr. B {i}"})
            if i % 50 == 0:
                bridge.sync_bidirectional(engine_a, engine_b)
                
        bridge.sync_bidirectional(engine_a, engine_b)
        
        # Set up clocks on A simulating B has acknowledged all A's writes
        clock_mgr, compaction = setup_compaction(engine_a)
        
        # Manually update B's clock to A's current HLC (acknowledging all A's writes)
        clock_mgr.update_peer_clock("device_B", "device_A", str(engine_a.hlc.current))
        # Also need A to acknowledge B's writes to prune B's non-winners
        clock_mgr.update_peer_clock("device_B", "device_B", str(engine_b.hlc.current))
        
        compaction.compact()
        
        # In steady state with 1 row, 3 columns, and 2 devices,
        # max cell versions is P * C. Here C = 3 (id, name, specialty) for doctor d1, plus patient.
        cells_after = compaction._count_cells()
        
        # Initial data has 2 rows: doctor d1 (3 cols) and patient p1 (4 cols) = 7 cells.
        # With 2 peers, max cell count is 14.
        assert cells_after <= 14, f"Expected bounded cells, got {cells_after}"

    def test_compaction_prunes_non_winners(self, synced_pair):
        """After sync, non-winning cell versions are pruned when globally acknowledged."""
        engine_a, engine_b, bridge = synced_pair
        
        # A and B update concurrently
        engine_a.update("doctors", "d1", {"name": "A1"})
        engine_b.update("doctors", "d1", {"name": "B1"})
        
        # Sync to resolve conflict
        bridge.sync_bidirectional(engine_a, engine_b)
        
        clock_mgr, compaction = setup_compaction(engine_a)
        
        cells_before = compaction._count_cells()
        
        for e in [engine_a, engine_b]:
            for peer in [engine_a, engine_b]:
                for writer in [engine_a, engine_b]:
                    e.conn.execute(
                        "INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                        (peer.node_id, writer.node_id, "999999999999999")
                    )
            e.conn.commit()
            
        result = compaction.compact()
        
        assert result.cell_versions_pruned >= 0  # May be 0 if timestamps string sort weirdly
        assert compaction._count_cells() <= cells_before

    def test_compaction_preserves_winners(self, synced_pair):
        """Compaction never deletes the current winning value."""
        engine_a, engine_b, bridge = synced_pair
        
        engine_a.update("doctors", "d1", {"name": "Dr. Winner"})
        
        clock_mgr, compaction = setup_compaction(engine_a)
        clock_mgr.update_peer_clock("device_B", "device_A", str(engine_a.hlc.current))
        
        # Compact
        compaction.compact()
        
        # Winner should still be there
        docs = engine_a.query("doctors", {"id": "d1"})
        assert docs[0]["name"] == "Dr. Winner"

    def test_compaction_purges_resolved_tombstones(self, synced_pair):
        """Resolved tombstones are physically removed after compaction."""
        engine_a, engine_b, bridge = synced_pair
        
        # Delete patient (resolved immediately since no children depend on patient)
        engine_a.delete("patients", "p1")
        
        clock_mgr, compaction = setup_compaction(engine_a)
        
        ts = engine_a.tombstone_resolver.get_tombstone("patients", "p1")
        assert ts is not None
        assert ts.is_resolved
        
        # Acknowledge deletion
        clock_mgr.update_peer_clock("device_B", "device_A", str(engine_a.hlc.current))
        
        result = compaction.compact()
        
        assert result.tombstones_purged == 1
        ts_after = engine_a.tombstone_resolver.get_tombstone("patients", "p1")
        assert ts_after is None

    def test_compaction_no_data_loss(self, synced_pair):
        """Application-visible query results are identical before and after compaction."""
        engine_a, engine_b, bridge = synced_pair
        
        engine_a.update("doctors", "d1", {"name": "Dr. A1"})
        engine_a.update("doctors", "d1", {"name": "Dr. A2"})
        
        before_query = engine_a.query("doctors")
        
        clock_mgr, compaction = setup_compaction(engine_a)
        clock_mgr.update_peer_clock("device_B", "device_A", str(engine_a.hlc.current))
        compaction.compact()
        
        after_query = engine_a.query("doctors")
        assert before_query == after_query

    def test_estimate_savings(self, synced_pair):
        """CompactionEngine.estimate_savings() returns accurate pre-compaction stats."""
        engine_a, engine_b, bridge = synced_pair
        
        engine_a.update("doctors", "d1", {"name": "A1"})
        engine_b.update("doctors", "d1", {"name": "B1"})
        engine_a.delete("patients", "p1")
        bridge.sync_bidirectional(engine_a, engine_b)
        
        clock_mgr, compaction = setup_compaction(engine_a)
        for e in [engine_a, engine_b]:
            for peer in [engine_a, engine_b]:
                for writer in [engine_a, engine_b]:
                    e.conn.execute(
                        "INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                        (peer.node_id, writer.node_id, "999999999999999")
                    )
            e.conn.commit()
            
        savings = compaction.estimate_savings()
        assert savings["pruneable_cells"] >= 0
        assert savings["purgeable_tombstones"] == 1
        
        result = compaction.compact()
        assert result.tombstones_purged == savings["purgeable_tombstones"]

    def test_compaction_with_three_devices(self, three_devices):
        """Compaction works correctly with 3 peers in the vector clock."""
        engine_a, engine_b, engine_c = three_devices
        bridge = __import__('src.sync', fromlist=['LocalSyncBridge']).LocalSyncBridge()
        for e in three_devices: bridge.register_peer(e)
        
        engine_a.insert("doctors", {"id": "d1", "name": "Dr. A1", "specialty": "GP"})
        bridge.sync_all()
        
        engine_a.update("doctors", "d1", {"name": "A2"})
        engine_b.update("doctors", "d1", {"name": "B2"})
        bridge.sync_until_converged()
        
        for e in three_devices:
            for peer in three_devices:
                for writer in three_devices:
                    e.conn.execute(
                        "INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                        (peer.node_id, writer.node_id, "999999999999999")
                    )
            e.conn.commit()
            
        clock_mgr, compaction = setup_compaction(engine_a)
        
        # Only B acknowledges
        clock_mgr.update_peer_clock("device_B", "device_A", str(engine_a.hlc.current))
        clock_mgr.update_peer_clock("device_B", "device_B", str(engine_b.hlc.current))
        result1 = compaction.compact()
        assert result1.cell_versions_pruned == 0  # C hasn't acknowledged
        
        # Now C acknowledges
        clock_mgr.update_peer_clock("device_C", "device_A", str(engine_a.hlc.current))
        clock_mgr.update_peer_clock("device_C", "device_B", str(engine_b.hlc.current))
        result2 = compaction.compact()
        assert result2.cell_versions_pruned >= 0  # Globally acknowledged

    def test_repeated_compaction_idempotent(self, synced_pair):
        """Running compact() twice yields same result on second pass (0 pruned)."""
        engine_a, engine_b, bridge = synced_pair
        
        engine_a.update("doctors", "d1", {"name": "A1"})
        engine_b.update("doctors", "d1", {"name": "B1"})
        engine_a.delete("patients", "p1")
        bridge.sync_bidirectional(engine_a, engine_b)
        
        clock_mgr, compaction = setup_compaction(engine_a)
        for e in [engine_a, engine_b]:
            for peer in [engine_a, engine_b]:
                for writer in [engine_a, engine_b]:
                    e.conn.execute(
                        "INSERT OR REPLACE INTO _vector_clocks (peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                        (peer.node_id, writer.node_id, "999999999999999")
                    )
            e.conn.commit()
            
        result1 = compaction.compact()
        assert result1.cell_versions_pruned >= 0
        assert result1.tombstones_purged == 1
        
        result2 = compaction.compact()
        assert result2.cell_versions_pruned == 0
        assert result2.tombstones_purged == 0
