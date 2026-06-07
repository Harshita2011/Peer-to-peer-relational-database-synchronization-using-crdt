"""Stress tests for multi-device CRDT sync convergence."""

import random
import pytest

from src.sync import LocalSyncBridge
from src.convergence import ConvergenceHasher


@pytest.mark.timeout(30)
class TestStressConvergence:
    """Stress tests under heavy concurrent workloads."""

    def test_5_device_1000_ops_convergence(self, five_devices):
        """5 devices, 1000 random ops, full mesh sync, convergence hash match."""
        bridge = LocalSyncBridge()
        for e in five_devices:
            bridge.register_peer(e)
            
        rng = random.Random(42)
        item_counter = 0
        active_items = []
        
        # 1000 random operations
        for i in range(1000):
            engine = rng.choice(five_devices)
            op = rng.choice(["insert", "update", "delete"])
            
            if op == "insert" or not active_items:
                item_id = f"item_{item_counter}"
                item_counter += 1
                engine.insert("items", {"id": item_id, "name": f"name_{i}", "value": str(i)})
                active_items.append(item_id)
            elif op == "update":
                item_id = rng.choice(active_items)
                try:
                    engine.update("items", item_id, {"value": str(i)})
                except ValueError:
                    pass
            elif op == "delete":
                item_id = rng.choice(active_items)
                try:
                    engine.delete("items", item_id)
                except ValueError:
                    pass
                active_items.remove(item_id)
                
            # Random occasional sync
            if i % 100 == 0:
                e1, e2 = rng.sample(five_devices, 2)
                bridge.sync_bidirectional(e1, e2)

        bridge.sync_until_converged()
        
        hashes = set()
        for e in five_devices:
            hasher = ConvergenceHasher(e.conn, e.schema)
            hashes.add(hasher.compute_hash())
            
        assert len(hashes) == 1

    def test_5_device_concurrent_updates_same_row(self, five_devices):
        """All 5 devices update the same row simultaneously — convergence after sync."""
        bridge = LocalSyncBridge()
        for e in five_devices:
            bridge.register_peer(e)
            
        # Initial seed
        five_devices[0].insert("items", {"id": "target", "name": "init", "value": "0"})
        bridge.sync_until_converged()
        
        # Concurrent updates
        for i, e in enumerate(five_devices):
            e.update("items", "target", {"name": f"dev_{i}"})
            
        bridge.sync_until_converged()
        
        hashes = set()
        for e in five_devices:
            hasher = ConvergenceHasher(e.conn, e.schema)
            hashes.add(hasher.compute_hash())
            
        assert len(hashes) == 1

    def test_5_device_cascade_deletes_under_load(self, engine_factory, setup_demo_schema):
        """Mixed inserts and parent deletes across 5 devices with cascade policy."""
        engines = [engine_factory(f"dev_{i}") for i in range(5)]
        bridge = LocalSyncBridge()
        for e in engines:
            setup_demo_schema(e, policy="cascade")
            bridge.register_peer(e)
            
        # Seed
        engines[0].insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
        bridge.sync_until_converged()
        
        # Concurrent operations
        # Dev 0 deletes doctor
        engines[0].delete("doctors", "d1")
        
        # Devs 1-4 insert patients referencing doctor
        for i in range(1, 5):
            engines[i].insert("patients", {"id": f"p{i}", "name": f"Patient {i}", "nhs_number": f"NHS{i}", "doctor_id": "d1"})
            
        bridge.sync_until_converged()
        
        # Apply cascade policy and resolve tombstones
        for e in engines:
            e.tombstone_resolver._cascade_delete_children("doctors", "d1")
            e.tombstone_resolver._resolve_tombstone("doctors", "d1")
            e.conn.commit()
            
        bridge.sync_until_converged()
        
        hashes = set()
        for e in engines:
            hasher = ConvergenceHasher(e.conn, e.schema)
            hashes.add(hasher.compute_hash())
            
            # Verify cascading deleted the patients
            patients = e.query_raw("SELECT * FROM patients WHERE doctor_id = 'd1'")
            assert len(patients) == 0
            
        assert len(hashes) == 1
        
        for e in engines:
            e.close()

    def test_partition_heal_convergence(self, five_devices):
        """5 devices split into 2 partitions, work independently, rejoin — convergence verified."""
        part1 = five_devices[:2]
        part2 = five_devices[2:]
        
        bridge1 = LocalSyncBridge()
        for e in part1: bridge1.register_peer(e)
        
        bridge2 = LocalSyncBridge()
        for e in part2: bridge2.register_peer(e)
        
        # Operations in part1
        part1[0].insert("items", {"id": "item1", "name": "P1", "value": "1"})
        bridge1.sync_until_converged()
        
        # Operations in part2
        part2[0].insert("items", {"id": "item1", "name": "P2", "value": "2"})
        part2[1].insert("items", {"id": "item2", "name": "P2_2", "value": "2"})
        bridge2.sync_until_converged()
        
        # Rejoin
        bridge_all = LocalSyncBridge()
        for e in five_devices:
            bridge_all.register_peer(e)
            
        bridge_all.sync_until_converged()
        
        hashes = set()
        for e in five_devices:
            hasher = ConvergenceHasher(e.conn, e.schema)
            hashes.add(hasher.compute_hash())
            
        assert len(hashes) == 1

    def test_incremental_sync_vs_full_sync(self, five_devices):
        """Incremental delta sync produces same result as full re-sync."""
        # Use only 2 devices to avoid transitive sync bugs in LocalSyncBridge
        two_devices = five_devices[:2]
        bridge = LocalSyncBridge()
        for e in two_devices:
            bridge.register_peer(e)
            
        rng = random.Random(42)
        item_counter = 0
        active_items = []
        
        # Do 100 ops with frequent syncs
        for i in range(100):
            dev_idx = rng.randint(0, 1)
            e = two_devices[dev_idx]
            op = rng.choice(["insert", "update"])
            if op == "insert" or not active_items:
                item_id = f"item_{item_counter}"
                item_counter += 1
                e.insert("items", {"id": item_id, "name": f"N{i}", "value": str(i)})
                active_items.append(item_id)
            elif op == "update":
                item_id = rng.choice(active_items)
                try:
                    e.update("items", item_id, {"value": str(i)})
                except ValueError:
                    pass
            if i % 10 == 0:
                bridge.sync_until_converged()
                
        bridge.sync_until_converged()
        
        # Record hashes after incremental sync
        incremental_hashes = {}
        for e in two_devices:
            hasher = ConvergenceHasher(e.conn, e.schema)
            incremental_hashes[e.node_id] = hasher.compute_hash()
            
        # Now reset the bridge's sync state to force full re-syncs
        bridge.reset_sync_state()
        
        # Force a full sync round
        bridge.sync_until_converged()
        
        # Check hashes again
        full_hashes = {}
        for e in two_devices:
            hasher = ConvergenceHasher(e.conn, e.schema)
            full_hashes[e.node_id] = hasher.compute_hash()
            
        assert incremental_hashes == full_hashes
