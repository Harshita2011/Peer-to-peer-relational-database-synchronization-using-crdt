"""Tests for tombstone-based FK preservation — the core innovation.

This test suite contains THE critical scenario that breaks CR-SQLite:
concurrent delete of a parent while another device inserts a child
referencing that parent during offline operation.

Every test here validates behavior that no existing CRDT-SQLite system handles.
"""

import pytest

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.convergence import ConvergenceHasher


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_devices_with_data():
    """Create two devices with shared initial state.
    
    Initial state on both devices:
    - Doctor 'd1' (Dr. Smith, GP)
    - Patient 'p1' (Alice, NHS001, doctor_id=d1)
    """
    engine_a = CRDTEngine(":memory:", "device_A")
    engine_b = CRDTEngine(":memory:", "device_B")
    bridge = LocalSyncBridge()
    
    # Register schema on both
    for engine in (engine_a, engine_b):
        engine.register_table(
            "doctors", primary_key="id",
            columns=["id", "name", "specialty"],
        )
        engine.register_table(
            "patients", primary_key="id",
            columns=["id", "name", "nhs_number", "doctor_id"],
            foreign_keys=[("doctor_id", "doctors", "id")],
            unique_cols=["nhs_number"],
            on_tombstone_policy="preserve",
        )
    
    # Insert initial data on device A
    engine_a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
    engine_a.insert("patients", {
        "id": "p1", "name": "Alice", "nhs_number": "NHS001", "doctor_id": "d1"
    })
    
    # Sync A → B so both have the same state
    bridge.register_peer(engine_a)
    bridge.register_peer(engine_b)
    bridge.sync(engine_a, engine_b)
    
    # Verify initial state on both devices
    assert len(engine_a.query("doctors")) == 1
    assert len(engine_b.query("doctors")) == 1
    assert len(engine_a.query("patients")) == 1
    assert len(engine_b.query("patients")) == 1
    
    yield engine_a, engine_b, bridge
    
    engine_a.close()
    engine_b.close()


@pytest.fixture
def three_devices_with_data():
    """Three devices with shared initial state for 3-way merge testing."""
    engines = [
        CRDTEngine(":memory:", f"device_{c}")
        for c in ["A", "B", "C"]
    ]
    bridge = LocalSyncBridge()
    
    for engine in engines:
        engine.register_table(
            "doctors", primary_key="id",
            columns=["id", "name", "specialty"],
        )
        engine.register_table(
            "patients", primary_key="id",
            columns=["id", "name", "nhs_number", "doctor_id"],
            foreign_keys=[("doctor_id", "doctors", "id")],
            unique_cols=["nhs_number"],
            on_tombstone_policy="preserve",
        )
        bridge.register_peer(engine)
    
    # Insert on device A
    engines[0].insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
    engines[0].insert("patients", {
        "id": "p1", "name": "Alice", "nhs_number": "NHS001", "doctor_id": "d1"
    })
    
    # Sync to all
    bridge.sync(engines[0], engines[1])
    bridge.sync(engines[0], engines[2])
    
    yield engines, bridge
    
    for e in engines:
        e.close()


# ─────────────────────────────────────────────────────────────────────
# THE CRITICAL TEST — This is the scenario that breaks CR-SQLite
# ─────────────────────────────────────────────────────────────────────

class TestConcurrentDeleteInsertFK:
    """The killer demo: concurrent parent DELETE + child INSERT."""

    def test_concurrent_delete_insert_fk_scenario(self, two_devices_with_data):
        """
        THE scenario that breaks CR-SQLite.
        
        Setup: Doctor 'd1' with Patient 'Alice' (p1) referencing it.
        
        1. Devices A and B both have this state
        2. Devices disconnect (no sync)
        3. Device A: DELETE doctor 'd1'
        4. Device B: INSERT patient 'Bob' (p2) into doctor 'd1'
        5. Devices reconnect and sync
        
        Expected:
        - Doctor 'd1' has a tombstone with ref_count >= 1
        - Bob's insert (p2) is preserved (NOT rejected)
        - Doctor is NOT physically deleted
        - Application can see the unresolved tombstone
        
        What CR-SQLite does: Either crashes with SQLITE_CONSTRAINT or
        silently orphans Bob's row (FK constraints are disabled on CRR tables).
        """
        engine_a, engine_b, bridge = two_devices_with_data
        
        # === DEVICES DISCONNECT ===
        # (We simply perform operations without syncing)
        
        # Device A: DELETE doctor 'd1'
        engine_a.delete("doctors", "d1")
        
        # Device B: INSERT new patient referencing doctor 'd1'
        engine_b.insert("patients", {
            "id": "p2", "name": "Bob", "nhs_number": "NHS002", "doctor_id": "d1"
        })
        
        # === DEVICES RECONNECT AND SYNC ===
        bridge.sync_bidirectional(engine_a, engine_b)
        # Do a second round to ensure convergence
        bridge.sync_bidirectional(engine_a, engine_b)
        
        # === VERIFY ON BOTH DEVICES ===
        for engine in (engine_a, engine_b):
            # Doctor 'd1' should have a tombstone
            tombstone = engine.tombstone_resolver.get_tombstone("doctors", "d1")
            assert tombstone is not None, "Tombstone should exist for deleted doctor"
            assert not tombstone.is_resolved, "Tombstone should be unresolved (children exist)"
            assert tombstone.ref_count >= 1, f"ref_count should be >= 1, got {tombstone.ref_count}"
            
            # Bob's patient record should exist
            patients = engine.query_raw(
                "SELECT * FROM patients WHERE id = ?", ("p2",)
            )
            assert len(patients) >= 1, "Bob's patient record should be preserved"
            
            # The unresolved tombstone should be visible to the application
            unresolved = engine.get_unresolved_tombstones()
            doctor_tombstones = [t for t in unresolved if t.row_id == "d1"]
            assert len(doctor_tombstones) >= 1, "Unresolved tombstone should be queryable"

    def test_three_way_concurrent_delete_and_two_inserts(self, three_devices_with_data):
        """
        3-way merge: A deletes parent, B and C each insert a child.
        
        All three operations happen concurrently (offline).
        After sync, the parent tombstone should have ref_count reflecting
        both B's and C's children.
        """
        engines, bridge = three_devices_with_data
        engine_a, engine_b, engine_c = engines
        
        # A: DELETE doctor 'd1'
        engine_a.delete("doctors", "d1")
        
        # B: INSERT patient 'Bob' referencing 'd1'
        engine_b.insert("patients", {
            "id": "p2", "name": "Bob", "nhs_number": "NHS002", "doctor_id": "d1"
        })
        
        # C: INSERT patient 'Charlie' referencing 'd1'
        engine_c.insert("patients", {
            "id": "p3", "name": "Charlie", "nhs_number": "NHS003", "doctor_id": "d1"
        })
        
        # Full mesh sync
        bridge.sync_all()
        bridge.sync_all()  # Second round for convergence
        
        # Verify on all three devices
        for engine in engines:
            tombstone = engine.tombstone_resolver.get_tombstone("doctors", "d1")
            assert tombstone is not None
            assert not tombstone.is_resolved
            # Should account for Alice (original), Bob, and Charlie
            # ref_count may vary based on whether Alice is also tracked
            assert tombstone.ref_count >= 2, \
                f"ref_count should be >= 2 (Bob + Charlie), got {tombstone.ref_count}"


class TestTombstoneRefCountAccuracy:
    """Tests for ref_count tracking accuracy."""

    def test_child_delete_decrements_ref_count(self, two_devices_with_data):
        """When a child is deleted, parent tombstone ref_count should decrease."""
        engine_a, engine_b, bridge = two_devices_with_data
        
        # Delete doctor (creates tombstone with ref_count for Alice)
        engine_a.delete("doctors", "d1")
        
        tombstone = engine_a.tombstone_resolver.get_tombstone("doctors", "d1")
        initial_ref_count = tombstone.ref_count
        
        # Delete Alice (child) — should decrement ref_count
        engine_a.delete("patients", "p1")
        
        tombstone = engine_a.tombstone_resolver.get_tombstone("doctors", "d1")
        # After deleting the only child, tombstone may be auto-resolved
        if tombstone is not None:
            assert tombstone.ref_count < initial_ref_count or tombstone.is_resolved

    def test_tombstone_resolves_when_all_children_deleted(self, two_devices_with_data):
        """Tombstone should be resolved when ref_count reaches 0."""
        engine_a, _, bridge = two_devices_with_data
        
        # Delete child first
        engine_a.delete("patients", "p1")
        
        # Then delete parent — no children left, should resolve immediately
        engine_a.delete("doctors", "d1")
        
        tombstone = engine_a.tombstone_resolver.get_tombstone("doctors", "d1")
        assert tombstone is not None
        assert tombstone.is_resolved, "Tombstone should be resolved (no children)"

    def test_ref_count_zero_immediate_purge(self):
        """Deleting a parent with no children should purge immediately."""
        engine = CRDTEngine(":memory:", "test_device")
        engine.register_table(
            "departments", primary_key="id",
            columns=["id", "name"],
        )
        engine.register_table(
            "employees", primary_key="id",
            columns=["id", "name", "dept_id"],
            foreign_keys=[("dept_id", "departments", "id")],
        )
        
        # Insert a department with NO employees
        engine.insert("departments", {"id": "dept1", "name": "Engineering"})
        
        # Delete it — should be immediately resolvable
        engine.delete("departments", "dept1")
        
        tombstone = engine.tombstone_resolver.get_tombstone("departments", "dept1")
        assert tombstone is not None
        assert tombstone.is_resolved, "Should be resolved (no children)"
        
        engine.close()

    def test_idempotent_re_delete(self, two_devices_with_data):
        """Re-deleting an already-tombstoned row should be idempotent."""
        engine_a, engine_b, bridge = two_devices_with_data
        
        # Delete on A
        engine_a.delete("doctors", "d1")
        tombstone_1 = engine_a.tombstone_resolver.get_tombstone("doctors", "d1")
        
        # Delete again on A (idempotent)
        engine_a.delete("doctors", "d1")
        tombstone_2 = engine_a.tombstone_resolver.get_tombstone("doctors", "d1")
        
        assert tombstone_2 is not None
        assert tombstone_2.ref_count == tombstone_1.ref_count


class TestTombstonePolicies:
    """Tests for configurable tombstone resolution policies."""

    def test_cascade_policy(self):
        """CASCADE should delete orphaned children when applied."""
        engine = CRDTEngine(":memory:", "test")
        engine.register_table(
            "departments", primary_key="id",
            columns=["id", "name"],
            on_tombstone_policy="cascade",
        )
        engine.register_table(
            "employees", primary_key="id",
            columns=["id", "name", "dept_id"],
            foreign_keys=[("dept_id", "departments", "id")],
        )
        
        engine.insert("departments", {"id": "d1", "name": "Engineering"})
        engine.insert("employees", {"id": "e1", "name": "Alice", "dept_id": "d1"})
        engine.insert("employees", {"id": "e2", "name": "Bob", "dept_id": "d1"})
        
        # Delete department — CASCADE policy
        engine.delete("departments", "d1")
        
        # Apply the policy
        result = engine.tombstone_resolver.apply_policy("departments", "d1")
        
        assert result.policy_applied == "cascade"
        assert result.children_cascaded >= 2
        
        engine.close()

    def test_nullify_policy(self):
        """NULLIFY should set FK columns to NULL."""
        engine = CRDTEngine(":memory:", "test")
        engine.register_table(
            "departments", primary_key="id",
            columns=["id", "name"],
            on_tombstone_policy="nullify",
        )
        engine.register_table(
            "employees", primary_key="id",
            columns=["id", "name", "dept_id"],
            foreign_keys=[("dept_id", "departments", "id")],
        )
        
        engine.insert("departments", {"id": "d1", "name": "Engineering"})
        engine.insert("employees", {"id": "e1", "name": "Alice", "dept_id": "d1"})
        
        engine.delete("departments", "d1")
        result = engine.tombstone_resolver.apply_policy("departments", "d1")
        
        assert result.policy_applied == "nullify"
        assert result.children_nullified >= 1
        
        engine.close()

    def test_preserve_policy(self):
        """PRESERVE should keep both tombstone and children alive."""
        engine = CRDTEngine(":memory:", "test")
        engine.register_table(
            "departments", primary_key="id",
            columns=["id", "name"],
        )
        engine.register_table(
            "employees", primary_key="id",
            columns=["id", "name", "dept_id"],
            foreign_keys=[("dept_id", "departments", "id")],
            on_tombstone_policy="preserve",
        )
        
        engine.insert("departments", {"id": "d1", "name": "Engineering"})
        engine.insert("employees", {"id": "e1", "name": "Alice", "dept_id": "d1"})
        
        engine.delete("departments", "d1")
        result = engine.tombstone_resolver.apply_policy("departments", "d1")
        
        assert result.policy_applied == "preserve"
        assert result.requires_resolution is True
        
        # Both should still exist
        tombstone = engine.tombstone_resolver.get_tombstone("departments", "d1")
        assert tombstone is not None
        assert not tombstone.is_resolved
        
        employees = engine.query_raw("SELECT * FROM employees WHERE dept_id = 'd1'")
        assert len(employees) >= 1
        
        engine.close()

    def test_callback_policy(self):
        """Custom callback should be invoked for resolution."""
        callback_log = []
        
        def my_callback(table, row_id, children):
            callback_log.append((table, row_id, len(children)))
            return True  # Signal that resolution is complete
        
        engine = CRDTEngine(":memory:", "test")
        engine.register_table(
            "departments", primary_key="id",
            columns=["id", "name"],
            on_tombstone_callback=my_callback,
        )
        engine.register_table(
            "employees", primary_key="id",
            columns=["id", "name", "dept_id"],
            foreign_keys=[("dept_id", "departments", "id")],
        )
        
        engine.insert("departments", {"id": "d1", "name": "Engineering"})
        engine.insert("employees", {"id": "e1", "name": "Alice", "dept_id": "d1"})
        
        engine.delete("departments", "d1")
        result = engine.tombstone_resolver.apply_policy("departments", "d1")
        
        assert result.policy_applied == "callback"
        assert len(callback_log) == 1
        assert callback_log[0][0] == "departments"
        assert callback_log[0][1] == "d1"
        
        engine.close()


class TestSyncWithTombstones:
    """Tests for tombstone behavior during sync."""

    def test_tombstone_syncs_to_peer(self, two_devices_with_data):
        """A tombstone created on one device should sync to another."""
        engine_a, engine_b, bridge = two_devices_with_data
        
        # Delete on A
        engine_a.delete("doctors", "d1")
        
        # Sync A → B
        bridge.sync(engine_a, engine_b)
        
        # B should have the tombstone
        tombstone = engine_b.tombstone_resolver.get_tombstone("doctors", "d1")
        assert tombstone is not None

    def test_convergence_after_conflict_resolution(self, two_devices_with_data):
        """After resolving tombstone conflicts, devices should converge."""
        engine_a, engine_b, bridge = two_devices_with_data
        
        # A deletes doctor, B inserts new patient
        engine_a.delete("doctors", "d1")
        engine_b.insert("patients", {
            "id": "p2", "name": "Bob", "nhs_number": "NHS002", "doctor_id": "d1"
        })
        
        # Sync both ways
        bridge.sync_bidirectional(engine_a, engine_b)
        bridge.sync_bidirectional(engine_a, engine_b)
        
        # Resolve on both with CASCADE
        for engine in (engine_a, engine_b):
            # Force cascade resolution
            engine.tombstone_resolver._cascade_delete_children("doctors", "d1")
            engine.tombstone_resolver._resolve_tombstone("doctors", "d1")
            engine.conn.commit()
        
        # After another sync, they should converge
        bridge.sync_bidirectional(engine_a, engine_b)
        
        # Both should have no unresolved tombstones
        for engine in (engine_a, engine_b):
            unresolved = engine.get_unresolved_tombstones()
            doctor_ts = [t for t in unresolved if t.row_id == "d1"]
            assert len(doctor_ts) == 0, "Should have no unresolved tombstones after resolution"
