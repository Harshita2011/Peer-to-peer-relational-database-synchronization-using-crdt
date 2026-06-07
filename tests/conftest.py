"""Pytest fixtures for multi-device CRDT sync simulation."""

import pytest
import sqlite3
import time

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.convergence import ConvergenceHasher


@pytest.fixture
def engine_factory():
    """Factory fixture for creating CRDTEngine instances with in-memory databases."""
    engines = []

    def _create(node_id: str) -> CRDTEngine:
        engine = CRDTEngine(":memory:", node_id)
        engines.append(engine)
        return engine

    yield _create

    for engine in engines:
        engine.close()


@pytest.fixture
def setup_demo_schema():
    """Setup the doctors -> patients -> prescriptions demo schema on an engine."""
    def _setup(engine: CRDTEngine, policy: str = "preserve") -> None:
        engine.register_table(
            "doctors", primary_key="id",
            columns=["id", "name", "specialty"],
        )
        engine.register_table(
            "patients", primary_key="id",
            columns=["id", "name", "nhs_number", "doctor_id"],
            foreign_keys=[("doctor_id", "doctors", "id")],
            unique_cols=["nhs_number"],
            on_tombstone_policy=policy,
        )
        engine.register_table(
            "prescriptions", primary_key="id",
            columns=["id", "medication", "dosage", "patient_id"],
            foreign_keys=[("patient_id", "patients", "id")],
            on_tombstone_policy=policy,
        )
    return _setup


@pytest.fixture
def two_devices(engine_factory, setup_demo_schema):
    """Create two synced devices with the demo schema."""
    engine_a = engine_factory("device_A")
    engine_b = engine_factory("device_B")
    setup_demo_schema(engine_a)
    setup_demo_schema(engine_b)
    return engine_a, engine_b


@pytest.fixture
def three_devices(engine_factory, setup_demo_schema):
    """Create three synced devices with the demo schema."""
    engine_a = engine_factory("device_A")
    engine_b = engine_factory("device_B")
    engine_c = engine_factory("device_C")
    setup_demo_schema(engine_a)
    setup_demo_schema(engine_b)
    setup_demo_schema(engine_c)
    return engine_a, engine_b, engine_c


@pytest.fixture
def sync_bridge():
    """Create a LocalSyncBridge instance."""
    return LocalSyncBridge()


@pytest.fixture
def synced_pair(two_devices, sync_bridge):
    """Two devices with initial data synced between them.

    Initial state:
    - Doctor 'd1' (Dr. Smith, General Practice)
    - Patient 'p1' (Alice, nhs=NHS001, doctor_id=d1)
    """
    engine_a, engine_b = two_devices
    bridge = sync_bridge
    bridge.register_peer(engine_a)
    bridge.register_peer(engine_b)

    # Insert initial data on device A
    engine_a.insert("doctors", {"id": "d1", "name": "Dr. Smith", "specialty": "GP"})
    engine_a.insert("patients", {
        "id": "p1", "name": "Alice", "nhs_number": "NHS001", "doctor_id": "d1"
    })

    # Sync to device B
    bridge.sync(engine_a, engine_b)

    return engine_a, engine_b, bridge


def verify_convergence(engines: list[CRDTEngine]) -> bool:
    """Verify all engines have converged to the same state."""
    hashes = []
    for engine in engines:
        hasher = ConvergenceHasher(engine.conn, engine.schema)
        hashes.append(hasher.compute_hash())
    return len(set(hashes)) == 1


@pytest.fixture
def setup_simple_schema():
    """Setup a simple schema (single table 'items', no FKs) on an engine."""
    def _setup(engine: CRDTEngine) -> None:
        engine.register_table(
            "items", primary_key="id",
            columns=["id", "name", "value"],
        )
    return _setup


@pytest.fixture
def five_devices(engine_factory, setup_simple_schema):
    """Create five synced devices with the simple schema."""
    engines = []
    for i in range(5):
        e = engine_factory(f"device_{i}")
        setup_simple_schema(e)
        engines.append(e)
    return engines
