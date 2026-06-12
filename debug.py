import sqlite3
import pytest
from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from tests.test_stress import ConvergenceHasher
import sys

def test():
    five_devices = [CRDTEngine(db_path=":memory:", node_id=f'device_{i}') for i in range(5)]
    for e in five_devices:
        e.register_table("items", primary_key="id", columns=["id", "name", "value"])
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
    for i, e in enumerate(five_devices):
        hasher = ConvergenceHasher(e.conn, e.schema)
        h = hasher.compute_hash()
        print(f"Device {i} hash: {h}")
        cursor = e.conn.execute("""
            SELECT col_name, value, hlc_ts, writer_id 
            FROM _crdt_cells 
            WHERE is_winner = 1 
            ORDER BY table_name, row_id, col_name
        """)
        print(f"Device {i} winner cells: {cursor.fetchall()}")
        cursor = e.conn.execute("""
            SELECT * FROM _operations ORDER BY local_seq
        """)
        ops = cursor.fetchall()
        print(f"Device {i} operations count: {len(ops)}")
        hashes.add(h)
    
    print(len(hashes) == 1)

if __name__ == "__main__":
    test()
