import sqlite3
import pytest
from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.merge import ClockRelation

def test():
    five_devices = [CRDTEngine(db_path=":memory:", node_id=f'device_{i}') for i in range(5)]
    for e in five_devices:
        e.register_table("items", primary_key="id", columns=["id", "name", "value"])
    bridge = LocalSyncBridge()
    for e in five_devices:
        bridge.register_peer(e)

    five_devices[0].insert("items", {"id": "target", "name": "init", "value": "0"})
    bridge.sync_until_converged()

    for i, e in enumerate(five_devices):
        e.update("items", "target", {"name": f"dev_{i}"})

    bridge.sync_until_converged()

    print("Checking winners:")
    for i, e in enumerate(five_devices):
        cursor = e.conn.execute("SELECT writer_id, hlc_ts, is_winner FROM _crdt_cells WHERE col_name = 'name'")
        print(f"Device {i}: {cursor.fetchall()}")

if __name__ == "__main__":
    test()
