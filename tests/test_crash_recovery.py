import pytest
import sqlite3
from unittest.mock import patch
from src.benchmark_adapter import open_peer

def test_crash_recovery_during_sync():
    """
    Simulates a node crash (exception) midway through apply_delta.
    Verifies that SQLite WAL rollback ensures atomic commits and no partial replication states.
    """
    peer_a = open_peer("peer_A")
    peer_b = open_peer("peer_B")
    
    schema = {
        "table_name": "users",
        "primary_key": "id",
        "columns": ["id", "name"],
        "foreign_keys": [],
        "unique_cols": [],
        "on_tombstone_policy": "preserve"
    }
    
    peer_a.apply_schema(schema)
    peer_b.apply_schema(schema)
    
    # Peer A executes several inserts
    peer_a.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
    peer_a.execute("INSERT INTO users (id, name) VALUES (?, ?)", (2, "Bob"))
    
    # Capture initial state of B
    b_hash_before = peer_b.snapshot_hash()
    
    delta_a = peer_a.engine.get_delta(since_seq=0, for_peer=peer_b.peer_id)
    
    # Inject a crash into peer B's engine.
    # We will mock CellMerger.merge_cell to raise an Exception on the second operation.
    original_merge = peer_b.engine.merger.merge_cell
    call_count = [0]
    
    def mock_merge_cell(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 3:  # Crash on the 3rd cell (during Bob's row)
            raise RuntimeError("Simulated Node Crash")
        return original_merge(*args, **kwargs)
        
    with patch.object(peer_b.engine.merger, 'merge_cell', side_effect=mock_merge_cell):
        with pytest.raises(RuntimeError, match="Simulated Node Crash"):
            peer_b.engine.apply_delta(delta_a, from_peer=peer_a.peer_id)
            
    # Because it crashed, the connection.commit() in apply_delta should not have been reached.
    # SQLite WAL should automatically rollback the entire transaction.
    
    b_hash_after = peer_b.snapshot_hash()
    
    # Assert that B's state is completely identical to before the crash
    assert b_hash_after == b_hash_before
    
    # Assert that no partial operations were left in the ops log
    cursor = peer_b.engine.conn.execute("SELECT COUNT(*) FROM _operations")
    op_count = cursor.fetchone()[0]
    assert op_count == 0

    peer_a.close()
    peer_b.close()
