import pytest
from src.benchmark_adapter import open_peer

def test_idempotent_replay():
    """
    Test that re-applying the same operation multiple times
    produces identical state and does not duplicate logs.
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
    
    # Peer A executes an insert
    peer_a.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
    
    # Get delta from A for B
    delta_a_1 = peer_a.engine.get_delta(since_seq=0, for_peer=peer_b.peer_id)
    
    # Apply to B once
    peer_b.engine.apply_delta(delta_a_1, from_peer=peer_a.peer_id)
    state_1 = peer_b.snapshot_state()
    hash_1 = peer_b.snapshot_hash()
    
    assert len(state_1["users"]) == 1
    assert state_1["users"][0]["name"] == "Alice"
    
    # Apply identical delta to B multiple times (replay)
    peer_b.engine.apply_delta(delta_a_1, from_peer=peer_a.peer_id)
    peer_b.engine.apply_delta(delta_a_1, from_peer=peer_a.peer_id)
    
    state_2 = peer_b.snapshot_state()
    hash_2 = peer_b.snapshot_hash()
    
    # Verify exact identical state and hash
    assert state_1 == state_2
    assert hash_1 == hash_2
    
    # Verify operations log hasn't duplicated entries
    cursor = peer_b.engine.conn.execute("SELECT COUNT(*) FROM _operations")
    op_count = cursor.fetchone()[0]
    
    # Insert creates multiple ops (1 for each cell). Verify they are stable.
    assert op_count == 2  # id, name cells

    peer_a.close()
    peer_b.close()
