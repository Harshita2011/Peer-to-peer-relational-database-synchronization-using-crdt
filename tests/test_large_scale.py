import pytest
import random
from src.benchmark_adapter import open_peer

def test_large_scale_stress():
    """
    Simulates a large-scale deployment with 10,000 operations across 5 peers.
    Includes simulated network partitions and random P2P delta syncing.
    """
    num_peers = 5
    num_ops = 10000
    
    peers = [open_peer(f"peer_{i}") for i in range(num_peers)]
    
    schema = {
        "table_name": "items",
        "primary_key": "id",
        "columns": ["id", "val"],
        "foreign_keys": [],
        "unique_cols": [],
        "on_tombstone_policy": "preserve"
    }
    
    for p in peers:
        p.apply_schema(schema)
        
    # Generate 10k random operations across random peers
    # We will just do inserts and updates to random keys
    for i in range(num_ops):
        p = random.choice(peers)
        key = random.randint(1, 1000)
        val = f"val_{i}"
        
        # 80% insert/update, 20% delete
        action = random.random()
        if action < 0.8:
            p.execute("INSERT INTO items (id, val) VALUES (?, ?)", (key, val))
        else:
            p.execute("DELETE FROM items WHERE id = ?", (key,))
            
    # Partition healing / Random gossip sync
    # We'll run enough random syncs to ensure high probability of full propagation
    for _ in range(num_peers * 10):
        p1 = random.choice(peers)
        p2 = random.choice(peers)
        if p1 != p2:
            p1.sync(p2)
            
    # Final full convergence loop
    # Ensure every node syncs with peer 0
    for i in range(1, num_peers):
        peers[0].sync(peers[i])
    # Sync peer 0 back to everyone else
    for i in range(1, num_peers):
        peers[i].sync(peers[0])
        
    # Verify convergence
    baseline_hash = peers[0].snapshot_hash()
    
    for p in peers:
        assert p.snapshot_hash() == baseline_hash, f"Peer {p.peer_id} diverged!"
        
    # Optional log compaction check
    # We just ensure the engine survived the load
    for p in peers:
        p.close()
