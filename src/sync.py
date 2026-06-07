"""Sync protocol with in-process bridge for benchmarking and HTTP for demo.

Provides two sync mechanisms:
1. LocalSyncBridge: In-process, passes Delta objects directly between engines.
   Used for all benchmarks (zero network jitter, controlled measurements).
2. HTTPSyncServer: Flask-based REST API for visual demo (future).

The key rule: never mix them in the same experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.engine import CRDTEngine, Delta, SyncResult


@dataclass
class SyncStats:
    """Cumulative statistics for sync operations."""
    total_syncs: int = 0
    total_cells_synced: int = 0
    total_conflicts: int = 0
    total_tombstones_processed: int = 0
    total_artifacts_created: int = 0


class LocalSyncBridge:
    """In-process sync bridge for direct peer-to-peer delta exchange.

    Passes Delta objects directly between CRDTEngine instances without
    serialization or network overhead. Used for all benchmarks.

    Usage:
        bridge = LocalSyncBridge()
        bridge.register_peer(engine_a)
        bridge.register_peer(engine_b)
        
        # Sync A → B
        bridge.sync(engine_a, engine_b)
        
        # Full mesh sync (all pairs)
        bridge.sync_all()
    """

    def __init__(self):
        self.peers: dict[str, CRDTEngine] = {}
        self.stats = SyncStats()
        # Track the last known HLC per peer pair for delta optimization
        self._last_sync_hlc: dict[tuple[str, str], str] = {}

    def register_peer(self, engine: CRDTEngine) -> None:
        """Register an engine as a peer in the sync network.

        Args:
            engine: CRDTEngine instance to register.
        """
        self.peers[engine.node_id] = engine

    def unregister_peer(self, node_id: str) -> None:
        """Remove a peer from the sync network.

        Args:
            node_id: ID of the peer to remove.
        """
        self.peers.pop(node_id, None)
        # Clean up sync state
        keys_to_remove = [
            k for k in self._last_sync_hlc if node_id in k
        ]
        for k in keys_to_remove:
            del self._last_sync_hlc[k]

    def sync(
        self,
        source: CRDTEngine | str,
        target: CRDTEngine | str,
    ) -> SyncResult:
        """Sync changes from source to target.

        Sends a delta from source containing all changes since the last
        sync to the target, which merges them into its local state.

        Args:
            source: Source engine (or its node_id).
            target: Target engine (or its node_id).

        Returns:
            SyncResult from the target's apply_delta.
        """
        src = self._resolve_peer(source)
        tgt = self._resolve_peer(target)

        pair_key = (src.node_id, tgt.node_id)
        # Always use full sync ("0") because get_delta relies on original hlc_ts,
        # which drops transitive updates that have older hlc_ts than the last sync.
        since_hlc = "0"

        # Get delta from source
        delta = src.get_delta(since_hlc=since_hlc, for_peer=tgt.node_id)

        # Apply delta to target
        result = tgt.apply_delta(delta, from_peer=src.node_id)

        # Update sync state
        self._last_sync_hlc[pair_key] = delta.source_hlc or str(src.hlc.current)

        # Update stats
        self.stats.total_syncs += 1
        self.stats.total_cells_synced += result.merge_result.total_cells_merged
        self.stats.total_tombstones_processed += len(result.tombstone_results)

        return result

    def sync_bidirectional(
        self,
        peer_a: CRDTEngine | str,
        peer_b: CRDTEngine | str,
    ) -> tuple[SyncResult, SyncResult]:
        """Sync changes in both directions between two peers.

        Args:
            peer_a: First peer.
            peer_b: Second peer.

        Returns:
            Tuple of (A→B result, B→A result).
        """
        result_ab = self.sync(peer_a, peer_b)
        result_ba = self.sync(peer_b, peer_a)
        return result_ab, result_ba

    def sync_all(self) -> list[SyncResult]:
        """Perform a full mesh sync: every peer syncs with every other peer.

        This ensures all peers converge to the same state. May need
        multiple rounds if there are transitive dependencies.

        Returns:
            List of all SyncResults from the round.
        """
        results = []
        peer_ids = sorted(self.peers.keys())
        for i, src_id in enumerate(peer_ids):
            for j, tgt_id in enumerate(peer_ids):
                if i != j:
                    result = self.sync(src_id, tgt_id)
                    results.append(result)
        return results

    def sync_until_converged(self, max_rounds: int = 10) -> int:
        """Keep syncing all peers until convergence hashes match.

        Args:
            max_rounds: Maximum sync rounds before giving up.

        Returns:
            Number of rounds needed to converge, or -1 if failed.
        """
        from src.convergence import ConvergenceHasher

        for round_num in range(1, max_rounds + 1):
            self.sync_all()

            # Check convergence
            hashes = {}
            for node_id, engine in self.peers.items():
                hasher = ConvergenceHasher(engine.conn, engine.schema)
                hashes[node_id] = hasher.compute_hash()

            if len(set(hashes.values())) == 1:
                return round_num

        return -1

    def _resolve_peer(self, peer: CRDTEngine | str) -> CRDTEngine:
        """Resolve a peer reference to an engine instance."""
        if isinstance(peer, CRDTEngine):
            return peer
        if peer in self.peers:
            return self.peers[peer]
        raise ValueError(f"Unknown peer: {peer}")

    def reset_sync_state(self) -> None:
        """Reset all sync tracking state (forces full re-sync next time)."""
        self._last_sync_hlc.clear()
        self.stats = SyncStats()
