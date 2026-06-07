"""Vector clock management for compaction decisions.

Tracks per-peer acknowledgment of writes from all known writers.
This enables the compaction invariant: a cell version can be pruned
when ALL known peers have acknowledged a NEWER version for the same cell.

The vector clock table (_vector_clocks) stores entries as:
    (peer_id, writer_id, max_hlc_ts)

Where max_hlc_ts is the highest HLC timestamp that peer_id has
acknowledged from writer_id.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class PeerClockState:
    """The clock state of a single peer."""
    peer_id: str
    writer_clocks: dict[str, str]  # writer_id → max_hlc_ts


class ClockManager:
    """Tracks vector clocks across all known peers.

    Used by the CompactionEngine to determine which cell versions
    are causally stable (acknowledged by all peers) and can be pruned.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def update_peer_clock(self, peer_id: str, writer_id: str, hlc_ts: str) -> None:
        """Record that peer_id has acknowledged up to hlc_ts from writer_id.

        Only advances the clock — if the existing entry has a higher HLC,
        this is a no-op.

        Args:
            peer_id: The peer that acknowledged.
            writer_id: The writer whose data was acknowledged.
            hlc_ts: The highest HLC timestamp acknowledged.
        """
        existing = self.conn.execute(
            "SELECT max_hlc_ts FROM _vector_clocks WHERE peer_id = ? AND writer_id = ?",
            (peer_id, writer_id),
        ).fetchone()

        if existing is None:
            self.conn.execute(
                "INSERT INTO _vector_clocks (peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                (peer_id, writer_id, hlc_ts),
            )
        elif hlc_ts > existing[0]:
            self.conn.execute(
                "UPDATE _vector_clocks SET max_hlc_ts = ? WHERE peer_id = ? AND writer_id = ?",
                (hlc_ts, peer_id, writer_id),
            )

    def update_peer_clocks_bulk(self, peer_id: str, clocks: dict[str, str]) -> None:
        """Bulk update a peer's clock for multiple writers.

        Args:
            peer_id: The peer.
            clocks: Dict mapping writer_id → max_hlc_ts.
        """
        for writer_id, hlc_ts in clocks.items():
            self.update_peer_clock(peer_id, writer_id, hlc_ts)

    def get_peer_clock(self, peer_id: str) -> PeerClockState:
        """Get the full clock state for a single peer.

        Args:
            peer_id: The peer to query.

        Returns:
            PeerClockState with all writer clocks.
        """
        cursor = self.conn.execute(
            "SELECT writer_id, max_hlc_ts FROM _vector_clocks WHERE peer_id = ?",
            (peer_id,),
        )
        clocks = {row[0]: row[1] for row in cursor.fetchall()}
        return PeerClockState(peer_id=peer_id, writer_clocks=clocks)

    def get_all_peers(self) -> list[str]:
        """Return all known peer IDs."""
        cursor = self.conn.execute(
            "SELECT DISTINCT peer_id FROM _vector_clocks"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_minimum_acknowledged(self, writer_id: str) -> str | None:
        """Return the MINIMUM hlc_ts acknowledged by ALL peers for this writer.

        Cell versions from this writer older than this timestamp have been
        seen by all peers and are candidates for compaction.

        Args:
            writer_id: The writer to check.

        Returns:
            The minimum HLC string, or None if no peers have acknowledged.
        """
        peers = self.get_all_peers()
        if not peers:
            return None

        min_ts = None
        for peer_id in peers:
            row = self.conn.execute(
                "SELECT max_hlc_ts FROM _vector_clocks WHERE peer_id = ? AND writer_id = ?",
                (peer_id, writer_id),
            ).fetchone()

            if row is None:
                # This peer hasn't seen any data from this writer
                return None

            if min_ts is None or row[0] < min_ts:
                min_ts = row[0]

        return min_ts

    def is_globally_acknowledged(self, writer_id: str, hlc_ts: str) -> bool:
        """Check if a specific timestamp has been acknowledged by ALL peers.

        Args:
            writer_id: The writer that produced the timestamp.
            hlc_ts: The HLC timestamp to check.

        Returns:
            True if ALL peers have acknowledged this or a later timestamp.
        """
        min_ack = self.get_minimum_acknowledged(writer_id)
        if min_ack is None:
            return False
        return min_ack >= hlc_ts

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the clock tracking.

        This should be called when a peer permanently disconnects.

        Args:
            peer_id: The peer to remove.
        """
        self.conn.execute(
            "DELETE FROM _vector_clocks WHERE peer_id = ?",
            (peer_id,),
        )
