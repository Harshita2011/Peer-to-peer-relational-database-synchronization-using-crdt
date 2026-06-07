"""Hybrid Logical Clock (HLC) for causal ordering across distributed replicas.

Implements the algorithm from Kulkarni et al., "Logical Physical Clocks and
Consistent Snapshots in Globally Distributed Databases" (2014).

HLC timestamps combine physical wall-clock time with a logical counter to
guarantee causal ordering even when system clocks drift. Each timestamp is
a triple (physical_ms, logical_counter, node_id).

For SQLite storage, timestamps are serialized as strings in the format:
    "{physical:013d}:{logical:05d}:{node_id}"

This format enables correct lexicographic comparison via SQL '>' operator,
which is critical for the merge engine's Last-HLC-Wins resolution.

Usage:
    clock = HLC("device_A")
    ts = clock.now()                    # generate timestamp for local event
    ts = clock.receive(remote_ts)       # update clock on receiving remote timestamp
    HLC.compare(ts_a, ts_b)            # compare two timestamps: -1, 0, or 1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


# Physical time padding: 13 digits covers timestamps up to year 2286
_PHYSICAL_WIDTH = 13
# Logical counter padding: 5 digits allows up to 99999 events per millisecond
_LOGICAL_WIDTH = 5


@dataclass(frozen=True)
class HLCTimestamp:
    """An immutable HLC timestamp value."""

    physical: int
    logical: int
    node_id: str

    def __str__(self) -> str:
        """Serialize to the canonical string format for SQLite storage."""
        return f"{self.physical:0{_PHYSICAL_WIDTH}d}:{self.logical:0{_LOGICAL_WIDTH}d}:{self.node_id}"

    def __lt__(self, other: HLCTimestamp) -> bool:
        if self.physical != other.physical:
            return self.physical < other.physical
        if self.logical != other.logical:
            return self.logical < other.logical
        return self.node_id < other.node_id

    def __le__(self, other: HLCTimestamp) -> bool:
        return self == other or self < other

    def __gt__(self, other: HLCTimestamp) -> bool:
        return other < self

    def __ge__(self, other: HLCTimestamp) -> bool:
        return self == other or self > other

    @classmethod
    def from_string(cls, s: str) -> HLCTimestamp:
        """Parse an HLC timestamp from its string representation.

        Args:
            s: String in format "{physical}:{logical}:{node_id}"

        Returns:
            Parsed HLCTimestamp.

        Raises:
            ValueError: If the string format is invalid.
        """
        parts = s.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid HLC timestamp format: '{s}'. "
                f"Expected 'physical:logical:node_id'."
            )
        try:
            physical = int(parts[0])
            logical = int(parts[1])
        except ValueError as e:
            raise ValueError(
                f"Invalid HLC timestamp components in '{s}': {e}"
            ) from e
        return cls(physical=physical, logical=logical, node_id=parts[2])

    @classmethod
    def zero(cls, node_id: str = "unknown") -> HLCTimestamp:
        """Create a zero timestamp (used as initial state)."""
        return cls(physical=0, logical=0, node_id=node_id)


def _wall_ms() -> int:
    """Get current wall-clock time in milliseconds since epoch."""
    return int(time.time() * 1000)


class HLC:
    """Hybrid Logical Clock for a single node.

    Maintains monotonically increasing timestamps that encode both physical
    time and logical causality. Used by the CRDT engine to timestamp every
    cell write, enabling Last-HLC-Wins conflict resolution.

    Thread Safety:
        This class is NOT thread-safe. Each thread/async context should
        use its own HLC instance, or external synchronization must be used.
    """

    def __init__(self, node_id: str, wall_clock_fn: Optional[callable] = None):
        """Initialize an HLC for the given node.

        Args:
            node_id: Unique identifier for this node/device/replica.
            wall_clock_fn: Optional function that returns current time in ms.
                           Defaults to system wall clock. Useful for testing.
        """
        if not node_id:
            raise ValueError("node_id must be a non-empty string.")
        self.node_id = node_id
        self._wall_clock_fn = wall_clock_fn or _wall_ms
        self._physical: int = 0
        self._logical: int = 0

    @property
    def current(self) -> HLCTimestamp:
        """Return the current clock state without advancing it."""
        return HLCTimestamp(
            physical=self._physical,
            logical=self._logical,
            node_id=self.node_id,
        )

    def now(self) -> HLCTimestamp:
        """Generate a new HLC timestamp for a local event.

        The algorithm ensures:
        - If wall clock advanced: use new physical time, reset logical to 0
        - If wall clock hasn't advanced: increment logical counter

        This guarantees strictly monotonic timestamps for sequential local events.

        Returns:
            A new HLCTimestamp strictly greater than any previous timestamp
            generated by this clock.
        """
        wall = self._wall_clock_fn()
        if wall > self._physical:
            self._physical = wall
            self._logical = 0
        else:
            self._logical += 1
        return HLCTimestamp(
            physical=self._physical,
            logical=self._logical,
            node_id=self.node_id,
        )

    def receive(self, remote: HLCTimestamp | str) -> HLCTimestamp:
        """Update the clock upon receiving a remote timestamp.

        Implements the HLC receive algorithm:
        - physical = max(local_physical, remote_physical, wall_clock)
        - logical is updated based on which component(s) are maximal

        This ensures that the returned timestamp is greater than both
        the previous local timestamp and the remote timestamp.

        Args:
            remote: The remote HLC timestamp (as HLCTimestamp or string).

        Returns:
            A new HLCTimestamp reflecting the updated clock state.
        """
        if isinstance(remote, str):
            remote = HLCTimestamp.from_string(remote)

        wall = self._wall_clock_fn()
        old_physical = self._physical
        remote_physical = remote.physical
        remote_logical = remote.logical

        self._physical = max(old_physical, remote_physical, wall)

        if self._physical == old_physical == remote_physical:
            # All three are equal — take the max logical and increment
            self._logical = max(self._logical, remote_logical) + 1
        elif self._physical == old_physical:
            # Local physical was highest — just increment local logical
            self._logical += 1
        elif self._physical == remote_physical:
            # Remote physical was highest — start from remote logical + 1
            self._logical = remote_logical + 1
        else:
            # Wall clock was highest — reset logical
            self._logical = 0

        return HLCTimestamp(
            physical=self._physical,
            logical=self._logical,
            node_id=self.node_id,
        )

    def update_to(self, ts: HLCTimestamp | str) -> None:
        """Ensure the clock is at least as advanced as the given timestamp.

        Used when loading state from disk to restore clock position.

        Args:
            ts: Timestamp to advance to (does not go backward).
        """
        if isinstance(ts, str):
            ts = HLCTimestamp.from_string(ts)
        if ts.physical > self._physical:
            self._physical = ts.physical
            self._logical = ts.logical
        elif ts.physical == self._physical and ts.logical > self._logical:
            self._logical = ts.logical

    @staticmethod
    def compare(a: HLCTimestamp | str, b: HLCTimestamp | str) -> int:
        """Compare two HLC timestamps.

        Comparison order:
        1. Physical time (higher wins)
        2. Logical counter (higher wins)
        3. Node ID (lexicographic, for deterministic tiebreaking)

        Args:
            a: First timestamp.
            b: Second timestamp.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b.
        """
        if isinstance(a, str):
            a = HLCTimestamp.from_string(a)
        if isinstance(b, str):
            b = HLCTimestamp.from_string(b)

        if a.physical != b.physical:
            return -1 if a.physical < b.physical else 1
        if a.logical != b.logical:
            return -1 if a.logical < b.logical else 1
        if a.node_id != b.node_id:
            return -1 if a.node_id < b.node_id else 1
        return 0

    @staticmethod
    def max_ts(a: HLCTimestamp | str, b: HLCTimestamp | str) -> HLCTimestamp:
        """Return the greater of two timestamps."""
        if isinstance(a, str):
            a = HLCTimestamp.from_string(a)
        if isinstance(b, str):
            b = HLCTimestamp.from_string(b)
        return a if HLC.compare(a, b) >= 0 else b

    def __repr__(self) -> str:
        return f"HLC(node_id='{self.node_id}', physical={self._physical}, logical={self._logical})"
