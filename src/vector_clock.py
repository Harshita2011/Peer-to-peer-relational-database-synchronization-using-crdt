"""Vector Clock implementation for causal ordering and concurrency detection.

Implements a standard Version Vector / Vector Clock map of {node_id: sequence_number}.
Provides operations to compare two clocks to determine if one dominates the other
or if they are concurrent.

Used by the Cell-Level Multi-Value Register and Remove-Wins Tombstone components.
"""

from __future__ import annotations
import json
from enum import Enum


class ClockRelation(Enum):
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    EQUAL = "equal"
    CONCURRENT = "concurrent"


class VectorClock:
    """A vector clock mapping node_id to an integer sequence number."""

    def __init__(self, state: dict[str, int] | None = None):
        self.state = state.copy() if state else {}

    @classmethod
    def from_string(cls, s: str) -> VectorClock:
        """Parse from JSON string."""
        if not s:
            return cls()
        return cls(json.loads(s))

    def __str__(self) -> str:
        """Serialize to JSON string for SQLite storage."""
        return json.dumps(self.state, sort_keys=True)

    def increment(self, node_id: str) -> VectorClock:
        """Return a new VectorClock with the node's sequence incremented."""
        new_state = self.state.copy()
        new_state[node_id] = new_state.get(node_id, 0) + 1
        return VectorClock(new_state)

    def merge(self, other: VectorClock) -> VectorClock:
        """Return a new VectorClock representing the point-wise maximum."""
        new_state = self.state.copy()
        for node_id, seq in other.state.items():
            if seq > new_state.get(node_id, 0):
                new_state[node_id] = seq
        return VectorClock(new_state)

    def compare(self, other: VectorClock) -> ClockRelation:
        """Compare this vector clock to another.

        Returns:
            EQUAL if all sequences match exactly.
            GREATER_THAN if this clock has all knowledge of the other, and at least one strictly greater sequence.
            LESS_THAN if the other clock has all knowledge of this, and at least one strictly greater sequence.
            CONCURRENT if both clocks have at least one sequence greater than the other.
        """
        is_greater = False
        is_less = False

        all_nodes = set(self.state.keys()).union(set(other.state.keys()))

        for node_id in all_nodes:
            self_seq = self.state.get(node_id, 0)
            other_seq = other.state.get(node_id, 0)

            if self_seq > other_seq:
                is_greater = True
            elif self_seq < other_seq:
                is_less = True

            if is_greater and is_less:
                return ClockRelation.CONCURRENT

        if is_greater:
            return ClockRelation.GREATER_THAN
        elif is_less:
            return ClockRelation.LESS_THAN
        else:
            return ClockRelation.EQUAL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self.compare(other) == ClockRelation.EQUAL
