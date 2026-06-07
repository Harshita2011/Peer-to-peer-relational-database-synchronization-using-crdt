"""Tests for the Hybrid Logical Clock (HLC) module."""

import time
import pytest

from src.hlc import HLC, HLCTimestamp


class TestHLCTimestamp:
    """Tests for the HLCTimestamp data class."""

    def test_string_serialization(self):
        ts = HLCTimestamp(physical=1000000000000, logical=42, node_id="device_A")
        s = str(ts)
        assert s == "1000000000000:00042:device_A"

    def test_string_parsing(self):
        ts = HLCTimestamp.from_string("1000000000000:00042:device_A")
        assert ts.physical == 1000000000000
        assert ts.logical == 42
        assert ts.node_id == "device_A"

    def test_roundtrip(self):
        original = HLCTimestamp(physical=1717777777777, logical=99, node_id="node_X")
        parsed = HLCTimestamp.from_string(str(original))
        assert parsed == original

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            HLCTimestamp.from_string("invalid")

    def test_zero(self):
        ts = HLCTimestamp.zero("test")
        assert ts.physical == 0
        assert ts.logical == 0
        assert ts.node_id == "test"

    def test_comparison_physical(self):
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=200, logical=0, node_id="A")
        assert a < b
        assert b > a
        assert not a > b

    def test_comparison_logical(self):
        a = HLCTimestamp(physical=100, logical=1, node_id="A")
        b = HLCTimestamp(physical=100, logical=2, node_id="A")
        assert a < b

    def test_comparison_node_id(self):
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=100, logical=0, node_id="B")
        assert a < b

    def test_lexicographic_ordering_matches_semantic_ordering(self):
        """Verify that string comparison matches semantic comparison.
        This is critical for SQLite queries using > operator."""
        ts1 = HLCTimestamp(physical=100, logical=0, node_id="A")
        ts2 = HLCTimestamp(physical=200, logical=0, node_id="A")
        ts3 = HLCTimestamp(physical=200, logical=5, node_id="A")
        ts4 = HLCTimestamp(physical=200, logical=5, node_id="B")

        timestamps = [ts4, ts2, ts1, ts3]
        sorted_semantic = sorted(timestamps)
        sorted_string = sorted(timestamps, key=lambda t: str(t))
        assert sorted_semantic == sorted_string

    def test_equality(self):
        """Two HLCTimestamps with identical fields are equal."""
        a = HLCTimestamp(physical=500, logical=3, node_id="X")
        b = HLCTimestamp(physical=500, logical=3, node_id="X")
        assert a == b
        assert not a < b
        assert not a > b

    def test_le_ge(self):
        """<= and >= operators work correctly."""
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=200, logical=0, node_id="A")
        c = HLCTimestamp(physical=100, logical=0, node_id="A")
        assert a <= b
        assert b >= a
        assert a <= c
        assert a >= c

    def test_zero_default_node_id(self):
        """zero() with no args uses 'unknown' as node_id."""
        ts = HLCTimestamp.zero()
        assert ts.node_id == "unknown"

    def test_from_string_colon_in_node_id(self):
        """Node IDs containing colons are parsed correctly (split on first 2 only)."""
        ts = HLCTimestamp(physical=999, logical=1, node_id="host:port:extra")
        roundtripped = HLCTimestamp.from_string(str(ts))
        assert roundtripped.node_id == "host:port:extra"
        assert roundtripped.physical == 999
        assert roundtripped.logical == 1


class TestHLC:
    """Tests for the HLC clock."""

    def test_monotonicity(self):
        """now() always returns strictly increasing timestamps."""
        clock = HLC("test", wall_clock_fn=lambda: 1000)
        timestamps = [clock.now() for _ in range(100)]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_wall_clock_advance(self):
        """Physical time advance resets logical counter."""
        t = [1000]
        clock = HLC("test", wall_clock_fn=lambda: t[0])

        ts1 = clock.now()  # 1000:0
        ts2 = clock.now()  # 1000:1
        assert ts2.logical == 1

        t[0] = 2000
        ts3 = clock.now()  # 2000:0
        assert ts3.physical == 2000
        assert ts3.logical == 0

    def test_receive_remote_ahead(self):
        """Receiving a timestamp from the future advances the clock."""
        clock = HLC("local", wall_clock_fn=lambda: 1000)
        clock.now()  # Initialize

        remote = HLCTimestamp(physical=5000, logical=10, node_id="remote")
        ts = clock.receive(remote)

        assert ts.physical == 5000
        assert ts.logical == 11  # remote_logical + 1
        assert ts.node_id == "local"

    def test_receive_remote_behind(self):
        """Receiving an old timestamp doesn't move clock backward."""
        clock = HLC("local", wall_clock_fn=lambda: 5000)
        clock.now()  # Initialize at 5000

        remote = HLCTimestamp(physical=1000, logical=0, node_id="remote")
        ts = clock.receive(remote)

        assert ts.physical == 5000
        assert ts.logical >= 1  # Advanced from local state

    def test_receive_same_physical(self):
        """Receiving a timestamp with same physical time merges logical counters."""
        clock = HLC("local", wall_clock_fn=lambda: 1000)
        clock.now()  # 1000:0
        clock.now()  # 1000:1
        clock.now()  # 1000:2

        remote = HLCTimestamp(physical=1000, logical=5, node_id="remote")
        ts = clock.receive(remote)

        assert ts.physical == 1000
        assert ts.logical == 6  # max(2, 5) + 1

    def test_receive_string_format(self):
        """Receive works with string timestamps."""
        clock = HLC("local", wall_clock_fn=lambda: 1000)
        ts = clock.receive("0000000005000:00010:remote")
        assert ts.physical == 5000

    def test_compare(self):
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=200, logical=0, node_id="B")
        assert HLC.compare(a, b) == -1
        assert HLC.compare(b, a) == 1
        assert HLC.compare(a, a) == 0

    def test_compare_strings(self):
        assert HLC.compare("0000000000100:00000:A", "0000000000200:00000:B") == -1

    def test_max_ts(self):
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=200, logical=0, node_id="B")
        assert HLC.max_ts(a, b) == b
        assert HLC.max_ts(b, a) == b

    def test_empty_node_id_raises(self):
        with pytest.raises(ValueError):
            HLC("")

    def test_update_to(self):
        clock = HLC("test", wall_clock_fn=lambda: 100)
        clock.now()  # 100:0

        clock.update_to(HLCTimestamp(physical=500, logical=10, node_id="other"))
        ts = clock.now()
        assert ts.physical >= 500

    def test_current_property(self):
        clock = HLC("test", wall_clock_fn=lambda: 1000)
        ts = clock.now()
        current = clock.current
        assert current.physical == ts.physical
        assert current.logical == ts.logical

    def test_update_to_does_not_go_backward(self):
        """update_to with an older timestamp does not regress the clock."""
        clock = HLC("test", wall_clock_fn=lambda: 5000)
        clock.now()  # 5000:0

        clock.update_to(HLCTimestamp(physical=100, logical=0, node_id="old"))
        current = clock.current
        assert current.physical == 5000
        assert current.logical == 0

    def test_update_to_string(self):
        """update_to accepts string-format timestamps."""
        clock = HLC("test", wall_clock_fn=lambda: 100)
        clock.now()
        clock.update_to("0000000000500:00010:other")
        ts = clock.now()
        assert ts.physical >= 500

    def test_max_ts_with_strings(self):
        """max_ts works with string inputs."""
        result = HLC.max_ts("0000000000100:00000:A", "0000000000200:00000:B")
        assert result.physical == 200
        assert result.node_id == "B"

    def test_repr(self):
        """HLC repr includes node_id and state."""
        clock = HLC("my_node", wall_clock_fn=lambda: 42)
        clock.now()
        r = repr(clock)
        assert "my_node" in r
        assert "42" in r

    def test_max_ts_equal(self):
        """max_ts with equal timestamps returns the first one."""
        a = HLCTimestamp(physical=100, logical=0, node_id="A")
        b = HLCTimestamp(physical=100, logical=0, node_id="A")
        result = HLC.max_ts(a, b)
        assert result == a

    def test_compare_logical_tiebreak(self):
        """compare resolves logical counter ties correctly."""
        a = HLCTimestamp(physical=100, logical=3, node_id="A")
        b = HLCTimestamp(physical=100, logical=7, node_id="A")
        assert HLC.compare(a, b) == -1
        assert HLC.compare(b, a) == 1

    def test_compare_node_id_tiebreak(self):
        """compare resolves node_id ties correctly."""
        a = HLCTimestamp(physical=100, logical=0, node_id="alpha")
        b = HLCTimestamp(physical=100, logical=0, node_id="beta")
        assert HLC.compare(a, b) == -1
        assert HLC.compare(b, a) == 1
