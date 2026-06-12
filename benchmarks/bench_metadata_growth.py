#!/usr/bin/env python3
"""Benchmark: Metadata growth, compaction effectiveness, and storage overhead.

Measures:
1. How metadata (CRDT cells / app rows) grows over mixed insert/update/delete ops
2. How compaction reduces accumulated metadata
3. Storage overhead of CRDT engine vs plain SQLite

Produces:
- paper/figures/fig_metadata_ratio.png    — dual-axis metadata ratio + tombstones
- paper/figures/fig_storage_overhead.png  — horizontal bar chart CRDT vs plain SQLite
- paper/figures/fig_compaction_savings.png — compaction before/after line chart
- benchmarks/results/bench_metadata_growth.json — raw data
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.clock_manager import ClockManager
from src.compaction import CompactionEngine

from bench_utils import (
    setup_engines,
    setup_bridge,
    save_results,
    apply_plot_style,
    save_figure,
    COLORS,
    print_header,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOTAL_OPS = 5000
SAMPLE_EVERY = 250       # record metrics every N ops
SYNC_EVERY = 500          # sync between peers every N ops
INSERT_PCT = 0.60
UPDATE_PCT = 0.30
# DELETE_PCT = 0.10 (remainder)

STORAGE_ROWS = 1000       # rows for storage overhead test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_crdt_cells(engine: CRDTEngine) -> int:
    """Total rows in _crdt_cells for this engine."""
    row = engine.conn.execute("SELECT COUNT(*) FROM _crdt_cells").fetchone()
    return row[0] if row else 0


def _count_tombstones(engine: CRDTEngine) -> int:
    """Unresolved tombstones for this engine."""
    row = engine.conn.execute(
        "SELECT COUNT(*) FROM _tombstones"
    ).fetchone()
    return row[0] if row else 0


def _count_app_rows(engine: CRDTEngine, table: str) -> int:
    """Live (non-tombstoned) rows via engine.query."""
    return len(engine.query(table))


def _db_size_bytes(path: str) -> int:
    """Database file size via PRAGMA page_count * page_size."""
    conn = sqlite3.connect(path)
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()
    return page_count * page_size


# ---------------------------------------------------------------------------
# 1.  Metadata Ratio Over Time
# ---------------------------------------------------------------------------

def bench_metadata_ratio(
    engines: list[CRDTEngine],
    bridge: LocalSyncBridge,
    rng: random.Random,
) -> dict:
    """Run mixed ops and track metadata ratio + tombstones over time."""

    print_header("Metadata Ratio Over Time")
    engine_a, engine_b = engines[0], engines[1]
    table = "items"

    # We'll alternate which engine receives an op
    active_ids: list[str] = []          # IDs that are alive (not deleted)

    snapshots: list[dict] = []          # (op_count, app_rows, crdt_cells, tombstones, ratio)

    for op_idx in range(1, TOTAL_OPS + 1):
        # Pick engine for this op (round-robin)
        engine = engine_a if op_idx % 2 == 0 else engine_b

        roll = rng.random()

        if roll < INSERT_PCT or len(active_ids) == 0:
            # ----- INSERT -----
            row_id = engine.insert(table, {
                "name": f"item_{op_idx}",
                "value": rng.randint(1, 10000),
            })
            active_ids.append(row_id)

        elif roll < INSERT_PCT + UPDATE_PCT:
            # ----- UPDATE -----
            if active_ids:
                target_id = rng.choice(active_ids)
                try:
                    engine.update(table, target_id, {
                        "value": rng.randint(1, 10000),
                    })
                except ValueError:
                    # Row might have been deleted on another engine and not synced yet
                    pass

        else:
            # ----- DELETE -----
            if active_ids:
                target_id = rng.choice(active_ids)
                try:
                    engine.delete(table, target_id)
                    active_ids.remove(target_id)
                except (ValueError, Exception):
                    pass

        # --- Periodic snapshot ---
        if op_idx % SAMPLE_EVERY == 0:
            app_rows = _count_app_rows(engine_a, table)
            crdt_cells = _count_crdt_cells(engine_a)
            tombstones = _count_tombstones(engine_a)
            ratio = crdt_cells / max(app_rows, 1)

            snapshots.append({
                "op_count": op_idx,
                "app_rows": app_rows,
                "crdt_cells": crdt_cells,
                "tombstones": tombstones,
                "ratio": round(ratio, 2),
            })
            print(f"  ops={op_idx:5d}  rows={app_rows:5d}  "
                  f"cells={crdt_cells:6d}  tombstones={tombstones:4d}  "
                  f"ratio={ratio:.2f}")

        # --- Periodic sync ---
        if op_idx % SYNC_EVERY == 0:
            bridge.sync_bidirectional(engine_a, engine_b)
            print(f"  [sync at op {op_idx}]")

    return {"metadata_ratio_snapshots": snapshots}


# ---------------------------------------------------------------------------
# 2.  Compaction Effectiveness
# ---------------------------------------------------------------------------

def bench_compaction(
    engines: list[CRDTEngine],
    bridge: LocalSyncBridge,
    rng: random.Random,
) -> dict:
    """At each sync checkpoint, run compaction and record before/after."""

    print_header("Compaction Effectiveness")
    engine_a, engine_b = engines[0], engines[1]
    table = "items"

    active_ids: list[str] = []
    compaction_snapshots: list[dict] = []

    for op_idx in range(1, TOTAL_OPS + 1):
        engine = engine_a if op_idx % 2 == 0 else engine_b
        roll = rng.random()

        if roll < INSERT_PCT or len(active_ids) == 0:
            row_id = engine.insert(table, {
                "name": f"citem_{op_idx}",
                "value": rng.randint(1, 10000),
            })
            active_ids.append(row_id)
        elif roll < INSERT_PCT + UPDATE_PCT:
            if active_ids:
                target_id = rng.choice(active_ids)
                try:
                    engine.update(table, target_id, {"value": rng.randint(1, 10000)})
                except ValueError:
                    pass
        else:
            if active_ids:
                target_id = rng.choice(active_ids)
                try:
                    engine.delete(table, target_id)
                    active_ids.remove(target_id)
                except (ValueError, Exception):
                    pass

        # --- Sync + compact at every SYNC_EVERY ops ---
        if op_idx % SYNC_EVERY == 0:
            bridge.sync_bidirectional(engine_a, engine_b)

            # Measure compaction on engine_a
            cells_before = _count_crdt_cells(engine_a)

            # Simulate global acknowledgement: every peer has seen every writer's max HLC
            for peer in engines:
                for writer in engines:
                    engine_a.conn.execute(
                        "INSERT OR REPLACE INTO _vector_clocks "
                        "(peer_id, writer_id, max_hlc_ts) VALUES (?, ?, ?)",
                        (peer.node_id, writer.node_id, "999999999999999"),
                    )
            engine_a.conn.commit()

            clock_mgr = ClockManager(engine_a.conn)
            compactor = CompactionEngine(engine_a.conn, clock_mgr)
            result = compactor.compact()

            cells_after = _count_crdt_cells(engine_a)
            savings = result.savings_pct

            compaction_snapshots.append({
                "op_count": op_idx,
                "cells_before": cells_before,
                "cells_after": cells_after,
                "pruned": result.cell_versions_pruned,
                "savings_pct": round(savings, 2),
            })
            print(f"  ops={op_idx:5d}  before={cells_before:6d}  "
                  f"after={cells_after:6d}  pruned={result.cell_versions_pruned:5d}  "
                  f"savings={savings:.1f}%")

    return {"compaction_snapshots": compaction_snapshots}


# ---------------------------------------------------------------------------
# 3.  Storage Overhead vs Plain SQLite
# ---------------------------------------------------------------------------

def bench_storage_overhead(rng: random.Random) -> dict:
    """Compare file-based CRDT engine size to plain SQLite with same data."""

    print_header("Storage Overhead vs Plain SQLite")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / "_tmp_storage_bench"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        crdt_path_a = str(tmp_dir / "crdt_a.db")
        crdt_path_b = str(tmp_dir / "crdt_b.db")
        plain_path = str(tmp_dir / "plain.db")

        # --- CRDT engines ---
        ea = CRDTEngine(crdt_path_a, "storA")
        eb = CRDTEngine(crdt_path_b, "storB")
        for e in (ea, eb):
            e.register_table("items", primary_key="id",
                             columns=["id", "name", "value"])

        bridge = LocalSyncBridge()
        bridge.register_peer(ea)
        bridge.register_peer(eb)

        rows_data = []
        for i in range(STORAGE_ROWS):
            name = f"item_{i}"
            value = rng.randint(1, 100000)
            rows_data.append((name, value))
            ea.insert("items", {"name": name, "value": value})

        bridge.sync_bidirectional(ea, eb)

        # Flush WAL
        ea.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        eb.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        ea.close()
        eb.close()

        crdt_size_a = _db_size_bytes(crdt_path_a)
        crdt_size_b = _db_size_bytes(crdt_path_b)
        crdt_size = (crdt_size_a + crdt_size_b) / 2  # average

        # --- Plain SQLite ---
        conn = sqlite3.connect(plain_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE items (id TEXT PRIMARY KEY, name TEXT, value TEXT)"
        )
        for i, (name, value) in enumerate(rows_data):
            conn.execute(
                "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                (f"plain_{i}", name, str(value)),
            )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        plain_size = _db_size_bytes(plain_path)

        overhead_x = crdt_size / max(plain_size, 1)

        print(f"  Plain SQLite : {plain_size / 1024:.1f} KB")
        print(f"  CRDT Engine  : {crdt_size / 1024:.1f} KB")
        print(f"  Overhead     : {overhead_x:.2f}x")

        return {
            "plain_sqlite_bytes": plain_size,
            "crdt_engine_bytes": int(crdt_size),
            "overhead_multiplier": round(overhead_x, 2),
        }

    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metadata_ratio(snapshots: list[dict]) -> None:
    """Dual-axis: metadata ratio (left) + tombstone count (right) over ops."""
    apply_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ops = [s["op_count"] for s in snapshots]
    ratios = [s["ratio"] for s in snapshots]
    tombstones = [s["tombstones"] for s in snapshots]

    # Left axis — ratio
    color_ratio = COLORS[0]
    ax1.plot(ops, ratios, color=color_ratio, marker="o", linewidth=2.2,
             markersize=6, label="Metadata Ratio (cells / rows)", zorder=3)
    ax1.set_xlabel("Operations", fontsize=12)
    ax1.set_ylabel("Metadata Ratio  (cells / rows)", fontsize=12, color=color_ratio)
    ax1.tick_params(axis="y", labelcolor=color_ratio)
    ax1.set_xlim(ops[0] - 100, ops[-1] + 100)

    # Right axis — tombstones
    ax2 = ax1.twinx()
    color_tomb = COLORS[3]
    ax2.plot(ops, tombstones, color=color_tomb, marker="s", linewidth=2.2,
             markersize=6, linestyle="--", label="Tombstone Count", zorder=3)
    ax2.set_ylabel("Tombstone Count", fontsize=12, color=color_tomb)
    ax2.tick_params(axis="y", labelcolor=color_tomb)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=10, framealpha=0.85)

    ax1.set_title("Metadata Ratio & Tombstones Over Time", fontsize=14, pad=12)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "fig_metadata_ratio.png")
    plt.close(fig)


def plot_storage_overhead(storage: dict) -> None:
    """Horizontal bar chart: Plain SQLite vs CRDT Engine (KB)."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["Plain SQLite", "CRDT Engine"]
    sizes_kb = [
        storage["plain_sqlite_bytes"] / 1024,
        storage["crdt_engine_bytes"] / 1024,
    ]
    bar_colors = [COLORS[2], COLORS[0]]

    bars = ax.barh(labels, sizes_kb, color=bar_colors, height=0.45, edgecolor="white",
                   linewidth=0.6, zorder=3)

    # Annotate overhead multiplier on CRDT bar
    overhead = storage["overhead_multiplier"]
    crdt_bar = bars[1]
    ax.text(
        crdt_bar.get_width() + max(sizes_kb) * 0.03,
        crdt_bar.get_y() + crdt_bar.get_height() / 2,
        f"{overhead:.1f}×",
        va="center", ha="left", fontsize=13, fontweight="bold",
        color=COLORS[0],
    )

    # Value labels inside bars
    for bar, val in zip(bars, sizes_kb):
        ax.text(
            bar.get_width() * 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} KB",
            va="center", ha="center", fontsize=11, color="white", fontweight="bold",
        )

    ax.set_xlabel("Database Size (KB)", fontsize=12)
    ax.set_title("Storage Overhead: CRDT Engine vs Plain SQLite", fontsize=14, pad=12)
    ax.set_xlim(0, max(sizes_kb) * 1.25)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "fig_storage_overhead.png")
    plt.close(fig)


def plot_compaction_savings(snapshots: list[dict]) -> None:
    """Line chart: cells before vs after compaction, with savings % annotations."""
    apply_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ops = [s["op_count"] for s in snapshots]
    before = [s["cells_before"] for s in snapshots]
    after = [s["cells_after"] for s in snapshots]
    savings = [s["savings_pct"] for s in snapshots]

    ax1.plot(ops, before, color=COLORS[1], marker="o", linewidth=2.2,
             markersize=6, label="Before Compaction", zorder=3)
    ax1.plot(ops, after, color=COLORS[0], marker="s", linewidth=2.2,
             markersize=6, label="After Compaction", zorder=3)

    # Fill between for visual emphasis
    ax1.fill_between(ops, after, before, alpha=0.12, color=COLORS[1])

    ax1.set_xlabel("Operations", fontsize=12)
    ax1.set_ylabel("CRDT Cells", fontsize=12)
    ax1.set_xlim(ops[0] - 100, ops[-1] + 100)

    # Annotate savings % at each checkpoint
    for x, y_b, y_a, s in zip(ops, before, after, savings):
        mid_y = (y_b + y_a) / 2
        if s > 0:
            ax1.annotate(
                f"−{s:.0f}%",
                xy=(x, mid_y),
                fontsize=9, fontweight="bold",
                color=COLORS[4] if len(COLORS) > 4 else COLORS[0],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none"),
            )

    ax1.legend(fontsize=10, loc="upper left", framealpha=0.85)
    ax1.set_title("Compaction Effectiveness Over Time", fontsize=14, pad=12)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "fig_compaction_savings.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  BENCHMARK: Metadata Growth & Compaction")
    print("=" * 70)
    t0 = time.perf_counter_ns()

    # --- Setup for tests 1 & 2 (in-memory, 'simple' schema) ---
    # Test 1: Metadata ratio
    rng1 = random.Random(SEED)
    engines_1 = setup_engines(2, prefix="meta")
    bridge_1 = setup_bridge(engines_1)

    ratio_results = bench_metadata_ratio(engines_1, bridge_1, rng1)

    # Test 2: Compaction (fresh engines so results are independent)
    rng2 = random.Random(SEED)
    engines_2 = setup_engines(2, prefix="comp")
    bridge_2 = setup_bridge(engines_2)

    compaction_results = bench_compaction(engines_2, bridge_2, rng2)

    # Test 3: Storage overhead (file-based)
    rng3 = random.Random(SEED)
    storage_results = bench_storage_overhead(rng3)

    # --- Aggregate & save ---
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    all_results = {
        **ratio_results,
        **compaction_results,
        "storage_overhead": storage_results,
        "config": {
            "total_ops": TOTAL_OPS,
            "sample_every": SAMPLE_EVERY,
            "sync_every": SYNC_EVERY,
            "insert_pct": INSERT_PCT,
            "update_pct": UPDATE_PCT,
            "delete_pct": round(1 - INSERT_PCT - UPDATE_PCT, 2),
            "storage_rows": STORAGE_ROWS,
            "seed": SEED,
        },
        "elapsed_ms": round(elapsed_ms, 1),
    }

    save_results(all_results, "bench_metadata_growth.json")

    # --- Plots ---
    print_header("Generating Figures")
    plot_metadata_ratio(ratio_results["metadata_ratio_snapshots"])
    plot_storage_overhead(storage_results)
    plot_compaction_savings(compaction_results["compaction_snapshots"])

    print(f"\n{'=' * 70}")
    print(f"  DONE  ({elapsed_ms / 1000:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
