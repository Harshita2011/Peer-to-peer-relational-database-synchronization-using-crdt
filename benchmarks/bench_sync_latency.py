#!/usr/bin/env python3
"""Sync Latency Benchmark — measures write, delta, and end-to-end sync latency.

Sections:
  1. Local Write Latency (P50/P90/P99) for insert/update/delete
  2. Delta Generation & Application Time vs row count
  3. End-to-End Sync Latency vs peer count

Produces:
  paper/figures/fig_write_latency.png
  paper/figures/fig_sync_latency.png
  paper/figures/fig_delta_gen_apply.png
  paper/figures/fig_peer_scaling.png
  benchmarks/results/sync_latency.json
"""

from __future__ import annotations

import sys
import time
import random
import numpy as np
from pathlib import Path

# ── path fixup ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from bench_utils import (
    BenchmarkResult,
    run_benchmark,
    setup_engines,
    setup_bridge,
    save_results,
    apply_plot_style,
    save_figure,
    COLORS,
    print_header,
    print_results_table,
)

import matplotlib.pyplot as plt

# ── constants ───────────────────────────────────────────────────────────
RNG_SEED = 42
WRITE_ROW_COUNTS = [100, 1_000, 10_000]
DELTA_ROW_COUNTS = [100, 500, 1_000, 5_000]
PEER_COUNTS = [2, 5, 10]
WRITE_ITERATIONS = 100


# ── helpers ─────────────────────────────────────────────────────────────

def _make_engine(node_id: str) -> CRDTEngine:
    """Create an in-memory engine with a simple schema."""
    engine = CRDTEngine(":memory:", node_id)
    engine.register_table(
        "items",
        primary_key="id",
        columns=["id", "name", "value", "category"],
    )
    return engine


def _populate(engine: CRDTEngine, n: int, rng: random.Random, prefix: str = "") -> list[str]:
    """Insert *n* rows and return their IDs."""
    ids: list[str] = []
    for i in range(n):
        row_id = engine.insert("items", {
            "id": f"{prefix}row_{i}",
            "name": f"item_{rng.randint(0, 1_000_000)}",
            "value": str(rng.randint(1, 10_000)),
            "category": rng.choice(["A", "B", "C", "D"]),
        })
        ids.append(row_id)
    return ids


def _percentiles(timings_ns: list[int]) -> dict[str, float]:
    """Return P50/P90/P99 in **microseconds**."""
    arr = np.array(timings_ns, dtype=np.float64) / 1_000  # ns → µs
    return {
        "p50_us": float(np.percentile(arr, 50)),
        "p90_us": float(np.percentile(arr, 90)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_us": float(np.mean(arr)),
    }


# =====================================================================
# 1. Local Write Latency
# =====================================================================

def bench_write_latency() -> list[BenchmarkResult]:
    """Measure insert / update / delete latency at different DB sizes."""
    print_header("1. Local Write Latency (P50/P90/P99)")
    results: list[BenchmarkResult] = []
    rng = random.Random(RNG_SEED)

    for row_count in WRITE_ROW_COUNTS:
        print(f"\n  ▸ Pre-populating {row_count:,} rows …", end=" ", flush=True)
        engine = _make_engine("write_bench")
        existing_ids = _populate(engine, row_count, rng, prefix=f"rc{row_count}_")
        print("done.")

        # --- INSERT latency ---
        insert_times: list[int] = []
        for i in range(WRITE_ITERATIONS):
            row = {
                "id": f"ins_{row_count}_{i}",
                "name": f"new_{rng.randint(0, 1_000_000)}",
                "value": str(rng.randint(1, 10_000)),
                "category": rng.choice(["A", "B", "C", "D"]),
            }
            t0 = time.perf_counter_ns()
            engine.insert("items", row)
            t1 = time.perf_counter_ns()
            insert_times.append(t1 - t0)

        stats = _percentiles(insert_times)
        results.append(BenchmarkResult(
            name=f"insert_{row_count}",
            value=stats["p50_us"],
            unit="µs",
            params={"row_count": row_count, "op": "insert", **stats},
        ))
        print(f"    INSERT  P50={stats['p50_us']:.1f} µs  P90={stats['p90_us']:.1f} µs  P99={stats['p99_us']:.1f} µs")

        # --- UPDATE latency ---
        update_times: list[int] = []
        sample_ids = rng.sample(existing_ids, min(WRITE_ITERATIONS, len(existing_ids)))
        for rid in sample_ids:
            changes = {"value": str(rng.randint(1, 99_999))}
            t0 = time.perf_counter_ns()
            try:
                engine.update("items", rid, changes)
            except ValueError:
                pass  # row may have been deleted
            t1 = time.perf_counter_ns()
            update_times.append(t1 - t0)

        stats = _percentiles(update_times)
        results.append(BenchmarkResult(
            name=f"update_{row_count}",
            value=stats["p50_us"],
            unit="µs",
            params={"row_count": row_count, "op": "update", **stats},
        ))
        print(f"    UPDATE  P50={stats['p50_us']:.1f} µs  P90={stats['p90_us']:.1f} µs  P99={stats['p99_us']:.1f} µs")

        # --- DELETE latency ---
        delete_times: list[int] = []
        delete_ids = rng.sample(existing_ids, min(WRITE_ITERATIONS, len(existing_ids)))
        for rid in delete_ids:
            t0 = time.perf_counter_ns()
            try:
                engine.delete("items", rid)
            except ValueError:
                pass  # already deleted
            t1 = time.perf_counter_ns()
            delete_times.append(t1 - t0)

        stats = _percentiles(delete_times)
        results.append(BenchmarkResult(
            name=f"delete_{row_count}",
            value=stats["p50_us"],
            unit="µs",
            params={"row_count": row_count, "op": "delete", **stats},
        ))
        print(f"    DELETE  P50={stats['p50_us']:.1f} µs  P90={stats['p90_us']:.1f} µs  P99={stats['p99_us']:.1f} µs")

        engine.close()

    return results


# =====================================================================
# 2. Delta Generation & Application Time
# =====================================================================

def bench_delta_gen_apply() -> list[BenchmarkResult]:
    """Measure get_delta() and apply_delta() time vs row count."""
    print_header("2. Delta Generation & Application Time")
    results: list[BenchmarkResult] = []
    rng = random.Random(RNG_SEED)

    for row_count in DELTA_ROW_COUNTS:
        print(f"\n  ▸ {row_count:,} rows …", end=" ", flush=True)

        engine_a = _make_engine("delta_A")
        engine_b = _make_engine("delta_B")
        _populate(engine_a, row_count, rng, prefix=f"drc{row_count}_")

        # --- get_delta ---
        t0 = time.perf_counter_ns()
        # Use a dummy seq "since" the beginning
        delta = engine_a.get_delta(since_seq=0, for_peer="delta_B")
        t1 = time.perf_counter_ns()
        gen_us = (t1 - t0) / 1_000

        results.append(BenchmarkResult(
            name=f"get_delta_{row_count}",
            value=gen_us,
            unit="µs",
            params={"row_count": row_count, "delta_size": delta.size},
        ))

        # --- apply_delta ---
        t0 = time.perf_counter_ns()
        engine_b.apply_delta(delta, from_peer="delta_A")
        t1 = time.perf_counter_ns()
        apply_us = (t1 - t0) / 1_000

        results.append(BenchmarkResult(
            name=f"apply_delta_{row_count}",
            value=apply_us,
            unit="µs",
            params={"row_count": row_count, "delta_size": delta.size},
        ))

        print(f"delta_size={delta.size:,}  get_delta={gen_us:,.0f} µs  apply_delta={apply_us:,.0f} µs")

        engine_a.close()
        engine_b.close()

    return results


# =====================================================================
# 3. End-to-End Sync Latency
# =====================================================================

def bench_e2e_sync() -> list[BenchmarkResult]:
    """Measure sync_all and sync_until_converged across peer counts."""
    print_header("3. End-to-End Sync Latency")
    results: list[BenchmarkResult] = []
    rng = random.Random(RNG_SEED)
    rows_per_peer = 100

    for n_peers in PEER_COUNTS:
        print(f"\n  ▸ {n_peers} peers, {rows_per_peer} rows each …", end=" ", flush=True)

        bridge = LocalSyncBridge()
        engines: list[CRDTEngine] = []
        for p in range(n_peers):
            eng = _make_engine(f"peer_{p}")
            bridge.register_peer(eng)
            _populate(eng, rows_per_peer, rng, prefix=f"p{p}_")
            engines.append(eng)

        # --- sync_all ---
        t0 = time.perf_counter_ns()
        bridge.sync_all()
        t1 = time.perf_counter_ns()
        sync_all_us = (t1 - t0) / 1_000

        results.append(BenchmarkResult(
            name=f"sync_all_{n_peers}",
            value=sync_all_us,
            unit="µs",
            params={"n_peers": n_peers, "rows_per_peer": rows_per_peer},
        ))

        # Reset bridge state for a clean convergence measurement
        bridge.reset_sync_state()

        # --- sync_until_converged ---
        t0 = time.perf_counter_ns()
        rounds = bridge.sync_until_converged(max_rounds=10)
        t1 = time.perf_counter_ns()
        converge_us = (t1 - t0) / 1_000

        results.append(BenchmarkResult(
            name=f"sync_converge_{n_peers}",
            value=converge_us,
            unit="µs",
            params={"n_peers": n_peers, "rounds": rounds, "rows_per_peer": rows_per_peer},
        ))

        print(f"sync_all={sync_all_us:,.0f} µs  converge={converge_us:,.0f} µs ({rounds} rounds)")

        for eng in engines:
            eng.close()

    return results


# =====================================================================
# Plotting
# =====================================================================

def plot_write_latency(results: list[BenchmarkResult]) -> None:
    """fig_write_latency.png — grouped bar chart of write latency."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ops = ["insert", "update", "delete"]
    op_colors = [COLORS[0], COLORS[1], COLORS[2]]
    x = np.arange(len(WRITE_ROW_COUNTS))
    bar_width = 0.25

    for i, op in enumerate(ops):
        p50s = []
        p99s = []
        for rc in WRITE_ROW_COUNTS:
            match = [r for r in results if r.name == f"{op}_{rc}"]
            if match:
                p50s.append(match[0].params["p50_us"])
                p99s.append(match[0].params["p99_us"])
            else:
                p50s.append(0)
                p99s.append(0)

        # P99 as error bar (top whisker only)
        yerr_low = [0] * len(p50s)
        yerr_high = [p99 - p50 for p50, p99 in zip(p50s, p99s)]
        ax.bar(
            x + i * bar_width,
            p50s,
            bar_width,
            label=op.capitalize(),
            color=op_colors[i],
            alpha=0.9,
            yerr=[yerr_low, yerr_high],
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

    ax.set_xlabel("Pre-populated Row Count", fontsize=12)
    ax.set_ylabel("Latency (µs)", fontsize=12)
    ax.set_title("Local Write Latency — P50 (bars) + P99 (whiskers)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{rc:,}" for rc in WRITE_ROW_COUNTS])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig_write_latency.png")
    plt.close(fig)
    print("  ✓ Saved fig_write_latency.png")


def plot_sync_latency(delta_results: list[BenchmarkResult]) -> None:
    """fig_sync_latency.png — bidirectional sync time vs delta size."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    # For each row count we have get_delta + apply_delta; total ≈ one sync_bidirectional
    delta_sizes = []
    sync_times_us = []
    for rc in DELTA_ROW_COUNTS:
        gen = [r for r in delta_results if r.name == f"get_delta_{rc}"]
        app = [r for r in delta_results if r.name == f"apply_delta_{rc}"]
        if gen and app:
            total_us = gen[0].value + app[0].value
            size = gen[0].params.get("delta_size", rc)
            delta_sizes.append(size)
            sync_times_us.append(total_us)

    ax.plot(delta_sizes, sync_times_us, "o-", color=COLORS[0], linewidth=2,
            markersize=8, label="sync_bidirectional (get + apply)")
    ax.set_xlabel("Delta Size (entries)", fontsize=12)
    ax.set_ylabel("Latency (µs)", fontsize=12)
    ax.set_title("Sync Latency vs Delta Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig_sync_latency.png")
    plt.close(fig)
    print("  ✓ Saved fig_sync_latency.png")


def plot_delta_gen_apply(delta_results: list[BenchmarkResult]) -> None:
    """fig_delta_gen_apply.png — get_delta / apply_delta time vs rows."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    gen_times = []
    apply_times = []
    rcs = []
    for rc in DELTA_ROW_COUNTS:
        gen = [r for r in delta_results if r.name == f"get_delta_{rc}"]
        app = [r for r in delta_results if r.name == f"apply_delta_{rc}"]
        if gen and app:
            rcs.append(rc)
            gen_times.append(gen[0].value)
            apply_times.append(app[0].value)

    ax.plot(rcs, gen_times, "s-", color=COLORS[0], linewidth=2, markersize=8, label="get_delta()")
    ax.plot(rcs, apply_times, "D-", color=COLORS[1], linewidth=2, markersize=8, label="apply_delta()")
    ax.set_xlabel("Row Count", fontsize=12)
    ax.set_ylabel("Latency (µs)", fontsize=12)
    ax.set_title("Delta Generation vs Application Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig_delta_gen_apply.png")
    plt.close(fig)
    print("  ✓ Saved fig_delta_gen_apply.png")


def plot_peer_scaling(e2e_results: list[BenchmarkResult]) -> None:
    """fig_peer_scaling.png — sync_all time vs peer count."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    times = []
    labels = []
    for np_ in PEER_COUNTS:
        match = [r for r in e2e_results if r.name == f"sync_all_{np_}"]
        if match:
            times.append(match[0].value)
            labels.append(str(np_))

    x = np.arange(len(labels))
    bars = ax.bar(x, times, color=COLORS[0], alpha=0.9, width=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="white",
        )

    ax.set_xlabel("Number of Peers", fontsize=12)
    ax.set_ylabel("sync_all() Latency (µs)", fontsize=12)
    ax.set_title("Sync Scaling — Full Mesh sync_all()", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig_peer_scaling.png")
    plt.close(fig)
    print("  ✓ Saved fig_peer_scaling.png")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 64)
    print("  CRDT Sync-Latency Benchmark Suite")
    print("=" * 64)

    all_results: list[BenchmarkResult] = []

    # 1. Write latency
    write_results = bench_write_latency()
    all_results.extend(write_results)

    # 2. Delta gen / apply
    delta_results = bench_delta_gen_apply()
    all_results.extend(delta_results)

    # 3. End-to-end sync
    e2e_results = bench_e2e_sync()
    all_results.extend(e2e_results)

    # ── Save JSON ────────────────────────────────────────────────────
    save_results(all_results, "sync_latency.json")
    print("\n  ✓ Saved benchmarks/results/sync_latency.json")

    # ── Plot ─────────────────────────────────────────────────────────
    print_header("Generating Figures")
    plot_write_latency(write_results)
    plot_sync_latency(delta_results)
    plot_delta_gen_apply(delta_results)
    plot_peer_scaling(e2e_results)

    # ── Summary table ────────────────────────────────────────────────
    print_header("Summary")
    print_results_table(all_results)


if __name__ == "__main__":
    main()
