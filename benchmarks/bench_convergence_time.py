#!/usr/bin/env python3
"""Benchmark: Convergence Time

Measures how quickly the CRDT sync engine converges across multiple scenarios:
1. Rounds to converge — number of sync_all() rounds needed
2. Wall-clock convergence time — total ms for sync_until_converged()
3. Partition-heal recovery — time to re-converge after a network partition
4. Hash computation overhead — ConvergenceHasher.compute_hash() latency

Produces:
- paper/figures/fig_convergence_rounds.png   (heatmap)
- paper/figures/fig_convergence_time.png     (line chart)
- paper/figures/fig_partition_recovery.png   (bar chart)
- benchmarks/results/convergence_time.json   (raw data)
"""

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bench_utils import (
    setup_engines,
    setup_bridge,
    save_results,
    apply_plot_style,
    save_figure,
    COLORS,
    print_header,
)

from src.engine import CRDTEngine
from src.sync import LocalSyncBridge
from src.convergence import ConvergenceHasher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
PEER_COUNTS = [2, 3, 5, 10]
OPS_PER_PEER = [10, 50, 100, 500]
PARTITION_PEER_COUNTS = [4, 6, 10]
HASH_ROW_COUNTS = [100, 500, 1000, 5000, 10000]
GRID_ITERS = 3
PARTITION_ITERS = 3
HASH_ITERS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_engine(node_id: str) -> CRDTEngine:
    """Create an in-memory CRDTEngine with the 'simple' schema."""
    engine = CRDTEngine(":memory:", node_id)
    engine.register_table(
        "simple",
        primary_key="id",
        columns=["id", "name", "value"],
    )
    return engine


def _insert_rows(engine: CRDTEngine, count: int, rng: random.Random, prefix: str):
    """Insert `count` rows with unique IDs into the 'simple' table."""
    for i in range(count):
        row_id = f"{prefix}_{i}"
        engine.insert("simple", {
            "id": row_id,
            "name": f"name_{rng.randint(0, 999999)}",
            "value": str(rng.randint(0, 999999)),
        })


# ===================================================================
# Benchmark 1: Rounds to Converge
# ===================================================================
def bench_rounds_to_converge(rng: random.Random) -> dict:
    """Measure the number of sync_all() rounds needed for convergence."""
    print_header("Benchmark 1: Rounds to Converge")
    results = {}  # key: "peers_N_ops_M" -> mean rounds

    matrix = []  # 2D list [peer_count_idx][ops_idx]
    for pc in PEER_COUNTS:
        row = []
        for ops in OPS_PER_PEER:
            rounds_list = []
            for it in range(GRID_ITERS):
                try:
                    # Create engines
                    engines = [_make_engine(f"node_{pc}_{ops}_{it}_{j}") for j in range(pc)]

                    # Each engine inserts ops rows independently
                    for idx, eng in enumerate(engines):
                        _insert_rows(eng, ops, rng, prefix=f"n{idx}_it{it}")

                    # Create bridge, register all
                    bridge = LocalSyncBridge()
                    for eng in engines:
                        bridge.register_peer(eng)

                    rounds = bridge.sync_until_converged(max_rounds=50)
                    rounds_list.append(rounds if rounds > 0 else 50)

                    # Cleanup
                    for eng in engines:
                        eng.close()
                except Exception as e:
                    print(f"  [WARN] peers={pc}, ops={ops}, iter={it}: {e}")
                    rounds_list.append(-1)

            mean_rounds = statistics.mean([r for r in rounds_list if r > 0]) if any(r > 0 for r in rounds_list) else -1
            row.append(round(mean_rounds, 2))
            key = f"peers_{pc}_ops_{ops}"
            results[key] = round(mean_rounds, 2)
            print(f"  peers={pc:>2}, ops_per_peer={ops:>4} → {mean_rounds:.1f} rounds (mean of {GRID_ITERS})")
        matrix.append(row)

    results["_matrix"] = matrix
    return results


# ===================================================================
# Benchmark 2: Wall-Clock Convergence Time
# ===================================================================
def bench_convergence_time(rng: random.Random) -> dict:
    """Measure total wall-clock time (ms) for sync_until_converged()."""
    print_header("Benchmark 2: Wall-Clock Convergence Time")
    results = {}
    matrix = []

    for pc in PEER_COUNTS:
        row = []
        for ops in OPS_PER_PEER:
            times_ms = []
            for it in range(GRID_ITERS):
                try:
                    engines = [_make_engine(f"wc_{pc}_{ops}_{it}_{j}") for j in range(pc)]
                    for idx, eng in enumerate(engines):
                        _insert_rows(eng, ops, rng, prefix=f"wc{idx}_it{it}")

                    bridge = LocalSyncBridge()
                    for eng in engines:
                        bridge.register_peer(eng)

                    t0 = time.perf_counter_ns()
                    bridge.sync_until_converged(max_rounds=50)
                    t1 = time.perf_counter_ns()
                    elapsed_ms = (t1 - t0) / 1e6
                    times_ms.append(elapsed_ms)

                    for eng in engines:
                        eng.close()
                except Exception as e:
                    print(f"  [WARN] peers={pc}, ops={ops}, iter={it}: {e}")

            mean_ms = statistics.mean(times_ms) if times_ms else -1
            row.append(round(mean_ms, 2))
            results[f"peers_{pc}_ops_{ops}"] = round(mean_ms, 2)
            print(f"  peers={pc:>2}, ops_per_peer={ops:>4} → {mean_ms:>10.2f} ms")
        matrix.append(row)

    results["_matrix"] = matrix
    return results


# ===================================================================
# Benchmark 3: Partition-Heal Recovery
# ===================================================================
def bench_partition_recovery(rng: random.Random) -> dict:
    """Measure recovery time after a network partition heals."""
    print_header("Benchmark 3: Partition-Heal Recovery")
    results = {}

    for pc in PARTITION_PEER_COUNTS:
        times_ms = []
        converged_flags = []
        for it in range(PARTITION_ITERS):
            try:
                # Create all engines
                engines = [_make_engine(f"part_{pc}_{it}_{j}") for j in range(pc)]
                half = pc // 2
                partition_a = engines[:half]
                partition_b = engines[half:]

                # Each engine does 100 ops (inserts + updates)
                for idx, eng in enumerate(engines):
                    # 80 inserts
                    _insert_rows(eng, 80, rng, prefix=f"part{idx}_it{it}")
                    # 20 updates on existing rows
                    for u in range(20):
                        row_id = f"part{idx}_it{it}_{rng.randint(0, 79)}"
                        try:
                            eng.update("simple", row_id, {
                                "value": str(rng.randint(0, 999999)),
                            })
                        except (ValueError, Exception):
                            pass  # Row might not exist — skip

                # Sync within partition A
                bridge_a = LocalSyncBridge()
                for eng in partition_a:
                    bridge_a.register_peer(eng)
                bridge_a.sync_until_converged(max_rounds=20)

                # Sync within partition B
                bridge_b = LocalSyncBridge()
                for eng in partition_b:
                    bridge_b.register_peer(eng)
                bridge_b.sync_until_converged(max_rounds=20)

                # Heal: create unified bridge
                bridge_unified = LocalSyncBridge()
                for eng in engines:
                    bridge_unified.register_peer(eng)

                # Time the recovery
                t0 = time.perf_counter_ns()
                rounds = bridge_unified.sync_until_converged(max_rounds=50)
                t1 = time.perf_counter_ns()
                elapsed_ms = (t1 - t0) / 1e6
                times_ms.append(elapsed_ms)

                # Verify convergence with ConvergenceHasher
                hashes = set()
                for eng in engines:
                    hasher = ConvergenceHasher(eng.conn, eng.schema)
                    hashes.add(hasher.compute_hash())
                converged_flags.append(len(hashes) == 1)

                for eng in engines:
                    eng.close()

            except Exception as e:
                print(f"  [WARN] peers={pc}, iter={it}: {e}")

        if times_ms:
            mean_ms = statistics.mean(times_ms)
            std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        else:
            mean_ms, std_ms = -1, 0

        all_converged = all(converged_flags) if converged_flags else False
        results[f"peers_{pc}"] = {
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "converged": all_converged,
            "raw_ms": [round(t, 2) for t in times_ms],
        }
        print(f"  peers={pc:>2} → {mean_ms:>10.2f} ± {std_ms:.2f} ms  "
              f"(converged={all_converged})")

    return results


# ===================================================================
# Benchmark 4: Hash Computation Overhead
# ===================================================================
def bench_hash_overhead(rng: random.Random) -> dict:
    """Measure ConvergenceHasher.compute_hash() latency at varying scales."""
    print_header("Benchmark 4: Hash Computation Overhead")
    results = {}

    for row_count in HASH_ROW_COUNTS:
        try:
            engine = _make_engine(f"hash_{row_count}")
            _insert_rows(engine, row_count, rng, prefix=f"h{row_count}")

            hasher = ConvergenceHasher(engine.conn, engine.schema)
            times_us = []
            for _ in range(HASH_ITERS):
                t0 = time.perf_counter_ns()
                hasher.compute_hash()
                t1 = time.perf_counter_ns()
                times_us.append((t1 - t0) / 1e3)  # nanoseconds → microseconds

            p50 = sorted(times_us)[len(times_us) // 2]
            results[f"rows_{row_count}"] = {
                "p50_us": round(p50, 2),
                "mean_us": round(statistics.mean(times_us), 2),
                "min_us": round(min(times_us), 2),
                "max_us": round(max(times_us), 2),
            }
            print(f"  rows={row_count:>6} → P50 = {p50:>10.1f} µs")
            engine.close()
        except Exception as e:
            print(f"  [WARN] rows={row_count}: {e}")

    return results


# ===================================================================
# Plotting
# ===================================================================
def plot_convergence_rounds(matrix: list[list[float]]):
    """Heatmap: rounds to converge."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    data = np.array(matrix, dtype=float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Axis labels
    ax.set_xticks(range(len(OPS_PER_PEER)))
    ax.set_xticklabels([str(o) for o in OPS_PER_PEER])
    ax.set_yticks(range(len(PEER_COUNTS)))
    ax.set_yticklabels([str(p) for p in PEER_COUNTS])
    ax.set_xlabel("Operations per Peer", fontsize=12)
    ax.set_ylabel("Peer Count", fontsize=12)
    ax.set_title("Rounds to Converge", fontsize=14, fontweight="bold", pad=12)

    # Annotate each cell
    for i in range(len(PEER_COUNTS)):
        for j in range(len(OPS_PER_PEER)):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Rounds", fontsize=11)

    plt.tight_layout()
    save_figure(fig, "fig_convergence_rounds.png")
    plt.close(fig)
    print("  → Saved fig_convergence_rounds.png")


def plot_convergence_time(matrix: list[list[float]]):
    """Line chart: convergence time vs peer count for each ops_per_peer."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    data = np.array(matrix, dtype=float)  # shape: [len(PEER_COUNTS), len(OPS_PER_PEER)]

    for j, ops in enumerate(OPS_PER_PEER):
        color = COLORS[j % len(COLORS)]
        ax.plot(
            PEER_COUNTS,
            data[:, j],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"{ops} ops/peer",
            color=color,
        )

    ax.set_xlabel("Peer Count", fontsize=12)
    ax.set_ylabel("Convergence Time (ms)", fontsize=12)
    ax.set_title("Wall-Clock Convergence Time", fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Ops per Peer", fontsize=10, title_fontsize=11)
    ax.set_xticks(PEER_COUNTS)

    plt.tight_layout()
    save_figure(fig, "fig_convergence_time.png")
    plt.close(fig)
    print("  → Saved fig_convergence_time.png")


def plot_partition_recovery(partition_data: dict):
    """Bar chart: partition-heal recovery time by peer count."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    peer_labels = []
    means = []
    stds = []
    for pc in PARTITION_PEER_COUNTS:
        key = f"peers_{pc}"
        if key in partition_data:
            peer_labels.append(str(pc))
            means.append(partition_data[key]["mean_ms"])
            stds.append(partition_data[key]["std_ms"])

    x = np.arange(len(peer_labels))
    bars = ax.bar(
        x, means, yerr=stds,
        width=0.5,
        color=COLORS[0],
        edgecolor="white",
        linewidth=0.8,
        capsize=6,
        error_kw={"linewidth": 1.5, "capthick": 1.5},
        alpha=0.9,
    )

    # Value labels on top of bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + max(means) * 0.02,
            f"{m:.1f} ms",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(peer_labels)
    ax.set_xlabel("Peer Count", fontsize=12)
    ax.set_ylabel("Recovery Time (ms)", fontsize=12)
    ax.set_title("Partition-Heal Recovery Time", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    save_figure(fig, "fig_partition_recovery.png")
    plt.close(fig)
    print("  → Saved fig_partition_recovery.png")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 65)
    print("  CONVERGENCE TIME BENCHMARK SUITE")
    print("=" * 65)
    print()

    rng = random.Random(SEED)
    all_results = {}

    # --- Benchmark 1 ---
    rounds_data = bench_rounds_to_converge(rng)
    all_results["rounds_to_converge"] = {k: v for k, v in rounds_data.items() if k != "_matrix"}
    rounds_matrix = rounds_data["_matrix"]

    # --- Benchmark 2 ---
    time_data = bench_convergence_time(rng)
    all_results["convergence_time_ms"] = {k: v for k, v in time_data.items() if k != "_matrix"}
    time_matrix = time_data["_matrix"]

    # --- Benchmark 3 ---
    partition_data = bench_partition_recovery(rng)
    all_results["partition_recovery"] = partition_data

    # --- Benchmark 4 ---
    hash_data = bench_hash_overhead(rng)
    all_results["hash_overhead"] = hash_data

    # --- Save JSON results ---
    print_header("Saving Results")
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_results(all_results, "convergence_time.json")
    print("  → Saved benchmarks/results/convergence_time.json")

    # --- Generate figures ---
    print_header("Generating Figures")
    figures_dir = PROJECT_ROOT / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_convergence_rounds(rounds_matrix)
    except Exception as e:
        print(f"  [ERROR] fig_convergence_rounds: {e}")

    try:
        plot_convergence_time(time_matrix)
    except Exception as e:
        print(f"  [ERROR] fig_convergence_time: {e}")

    try:
        plot_partition_recovery(partition_data)
    except Exception as e:
        print(f"  [ERROR] fig_partition_recovery: {e}")

    print()
    print("=" * 65)
    print("  CONVERGENCE BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
