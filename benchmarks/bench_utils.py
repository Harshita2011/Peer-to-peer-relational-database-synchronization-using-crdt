"""bench_utils — Shared utilities for CRDT benchmark scripts.

Provides:
- BenchmarkTimer: context manager for ns-precision timing
- BenchmarkResult: dataclass with latency stats and formatting
- run_benchmark(): warmup + timed execution harness
- setup_engines(): create N in-memory CRDTEngine peers
- setup_bridge(): wire engines into a LocalSyncBridge
- save_results(): persist JSON to benchmarks/results/
- apply_plot_style(): premium dark matplotlib theme
- save_figure(): export figure as 300 DPI PNG
- COLORS: hex color palette
- print_header() / print_results_table(): formatted console output
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine import CRDTEngine  # noqa: E402
from src.sync import LocalSyncBridge  # noqa: E402

# ---------------------------------------------------------------------------
# Color palette & constants
# ---------------------------------------------------------------------------
COLORS: list[str] = [
    "#e94560",  # vibrant red-pink
    "#0f3460",  # deep navy
    "#16213e",  # dark indigo
    "#533483",  # muted purple
    "#e94560",  # vibrant red-pink (repeat for continuity)
    "#00b4d8",  # electric cyan
    "#06d6a0",  # mint green
    "#ffd166",  # warm gold
]

_RESULTS_DIR = _PROJECT_ROOT / "benchmarks" / "results"
_FIGURES_DIR = _PROJECT_ROOT / "paper" / "figures"


# ═══════════════════════════════════════════════════════════════════════════
# 1. BenchmarkTimer
# ═══════════════════════════════════════════════════════════════════════════
class BenchmarkTimer:
    """Context manager that records wall-clock elapsed time in nanoseconds.

    Usage::

        timer = BenchmarkTimer()
        with timer:
            do_work()
        print(timer.elapsed_ns)
    """

    def __init__(self) -> None:
        self.elapsed_ns: int = 0
        self._start: int = 0

    def __enter__(self) -> "BenchmarkTimer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ns = time.perf_counter_ns() - self._start


# ═══════════════════════════════════════════════════════════════════════════
# 2. BenchmarkResult
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkResult:
    """Collected latency data for a single benchmark."""

    name: str
    n: int = 1
    latencies_ns: list[int] = field(default_factory=list)
    value: float = 0.0
    unit: str = ""
    params: dict = field(default_factory=dict)

    # -- derived properties ------------------------------------------------

    @property
    def p50(self) -> float:
        """Median latency in nanoseconds."""
        return self._percentile(50)

    @property
    def p90(self) -> float:
        """90th-percentile latency in nanoseconds."""
        return self._percentile(90)

    @property
    def p99(self) -> float:
        """99th-percentile latency in nanoseconds."""
        return self._percentile(99)

    @property
    def mean_ns(self) -> float:
        """Arithmetic mean latency in nanoseconds."""
        if not self.latencies_ns:
            return 0.0
        return sum(self.latencies_ns) / len(self.latencies_ns)

    @property
    def std_ns(self) -> float:
        """Sample standard deviation of latencies in nanoseconds."""
        if len(self.latencies_ns) < 2:
            return 0.0
        mean = self.mean_ns
        variance = sum((x - mean) ** 2 for x in self.latencies_ns) / (
            len(self.latencies_ns) - 1
        )
        return math.sqrt(variance)

    @property
    def throughput_ops_sec(self) -> float:
        """Operations per second based on mean latency."""
        if self.mean_ns == 0:
            return 0.0
        return 1e9 / self.mean_ns

    # -- formatting --------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted table row for console output."""
        if self.latencies_ns:
            return (
                f"| {self.name:<30s} "
                f"| {self.n:>6d} "
                f"| {self.mean_ns / 1e6:>10.3f} "
                f"| {self.p50 / 1e6:>10.3f} "
                f"| {self.p90 / 1e6:>10.3f} "
                f"| {self.p99 / 1e6:>10.3f} "
                f"| {self.throughput_ops_sec:>12.1f} |"
            )
        else:
            p50 = self.params.get("p50_us", self.value) / 1000
            p90 = self.params.get("p90_us", self.value) / 1000
            p99 = self.params.get("p99_us", self.value) / 1000
            mean = self.params.get("mean_us", self.value) / 1000
            return (
                 f"| {self.name:<30s} "
                 f"| {self.params.get('row_count', 1):>6d} "
                 f"| {mean:>10.3f} "
                 f"| {p50:>10.3f} "
                 f"| {p90:>10.3f} "
                 f"| {p99:>10.3f} "
                 f"| {0.0:>12.1f} |"
            )

    # -- internals ---------------------------------------------------------

    def _percentile(self, p: float) -> float:
        """Compute the *p*-th percentile using linear interpolation."""
        if not self.latencies_ns:
            return 0.0
        sorted_vals = sorted(self.latencies_ns)
        k = (p / 100) * (len(sorted_vals) - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(sorted_vals[lo])
        frac = k - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ═══════════════════════════════════════════════════════════════════════════
# 3. run_benchmark
# ═══════════════════════════════════════════════════════════════════════════
def run_benchmark(
    name: str,
    fn: Callable[[], Any],
    n: int = 20,
    warmup: int = 5,
) -> BenchmarkResult:
    """Execute *fn* with warmup, then collect *n* latency samples.

    Args:
        name: Human-readable benchmark label.
        fn: Zero-argument callable to benchmark.
        n: Number of timed iterations.
        warmup: Number of untimed warmup calls (discarded).

    Returns:
        A :class:`BenchmarkResult` with the collected latencies.
    """
    # Warmup phase — discard results
    for _ in range(warmup):
        fn()

    # Timed phase
    latencies: list[int] = []
    for i in range(n):
        timer = BenchmarkTimer()
        with timer:
            fn()
        latencies.append(timer.elapsed_ns)
        if (i + 1) % max(1, n // 5) == 0:
            print(f"  [{name}] {i + 1}/{n} iterations done")

    result = BenchmarkResult(name=name, n=n, latencies_ns=latencies)
    print(
        f"  ✓ {name}: mean={result.mean_ns / 1e6:.3f} ms  "
        f"p99={result.p99 / 1e6:.3f} ms  "
        f"throughput={result.throughput_ops_sec:.1f} ops/s"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. setup_engines
# ═══════════════════════════════════════════════════════════════════════════
def setup_engines(
    n_peers: int,
    schema_type: str = "simple",
    prefix: str = "node",
) -> list[CRDTEngine]:
    """Create *n_peers* in-memory :class:`CRDTEngine` instances.

    Args:
        n_peers: Number of engines to create.
        schema_type:
            ``'simple'`` — single ``items`` table with columns
            ``[id, name, value]``.
            ``'relational'`` — three tables:
            ``doctors`` → ``patients`` → ``prescriptions`` with FK chains.
        prefix: Prefix for engine node IDs.

    Returns:
        List of ready-to-use engine instances.
    """
    engines: list[CRDTEngine] = []
    for i in range(n_peers):
        engine = CRDTEngine(":memory:", f"{prefix}_{i}")
        _register_schema(engine, schema_type)
        engines.append(engine)
    return engines


def _register_schema(engine: CRDTEngine, schema_type: str) -> None:
    """Register tables on *engine* according to *schema_type*."""
    if schema_type == "simple":
        engine.register_table(
            "items",
            primary_key="id",
            columns=["id", "name", "value"],
        )
    elif schema_type == "relational":
        engine.register_table(
            "doctors",
            primary_key="id",
            columns=["id", "name", "specialty"],
        )
        engine.register_table(
            "patients",
            primary_key="id",
            columns=["id", "name", "nhs_number", "doctor_id"],
            foreign_keys=[("doctor_id", "doctors", "id")],
        )
        engine.register_table(
            "prescriptions",
            primary_key="id",
            columns=["id", "drug_name", "dosage", "patient_id"],
            foreign_keys=[("patient_id", "patients", "id")],
        )
    else:
        raise ValueError(f"Unknown schema_type: {schema_type!r}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. setup_bridge
# ═══════════════════════════════════════════════════════════════════════════
def setup_bridge(engines: list[CRDTEngine]) -> LocalSyncBridge:
    """Create a :class:`LocalSyncBridge` and register all *engines*.

    Args:
        engines: List of CRDTEngine peers.

    Returns:
        Fully wired LocalSyncBridge.
    """
    bridge = LocalSyncBridge()
    for engine in engines:
        bridge.register_peer(engine)
    return bridge


# ═══════════════════════════════════════════════════════════════════════════
# 6. save_results
# ═══════════════════════════════════════════════════════════════════════════
def save_results(data: dict, filename: str) -> Path:
    """Persist *data* as JSON in ``benchmarks/results/``.

    Creates the directory if it does not exist.

    Args:
        data: Arbitrary dict (must be JSON-serialisable).
        filename: File name (e.g. ``"sync_latency.json"``).

    Returns:
        Absolute path to the saved file.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  📄 Results saved → {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# 7. apply_plot_style
# ═══════════════════════════════════════════════════════════════════════════
def apply_plot_style() -> None:
    """Apply a premium dark matplotlib theme for publication-quality charts.

    Sets:
    - Dark background (#1a1a2e)
    - Subtle grid with low alpha
    - Custom colour cycle from :data:`COLORS`
    - DejaVu Sans font at size 11
    - Tight layout by default
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    dark_bg = "#1a1a2e"
    text_color = "#e0e0e0"
    grid_color = "#333355"

    plt.style.use("dark_background")

    mpl.rcParams.update(
        {
            # Background
            "figure.facecolor": dark_bg,
            "axes.facecolor": dark_bg,
            "savefig.facecolor": dark_bg,
            # Text
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            # Grid
            "axes.grid": True,
            "grid.color": grid_color,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            # Colour cycle
            "axes.prop_cycle": mpl.cycler(color=COLORS),
            # Layout
            "figure.autolayout": True,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. save_figure
# ═══════════════════════════════════════════════════════════════════════════
def save_figure(fig: Any, name: str) -> Path:
    """Save a matplotlib figure to ``paper/figures/{name}.png`` at 300 DPI.

    Creates the output directory if needed.

    Args:
        fig: A :class:`matplotlib.figure.Figure`.
        name: Base filename (without extension).

    Returns:
        Absolute path to the saved PNG.
    """
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _FIGURES_DIR / f"{name}.png"
    fig.savefig(
        filepath,
        dpi=300,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    print(f"  📊 Figure saved → {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# 10. print_header
# ═══════════════════════════════════════════════════════════════════════════
def print_header(title: str) -> None:
    """Print a formatted section header to stdout.

    Example output::

        ══════════════════════════════════════════
          SYNC LATENCY BENCHMARK
        ══════════════════════════════════════════
    """
    width = max(len(title) + 4, 42)
    bar = "═" * width
    print()
    print(bar)
    print(f"  {title.upper()}")
    print(bar)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 11. print_results_table
# ═══════════════════════════════════════════════════════════════════════════
def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted ASCII table of benchmark results.

    Columns: Name, N, Mean(ms), P50(ms), P90(ms), P99(ms), Throughput(ops/s).
    """
    header = (
        f"| {'Benchmark':<30s} "
        f"| {'N':>6s} "
        f"| {'Mean(ms)':>10s} "
        f"| {'P50(ms)':>10s} "
        f"| {'P90(ms)':>10s} "
        f"| {'P99(ms)':>10s} "
        f"| {'Ops/sec':>12s} |"
    )
    sep = "+" + "-" * 32 + "+" + "-" * 8 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 14 + "+"

    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(r.summary())
    print(sep)
    print()
