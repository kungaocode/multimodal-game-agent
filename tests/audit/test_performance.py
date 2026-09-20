"""Tests for performance utilities."""

from security.audit.performance import PerformanceProfiler, benchmark, memory_usage, profiled
from tests.audit.samples.safe_demo import heavy_function


def test_benchmark_runs_and_passes():
    decorated = benchmark(iterations=3, timeout_seconds=5.0)(heavy_function)
    result = decorated(500)
    assert result.passed
    assert result.iterations == 3
    assert result.mean_seconds > 0


def test_profiled_records_output():
    decorated = profiled(limit=5)(heavy_function)
    decorated(100)
    assert hasattr(decorated, "last_profile")
    assert "function calls" in decorated.last_profile


def test_memory_usage_records_peak():
    with memory_usage(label="demo") as _state:
        _data = [i * i for i in range(10000)]
    assert hasattr(memory_usage, "last")
    assert memory_usage.last["peak_mb"] >= 0


def test_profiler_imports_target():
    profiler = PerformanceProfiler()
    report = profiler.run_imported_target(
        "tests.audit.samples.safe_demo", "heavy_function", 200
    )
    assert report["passed"]
    assert report["function"] == "heavy_function"
