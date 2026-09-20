"""Performance and efficiency analysis utilities."""

from __future__ import annotations

import cProfile
import importlib
import io
import pstats
import signal
import statistics
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator

from .models import AuditConfig


class TimeoutError(RuntimeError):
    """Raised when a benchmarked function exceeds its timeout."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Function exceeded the configured timeout.")


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    stddev_seconds: float | None
    passed: bool = True
    error: str = ""


def benchmark(
    *,
    iterations: int = 10,
    timeout_seconds: float = 30.0,
) -> Callable[[Callable], Callable]:
    """Decorator that benchmarks a function over multiple iterations.

    The decorated function returns a BenchmarkResult instead of its normal
    return value. It raises a TimeoutError if any single iteration exceeds
    ``timeout_seconds``.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> BenchmarkResult:
            durations: list[float] = []
            error = ""
            passed = True
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            try:
                for _ in range(iterations):
                    signal.alarm(int(timeout_seconds))
                    start = time.perf_counter()
                    try:
                        func(*args, **kwargs)
                    finally:
                        signal.alarm(0)
                    durations.append(time.perf_counter() - start)
            except TimeoutError as exc:
                passed = False
                error = str(exc)
            except Exception as exc:
                passed = False
                error = f"{type(exc).__name__}: {exc}"
            finally:
                signal.signal(signal.SIGALRM, old_handler)
            if durations:
                mean = statistics.mean(durations)
                std = statistics.stdev(durations) if len(durations) > 1 else None
                result = BenchmarkResult(
                    name=func.__name__,
                    iterations=len(durations),
                    total_seconds=sum(durations),
                    mean_seconds=mean,
                    min_seconds=min(durations),
                    max_seconds=max(durations),
                    stddev_seconds=std,
                    passed=passed,
                    error=error,
                )
            else:
                result = BenchmarkResult(
                    name=func.__name__,
                    iterations=0,
                    total_seconds=0.0,
                    mean_seconds=0.0,
                    min_seconds=0.0,
                    max_seconds=0.0,
                    stddev_seconds=None,
                    passed=passed,
                    error=error,
                )
            wrapper.last_result = result
            return result
        return wrapper
    return decorator


def profiled(sort_by: str = "cumulative", limit: int = 20) -> Callable[[Callable], Callable]:
    """Decorator that profiles a single function call with cProfile."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = cProfile.Profile()
            result = profiler.runcall(func, *args, **kwargs)
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(limit)
            wrapper.last_profile = stream.getvalue()
            return result
        return wrapper
    return decorator


@contextmanager
def memory_usage(label: str = "") -> Iterator[dict[str, Any]]:
    """Context manager that measures peak memory with tracemalloc."""
    tracemalloc.start()
    try:
        yield {"label": label, "running": True}
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage.last = {
            "label": label,
            "current_bytes": current,
            "peak_bytes": peak,
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
        }


class PerformanceProfiler:
    """Programmatic entry point for auditing performance of target functions."""

    def __init__(self, config: AuditConfig | None = None) -> None:
        self.config = config or AuditConfig()

    def run_imported_target(
        self,
        module_path: str,
        function_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)
        except Exception as exc:
            return {
                "module": module_path,
                "function": function_name,
                "error": f"Import failed: {exc}",
                "passed": False,
            }
        benchmark_decorator = benchmark(
            iterations=self.config.perf_iterations,
            timeout_seconds=self.config.perf_timeout_seconds,
        )
        wrapped = benchmark_decorator(func)
        result = wrapped(*args, **kwargs)
        duration = time.perf_counter() - start
        return {
            "module": module_path,
            "function": function_name,
            "passed": result.passed,
            "iterations": result.iterations,
            "mean_seconds": result.mean_seconds,
            "min_seconds": result.min_seconds,
            "max_seconds": result.max_seconds,
            "stddev_seconds": result.stddev_seconds,
            "error": result.error,
            "total_seconds": round(duration, 3),
            "memory_peak_mb": None,
        }
