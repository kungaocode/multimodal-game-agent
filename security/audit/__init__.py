"""Code audit module: static security and performance analysis.

All new modules must pass this audit before being merged.
"""

from .scanner import AuditScanner, AuditConfig, Finding
from .performance import PerformanceProfiler, profiled, benchmark
from .reporter import AuditReporter

__all__ = [
    "AuditScanner",
    "AuditConfig",
    "Finding",
    "PerformanceProfiler",
    "profiled",
    "benchmark",
    "AuditReporter",
]
