"""Command line interface for the code audit module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import AuditConfig, Severity
from .reporter import AuditReporter
from .scanner import AuditScanner


def parse_perf_targets(raw: list[str] | None) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for item in raw or []:
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Performance target must be module:function, got {item}"
            )
        module, function = item.split(":", 1)
        targets.append((module, function))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit", description="Static security and performance audit"
    )
    parser.add_argument(
        "--path", nargs="+", default=None, help="Paths to scan (default from pyproject.toml)"
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None, help="Override exclude patterns"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "console"],
        default="console",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Write report to file instead of stdout"
    )
    parser.add_argument(
        "--threshold",
        default=None,
        choices=[s.value for s in Severity],
        help="Minimum severity that fails the audit",
    )
    parser.add_argument(
        "--perf", action="append", default=None, help="Benchmark target as module:function"
    )
    parser.add_argument("--max-critical", type=int, default=None)
    parser.add_argument("--max-high", type=int, default=None)
    parser.add_argument("--max-medium", type=int, default=None)
    args = parser.parse_args(argv)

    scanner = AuditScanner.from_pyproject_path()
    config = scanner.config
    if args.path:
        config.include = args.path
        if args.exclude is None:
            config.exclude = []
    if args.exclude is not None:
        config.exclude = args.exclude

    if args.threshold:
        config.severity_threshold = Severity(args.threshold)
    if args.max_critical is not None:
        config.max_critical = args.max_critical
    if args.max_high is not None:
        config.max_high = args.max_high
    if args.max_medium is not None:
        config.max_medium = args.max_medium

    roots = [Path(p) for p in config.include]
    perf_targets = parse_perf_targets(args.perf)
    result = scanner.scan(roots=roots, perf_targets=perf_targets)
    reporter = AuditReporter(result, config)
    if args.format == "json":
        text = reporter.to_json()
    elif args.format == "markdown":
        text = reporter.to_markdown()
    else:
        text = reporter.to_console()
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 1 if result.exceeds_thresholds(config)["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
