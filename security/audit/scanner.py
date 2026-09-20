"""Generic scanner dispatcher and file discovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .c_rules import scan_c_file
from .models import AuditConfig, Finding, Severity
from .performance import PerformanceProfiler
from .python_rules import scan_python_file


@dataclass
class AuditResult:
    findings: list[Finding] = field(default_factory=list)
    perf_reports: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    files_scanned: int = 0

    def by_severity(self) -> dict[Severity, list[Finding]]:
        buckets: dict[Severity, list[Finding]] = {s: [] for s in Severity}
        for finding in self.findings:
            buckets[finding.severity].append(finding)
        return buckets

    def summary(self) -> dict[str, Any]:
        buckets = self.by_severity()
        return {
            "files_scanned": self.files_scanned,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_findings": len(self.findings),
            "severity_counts": {s.value: len(buckets[s]) for s in Severity},
            "performance_reports": len(self.perf_reports),
        }

    def exceeds_thresholds(self, config: AuditConfig) -> dict[str, Any]:
        buckets = self.by_severity()
        violations: list[str] = []
        if config.max_critical is not None and len(buckets[Severity.CRITICAL]) > config.max_critical:
            violations.append(
                f"CRITICAL findings {len(buckets[Severity.CRITICAL])} > max {config.max_critical}"
            )
        if config.max_high is not None and len(buckets[Severity.HIGH]) > config.max_high:
            violations.append(
                f"HIGH findings {len(buckets[Severity.HIGH])} > max {config.max_high}"
            )
        if config.max_medium is not None and len(buckets[Severity.MEDIUM]) > config.max_medium:
            violations.append(
                f"MEDIUM findings {len(buckets[Severity.MEDIUM])} > max {config.max_medium}"
            )
        threshold_violations = [
            f.location
            for f in self.findings
            if f.severity >= config.severity_threshold
        ]
        return {
            "failed": bool(violations) or bool(threshold_violations),
            "violations": violations,
            "threshold_locations": threshold_violations,
        }


class AuditScanner:
    """Discover files and run language-specific static analysis."""

    PYTHON_EXTENSIONS = {".py", ".pyw"}
    C_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"}

    def __init__(self, config: AuditConfig | None = None) -> None:
        self.config = config or AuditConfig()

    def discover_files(self, roots: list[Path] | None = None) -> list[Path]:
        roots = roots or [Path(p) for p in self.config.include]
        files: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            if root.is_file():
                candidates = [root]
            else:
                candidates = list(root.rglob("*"))
            for path in candidates:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                if not self.config.should_scan(path):
                    continue
                ext = path.suffix.lower()
                if ext in self.PYTHON_EXTENSIONS or ext in self.C_EXTENSIONS:
                    files.append(path)
                    seen.add(resolved)
        return files

    def scan_file(self, path: Path) -> list[Finding]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            from .models import Location

            return [
                Finding(
                    rule_id="audit-read-error",
                    severity=Severity.INFO,
                    title="File could not be read",
                    description=str(exc),
                    location=Location(file=str(path), line=0),
                    language="unknown",
                )
            ]
        ext = path.suffix.lower()
        if ext in self.PYTHON_EXTENSIONS:
            return scan_python_file(str(path), source)
        if ext in self.C_EXTENSIONS:
            return scan_c_file(str(path), source)
        return []

    def scan(
        self,
        roots: list[Path] | None = None,
        perf_targets: list[tuple[str, str]] | None = None,
    ) -> AuditResult:
        start = time.perf_counter()
        files = self.discover_files(roots)
        findings: list[Finding] = []
        for path in files:
            findings.extend(self.scan_file(path))
        perf_reports: list[dict[str, Any]] = []
        if perf_targets:
            profiler = PerformanceProfiler(config=self.config)
            for module_path, function_name in perf_targets:
                report = profiler.run_imported_target(module_path, function_name)
                perf_reports.append(report)
        duration = time.perf_counter() - start
        return AuditResult(
            findings=findings,
            perf_reports=perf_reports,
            duration_seconds=duration,
            files_scanned=len(files),
        )

    @classmethod
    def from_pyproject_path(cls, path: Path = Path("pyproject.toml")) -> "AuditScanner":
        config = AuditConfig()
        if path.exists():
            config = AuditConfig.from_pyproject(path.read_text(encoding="utf-8"))
        return cls(config=config)
