"""Data models for the code audit module."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    @classmethod
    def rank(cls, value: str) -> int:
        order = [cls.INFO, cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        try:
            return order.index(cls(value))
        except ValueError:
            return -1

    def __ge__(self, other: "Severity") -> bool:
        return Severity.rank(self.value) >= Severity.rank(other.value)

    def __gt__(self, other: "Severity") -> bool:
        return Severity.rank(self.value) > Severity.rank(other.value)

    def __le__(self, other: "Severity") -> bool:
        return Severity.rank(self.value) <= Severity.rank(other.value)

    def __lt__(self, other: "Severity") -> bool:
        return Severity.rank(self.value) < Severity.rank(other.value)


@dataclass(frozen=True)
class Location:
    file: str
    line: int
    column: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    title: str
    description: str
    location: Location
    language: str
    snippet: str = ""
    fix_suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "location": {
                "file": self.location.file,
                "line": self.location.line,
                "column": self.location.column,
            },
            "language": self.language,
            "snippet": self.snippet,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class AuditConfig:
    """Configuration for the code audit scanner."""

    include: list[str] = field(default_factory=lambda: ["."])
    exclude: list[str] = field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "build",
            "dist",
            "*.egg-info",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
        ]
    )
    severity_threshold: Severity = Severity.MEDIUM
    max_critical: int = 0
    max_high: int = 0
    max_medium: int | None = None
    perf_timeout_seconds: float = 30.0
    perf_memory_mb: float = 512.0
    perf_iterations: int = 10

    def should_scan(self, path: Path) -> bool:
        """Return True if the path is included and not excluded."""
        for pattern in self.exclude:
            if self._match(path, pattern):
                return False
        if not self.include:
            return True
        for pattern in self.include:
            if self._match(path, pattern):
                return True
        return False

    @staticmethod
    def _match(path: Path, pattern: str) -> bool:
        if pattern in (path.name, str(path)):
            return True
        if any(pattern == part for part in path.parts):
            return True
        if fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
        pattern_parts = Path(pattern).parts
        if len(pattern_parts) <= len(path.parts):
            if path.parts[: len(pattern_parts)] == pattern_parts:
                return True
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditConfig":
        cfg = cls()
        if "include" in data:
            cfg.include = list(data["include"])
        if "exclude" in data:
            cfg.exclude = list(data["exclude"])
        if "severity_threshold" in data:
            cfg.severity_threshold = Severity(data["severity_threshold"].upper())
        if "max_critical" in data:
            cfg.max_critical = int(data["max_critical"])
        if "max_high" in data:
            cfg.max_high = int(data["max_high"])
        if "max_medium" in data:
            cfg.max_medium = int(data["max_medium"])
        if "perf_timeout_seconds" in data:
            cfg.perf_timeout_seconds = float(data["perf_timeout_seconds"])
        if "perf_memory_mb" in data:
            cfg.perf_memory_mb = float(data["perf_memory_mb"])
        if "perf_iterations" in data:
            cfg.perf_iterations = int(data["perf_iterations"])
        return cfg

    @classmethod
    def from_pyproject(cls, pyproject_text: str) -> "AuditConfig":
        import tomllib

        data = tomllib.loads(pyproject_text)
        audit_data = data.get("tool", {}).get("audit", {})
        return cls.from_dict(audit_data)
