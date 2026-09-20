"""Report generators for audit results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .models import AuditConfig, Finding, Severity
from .scanner import AuditResult


class AuditReporter:
    """Render an AuditResult to JSON, Markdown or console text."""

    def __init__(self, result: AuditResult, config: AuditConfig) -> None:
        self.result = result
        self.config = config

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": self.result.summary(),
            "violations": self.result.exceeds_thresholds(self.config),
            "findings": [f.to_dict() for f in self.result.findings],
            "performance_reports": self.result.perf_reports,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        summary = self.result.summary()
        violations = self.result.exceeds_thresholds(self.config)
        lines = [
            "# Code Audit Report",
            "",
            f"- **Generated at:** {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
            f"- **Files scanned:** {summary['files_scanned']}",
            f"- **Total findings:** {summary['total_findings']}",
            f"- **Duration:** {summary['duration_seconds']}s",
            "",
            "## Severity Counts",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev in Severity:
            lines.append(f"| {sev.value} | {summary['severity_counts'][sev.value]} |")
        lines.extend(["", "## Threshold Check", ""])
        if violations["failed"]:
            lines.append("**FAILED** — audit thresholds exceeded.")
            for v in violations["violations"]:
                lines.append(f"- {v}")
        else:
            lines.append("**PASSED** — no threshold violations.")
        lines.extend(["", "## Findings", ""])
        if not self.result.findings:
            lines.append("No findings.")
        for f in self.result.findings:
            lines.extend([
                f"### {f.rule_id} — {f.severity.value}",
                "",
                f"- **Location:** {f.location}",
                f"- **Title:** {f.title}",
                f"- **Description:** {f.description}",
                f"- **Snippet:** `{f.snippet}`",
                f"- **Fix suggestion:** {f.fix_suggestion}",
                "",
            ])
        if self.result.perf_reports:
            lines.extend(["", "## Performance Reports", ""])
            for r in self.result.perf_reports:
                lines.append(f"```json\n{json.dumps(r, indent=2, ensure_ascii=False)}\n```")
        return "\n".join(lines)

    def to_console(self) -> str:
        lines = []
        summary = self.result.summary()
        violations = self.result.exceeds_thresholds(self.config)
        lines.append(
            f"Files scanned: {summary['files_scanned']} | "
            f"Findings: {summary['total_findings']} | "
            f"Duration: {summary['duration_seconds']}s"
        )
        for sev, count in summary["severity_counts"].items():
            lines.append(f"  {sev}: {count}")
        if violations["failed"]:
            lines.append("AUDIT FAILED")
            for v in violations["violations"]:
                lines.append(f"  - {v}")
        else:
            lines.append("AUDIT PASSED")
        for f in self.result.findings:
            lines.append(f"[{f.severity.value}] {f.rule_id} at {f.location}: {f.title}")
        return "\n".join(lines)
