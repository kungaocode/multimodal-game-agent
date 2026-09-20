"""Tests for the static security scanner."""

from pathlib import Path

from security.audit.c_rules import scan_c_file
from security.audit.models import AuditConfig, Severity
from security.audit.python_rules import scan_python_file
from security.audit.scanner import AuditScanner

SAMPLES = Path(__file__).parent / "samples"


def test_python_scanner_detects_vulnerabilities():
    source = (SAMPLES / "unsafe_python.py").read_text()
    findings = scan_python_file("unsafe_python.py", source)
    rule_ids = {f.rule_id for f in findings}
    assert "python-deserialization-pickle-loads" in rule_ids
    assert "python-dangerous-eval" in rule_ids
    assert "python-subprocess-shell" in rule_ids
    assert "python-hardcoded-secret" in rule_ids
    assert "python-bare-except" in rule_ids
    assert "python-resource-leak" in rule_ids


def test_c_scanner_detects_uaf_and_overflows():
    source = (SAMPLES / "unsafe_cpp.cpp").read_text()
    findings = scan_c_file("unsafe_cpp.cpp", source)
    rule_ids = {f.rule_id for f in findings}
    assert "c-use-after-free" in rule_ids
    assert "c-unsafe-api-strcpy" in rule_ids
    assert "c-format-string" in rule_ids
    assert "c-memory-leak" in rule_ids


def test_scanner_discovers_samples():
    config = AuditConfig(include=[str(SAMPLES)], exclude=[])
    scanner = AuditScanner(config)
    files = scanner.discover_files()
    names = {p.name for p in files}
    assert "unsafe_python.py" in names
    assert "unsafe_cpp.cpp" in names


def test_threshold_violation_for_critical_findings():
    config = AuditConfig(
        include=[str(SAMPLES)],
        exclude=[],
        severity_threshold=Severity.CRITICAL,
        max_critical=0,
        max_high=0,
    )
    scanner = AuditScanner(config)
    result = scanner.scan(roots=[SAMPLES])
    violations = result.exceeds_thresholds(config)
    assert violations["failed"]
    assert any("CRITICAL" in v for v in violations["violations"])
