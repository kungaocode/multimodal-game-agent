"""Regex-based static security scanner for C and C++ source code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Finding, Location, Severity

UNSAFE_APIS = {"strcpy", "strcat", "gets", "sprintf", "scanf"}

ASSIGN_RE = re.compile(
    r"(?:^|[^=])\b([A-Za-z_]\w*)\s*=(?!=)",
    re.MULTILINE,
)
ALLOC_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*=?\s*(?:\([^)]*\))?\s*\b(malloc|calloc|realloc|strdup|new)\b"
)
FREE_RE = re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)")
DELETE_RE = re.compile(r"\bdelete\b(?:\s*\[\])?\s*([A-Za-z_]\w*)")
UNSAFE_API_RE = re.compile(r"\b(strcpy|strcat|gets|sprintf|scanf)\s*\(")
FORMAT_FUNC_RE = re.compile(r"\b(f?printf|f?scanf|snprintf|syslog)\s*\(")
VAR_USE_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass
class CFunctionScope:
    name: str = ""
    malloced: dict[str, int] = field(default_factory=dict)
    freed: dict[str, int] = field(default_factory=dict)
    reassigned: set[str] = field(default_factory=set)


def _strip_comments(line: str) -> str:
    """Remove // and /* ... */ style comments from a single logical line."""
    line = re.sub(r"//.*", "", line)
    return line


def _findings_for_line(filename: str, line_no: int, line: str) -> list[Finding]:
    findings: list[Finding] = []
    stripped = line.strip()
    if not stripped or stripped.startswith("//") or stripped.startswith("*"):
        return findings
    for match in UNSAFE_API_RE.finditer(line):
        func = match.group(1)
        sev = Severity.CRITICAL if func == "gets" else Severity.HIGH
        findings.append(
            Finding(
                rule_id=f"c-unsafe-api-{func}",
                severity=sev,
                title=f"Unsafe C API: {func}",
                description=f"{func}() can cause buffer overflows or unbounded reads.",
                location=Location(file=filename, line=line_no, column=match.start()),
                language="c/c++",
                snippet=stripped,
                fix_suggestion=f"Use {func.replace('str', 'strn')} or a bounds-checked alternative.",
            )
        )
    for match in FORMAT_FUNC_RE.finditer(line):
        after = line[match.end() :]
        args = _split_args(after)
        fmt_index = 1 if match.group(1).startswith("f") else 0
        if len(args) > fmt_index:
            fmt_arg = args[fmt_index].strip()
            if not STRING_LITERAL_RE.match(fmt_arg) and VAR_USE_RE.match(fmt_arg):
                findings.append(
                    Finding(
                        rule_id="c-format-string",
                        severity=Severity.HIGH,
                        title="Potential format string vulnerability",
                        description="Format argument is a variable, allowing format string injection.",
                        location=Location(file=filename, line=line_no, column=match.start()),
                        language="c/c++",
                        snippet=stripped,
                        fix_suggestion="Use a string literal for the format string.",
                    )
                )
    return findings


def _split_args(fragment: str) -> list[str]:
    """Split the inside of a function call up to the matching ')'."""
    depth = 1
    buf = ""
    args: list[str] = []
    in_string = False
    escape = False
    for ch in fragment:
        if in_string:
            buf += ch
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            buf += ch
            continue
        if ch == "(":
            depth += 1
            buf += ch
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                if buf.strip():
                    args.append(buf)
                break
            buf += ch
            continue
        if ch == "," and depth == 1:
            args.append(buf)
            buf = ""
            continue
        buf += ch
    return args


def _is_function_start(line: str) -> str | None:
    match = re.match(
        r"^\s*(?:static\s+|inline\s+|const\s+|volatile\s+|unsigned\s+|signed\s+)*"
        r"(?:[\w\*]+(?:\s+const)?\s+)+"
        r"([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:const\s*)?\{?\s*$",
        line,
    )
    if match:
        return match.group(1)
    return None


def scan_c_file(filename: str, source: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()
    scope_stack: list[CFunctionScope] = []
    brace_level = 0
    for line_no, raw_line in enumerate(lines, start=1):
        line = _strip_comments(raw_line)
        findings.extend(_findings_for_line(filename, line_no, line))
        func_name = _is_function_start(line)
        if func_name and not scope_stack:
            scope_stack.append(CFunctionScope(name=func_name))
        if scope_stack:
            brace_level += line.count("{") - line.count("}")
            scope = scope_stack[-1]
            for match in ALLOC_RE.finditer(line):
                var = match.group(1)
                if var in scope.reassigned or var not in scope.malloced:
                    scope.malloced[var] = line_no
                    scope.freed.pop(var, None)
            for match in FREE_RE.finditer(line):
                var = match.group(1)
                if var in scope.freed:
                    findings.append(
                        Finding(
                            rule_id="c-double-free",
                            severity=Severity.CRITICAL,
                            title="Double free",
                            description=f"{var} is freed again without reallocation.",
                            location=Location(file=filename, line=line_no, column=match.start()),
                            language="c/c++",
                            snippet=line.strip(),
                            fix_suggestion="Set pointer to NULL after free or remove duplicate free.",
                        )
                    )
                else:
                    scope.freed[var] = line_no
            for match in DELETE_RE.finditer(line):
                var = match.group(1)
                if var in scope.freed:
                    findings.append(
                        Finding(
                            rule_id="c-double-delete",
                            severity=Severity.CRITICAL,
                            title="Double delete",
                            description=f"{var} is deleted again without reallocation.",
                            location=Location(file=filename, line=line_no, column=match.start()),
                            language="c/c++",
                            snippet=line.strip(),
                            fix_suggestion="Avoid deleting the same pointer twice.",
                        )
                    )
                else:
                    scope.freed[var] = line_no
            for match in ASSIGN_RE.finditer(line):
                var = match.group(1)
                if var in scope.freed:
                    scope.freed.pop(var, None)
                    scope.reassigned.add(var)
            for var in list(scope.freed.keys()):
                if var in line and not FREE_RE.search(line) and not _is_assignment_to(line, var):
                    if re.search(rf"\b{re.escape(var)}\b(?!\s*\()", line):
                        findings.append(
                            Finding(
                                rule_id="c-use-after-free",
                                severity=Severity.CRITICAL,
                                title="Use after free",
                                description=f"{var} is used after being freed on line {scope.freed[var]}.",
                                location=Location(file=filename, line=line_no),
                                language="c/c++",
                                snippet=line.strip(),
                                fix_suggestion="Set pointer to NULL after free or reassign before use.",
                            )
                        )
                        scope.freed.pop(var, None)
            if brace_level <= 0:
                for var, alloc_line in scope.malloced.items():
                    if var not in scope.freed:
                        findings.append(
                            Finding(
                                rule_id="c-memory-leak",
                                severity=Severity.MEDIUM,
                                title="Possible memory leak",
                                description=f"{var} allocated on line {alloc_line} is not freed in {scope.name}.",
                                location=Location(file=filename, line=alloc_line),
                                language="c/c++",
                                snippet=lines[alloc_line - 1].strip() if alloc_line <= len(lines) else "",
                                fix_suggestion="Ensure every allocation has a matching free/deletion path.",
                            )
                        )
                scope_stack.clear()
                brace_level = 0
    return findings


def _is_assignment_to(line: str, var: str) -> bool:
    return bool(re.search(rf"\b{re.escape(var)}\s*=(?!=)", line))
