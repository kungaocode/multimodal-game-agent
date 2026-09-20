"""AST-based security scanner for Python source code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .models import Finding, Location, Severity

DANGEROUS_FUNCTIONS = {
    "eval": (Severity.CRITICAL, "Dangerous function call", "Avoid eval(); it can execute arbitrary code."),
    "exec": (Severity.CRITICAL, "Dangerous function call", "Avoid exec(); it can execute arbitrary code."),
    "compile": (Severity.HIGH, "Compilation call", "Ensure compile() source is trusted."),
    "__import__": (Severity.HIGH, "Dynamic import", "Avoid __import__(); use importlib with validation."),
    "input": (Severity.MEDIUM, "Unsafe input", "input() is unsafe in Python 2 compatibility mode; validate carefully."),
    "os.system": (Severity.CRITICAL, "Shell command execution", "Avoid os.system(); use subprocess with a fixed argv."),
    "os.popen": (Severity.CRITICAL, "Shell command execution", "Avoid os.popen(); use subprocess with a fixed argv."),
}

DESERIALIZATION_PATTERNS = {
    ("pickle", "loads"): (Severity.CRITICAL, "Unsafe deserialization", "Never unpickle untrusted data."),
    ("cPickle", "loads"): (Severity.CRITICAL, "Unsafe deserialization", "Never unpickle untrusted data."),
    ("_pickle", "loads"): (Severity.CRITICAL, "Unsafe deserialization", "Never unpickle untrusted data."),
    ("yaml", "load"): (Severity.HIGH, "Unsafe YAML loading", "Use yaml.safe_load() instead."),
    ("marshal", "loads"): (Severity.HIGH, "Unsafe deserialization", "Never unmarshal untrusted data."),
}

WEAK_HASHES = {"md5", "sha1"}
SQL_METHODS = {"execute", "executemany", "executescript", "raw"}

SECRET_NAME_RE = re.compile(
    r"(password|passwd|pwd|secret|token|api_key|apikey|private_key|access_key|auth_token)$",
    re.IGNORECASE,
)


@dataclass
class FunctionScope:
    opens: dict[str, int] = field(default_factory=dict)
    closed: set[str] = field(default_factory=set)


class PythonSecurityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, source: str) -> None:
        self.filename = filename
        self.source = source
        self.lines = source.splitlines()
        self.findings: list[Finding] = []
        self.imports: dict[str, str] = {}
        self.scope_stack: list[FunctionScope] = []

    def add(
        self,
        rule_id: str,
        severity: Severity,
        title: str,
        description: str,
        node: ast.AST,
        fix_suggestion: str = "",
    ) -> None:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = self.lines[line - 1] if 1 <= line <= len(self.lines) else ""
        self.findings.append(
            Finding(
                rule_id=rule_id,
                severity=severity,
                title=title,
                description=description,
                location=Location(file=self.filename, line=line, column=col),
                language="python",
                snippet=snippet.strip(),
                fix_suggestion=fix_suggestion,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = f"{module}.{alias.name}" if module else alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._resolve_call_name(node.func)
        if func_name in DANGEROUS_FUNCTIONS:
            sev, title, desc = DANGEROUS_FUNCTIONS[func_name]
            self.add(
                f"python-dangerous-{func_name.replace('.', '-')}",
                sev,
                title,
                desc,
                node,
                f"Replace {func_name}() with a safer alternative.",
            )
        parts = func_name.rsplit(".", 1)
        key = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        if key in DESERIALIZATION_PATTERNS:
            sev, title, desc = DESERIALIZATION_PATTERNS[key]
            self.add(
                f"python-deserialization-{parts[0]}-{parts[1]}",
                sev,
                title,
                desc,
                node,
                "Use safe parsing alternatives (e.g., json, yaml.safe_load).",
            )
        if func_name in {"subprocess.call", "subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call"}:
            if any(kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords):
                self.add(
                    "python-subprocess-shell",
                    Severity.CRITICAL,
                    "Subprocess with shell=True",
                    "shell=True enables command injection if arguments are attacker-controlled.",
                    node,
                    "Use shell=False and pass a list of arguments.",
                )
        if func_name in {"hashlib.md5", "hashlib.sha1"} or (
            func_name == "hashlib.new"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).lower() in WEAK_HASHES
        ):
            self.add(
                "python-weak-hash",
                Severity.MEDIUM,
                "Weak cryptographic hash",
                f"{func_name} is not collision-resistant.",
                node,
                "Use hashlib.sha256 or a purpose-built password hashing library.",
            )
        if func_name in {"tempfile.mktemp"} or (
            func_name in {"mktemp"} and self.imports.get("mktemp") == "tempfile.mktemp"
        ):
            self.add(
                "python-unsafe-tempfile",
                Severity.HIGH,
                "Insecure temporary file creation",
                "mktemp() is subject to race conditions.",
                node,
                "Use tempfile.mkstemp() or NamedTemporaryFile(delete=False).",
            )
        if any(kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords):
            self.add(
                "python-no-tls-verify",
                Severity.HIGH,
                "TLS certificate verification disabled",
                "verify=False disables SSL/TLS certificate checks.",
                node,
                "Remove verify=False or pin a CA bundle.",
            )
        if func_name in {"os.chmod"} and len(node.args) >= 2:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and isinstance(mode.value, int):
                if mode.value & 0o002:
                    self.add(
                        "python-world-writable",
                        Severity.MEDIUM,
                        "World-writable permissions",
                        f"os.chmod() sets mode {oct(mode.value)} which grants write to others.",
                        node,
                        "Use restrictive permissions (e.g., 0o600).",
                    )
        if self._is_sql_method(func_name) and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.JoinedStr) or (
                isinstance(first_arg, ast.Call)
                and isinstance(first_arg.func, ast.Attribute)
                and first_arg.func.attr == "format"
            ):
                self.add(
                    "python-sql-injection",
                    Severity.CRITICAL,
                    "Possible SQL injection",
                    "SQL query is built from a formatted string.",
                    node,
                    "Use parameterized queries.",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = self._get_target_name(target)
            if name and SECRET_NAME_RE.search(name):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    self.add(
                        "python-hardcoded-secret",
                        Severity.HIGH,
                        "Hardcoded secret",
                        f"Variable '{name}' appears to hold a hardcoded credential.",
                        node,
                        "Load secrets from environment variables or a secrets manager.",
                    )
            if isinstance(node.value, ast.Call) and self._resolve_call_name(node.value.func) in {"open", "io.open"}:
                if self.scope_stack and name:
                    self.scope_stack[-1].opens[name] = node.lineno
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call) and self._resolve_call_name(item.context_expr.func) in {"open", "io.open"}:
                name = self._get_target_name(item.optional_vars) if item.optional_vars else None
                if name and self.scope_stack and name in self.scope_stack[-1].opens:
                    self.scope_stack[-1].opens.pop(name, None)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.add(
                "python-bare-except",
                Severity.MEDIUM,
                "Bare except clause",
                "Bare 'except:' catches KeyboardInterrupt and SystemExit.",
                node,
                "Catch specific exceptions (e.g., except ValueError).",
            )
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception" and len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                self.add(
                    "python-swallowed-exception",
                    Severity.LOW,
                    "Exception swallowed silently",
                    "Broad Exception is caught and ignored.",
                    node,
                    "Log the exception or re-raise after cleanup.",
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_scope(node)

    def _enter_scope(self, node: ast.AST) -> None:
        self.scope_stack.append(FunctionScope())
        self.generic_visit(node)
        scope = self.scope_stack.pop()
        for name, line in scope.opens.items():
            if name not in scope.closed:
                self.add(
                    "python-resource-leak",
                    Severity.MEDIUM,
                    "Possible resource leak",
                    f"File opened on line {line} may not be closed.",
                    node,
                    "Use a context manager (with open(...) as f:).",
                )

    def _resolve_call_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            full = self.imports.get(node.id, node.id)
            return full
        if isinstance(node, ast.Attribute):
            base = self._resolve_call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _get_target_name(self, node: ast.expr | ast.Tuple | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Tuple):
            return None
        return None

    def _is_sql_method(self, func_name: str) -> bool:
        parts = func_name.rsplit(".", 1)
        return parts[-1] in SQL_METHODS if parts else False


def scan_python_file(filename: str, source: str) -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = exc.lineno or 0
        return [
            Finding(
                rule_id="python-syntax-error",
                severity=Severity.INFO,
                title="Python syntax error",
                description=str(exc),
                location=Location(file=filename, line=line),
                language="python",
            )
        ]
    visitor = PythonSecurityVisitor(filename, source)
    visitor.visit(tree)
    return visitor.findings
