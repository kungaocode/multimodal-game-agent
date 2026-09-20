# security/audit — 代码审计模块

本模块用于对项目中的 **Python** 与 **C/C++** 代码进行强制性静态安全审计和执行效率测试。任何新模块在合并前必须通过审计。

## 覆盖范围

### Python 静态安全规则

| Rule ID | 严重级别 | 说明 |
|---------|----------|------|
| `python-dangerous-*` | CRITICAL | eval / exec / os.system / os.popen 等危险调用 |
| `python-subprocess-shell` | CRITICAL | subprocess.* 使用 `shell=True` |
| `python-deserialization-*` | CRITICAL/HIGH | pickle.loads / yaml.load / marshal.loads 等不安全的反序列化 |
| `python-hardcoded-secret` | HIGH | 硬编码密码 / token / key |
| `python-no-tls-verify` | HIGH | `verify=False` 关闭 TLS 校验 |
| `python-sql-injection` | CRITICAL | 使用 f-string / .format 拼接 SQL |
| `python-weak-hash` | MEDIUM | md5 / sha1 等弱哈希 |
| `python-world-writable` | MEDIUM | `os.chmod` 设置为全局可写 |
| `python-unsafe-tempfile` | HIGH | `mktemp()` 竞态条件 |
| `python-bare-except` | MEDIUM | 裸 except |
| `python-swallowed-exception` | LOW | 捕获 Exception 后静默吞掉 |
| `python-resource-leak` | MEDIUM | `open()` 未关闭或不用上下文管理器 |

### C/C++ 静态安全规则

| Rule ID | 严重级别 | 说明 |
|---------|----------|------|
| `c-use-after-free` | CRITICAL | 释放后使用指针 |
| `c-double-free` / `c-double-delete` | CRITICAL | 重复释放/删除 |
| `c-buffer-overflow` | HIGH | strcpy / strcat / gets / sprintf / scanf 等 |
| `c-format-string` | HIGH | printf 使用变量作为格式字符串 |
| `c-memory-leak` | MEDIUM | malloc/new 未释放 |
| `c-unsafe-api-*` | HIGH/CRITICAL | 不安全 C API |

### 性能测试

- `benchmark()` 装饰器：多次运行 + 超时保护 + 均值/标准差/最大最小值
- `profiled()` 装饰器：cProfile 单函数分析
- `memory_usage()` 上下文：tracemalloc 峰值内存测量
- `PerformanceProfiler`：程序化对 `module:function` 进行基准测试

## 使用方式

### 命令行

```bash
# 使用 pyproject.toml 默认配置扫描
python -m security.audit

# 扫描指定目录
python -m security.audit --path security perception

# 生成 Markdown 报告
python -m security.audit --format markdown --output AUDIT_REPORT.md

# 调整阈值
python -m security.audit --threshold HIGH --max-critical 0 --max-high 2

# 对目标函数做性能基准
python -m security.audit --perf security.audit.samples.safe_demo:heavy_function
```

### 作为 pytest 用例

```bash
python -m pytest tests/audit -v
```

### 程序化调用

```python
from security.audit import AuditScanner, AuditReporter, AuditConfig

scanner = AuditScanner(AuditConfig(include=["perception", "state"]))
result = scanner.scan()
reporter = AuditReporter(result, scanner.config)
print(reporter.to_console())
```

## 配置

在 `pyproject.toml` 的 `[tool.audit]` 段配置：

```toml
[tool.audit]
include = ["security", "perception", "state", "reasoning", "decision", "tasks", "executor", "evaluation", "tests"]
exclude = ["__pycache__", ".git", ".venv"]
severity_threshold = "MEDIUM"
max_critical = 0
max_high = 0
perf_timeout_seconds = 30.0
perf_memory_mb = 512.0
perf_iterations = 10
```

## 强制策略

1. 任何新增 Python / C / C++ 文件必须被审计扫描到。
2. CI 中运行 `python -m security.audit`，返回非零则阻止合并。
3. CRITICAL 和 HIGH 问题必须修复或得到明确豁免。
4. 性能测试报告随版本发布保存，用于回归对比。
