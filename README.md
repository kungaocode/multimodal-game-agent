# Multimodal Game Agent

基于多模态大模型的跨平台游戏环境感知与安全智能任务规划系统。

## 项目进度

阶段状态、测试进度、最近交付与下一步建议见 [PROGRESS.md](PROGRESS.md)。

## 模块速览

| 目录 | 职责 | 状态 |
|------|------|------|
| `perception/` | YOLO 检测、等级分类、PaddleOCR、视觉大模型、GameState 提取 | ✅ |
| `state/` | 结构化游戏状态（GameState / ResourceStatus） | ✅ |
| `decision/` | 打资源 FSM + 资源目标优先度打分（阶段 4） | ✅ |
| `executor/` | 结构化动作 + 动作校验器 + 模拟环境（阶段 8） | ✅ |
| `tasks/` | ResourceTask 打资源闭环：感知→决策→校验→执行→验证（阶段 3/7） | ✅ |
| `api/` | FastAPI 感知服务（/perceive） | 🔄 |
| `security/audit/` | 强制代码审计 | ✅ |
| `tools/labeling/` `asset_crawl/` | 标注工具、数据爬虫 | ✅ |

## 快速运行

```bash
conda activate multimodal-agent

# 全部测试（当前 171 个）
python -m pytest tests -ra

# 在模拟环境跑一次完整打资源闭环（演示）
python - <<'PY'
from executor.simulator import Simulator, default_farm_world
from tasks.resource import ResourceTask

result = ResourceTask().run(Simulator(default_farm_world()))
print(result.summary()["status"], result.summary()["total_loot"])
PY
```

## 代码审计（强制流程）

本项目所有 Python / C / C++ 代码在合并前必须通过 `security.audit` 审计：

```bash
# 扫描全部默认目录
python -m security.audit

# 扫描指定目录并生成 Markdown 报告
python -m security.audit --path security perception state --format markdown --output AUDIT_REPORT.md

# 作为 pytest 用例运行（失败阈值由 pyproject.toml [tool.audit] 控制）
python -m pytest tests/audit -v
```

详细说明见 [security/audit/README.md](security/audit/README.md)。
