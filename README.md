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
| `tasks/` | ResourceTask / DonationTask / AgentLoop：三环闭环与 ADB 接线（阶段 3/7/10） | ✅ |
| `decision/` | 采集/打资源/捐兵 FSM + 资源阈值 + 仲裁器（组合村庄环优先） | ✅ |
| `api/` | FastAPI 感知服务（/perceive） | 🔄 |
| `security/audit/` | 强制代码审计 | ✅ |
| `tools/labeling/` `asset_crawl/` | 标注工具、数据爬虫 | ✅ |

## 快速运行

```bash
conda activate multimodal-agent

# 全部测试
python -m pytest tests -ra
```

启动感知 API（自动读取 `api.txt` 第 4 行密钥并复用 `multimodal-agent` 环境）：

```bash
./run_api.sh
```

等价于在项目根目录下执行：

```bash
export VISION_MODEL_BASE_URL="https://ws-avxkjb2tq5lq1gwm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export VISION_MODEL_API_KEY="$(sed -n '4p' api.txt)"
export VISION_MODEL_NAME="qwen3.8-omni-flash"
export VISION_MODEL_TIMEOUT=120
/home/kun/miniconda3/envs/multimodal-agent/bin/python -m uvicorn api.main:app \
  --host 0.0.0.0 --port 8000
```

```bash
# 在模拟环境跑一次完整打资源闭环（演示）
python - <<'PY'
from executor.simulator import Simulator, default_farm_world
from tasks.resource import ResourceTask

result = ResourceTask().run(Simulator(default_farm_world()))
print(result.summary()["status"], result.summary()["total_loot"])
PY
```

ADB 三环 CLI（连接真机/模拟器后执行；当前按用户要求停在设备接线前）：

```bash
python -m tasks.adb_agent --rounds 8 --max-steps 60
```

运行顺序：村庄环先收集免费采集图标 → 资源阈值检查 → 捐兵检测；捐兵因缺资源
`BLOCKED` 或资源跌破低保线 `LOW` 时，立即调用打资源环补缺口，战斗结束后自动回到
村庄环重新采集 + 捐兵检测。

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
