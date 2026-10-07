# 项目进度（Progress）

> 最后更新：2026-10-07（ADB 三环接线：组合村庄环 + 阻塞切战斗）
> 对应阶段拆分计划（阶段 0~15）

## 本地 YOLO 能力核查（2026-10-06）

**结论：当前 `fsm_v4` / `fsm_v5` 尚不能独立完成「打资源 + 捐兵」端到端闭环。**
框架本身已覆盖两个模块所需类别，缺的是真实场景训练数据与模型质量，不适宜现阶段直接用云端 4B
全文替换；已具备的级联感知（本地优先 + VL 兜底 + 样本回流）是更稳的实现方式。

核查证据：
- 历史 coach 会话共 185 步：本地动作全部为 `wait`，`tap=0`、`deploy=0`；真机执行全部来自教练纠正。
- `fsm_v4`（10 epochs，coach_training_v4 仅 14 图/47 框且无资源建筑标注）：mAP50=0.4975，虚高。
- `fsm_v5`（10 epochs，video_batch_v5 84 图/488 框）：mAP50=0.0846、mAP50-95=0.0292。
- 实拍推理：打资源战斗帧在 conf=0.1/0.25 均 0 检出；捐兵帧只把多个绿条误检为「增援按钮」，
  消息列表/关闭/确认/兵种选择栏均未检出。
- 17 类标注总量偏少且不均衡：返回按钮 6、搜索对手 12、下一个 12、关闭 10、捐赠确认 11。

## 最近交付（2026-10-06 · 下兵掠夺核心逻辑 + 捐兵闭环）

### 本轮核心改动

**下兵与掠夺判断（主线一）**
- `decision/loot_estimator.py`：新增确定性可获取性估算器。输入可下兵掩码/城墙掩码或墙段/防御坐标/容量/已抢量/兵力预算，输出落点、落点距建筑距离、城墙穿越兵力、步行损耗、所需兵力、预期收益、风险分和效率分；不同落点直接改变收益系数与兵力成本。
- `decision/battle_memory.py`：新增战斗内目标记忆，维护容量、已抢/剩余、下兵轮次、连续可见帧和打空判定，供 planner 排除已打空目标并支撑撤退判空。
- `decision/deploy_planner.py`：分数排序后接入 `LootEstimator` + `BattleMemory`，同一目标连续部署轮换，教练候选回填，`commit=False/commit()` 保证只有执行成功的波次才记入记忆。
- `executor/simulator.py`：模拟器不再要求落点等于建筑中心；增加 `radius`、`wall_segments`、空投距离/步行损耗/墙体代价/收益衰减，落点不同会得到不同的掠夺结果和兵力消耗。
- `tools/coach_masks.py`：新增 `wall_mask()` 灰度墙体识别，供估算器计算城墙穿越。
- 结束判定：`_battle_end_guard` 与记忆联动，连续 N 帧无可获取目标才放行结束/返回，已打空资源不会被重复下兵。

**捐兵闭环（主线二）**
- `executor/donation_simulator.py`：新增五屏捐兵确定性模拟器（村庄→消息列表→请求详情→增援选择→完成），必须选兵种才能确认；支持多请求自动回到消息列表继续下一单。
- `tasks/donation.py`：新增 `DonationTask`，串起感知→FSM→校验→执行→验证→日志。
- `decision/donation_fsm.py`：补全捐兵 FSM，所有 tap 带坐标和目标名；增加 `more_requests` 信号；进入新请求时重置兵种选择状态。
- API 与 CLI 共用同一 `_apply_battle_decision_chain`，`api/sessions.py` 已接入回填/planner/结束护栏/执行后 commit，并新增 `end_empty_frames`、`troop_budget` 参数。

**验证**
- `python -m pytest tests -q -o addopts=''`：213 passed
- `python -m security.audit`：0 findings
- 新增测试：`tests/test_loot_estimator.py`、`tests/test_battle_memory.py`、`tests/test_donation_fsm.py`、`tests/test_donation_task.py`

### 三环调度补测（2026-10-06）

- `state/game_state.py`：新增 `ResourceThresholds`，金/圣水/黑油分别配置 `low / resume`，捐兵缺口另带安全缓冲。
- `decision/resource_gate.py`：确定性滞回：低于 `low` 进入打资源环，回升到 `resume` 才解除，避免临界值附近抖动。
- `decision/arbiter.py` + `tasks/agent_loop.py`：三环仲裁统一为 采集 > 打资源 > 捐兵；采集环与打资源环互斥；
  捐兵 `BLOCKED` 的缺口转成 `FarmObjective`，资源补足后自动放行捐兵环。
- `decision/collector_fsm.py` / `executor/collector_simulator.py` / `tasks/collector.py`：本地 YOLO 只需报出
  金/圣水/黑油采集图标，FSM 负责点图标并验证顶栏到账。
- `tasks/donation.py`：`BLOCKED` 结果现在携带完整 `resource_type / required / available / missing`，
  仲裁器按缺口生成打资源目标而不是按整条需求重复刷资源。
- `tasks/resource.py`：支持只打目标资源类型；混合村庄中非目标资源不会阻塞战斗结束，且已满足目标时直接返回，不再空等。

### qwen3.8-omni-flash 云端联调（2026-10-06）

- 用 `api.txt` 第 4 行密钥 + `https://ws-avxkjb2tq5lq1gwm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` 接入，
  `qwen3.8-omni-flash` 经现有 `VisionModel.chat()` 直接返回 JSON，无需改模型结构。
- 真实回归图验证：farm 截图能返回带像素 `coords` 的金矿/圣水/黑油等资源建筑与按钮；
  `/perceive?mode=llm` + `/plan`（FastAPI 真实链路）联调通过，规则层无候选时可用 LLM 候选补入。
- 兼容性修正：
  - `perception/vision_model.py`：默认超时 60s → 120s，新增 `VISION_MODEL_TIMEOUT` 环境变量；
    JSON 根节点为数组时按 `objects` 兼容，避免 omni 系列偶发数组输出导致崩溃。
  - `perception/cascade.py`：LLM 物件坐标兼容 `coords` / `position_estimate` / `bbox` 及 `"[x, y]"`、`"x, y"` 字符串表示。
- `run_api.sh`：一键启动 FastAPI；自动 cd 到项目根目录、复用 `multimodal-agent` 环境、读取 `api.txt` 密钥。
- 验证：`python -m pytest tests -q -o addopts=''` → 215 passed；`python -m security.audit` → 0 findings。
- 云端模型用途固定在画面感知/目标识别；三环仲裁仍由本地 `AgentArbiter` 等确定性逻辑负责。

### ADB 三环接线（2026-10-07）

- `tasks/adb_agent.py`：新增 `AdbAgent`，把真机 ADB 接到 `AgentLoop` 四个端口
  （`perceive_village / run_collect / run_farm / run_donate / run_village`），
  复用 `AdbExecutor`、`CascadeExtractor`（本地 YOLO + 云端 Qwen 兜底）、
  `BattleFSM`、`DonationFSM`、`CollectorFSM` 与 CLI/API 共用的下兵决策链。
- `tasks/agent_loop.py`：`AgentPorts` 新增可选 `run_village`，有该端口时优先走
  「组合村庄环」：先采集免费图标 → 资源阈值检查 → 捐兵检测；`BLOCKED`/`LOW`
  立即调用 `run_farm` 补缺口，战斗结束下一轮自动回到村庄环。
- 修复 `AdbAgent._perceive_state` 未写 `_state_cache` 的问题，`perceive_village`
  现在读取最近一帧缓存，避免一次快照重复截屏。
- 验证：`python -m pytest tests -q -o addopts=''` → 221 passed；
  `python -m py_compile tasks/adb_agent.py tasks/agent_loop.py decision/arbiter.py`
  通过；新增 `tests/test_adb_agent_ports.py` 覆盖缓存回归、BLOCKED→FARM→重新村庄、
  LOW→FARM 三种接线。

### 当前进度（同步后）

| 模块 | 状态 | 本轮备注 |
|------|------|---------|
| 阶段 0-4 / 7 / 8 | ✅ | 环境/数据/感知/状态机/资源打分/模拟闭环/动作校验均完成 |
| 阶段 6 / 级联感知 | ✅ | qwen3.8-omni-flash 真实联调 + cascade 路由 + 样本回流 + `/cascade/stats` |
| CLI coach 会话 | ✅ | 已含资源回填、DeployPlanner、可获取性估算、战斗记忆、判空护栏 |
| 后端 API（阶段 9） | 🔄 | `/perceive`、`/plan`、`/cascade/stats`、`/session/*` 可用；API 会话已复用 CLI 决策链；`/execute`、`/verify` 仍缺失 |
| 捐兵（阶段 10） | ✅ | FSM + 校验器白名单 + 五屏确定性模拟器 + DonationTask 闭环 |
| 三环调度 | ✅ | 采集优先、打资源/捐兵仲裁、资源阈值滞回、阻塞缺口注入目标；`AgentPorts` 已留好 API 接线边界 |
| Android / HarmonyOS（阶段 11/12） | 🔄 | `tasks/adb_agent.py` 接线完成，等用户连接真机/模拟器后冒烟 |
| 阶段 13 评估 | 🔄 | 状态转换工具与基线在；真机本地 YOLO 检出力弱 |
| 阶段 14 安全 | ✅ | 审计通过；validator/风险分级在 |
| 阶段 15 真机 | ⬜ | 未在本轮执行 |
| 数据资产 | ✅ | 20 个含 `session_summary.json` 的教练目录；resource 5 视频 + donate 6 视频可 dry-run |

### 遗留问题（按优先级）

| # | 级别 | 问题 | 位置 | 说明 |
|---|------|------|------|------|
| P2-2 | 中 | `/execute`、`/verify` 未实现 | `api/main.py` | 阶段 9 收尾项缺失，移动端没有单动作执行与验证接口 |
| P3-1 | 低 | 真机本地模型检出力弱 | PROGRESS 历史 Eval | 重训数据不足；`fsm_real_buttons` 与 `final_test_dataset` 疑似同源，需拆分核实 |
| P3-2 | 低 | 捐兵真实录制视频中 `more_requests` 尚未由 OCR/检测器产出 | `perception/fsm_labels.py` | 模拟器已用该信号打通多请求；真机识别仍需补一个「还有新请求」信号，否则默认按单请求回村庄 |
| P3-3 | 低 | `api.txt` 含凭据（已 gitignore，未入库） | 工作区根目录 | 建议迁到环境变量并轮换密钥；本轮未打印/提交 |

### 后续推进与测试计划（2026-10 起）

| 阶段 | 工作项 | 验收 / 测试 |
|------|--------|-----------|
| 1 真机下兵验收 | dry-run resource 视频并记录落点、墙体代价、重复轮换和判空帧数 | 落点 100% 在可下兵草地；无刚开战就结束；同一目标轮换符合 `max_consecutive_per_target` |
| 2 真机捐兵验收 | dry-run donate 视频并核对五屏转移，补真实 `more_requests` 感知 | 每一单都选兵种再确认；多请求不提前回村；无全 wait |
| 3 接 API 三环 | 用 `AgentPorts` 接真实 ADB/OCR/Qwen：感知村庄、执行采集、执行打资源、恢复捐兵 | mock + 模拟器契约测试通过（`tests/test_adb_agent_ports.py`）；下一步等设备连接后做真机三环冒烟 |
| 4 补 P2-2 | 实现 `/execute`（validator + ADB/simulator）与 `/verify`（前后 GameState 对比） | API 契约测试 + simulator 回归 |
| 5 数据/模型 | 拆分 train/val，用现有 coach 会话 + 新采集回流重训 v6 | YOLO 真机关键按钮/资源检出率上升；教练纠正率下降；汇总对比表 |
| 6 真机 | 恢复逻辑（桌面→拉起 App）、结束判定 3 帧护栏、误点右下角图标回归 | 按「下次上机验收口径」逐条记录 |

## 阶段总览

| 阶段 | 内容 | 状态 | 说明 |
|------|------|------|------|
| 0 | 环境与数据准备 | ✅ | conda 环境 `multimodal-agent`；43 类检测数据集、farm 8 类合成集、爬虫数据（dataset/） |
| 1 | 视觉识别模块 | ✅ | YOLO 检测器 + ResNet18 等级分类器（已训练）+ PaddleOCR 封装 + GameStateExtractor；`perception/` |
| 2 | 状态管理与 FSM | ✅ | `state/game_state.py`（GameState/ResourceStatus）+ `decision/farm_fsm.py`（村庄待机/战斗中/战斗结束/停止） |
| 3 | 资源任务模块 | ✅ | `tasks/resource.py`：感知→决策→校验→执行→验证闭环（模拟环境） |
| 4 | 资源目标优先度算法 | ✅ | `decision/resource_policy.py`：0.40 价值 + 0.30 可达 + 0.20 安全 + 0.10 距离，权重可配，自动跳过己方已满资源 |
| 5 | 升级任务模块 | ⬜ | 计划中（需检测工人与升级 UI） |
| 6 | 国产多模态大模型 | ✅ | `perception/vision_model.py` 已支持 Qwen-VL 类 API + JSON 输出；qwen3-vl-plus 真实 API 联调通过（含 JSON Output 与 400 降级重试）；高层规划并入 `/plan`（阶段 9）；**级联路由**（本地优先 + LLM 兜底 + 样本回流）落地 |
| 7 | Agent 闭环 | ✅ | ResourceTask + Simulator 完成打资源闭环（阶段 3/4/7 一起交付） |
| 8 | 行为执行器与模拟环境 | ✅ | `executor/`：AgentAction schema + ActionValidator 白名单/越界校验 + Simulator 确定性仿真 |
| 9 | 后端 API 封装 | 🔄 | `/health`、`/perceive`、`/perceive/base64`、`/plan` 均可用并已真实联调；`/execute`、`/verify` 未加 |
| 10 | 捐兵模块 | ✅ | 捐兵 FSM + 五屏确定性模拟器 + DonationTask 闭环；真实 `more_requests` 感知待补 |
| 11 | Android 客户端 | ⬜ | 未开始 |
| 12 | HarmonyOS 客户端 | ⬜ | 未开始 |
| 13 | 实验评估体系 | 🔄 | 状态转换识别测试工具+基线完成；后续计划见「接下来测试重点」 |
| 14 | AI Agent 安全研究 | 🔄 | security/audit 强制审计已通过；动作校验器/风险分级已落地（executor/validator）；级联路由含触发率/成本统计与样本审计 |
| 15 | 真机测试与交付 | ⬜ | 未开始 |

## 最近交付（2026-09-06 · 战斗开始后的行为逻辑）

> 解决第6轮「过早结束战斗」「资源列表缺失」；改动前已备份 3 个源文件

**代码改动（3 个源文件）**
1. `decision/battle_fsm.py`：新增极小组件 `EmptyTargetGuard`（连续 N 帧无可获取目标才放行结束，`threshold` 默认 3，可配）；`BattleFSM.__init__` 加 `empty_frames_threshold`；BATTLE 分支改为「先算法判可获取目标→有则 deploy 并清空判空计数→return_button 被动结算→end_battle_button+连续判空达阈值才主动结束」；进入新战斗时守卫复位；`BattleSignals.enemy_resources` 语义注释明确每项都是「算法会打分的候选」。
2. `decision/coach.py`：`CoachVerdict` 新增 `resource_buildings` 字段；prompt JSON schema 增补该 key；重写「战斗中」规则（红「结束战斗」按钮整场常驻≠结束信号；每帧枚举全部可见资源建筑；无资源建筑时才允许结束）；`_parse_verdict` 解析/过滤（归一化 center→像素，非 6 类/越界/缺失丢弃，旧画布像素坐标 viewed→orig 兜底）。
3. `tools/coach_session.py`：coach review 后、deploy_planner.plan 前插入 ① 资源回填（`verdict.resource_buildings` 并入 `signals.enemy_resources` 去重）② 模块级 `_battle_end_guard` 护栏（与 BattleFSM **共享同一守卫实例**：只有 `deploy_planner.plan` 判「连续 N 帧无可获取目标」才放行结束/返回，否则强制转下兵或 wait）；新增 argparse `--end-empty-frames`（默认 3）透传 BattleFSM；`战斗结束` 时重置 guard 与 planner。

**验证**
- `python -m pytest tests`：**171 passed**（基线 149 + 新增 22：battle_fsm 结束/守卫/顺序用例、coach resource_buildings 解析、coach_session 护栏 dry-run）
- `python -m security.audit`：0 发现
- 真机未跑（按方案 §1/§7：先出方案与离线测试，暂不再自动真机点击）

**下次上机验收口径（跑通一次完整战斗）**
1. 进入战斗后日志出现多资源打分与 deploy，没有「刚开战就 tap 结束」；
2. 打光过程中某帧视野暂无资源时，连续 ≥3 帧无目标才 tap 结束；
3. 结束只发生在：连续判空 / 结算界面（return_button）；随后回村进入下一轮再次进攻；
4. 每条 deploy 日志含 target_score / repeat / reason，落点 100% 在可下兵草地（不进红区/UI）。

## 最近交付（2026-09-04 · 下兵最终选择层）

**代码改动**
1. 新增 `decision/deploy_planner.py`：
   - 在 `ResourcePolicy` 排序之后做最终 deploy 决策；
   - 教练给出的目标即使本地漏检，也会作为补充候选重新打分，不再盲目执行；
   - 己方已满的资源类型继续跳过；同一目标连续部署达到上限且有备选时自动轮换；
   - 输出 `target_score / target_breakdown / target_repeat_count / target_reason` 便于逐帧复盘。
2. `tools/coach_session.py`：
   - 本地 `BattleFSM` 改用 `ResourcePolicy`；
   - 本地与教练纠正后的 deploy 都统一经过 `DeployPlanner`；
   - 规划后的落点只执行一次真实点击链：先 `troop_coords`，再贴建筑草地落点。
3. `decision/battle_fsm.py`：
   - 搜索中“值得打”的 deploy 不再取原始列表第一个，改用策略排序后的第一目标。

**验证**
- `python -m pytest tests -ra`：147 passed
- `python -m security.audit`：0 findings
- 2026-09-04 本地逻辑冒烟（720x1280 合成战场，两个金矿，第一个被红区边框包围）：
  - 3 次目标选择：`(300,600) → (300,600) → (900,600)`
  - reasons：`top_scored_target → repeat_limit_reached → rotated_after_repeat_limit`
  - 所有落点均吸附到禁落掩码外草地；模拟 ADB 共 6 次点击，均为先兵种栏后落点。

**下次上机验收**
1. 多资源可见时，日志应出现高优先目标；同一目标最多连续 2 次，第 3 次切到次优目标。
2. 本地漏检但教练给出 6 类资源名时，仍能补充候选并按可达性/价值打分。
3. 每条 deploy 日志应包含 score/repeat/reason，最终截图落点继续全部落在可下兵草地。

## 最近交付（2026-09-03 深晚 · 第6轮最终真机会话，用户已暂停待修复）

**第6轮 Coach 上机会话** `dataset/coach_sessions/20260903_113129_b9b98cf7/`（16/30 步，11 次执行，教练纠正率 100%）
- 成功点：
  - 新落点规划生效：战斗中 deploy 金矿时 `landing_plan=building_adjacent`（先排除红区/UI 不可下兵，再就近资源建筑吸附，落点在目标附近草地）
  - 村庄 → 进攻按钮 → 搜索对手 → 进入战斗的链路依然正确（step1~3）
- 失败点（用户已确认）：
  1. **进攻按钮误点**：step4/step11 教练把右下角 `(2304,1109)` 当成进攻按钮（触发购买/金币不足弹窗），需排除右下角「加速/购买」类图标
  2. **停止战斗判定过松**：step8 开战约 30 秒（倒计时还剩 2:50）、资源未打满，教练仅因「看到灰色返回箭头 + 视野内暂时无敌方资源建筑」就判定战斗结束并返回；要求：有可见敌方资源建筑或摧毁进度未打满时不得返回；只有红色「结束战斗」按钮（或有明确结算界面）才算战斗结束；灰色返回箭头/视野空窗不能当结束信号
  3. **退出到桌面无恢复逻辑**：step13~16 手机已退回桌面（App 图标可见），本地模型仍判「村庄待机」、教练只 wait，没有「检测到桌面 → 重新拉起 App/回到村庄」的动作
- 统计：16 步 / 11 次执行 / 教练纠正率 100%（本地模型全程 wait，待回流重训）

## 接下来测试重点（按优先级，用户确认）

1. **① 下兵逻辑（最高优先）**
   - 现状：`_plan_deploy_point` 目标选择仍偏简单——第6轮两次 deploy 都选到同一金矿 `(1621,353)`；需让 deploy 目标/数量按资源价值与可达性动态轮换（接入已就绪的 `ResourcePolicy.landable_provider` 做规则层目标选择）
   - 验收：deploy 落点 100% 落在可下兵草地（绝不进红区闭合面积/UI 带/屏幕外）；同一目标不连续重复部署超过 N 次；每次都先点兵种图标再落点
2. **② 区域选择（次优先）**
   - 现状：第4轮后已修复「红区闭合面积内部全禁 + 落点吸附最近草地」，但多目标并存时选哪个资源建筑仍需真机验证
   - 验收：多资源建筑可见时优先选价值最高且可达的；落点贴目标建筑边缘的草地（截图逐帧验证）
3. **③ 结束判定（最后）**
   - 现状：第6轮 step8 过松（战斗 30 秒就返回）；需改 `decision/coach.py` 提示词 + 判定规则
   - 验收：有可见敌方资源建筑 / 摧毁率未打满 → 继续 deploy 不返回；只有红色「结束战斗」按钮或结算界面 → 才算战斗结束；误把灰色返回箭头当结束信号为 FAIL

## 最近交付（2026-09-03 下午 · 第5轮真机会话 + 红区/资源优先度修复）

**第5轮 Coach 上机会话** `dataset/coach_sessions/20260903_110901_19dd07a4/`（30/30 步，24 次执行）
- 完整闭环：村庄待机 →（正确 tap 左下角进攻按钮）→ 模式选择面板 →（正确 tap 黄色搜索对手）→ 战斗中
  → 连续 deploy（先点兵种图标再落草地，`troop_source=coach`）→ 战斗结束 → 返回 → 再进攻 → 再战斗
- 第4轮两大问题已修复验证：进攻按钮不再乱点右下角图标/日志字样；红区落点吸附到最近草地（不再甩到底部 UI 红灯带）
- 教练纠正率仍 100%（本地模型每步 wait，待第6轮继续采集后回流重训）

**代码修复（红区识别 / 可下兵判定 / 资源优先度）**
1. `tools/coach_session.py`：
   - 新增 `_no_land_mask`：红区**围成的面积内部**（洪水填充闭合区域）+ 红区像素 + 边界膨胀 + 底部 UI 带 + 屏幕外缘，全部判为不可下兵
   - `_snap_deploy_point` 改用不可下兵掩码（闭合面积内即使有草地也排除）
   - 新增 `_plan_deploy_point`：**先排除不可下兵位置，再就近资源建筑选落点**；目标中心用检测框（多实例取最近），落点=目标附近最近草地（兜底任意可下兵点）；教练落点可下兵且贴目标则信任
   - `_execute_action` deploy 分支接入落点规划，返回 `landing_plan=building_adjacent`
2. `decision/resource_policy.py`（资源优先度修正）：
   - 新增 `landable_provider`：可达性=目标中心到**最近可下兵点距离**（先排除红区/UI 再选资源），默认权重改为 (0.40 价值, 0.35 可达, 0.15 安全, 0.10 距离)
   - `breakdown` 增加 `landable_dist` 分量，便于日志/实验分析
3. `api/planner.py`：`build_plan` 新增可选 `landable_mask`，计划层同样先判可下兵再排序
4. `decision/coach.py`：教练规则补充——红区是闭合面积内部全禁；deploy 落点给目标建筑中心即可（执行器自动吸附贴建筑）；目标建筑名限 6 类枚举
5. 新增测试 `tests/test_coach_session_landable.py`（7 个）：闭合面积排除、内部草地不可下兵、贴建筑落点、检测框选目标、landable 感知排序；**全部 141 个测试通过**
6. 后端已用新代码重启（0.0.0.0:8000）

## 最近交付（2026-09-03 上午 · 首轮真机上机会话）

**首轮 ADB Coach 上机会话成功** `dataset/coach_sessions/20260903_101446_3f89db17/`
- 设备：Android 真机（USB ADB）；游戏 Clash of Clans 前台
- 链路：ADB 截屏 → fsm_v4 本地感知 → qwen3-vl-plus 教练审查 → 纠正动作真机执行 → 数据自动记录（**无需录屏**）
- 结果：8/8 步完成；8 截图+8 本地输出+8 教练输出；教练批改 7/8（纠错率 87.5%）；实际执行 4 次点击
- 流程：村庄待机 →（教练纠正点击进攻/搜索对手/进攻）→ 战斗中（开战倒计时）→ 因 max_steps=8 在第 8 步停止
- 结论：本地 fsm_v4 仍弱（符合预期），强模型教练全程在带；每步纠错即高质量蒸馏样本，
  证明「强模型指导→主动采集→回流训练」上机路径可行；下一步按计划多轮采集后重训 v6

## 最近交付（2026-09-03）

10. **视频批量标注 + v5 首轮训练**
    - 11 个视频（5 打资源 + 6 捐兵）→ 84 关键帧 / 488 框，输出 `dataset/video_batch_v5/`
    - v5 检测器：`runs/detect/fsm_v5/weights/best.pt`，mAP50 0.084 / mAP50-95 0.029，数据量不足，未达真机门槛
    - v5 状态分类器：`runs/state_classifier_v5/best.pt`，val accuracy 0.8125
    - 打资源与捐兵 v5 dry-run 均跑通：`dataset/coach_sessions/v5_regression_farm/`、`dataset/coach_sessions/v5_regression_donation/`
    - 控制台重训默认输出已改到 v5，批量标注/训练命令见 `tools/labeling/` 脚本

## 最近交付（2026-09-02）

6. **级联感知路由（本地优先 + 大模型兜底 + 样本回流）** `perception/cascade.py`（用户方案落地）
   - 常规流程（打资源/捐兵）高度相似 → 先本地 YOLO+ResNet（毫秒级、零成本、可离线）
   - 确定性路由：无建筑 / 平均置信度 < 阈值(默认0.5) / 未检出场景关键建筑 → 才调 qwen3-vl-plus
   - LLM 成功的样本（截图+提示词+输出+token 用量）自动回流 `dataset/llm_samples/`，供后续微调本地模型（教师-学生蒸馏 + 主动学习）
   - `/cascade/stats` 统计：本地/LLM 次数、LLM 触发率、触发原因、token 用量、回流样本数
   - 请求级 mode（auto/local/llm）+ 服务级 `CASCADE_MODE` 环境变量；实测：local 23 建筑 0 LLM；auto 置信度不足自动触发 LLM 补入 8 资源建筑

## 最近交付（2026-09-02）

4. **阶段 6 视觉大模型真实联调** `perception/vision_model.py`
   - qwen3-vl-plus（阿里云百炼 compatible-mode）真实 API 验证通过：`get /perceive` 内 YOLO 23 建筑 + VL 11 物体；`response_format=json_object` 原生支持，400/422 自动降级重试兜底
   - 新增 `VisionModel.chat()`：返回完整 JSON（summary/steps/coords），供 `/plan` 等高层规划使用
   - 修复：httpx 默认不再读环境代理变量（本机 socks5:// 代理会使 api 直接崩溃），设 `VISION_TRUST_ENV=1` 可恢复
5. **阶段 9 `/plan` 端点** `api/planner.py` + `api/main.py`
   - 输入 GameState → 规则层 ResourcePolicy 打分排序（自动跳过己方已满资源、防御参与安全分）
   - 可选 image_base64 + 已配置 VISION_MODEL_* → Qwen-VL 生成结构化 LLM 计划（summary/steps/coords）
   - 规则层无资源建筑时自动用 LLM 步骤补充候选（坐标标注为估算值）

## 最近交付（2026-09-02）

6. **级联感知路由（本地优先 + 大模型兜底 + 样本回流）** `perception/cascade.py`（用户方案落地）
   - 常规流程（打资源/捐兵）高度相似 → 先本地 YOLO+ResNet（毫秒级、零成本、可离线）
   - 确定性路由：无建筑 / 平均置信度 < 阈值(默认0.5) / 未检出场景关键建筑 → 才调 qwen3-vl-plus
   - LLM 成功的样本（截图+提示词+输出+token 用量）自动回流 `dataset/llm_samples/`，供后续微调本地模型（教师-学生蒸馏 + 主动学习）
   - `/cascade/stats` 统计：本地/LLM 次数、LLM 触发率、触发原因、token 用量、回流样本数
   - 请求级 mode（auto/local/llm）+ 服务级 `CASCADE_MODE` 环境变量；实测：local 23 建筑 0 LLM；auto 置信度不足自动触发 LLM 补入 8 资源建筑

## 最近交付（2026-09-02）

1. **阶段 4 资源目标优先度算法** `decision/resource_policy.py`
   - 打分公式：Target Score = 0.40×价值 + 0.30×可达性 + 0.20×安全性 + 0.10×距离
   - 分量全部归一化 [0,1]，权重/出兵点/画面尺寸/top_k 通过 `PolicyConfig` 配置
   - `ResourcePolicy` 兼容 `FarmFSM(policy=...)` 接口，自动跳过己方已满的资源类型
   - `ScoredTarget.breakdown` 输出分值拆解，便于日志与实验分析
2. **阶段 8 行为执行器与模拟环境** `executor/`
   - `action.py`：结构化 `AgentAction`（tap/deploy/wait/stop）+ JSON 序列化
   - `validator.py`：白名单 + 坐标边界校验 + 风险分级，拦截非法动作
   - `simulator.py`：确定性仿真环境（村庄待机/战斗中/战斗结束），`perceive()` 输出与 FSM 兼容的 `FarmSignals`
3. **阶段 3/7 打资源闭环** `tasks/resource.py`
   - `ResourceTask.run(simulator)` 完整闭环：感知→FSM 决策→校验→执行→前后状态验证→逐步日志
   - 结束条件：全部村庄搜刮完毕（SUCCESS）/ 资源已满（STOP）/ 动作被拦截（ERROR）/ 超步数（MAX_STEPS）

## 最近交付（2026-09-02 晚）

7. **状态转换识别测试（阶段 13 落地）** `tools/eval_state_transition.py` + `final_test_dataset/labels.json`
   - 9 张真实截图 ground truth（村庄待机×2 / 搜索中×2 / 战斗中×2 / 捐兵-消息列表×3）
   - 基线（43 类模型 + OCR）：打资源状态 6/6=100%、核心按钮召回 8/8=100%、敌方资源检出 0/6
   - `--sequence` 模式用截图序列驱动 `decision/battle_fsm.py` 验证完整转换链
8. **12 类 FSM 标签 + 数据管线** `perception/fsm_labels.py` + `tools/labeling/`
   - `llm_samples_to_yolo.py`：回流样本→YOLO（已产出 llm_finetune 2 图 9 框）
   - `ocr_buttons_to_yolo.py`：OCR 半自动按钮标注（已产出 fsm_real_buttons 9 图 29 框）
   - `finetune_fsm.py`：6 合成 + 回流 + OCR 按钮 → 12 类微调（farm 基线训练完成后运行）
9. **后续开发计划**（架构/API 配置/数据/训练/测试/里程碑/用户待办）

## 训练结果与诊断（2026-09-02 晚 · 两阶段训练已完成并自动停止）

**产物**
- farm 8 类基线：`runs/detect/farm/weights/best.pt`（40 epochs @512，CPU 训练完成）
- FSM 12 类识别器：`runs/detect/fsm/weights/best.pt`（20 epochs，6 合成资源 + LLM 回流 2 图 9 框 + OCR 按钮 9 图 29 框）

**合成验证集指标**（262 图 / 216 实例）
- all：P 0.693 / R 0.576 / mAP50 0.589 / mAP50-95 0.541
- 资源建筑 0~5：mAP50 ≈ 0.90~0.99（合成域内良好）
- 按钮类 6~11：仅「增援按钮」（样本 18）学到低置信度框（0.1~0.4），
  进攻/返回/搜索/下一个/结束战斗在验证集 P/R 全 0 —— **模型基本没学会状态转换标志**

**真实截图验收**（`tools/eval_state_transition.py`，9 张 final_test_dataset）
- 状态准确率 6/6=100%、按钮召回 8/8=100% —— **但全部来自 OCR 兜底**
- **YOLO 独立检出 0/9（按钮与资源全 0）**：本地模型真实能力未达标
- `--sequence` 链路村庄待机→搜索中可走通（修复了 sequence 的 OCR 兜底），
  战斗结束/返回环节因缺截图无法继续（待用户扩充）

**根因**
1. 按钮正样本太少且极不平衡（增援 18 vs 其他各类 1~3）；
2. **评测集与训练集重叠**：fsm_real_buttons 就来自 final_test_dataset 这 9 张图
   （训练见过其中 6 张），当前 100% 数字不能代表泛化能力，需独立评测集；
3. 合成 ↔ 真实域差异：资源建筑 0/6 与此前一致。

**下一步修法**
- 用户扩充每类按钮 ≥20 框 + 战斗结束/结算截图（独立评测集）；
- 重训时启用类别权重（按钮类 loss 加权）+ imgsz 640 + 独立验证集 —— 已记录待实现。

## 质量门槛（强制）

```bash
# 全部测试（当前 171 个）
source activate multimodal-agent && python -m pytest tests -ra

# 代码审计（合并前必须通过，当前 0 发现）
python -m security.audit
```

## 下一步建议（按优先级）

**打资源测试（本次）**
1. 跑 T1 Simulator 闭环 → T2 状态转换识别（--ocr 看 YOLO 独立 vs OCR 兜底差距）→ T3 Coach 视频 dry-run → T4 API 冒烟
2. 打资源测试前先把一段录屏放入 `dataset/video/resource/`（当前为空，T3 依赖）

**蒸馏训练（强模型指导 → 弱模型训练 → 渐进退场）**
3. 数据扩充：战斗进行中/结束/结算、捐兵详情/增援/完成，每类按钮 ≥20 框、资源 ≥50 帧，
   **独立验证集/评测集**（不再与训练集重叠）；强模型（qwen3-vl-plus + OCR）批量标注 `video_auto_label.py --llm`
4. 重训 12/17 类识别器：类别权重（按钮类加权）+ imgsz 640 + 热启动 v5（`train_yolo.py` / `finetune_fsm.py`）
5. 用 `tools/eval_state_transition.py` 验收 YOLO 独立检出（门槛：按钮 ≥60%、敌方资源 ≥50% 帧、状态准确率 100%）
6. 渐进退场：`CascadeConfig.min_avg_confidence`（0.5→0.4→0.3）+ 新增 Coach 抽样率（100%→50%→只审分歧→只审报错），
   以 `/cascade/stats` LLM 触发率曲线为准（稳态 <10%）；强模型始终保留兜底，回退一键可逆
7. 打资源端到端：值得/不值得决策命中率 ≥80% → 真机 ADB 执行（T5 / M5）
8. 捐兵 FSM（阶段 10）：消息列表/请求/增援按钮/关闭 + 默认气球兵
