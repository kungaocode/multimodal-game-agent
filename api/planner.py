"""打资源任务规划器（阶段 4/6/9 集成）。

把「感知结果 GameState」转成可执行的打资源计划：

1. policy（规则层，确定性、可离线测试）
   - 从 GameState 提取敌方资源建筑与己方防御建筑
   - 用 ResourcePolicy 按「0.40 价值 + 0.30 可达 + 0.20 安全 + 0.10 距离」打分排序
   - 自动跳过己方已满的那类资源建筑

2. llm_plan（可选，阶段 6 深化）
   - 把候选目标 + 己方资源情况整理成上下文，交给 Qwen-VL 生成结构化作战计划
   - 需要提供原始截图（image_base64）且已配置 VISION_MODEL_* 才启用
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from decision.farm_fsm import FarmSignals
from decision.resource_policy import (
    RESOURCE_TYPE_BY_BUILDING,
    PolicyConfig,
    ResourcePolicy,
    full_resource_names,
)
from state.game_state import GameState, ResourceStatus

# 常见防御建筑（来自 43 类检测数据集），用于安全性分量
DEFENSE_TYPES: frozenset[str] = frozenset(
    {
        "X连弩（十字连弩）",
        "加农炮",
        "地狱之塔",
        "地狱火炮",
        "复仇之塔",
        "复合机械塔",
        "多人箭塔",
        "天鹰火炮",
        "巨型地狱之塔",
        "巨型特斯拉电磁塔",
        "巨石碑",
        "投石炮",
        "法术塔",
        "火焰喷射器",
        "炸弹塔",
        "特斯拉电磁塔",
        "空气炮",
        "箭塔",
        "超级法师塔",
        "跳弹加农炮",
        "迫击炮",
        "防空火箭",
    }
)

# 己方资源别名 → 标准资源类型（gold / elixir / dark）
_RESOURCE_ALIASES: Mapping[str, str] = {
    "gold": "gold",
    "金": "gold",
    "金矿": "gold",
    "elixir": "elixir",
    "圣水": "elixir",
    "圣水收集器": "elixir",
    "dark": "dark",
    "dark_elixir": "dark",
    "黑油": "dark",
    "暗黑重油": "dark",
}

DEFAULT_IMAGE_SIZE: tuple[int, int] = (1280, 720)


def _normalize_full(full_resources: Sequence[str]) -> set[str]:
    """把用户传入的「已满资源」归一化为标准类型集合。"""
    out: set[str] = set()
    for item in full_resources:
        key = item.strip().lower()
        if key in _RESOURCE_ALIASES:
            out.add(_RESOURCE_ALIASES[key])
    return out


def signals_from_state(
    state: GameState,
    full_resources: Sequence[str] = (),
) -> tuple[FarmSignals, tuple[tuple[int, int], ...]]:
    """GameState → FarmSignals + 防御坐标。

    - 敌方资源建筑：GameState.buildings 中属于资源类型的，位置 (0,0) 视为未知并剔除
    - 防御坐标：GameState.buildings 中属于 DEFENSE_TYPES 的，供安全性分量使用
    """
    full = _normalize_full(full_resources)
    status = ResourceStatus(
        gold_full="gold" in full,
        elixir_full="elixir" in full,
        dark_full="dark" in full,
        detected=len(full),
    )
    enemy_resources = [
        (b.type, b.position)
        for b in state.buildings
        if b.type in RESOURCE_TYPE_BY_BUILDING and b.position != (0, 0)
    ]
    defenses = tuple(
        b.position for b in state.buildings if b.type in DEFENSE_TYPES and b.position != (0, 0)
    )
    signals = FarmSignals(enemy_resources=enemy_resources, resources=status)
    return signals, defenses


def build_plan(
    state: GameState,
    full_resources: Sequence[str] = (),
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    top_k: int | None = None,
    landable_mask: object | None = None,
) -> dict[str, Any]:
    """生成打资源计划（规则层）。返回可 JSON 序列化的 dict。

    landable_mask：可选 2D bool 掩码（True=可下兵，与 image_size 同画布）。
    传入后 ResourcePolicy 的「可达性」= 目标中心到最近可下兵点的距离，
    即先排除红区面积/UI 等不可下兵位置，再按资源价值选目标。
    """
    signals, defenses = signals_from_state(state, full_resources)
    config = PolicyConfig(image_size=image_size, top_k=top_k, defenses=defenses)
    if landable_mask is not None:
        mask = landable_mask

        def _landable_provider(_signals: FarmSignals) -> object:
            return mask

        policy = ResourcePolicy(config, landable_provider=_landable_provider)
    else:
        policy = ResourcePolicy(config)
    scored: list[Any] = policy.scored(signals)

    candidates = [
        {
            "name": t.name,
            "coords": list(t.coords),
            "score": t.score,
            "breakdown": dict(t.breakdown),
            "landable_dist": t.breakdown.get("landable_dist"),
        }
        for t in scored
    ]
    skipped = sorted(full_resource_names(signals.resources))
    return {
        "policy": {
            "candidates": candidates,
            "skipped_resource_buildings": skipped,
            "defenses_used": bool(defenses),
            "landable_used": landable_mask is not None,
            "image_size": list(image_size),
        },
        "warnings": [],
    }


def llm_plan_prompt(candidate_summary: dict[str, Any]) -> str:
    """把规则层候选整理成喂给 Qwen-VL 的结构化上下文。"""
    lines = ["你是部落冲突打资源指挥官，请根据已识别的敌方资源建筑制定下一步作战计划。", ""]
    lines.append("候选资源建筑（按优先度降序）：")
    for c in candidate_summary.get("candidates", []):
        lines.append(
            f"- {c['name']} 位置{c['coords']} 综合分{c['score']:.3f} "
            f"(价值{c['breakdown'].get('value', 0):.2f} 可达{c['breakdown'].get('accessibility', 0):.2f} "
            f"安全{c['breakdown'].get('safety', 0):.2f} 距离{c['breakdown'].get('distance', 0):.2f})"
        )
    if candidate_summary.get("skipped_resource_buildings"):
        lines.append("己方已满、本次不打的资源建筑："
                     + "、".join(candidate_summary["skipped_resource_buildings"]))
    lines.append("")
    lines.append(
        "请用 JSON 输出：{'summary': '一句话作战思路', "
        "'steps': [{'action': 'deploy|tap|wait|stop', 'target': '建筑名', "
        "'coords': [x, y], 'reason': '理由'}]}，只选得分最高的 1~3 个目标。"
    )
    return "\n".join(lines)


def candidates_from_llm_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把 LLM 计划的 deploy 步骤转成候选格式（来源标记 llm，坐标为估算值）。

    规则层（YOLO）没有检出资源建筑时，用大模型的步骤作为补充候选。
    """
    out: list[dict[str, Any]] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if str(s.get("action", "")).lower() != "deploy":
            continue
        coords = s.get("coords") or []
        try:
            cx, cy = int(coords[0]), int(coords[1])
        except (TypeError, ValueError, IndexError):
            cx, cy = 0, 0
        out.append(
            {
                "name": str(s.get("target", "未知")),
                "coords": [cx, cy],
                "score": 0.0,  # LLM 估算不参与规则打分
                "breakdown": {},
                "source": "llm",
                "reason": str(s.get("reason", "")),
            }
        )
    return out
