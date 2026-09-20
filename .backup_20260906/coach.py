"""Coach reviewer: Qwen-VL reviews local model's perception and decisions.

Every frame, the local model outputs (state, action, coords). The coach (Qwen-VL)
reviews and either approves or corrects. The corrected action is executed, and
the (screenshot, local_output, coach_output) triple is recorded for retraining.

Coach prompt asks Qwen-VL to:
  1. Verify the detected state is correct
  2. Verify the planned action makes sense for that state
  3. If wrong, provide corrected state + action + coordinates
  4. Also identify any objects the local model missed

Output is structured JSON for reliable parsing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from perception.fsm_labels import EXTENDED_FSM_ORDER

logger = logging.getLogger(__name__)

# 发送给教练的截图最大边长（像素）。缩小图片显著降低 Qwen-VL 调用延迟，
# 教练输出统一使用归一化坐标，缩放不影响坐标精度。
COACH_MAX_SIDE = int(os.getenv("COACH_MAX_SIDE", "1280"))


COACH_SYSTEM_PROMPT = (
    "You are a strict Clash of Clans resource-farming gameplay coach. The local model has "
    "analyzed a screenshot and produced a (state, action). Your job is to look at the "
    "screenshot YOURSELF and check whether the action matches the situation. If the screen "
    "shows a button or target the current state must act on, and the local action is wait or "
    "anything else, you MUST correct it (approved=false). Approving an action that ignores a "
    "clearly visible actionable button is a serious error. Be precise with coordinates "
    "(normalized 0..1 fractions)."
)


def _scale_legacy_pixel_coords(
    values: list[float],
    from_size: tuple[int, int] | None,
    to_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Convert pixel coords from the canvas the model actually saw to the execution canvas."""
    if from_size is None or to_size is None:
        return None
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        return None
    return int(round(values[0] * tw / fw)), int(round(values[1] * th / fh))


def _scale_legacy_bbox(
    values: list[float],
    image_size: tuple[int, int] | None,
) -> list[float]:
    """Scale a legacy 1280x960 pixel bbox into the current screenshot space."""
    if image_size is None:
        return values
    w, h = image_size
    return [
        values[0] * w / 1280.0,
        values[1] * h / 960.0,
        values[2] * w / 1280.0,
        values[3] * h / 960.0,
    ]


def _normalize_point(
    coords: Any,
    image_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Accept normalized [0..1] coords; fall back to the old 1280x960 canvas."""
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    try:
        x, y = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    if image_size is None:
        return int(round(x)), int(round(y))
    w, h = image_size
    return int(round(x * w)), int(round(y * h))


def build_coach_prompt(
    image_size: tuple[int, int],
    local_state: str | None,
    local_action_kind: str | None,
    local_action_target: str | None,
    local_action_coords: tuple[int, int] | None,
    local_detections: list[dict[str, Any]] | None = None,
    module: str = "farm",
    fsm_history: list[str] | None = None,
) -> str:
    """Build the coach review prompt with local model's output."""
    w, h = image_size
    lines = [
        "Review the local model's analysis of this game screenshot.",
        "",
        f"Image size: {w}x{h} pixels",
        f"Module: {'打资源 (farm)' if module == 'farm' else '捐兵 (donation)'}",
        f"Exact object types: {', '.join(EXTENDED_FSM_ORDER)}",
    ]
    if fsm_history:
        lines.append(f"FSM history: {' -> '.join(fsm_history[-5:])}")
    lines.append("")
    lines.append("Local model output:")
    lines.append(f"  State: {local_state or '未识别'}")
    if local_action_kind:
        lines.append(f"  Action: {local_action_kind}")
        if local_action_target:
            lines.append(f"  Target: {local_action_target}")
        if local_action_coords:
            lines.append(f"  Coords: ({local_action_coords[0]}, {local_action_coords[1]})")
    if local_detections:
        lines.append("  Detections:")
        for d in local_detections[:15]:
            lines.append(f"    - {d.get('type', '?')} at {d.get('coords', '?')} (conf={d.get('confidence', 0):.2f})")
    lines.append("")
    lines.append(
        "Please review and output JSON:\n"
        "{\n"
        '  "approved": true/false,\n'
        '  "corrected_state": "<村庄待机/搜索中/战斗中/战斗结束/捐兵-消息列表/捐兵-增援确认>",\n'
        '  "corrected_action": {\n'
        '    "kind": "tap/deploy/wait/stop",\n'
        '    "target": "<目标名称>",\n'
        '    "coords": [0.0, 1.0],\n'
        '    "troop_coords": [0.0, 1.0]  // 仅 deploy 时需要：底部兵种栏中非英雄兵种图标中心\n'
        "  },\n"
        '  "missed_objects": [\n'
        '    {"type": "<类别名>", "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 0.0~1.0}\n'
        "  ],\n"
        '  "reasoning": "<简要理由>"\n'
        "}\n"
        "\n"
        "Rules (farm = 打资源):\n"
        "- 村庄待机: 进攻按钮 = 村庄画面左下角那个橙色大按钮（带‘进攻！’文字）。"
        "只有它清晰可见时才 tap 它的中心；其他位置的图标"
        "（右下角锤子/金币图标、顶部标签、日志条目里的"
        "‘进攻’字样等）都不是进攻按钮，不要点。"
        "若本地动作为 wait 或其它，纠正为 tap 这个橙色进攻按钮。\n"
        "- 模式选择面板: 点击村庄进攻按钮后会弹出左侧模式选择面板"
        "（含‘常规战’及其下方黄色‘搜索对手’按钮）。"
        "看到该面板时，唯一正确动作是 tap 黄色‘搜索对手’"
        "按钮的中心，绝不能再点村庄左下角的进攻按钮。\n"
        "- 搜索中: 若‘搜索对手’按钮可见，唯一正确动作是 tap 它；"
        "若敌方资源建筑可见且值得打，deploy；若‘下一个’按钮"
        "可见且对手不值得打，tap 下一个。\n"
        "- 战斗中: if ANY enemy resource building (金矿/圣水收集器/储金罐/圣水瓶/暗黑重油罐/暗黑重油钻井) is "
        "visible, the ONLY correct action is deploy on one of them; never approve wait. If 返回按钮 or "
        "结束战斗 is visible and no enemy resources remain, tap it.\n"
        "- 战斗结束: tap 返回按钮.\n"
        "- 目标资源建筑名必须取自这 6 类之一（金矿/圣水收集器/储金罐/圣水瓶/暗黑重油罐/暗黑重油钻井），\n  不要发明其他建筑名。\n"
        "- deploy 时除了 coords 落点还必须给出 troop_coords，即底部兵种栏中一个非英雄兵种图标\n  （普通兵种，不要选英雄头像：国王/女皇/大守护者）的中心；执行时先点 troop_coords 选中兵种，\n  再点 coords 落点放兵。\n"
        "- 红色区域在游戏里会围成一个面积，面积内部（包括红色边界本身）全部禁止下兵；\n  deploy 落点 coords 必须落在红区面积之外的玩家侧草地上。\n"
        "- deploy 落点建议给「目标资源建筑中心或其紧邻处」；执行器会自动把落点吸附到红区面积外、\n  紧贴该建筑的最近可下兵点，所以你不必精确到像素，但 coords 必须指向该资源建筑附近，\n  而不是地图另一边。\n"
        "- NEVER approve a wait when the screenshot shows a button/target the current state must act on. If the "
        "local action repeated the same wait while the screen shows an actionable target, it is stuck: always correct it.\n"
        "- The type field in missed_objects must exactly match one of the exact object types above.\n"
        "- coords and bbox MUST be normalized fractions of the current image, not pixels.\n"
        "- For coords, use [x/width, y/height]. For bbox, use [left/width, top/height, right/width, bottom/height].\n"
        "- Every normalized value must be in the 0.0~1.0 range.\n"
        "- If the local model missed important objects, list them in missed_objects.\n"
        "- Do not invent objects you cannot see."
    )
    return "\n".join(lines)


@dataclass
class CoachVerdict:
    """Result of a coach review."""

    approved: bool = False
    corrected_state: str | None = None
    corrected_action_kind: str | None = None
    corrected_action_target: str | None = None
    corrected_action_coords: tuple[int, int] | None = None
    corrected_troop_coords: tuple[int, int] | None = None
    missed_objects: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    raw_response: dict[str, Any] | None = None
    error: str | None = None


class CoachReviewer:
    """Qwen-VL coach: reviews local model output, approves or corrects."""

    def __init__(self, vision_model: Any) -> None:
        self.vision = vision_model

    def review(
        self,
        screenshot: Image.Image,
        local_state: str | None = None,
        local_action_kind: str | None = None,
        local_action_target: str | None = None,
        local_action_coords: tuple[int, int] | None = None,
        local_detections: list[dict[str, Any]] | None = None,
        module: str = "farm",
        fsm_history: list[str] | None = None,
    ) -> CoachVerdict:
        """Send screenshot + local output to Qwen-VL for review."""
        orig = screenshot.size
        viewed = screenshot
        if max(orig) > COACH_MAX_SIDE:
            ratio = COACH_MAX_SIDE / float(max(orig))
            viewed = screenshot.resize(
                (max(1, int(orig[0] * ratio)), max(1, int(orig[1] * ratio))),
                Image.LANCZOS,
            )
        # 提示词里的 Image size 用模型实际看到的画布尺寸；执行时再把坐标映射回原图
        prompt = build_coach_prompt(
            image_size=viewed.size,
            local_state=local_state,
            local_action_kind=local_action_kind,
            local_action_target=local_action_target,
            local_action_coords=local_action_coords,
            local_detections=local_detections,
            module=module,
            fsm_history=fsm_history,
        )
        try:
            parsed = self.vision.chat(viewed, prompt=prompt)
        except Exception as exc:
            logger.warning("Coach review failed: %s", exc)
            return CoachVerdict(error=str(exc))
        if not isinstance(parsed, dict):
            return CoachVerdict(error="LLM returned non-dict response")
        return self._parse_verdict(parsed, image_size=orig, viewed_size=viewed.size)

    def _parse_verdict(
        self,
        parsed: dict[str, Any],
        image_size: tuple[int, int] | None = None,
        viewed_size: tuple[int, int] | None = None,
    ) -> CoachVerdict:
        """Parse LLM JSON output into CoachVerdict.

        image_size = 原截图尺寸（执行坐标的目标空间）
        viewed_size = 模型实际看到的画布尺寸（旧版像素坐标参考空间，默认等于 image_size）
        """
        if viewed_size is None:
            viewed_size = image_size
        verdict = CoachVerdict(raw_response=parsed)
        verdict.approved = bool(parsed.get("approved", False))
        verdict.corrected_state = parsed.get("corrected_state")
        verdict.reasoning = str(parsed.get("reasoning", ""))
        action = parsed.get("corrected_action")
        if isinstance(action, dict):
            verdict.corrected_action_kind = action.get("kind")
            verdict.corrected_action_target = action.get("target")
            coords = action.get("coords")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    values = [float(coords[0]), float(coords[1])]
                except (TypeError, ValueError):
                    values = []
                if values and values[0] <= 1.0 and values[1] <= 1.0:
                    verdict.corrected_action_coords = _normalize_point(values, image_size)
                else:
                    verdict.corrected_action_coords = _scale_legacy_pixel_coords(
                        values, viewed_size, image_size
                    )
            troop_coords = action.get("troop_coords")
            if isinstance(troop_coords, (list, tuple)) and len(troop_coords) >= 2:
                try:
                    values = [float(troop_coords[0]), float(troop_coords[1])]
                except (TypeError, ValueError):
                    values = []
                if values and values[0] <= 1.0 and values[1] <= 1.0:
                    verdict.corrected_troop_coords = _normalize_point(values, image_size)
                else:
                    verdict.corrected_troop_coords = _scale_legacy_pixel_coords(
                        values, viewed_size, image_size
                    )
        missed = parsed.get("missed_objects")
        if isinstance(missed, list):
            normalized_missed: list[dict[str, Any]] = []
            for item in missed:
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    try:
                        values = [float(v) for v in bbox[:4]]
                    except (TypeError, ValueError):
                        values = []
                    if values:
                        if all(0.0 <= v <= 1.0 for v in values):
                            if image_size is not None:
                                w, h = image_size
                                values = [values[0] * w, values[1] * h, values[2] * w, values[3] * h]
                        else:
                            values = _scale_legacy_bbox(values, image_size)
                        item = {**item, "bbox": values}
                normalized_missed.append(item)
            verdict.missed_objects = normalized_missed
        return verdict
