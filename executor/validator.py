"""动作校验器（阶段 8）：安全层的第一道闸门。

所有动作在执行前必须通过 ActionValidator：
- 动作种类必须在允许集合内；
- tap / deploy 必须带坐标，且坐标在画面范围内；
- 目标必须在白名单内；
- 输出风险等级供上层策略参考（deploy > tap > wait/stop）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .action import ALLOWED_ACTION_KINDS, FARM_ALLOWED_TARGETS

# 每种动作的基准风险等级
KIND_RISK: dict[str, str] = {
    "tap": "low",
    "deploy": "medium",
    "wait": "low",
    "stop": "low",
}


@dataclass(frozen=True)
class ValidationResult:
    allowed: bool
    reason: str = ""
    risk_level: str = "low"


class ActionValidator:
    """白名单 + 边界检查的动作校验器。"""

    def __init__(
        self,
        image_size: tuple[int, int] = (1280, 720),
        allowed_kinds: Sequence[str] | None = None,
        allowed_targets: Sequence[str] | None = None,
        margin: int = 5,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.allowed_kinds = tuple(allowed_kinds) if allowed_kinds is not None else ALLOWED_ACTION_KINDS
        self.allowed_targets = (
            tuple(allowed_targets) if allowed_targets is not None else FARM_ALLOWED_TARGETS
        )
        self.margin = margin

    def validate(self, action: Any) -> ValidationResult:
        kind = getattr(action, "kind", None)
        target = getattr(action, "target", None)
        coords = getattr(action, "coords", None)
        troop_coords = getattr(action, "troop_coords", None)

        if kind not in self.allowed_kinds:
            return ValidationResult(False, f"不允许的动作种类: {kind}", "high")
        if kind in ("tap", "deploy"):
            if coords is None:
                return ValidationResult(False, f"{kind} 动作缺少坐标", KIND_RISK.get(kind, "medium"))
            if not self._in_bounds(coords):
                return ValidationResult(False, f"坐标越界: {coords}（画面 {self.image_size}）", "medium")
            # troop_coords 只在 deploy（先点兵种栏再落点）时有意义。
            # 教练在 tap/wait 等非 deploy 动作上会照提示词填默认 [0,0]，
            # 若在 tap 上强查兵种坐标会把合法点击误拦成 wait。
            if kind == "deploy" and troop_coords is not None and not self._in_bounds(troop_coords):
                return ValidationResult(
                    False,
                    f"兵种坐标越界: {troop_coords}（画面 {self.image_size}）",
                    "medium",
                )
            if target is None or target not in self.allowed_targets:
                return ValidationResult(False, f"目标不在白名单: {target}", "medium")
        return ValidationResult(True, "ok", KIND_RISK.get(kind, "low"))

    def _in_bounds(self, coords: tuple[int, int]) -> bool:
        x, y = coords
        w, h = self.image_size
        m = self.margin
        return m <= x < w - m and m <= y < h - m
