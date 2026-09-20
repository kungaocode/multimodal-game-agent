"""下兵目标与落点的最终选择层。

`ResourcePolicy` 只负责对候选目标排序；真实执行前还需要处理三件事：

1. 教练给出的目标可能不在本地检出的候选里，要作为补充候选参与打分；
2. 不能无限连续下同一个目标，需要在有备选时按优先度轮换；
3. 最终落点必须由可下兵掩码吸附到目标建筑附近。

本模块不直接识别屏幕，只接收感知信号、候选检测和当前截图上下文。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .farm_fsm import FarmSignals
from .resource_policy import RESOURCE_NAMES, ResourcePolicy


MaskProvider = Callable[[FarmSignals, Any], object | None]
LandingPlanner = Callable[
    [str, tuple[int, int], Any, list[dict] | None, tuple[int, int] | None],
    tuple[int, int] | None,
]


@dataclass(frozen=True)
class DeployPlan:
    """一次已确认的下兵决策，供执行器与日志使用。"""

    target_name: str
    target_coords: tuple[int, int]
    landing_point: tuple[int, int]
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    repeat_count: int = 1
    reason: str = "top_scored_target"
    candidate_count: int = 0


class DeployPlanner:
    """按资源优先度选择目标，并维护连续重复目标的状态。

    ``requested_name``/``requested_coords`` 通常来自教练纠正；若目标有效，
    会优先尊重教练。当同一目标连续部署达到 ``max_consecutive_per_target``
    且存在其他候选时，改选综合分次高的目标。若只有一个候选，则继续打它，
    避免有目标可见时错误等待。
    """

    def __init__(
        self,
        policy: ResourcePolicy | None = None,
        mask_provider: MaskProvider | None = None,
        landing_planner: LandingPlanner | None = None,
        max_consecutive_per_target: int = 2,
        match_radius: float = 48.0,
    ) -> None:
        self.policy = policy or ResourcePolicy()
        self.mask_provider = mask_provider
        self.landing_planner = landing_planner
        self.max_consecutive_per_target = max(1, int(max_consecutive_per_target))
        self.match_radius = float(match_radius)
        self.last_target: tuple[str, tuple[int, int]] | None = None
        self.last_repeat_count = 0

    def plan(
        self,
        signals: FarmSignals,
        screenshot: Any = None,
        detections: list[dict] | None = None,
        requested_name: str | None = None,
        requested_coords: tuple[int, int] | list[int] | None = None,
    ) -> DeployPlan | None:
        """选择最终 deploy 目标与落点；没有可打目标时返回 None。"""
        effective_signals = self._with_requested_candidate(
            signals,
            requested_name,
            requested_coords,
            screenshot,
        )
        scored = self._scored(effective_signals, screenshot)
        if not scored:
            return None

        selected = self._select(scored, requested_name, requested_coords)
        if selected is None:
            return None

        previous = self.last_target
        previous_repeat_count = self.last_repeat_count
        same = previous is not None and self._same_target(
            selected.name, selected.coords, previous[0], previous[1]
        )
        repeat_count = self.last_repeat_count + 1 if same else 1
        self.last_target = (selected.name, selected.coords)
        self.last_repeat_count = repeat_count

        if same and repeat_count >= self.max_consecutive_per_target:
            reason = "repeat_limit_only_candidate" if len(scored) == 1 else "repeat_limit_reached"
        elif previous is not None and not same and previous_repeat_count >= self.max_consecutive_per_target:
            reason = "rotated_after_repeat_limit"
        elif requested_name == selected.name:
            reason = "requested_target_honored"
        elif len(scored) == 1:
            reason = "only_candidate"
        else:
            reason = "top_scored_target"

        target_coords = (int(selected.coords[0]), int(selected.coords[1]))
        landing = self._plan_landing(
            selected.name,
            target_coords,
            screenshot,
            detections,
            requested_coords,
        )
        return DeployPlan(
            target_name=selected.name,
            target_coords=target_coords,
            landing_point=landing,
            score=selected.score,
            breakdown=dict(selected.breakdown),
            repeat_count=repeat_count,
            reason=reason,
            candidate_count=len(scored),
        )

    def reset(self) -> None:
        """清空连续目标状态，通常用于进入下一场战斗。"""
        self.last_target = None
        self.last_repeat_count = 0

    def _with_requested_candidate(
        self,
        signals: FarmSignals,
        requested_name: str | None,
        requested_coords: tuple[int, int] | list[int] | None,
        screenshot: Any,
    ) -> FarmSignals:
        """教练目标不在本地检出中时，作为补充候选交给 ResourcePolicy 打分。"""
        if requested_name not in RESOURCE_NAMES or not requested_coords:
            return signals
        coords = self._clamp(requested_coords, screenshot)
        exists = any(
            self._same_target(requested_name, coords, name, candidate_coords)
            for name, candidate_coords in signals.enemy_resources
        )
        if exists:
            return signals
        resources = [*signals.enemy_resources, (requested_name, coords)]
        return replace(signals, enemy_resources=resources)

    def _scored(self, signals: FarmSignals, screenshot: Any) -> list:
        """按当前画布尺寸和可下兵掩码打分。"""
        policy = self.policy
        if screenshot is not None:
            try:
                image_size = tuple(screenshot.size)
            except AttributeError:
                image_size = None
            if image_size and image_size != tuple(policy.config.image_size):
                policy = ResourcePolicy(
                    config=replace(policy.config, image_size=image_size),
                    defense_provider=policy.defense_provider,
                    landable_provider=policy.landable_provider,
                )
        if self.mask_provider is not None:
            policy = ResourcePolicy(
                config=policy.config,
                defense_provider=policy.defense_provider,
                landable_provider=lambda _signals: self.mask_provider(_signals, screenshot),
            )
        return policy.scored(signals)

    def _select(
        self,
        scored: list,
        requested_name: str | None,
        requested_coords: tuple[int, int] | list[int] | None,
    ):
        requested = None
        if requested_name and requested_coords:
            requested = next(
                (
                    target
                    for target in scored
                    if target.name == requested_name
                    and self._same_target(
                        target.name,
                        target.coords,
                        requested_name,
                        tuple(requested_coords),
                    )
                ),
                None,
            )
            if requested is not None and not self._repeat_limit_reached(requested):
                return requested

        eligible = [target for target in scored if not self._repeat_limit_reached(target)]
        if eligible:
            return eligible[0]
        return scored[0]

    def _repeat_limit_reached(self, target) -> bool:
        previous = self.last_target
        return (
            previous is not None
            and self._same_target(target.name, target.coords, previous[0], previous[1])
            and self.last_repeat_count >= self.max_consecutive_per_target
        )

    def _plan_landing(
        self,
        target_name: str,
        target_coords: tuple[int, int],
        screenshot: Any,
        detections: list[dict] | None,
        requested_coords: tuple[int, int] | list[int] | None,
    ) -> tuple[int, int]:
        if self.landing_planner is None:
            return target_coords
        landing = self.landing_planner(
            target_name, target_coords, screenshot, detections, requested_coords
        )
        if landing is None:
            return target_coords
        return int(landing[0]), int(landing[1])

    def _same_target(
        self,
        name: str,
        coords: tuple[int, int] | list[int],
        other_name: str,
        other_coords: tuple[int, int] | list[int],
    ) -> bool:
        return name == other_name and math.dist(tuple(coords), tuple(other_coords)) <= self.match_radius

    def _clamp(
        self,
        coords: tuple[int, int] | list[int],
        screenshot: Any,
    ) -> tuple[int, int]:
        x, y = int(coords[0]), int(coords[1])
        if screenshot is None:
            return x, y
        try:
            w, h = screenshot.size
        except AttributeError:
            return x, y
        return min(max(x, 0), w - 1), min(max(y, 0), h - 1)
