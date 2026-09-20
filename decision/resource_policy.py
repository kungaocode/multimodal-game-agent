"""打资源目标优先度算法（阶段 4）。

按「先规则、后 LLM」原则，先用确定性规则给敌方资源建筑打分，再排序选目标：

    Target Score = w_v × 资源价值 + w_a × 可达性 + w_s × 安全性 + w_d × 距离

所有分量归一化到 [0, 1]，权重默认 (0.40, 0.35, 0.15, 0.10)，可在 PolicyConfig 中覆盖。

「可达性」= 目标中心到最近可下兵点的距离（需要传入 landable_provider 提供
「可下兵掩码」，即先强行排除不可下兵位置再选资源建筑）；未提供时退化为按
画面边缘距离的近似。落点越贴建筑、越容易下兵得分越高。

ResourcePolicy 实现与 decision.farm_fsm.FarmFSM 兼容的 policy 调用接口：
    policy(signals: FarmSignals) -> list[(建筑名, 中心点)]
并且在打分前自动跳过己方已满的那类资源建筑（如圣水满了就不打圣水收集器/圣水瓶）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from state.game_state import ResourceStatus

from .farm_fsm import FarmSignals, _BUILDING_RESOURCE

# 公开别名：建筑名 → 己方资源类型（gold / elixir / dark），用于「满了就跳过」
RESOURCE_TYPE_BY_BUILDING: Mapping[str, str] = _BUILDING_RESOURCE

# 打资源循环关心的 6 类资源建筑（与 tools/labeling/build_farm_dataset.FARM_ORDER 前 6 类一致）
RESOURCE_NAMES: tuple[str, ...] = (
    "圣水收集器",
    "金矿",
    "圣水瓶",
    "储金罐",
    "暗黑重油罐",
    "暗黑重油钻井",
)

# 基础资源价值（0~1）：仓库 > 矿井；暗黑重油最稀缺
BASE_RESOURCE_VALUE: Mapping[str, float] = {
    "圣水收集器": 0.55,
    "金矿": 0.55,
    "圣水瓶": 0.80,
    "储金罐": 0.80,
    "暗黑重油钻井": 0.70,
    "暗黑重油罐": 1.0,
}
_DEFAULT_RESOURCE_VALUE = 0.40

# 默认权重：价值 0.40 / 可达性 0.35 / 安全性 0.15 / 距离 0.10
DEFAULT_WEIGHTS: tuple[float, float, float, float] = (0.40, 0.35, 0.15, 0.10)

# 没有防御信息时的安全性中性分
_NEUTRAL_SAFETY = 0.5
# 距最近防御达到该像素距离即视为完全安全
_SAFE_DISTANCE_PX = 200.0


@dataclass(frozen=True)
class PolicyConfig:
    """打分策略配置：权重、画面尺寸、出兵点、是否截断 top_k、静态防御坐标。"""

    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS
    image_size: tuple[int, int] = (1280, 720)
    deploy_point: tuple[int, int] | None = None  # 默认画面底部中央
    top_k: int | None = None  # 只返回前 N 个候选；None 表示全部
    defenses: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ScoredTarget:
    """带打分拆解的目标，方便日志与实验分析。"""

    name: str
    coords: tuple[int, int]
    score: float
    breakdown: dict[str, float] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.breakdown.items())
        return f"ScoredTarget({self.name}, {self.coords}, score={self.score:.3f}, {parts})"


def resource_value(name: str) -> float:
    """资源价值分量（0~1）。"""
    return BASE_RESOURCE_VALUE.get(name, _DEFAULT_RESOURCE_VALUE)


def accessibility_score(
    coords: tuple[int, int],
    image_size: tuple[int, int],
    landable_dist: float | None = None,
) -> float:
    """可达性（0~1）：目标中心到最近可下兵点越近得分越高。

    landable_dist 为目标中心到最近可下兵点的像素距离（先排除红区面积/UI 后
    的落点可达性）。<=120px 视为完全可达；未提供时退化为按画面边缘距离近似。
    """
    if landable_dist is not None:
        reachable_px = 120.0
        return min(1.0, max(0.0, 1.0 - landable_dist / reachable_px))
    w, h = image_size
    x, y = coords
    edge = min(x, y, w - x, h - y)
    half_min = 0.5 * min(w, h)
    if half_min <= 0:
        return 0.0
    return 1.0 - min(1.0, edge / half_min)


def safety_score(
    coords: tuple[int, int],
    defenses: Sequence[tuple[int, int]] = (),
) -> float:
    """安全性（0~1）：离最近防御越远越安全；没有防御信息时给中性分 0.5。"""
    if not defenses:
        return _NEUTRAL_SAFETY
    nearest = min(math.dist(coords, d) for d in defenses)
    return min(1.0, nearest / _SAFE_DISTANCE_PX)


def distance_score(
    coords: tuple[int, int],
    deploy_point: tuple[int, int],
    image_size: tuple[int, int],
) -> float:
    """距离分量（0~1）：离出兵点越近得分越高，按画面对角线归一化。"""
    dist = math.dist(coords, deploy_point)
    norm = math.hypot(*image_size)
    if norm <= 0:
        return 0.0
    return min(1.0, max(0.0, (norm - dist) / norm))


def _default_deploy_point(config: PolicyConfig) -> tuple[int, int]:
    if config.deploy_point is not None:
        return config.deploy_point
    w, h = config.image_size
    return (w // 2, h - 10)


def score_target(
    name: str,
    coords: tuple[int, int],
    config: PolicyConfig,
    defenses: Sequence[tuple[int, int]] = (),
    deploy_point: tuple[int, int] | None = None,
    accessibility_dist: float | None = None,
) -> ScoredTarget:
    """对单个资源建筑计算综合得分并给出分量拆解。

    accessibility_dist：目标中心到最近可下兵点的像素距离（None 表示未知，
    使用画面边缘近似）。这是「先排除不可下兵位置、再选资源建筑」的关键输入。
    """
    dp = deploy_point or _default_deploy_point(config)
    w_v, w_a, w_s, w_d = config.weights
    value = resource_value(name)
    acc = accessibility_score(coords, config.image_size, accessibility_dist)
    safety = safety_score(coords, defenses)
    dist = distance_score(coords, dp, config.image_size)
    score = w_v * value + w_a * acc + w_s * safety + w_d * dist
    breakdown = {"value": value, "accessibility": acc, "safety": safety, "distance": dist}
    if accessibility_dist is not None:
        breakdown["landable_dist"] = accessibility_dist
    return ScoredTarget(name, coords, round(score, 6), breakdown)


def full_resource_names(status: ResourceStatus) -> set[str]:
    """己方已满的资源类型对应的敌方建筑名集合（这些建筑不打）。"""
    full: set[str] = set()
    if status.gold_full:
        full.update(n for n, rt in RESOURCE_TYPE_BY_BUILDING.items() if rt == "gold")
    if status.elixir_full:
        full.update(n for n, rt in RESOURCE_TYPE_BY_BUILDING.items() if rt == "elixir")
    if status.dark_full:
        full.update(n for n, rt in RESOURCE_TYPE_BY_BUILDING.items() if rt == "dark")
    return full


class ResourcePolicy:
    """可打分排序的打资源策略；兼容 FarmFSM(policy=...) 接口。

    defense_provider 允许动态提供当前敌方防御坐标（例如来自 GameState 或模拟环境），
    未提供时使用 config.defenses；两者都为空则安全性取中性分。

    landable_provider 允许动态提供「可下兵掩码」（2D bool，True=可下兵，
    需与 config.image_size 同画布）。提供后可达性=目标中心到最近可下兵点的
    距离——即先强行排除红区面积/UI 等不可下兵位置，再按资源价值选目标。
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        defense_provider: Callable[[FarmSignals], Sequence[tuple[int, int]]] | None = None,
        landable_provider: Callable[[FarmSignals], object | None] | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.defense_provider = defense_provider
        self.landable_provider = landable_provider

    def _defenses(self, signals: FarmSignals) -> tuple[tuple[int, int], ...]:
        if self.defense_provider is not None:
            return tuple(self.defense_provider(signals))
        return self.config.defenses

    def _landable_distances(
        self, signals: FarmSignals
    ) -> dict[tuple[int, int], float] | None:
        """目标中心 -> 最近可下兵点距离（px）；无掩码或导入失败时返回 None。"""
        if self.landable_provider is None:
            return None
        try:
            import numpy as np
            from scipy import ndimage

            landable = np.asarray(self.landable_provider(signals), dtype=bool)
        except Exception:
            return None
        if landable is None or not landable.any():
            return None
        h, w = landable.shape
        dist = ndimage.distance_transform_edt(~landable)
        out: dict[tuple[int, int], float] = {}
        for name, coords in signals.enemy_resources:
            x = min(max(int(coords[0]), 0), w - 1)
            y = min(max(int(coords[1]), 0), h - 1)
            out[name, coords] = float(dist[y, x])
        return out

    def scored(self, signals: FarmSignals) -> list[ScoredTarget]:
        """对敌方资源建筑打分并按综合分降序返回。"""
        skip = full_resource_names(signals.resources)
        defenses = self._defenses(signals)
        deploy_point = _default_deploy_point(self.config)
        land = self._landable_distances(signals)
        scored = [
            score_target(
                name,
                coords,
                self.config,
                defenses=defenses,
                deploy_point=deploy_point,
                accessibility_dist=land[(name, coords)] if land else None,
            )
            for name, coords in signals.enemy_resources
            if name not in skip
        ]
        scored.sort(key=lambda t: t.score, reverse=True)
        if self.config.top_k is not None:
            scored = scored[: self.config.top_k]
        return scored

    def __call__(self, signals: FarmSignals) -> list[tuple[str, tuple[int, int]]]:
        return [(t.name, t.coords) for t in self.scored(signals)]
