"""资源建筑可获取性与兵力需求估算层。

这里把「能不能轻松抢到」拆成可测试的确定性模型：

    可获取 = 有可下兵落点
         且 落点距建筑足够近（落点不同 → 步行损耗不同）
         且 城墙穿越代价可接受
         且 防御火力风险可接受
         且 估算所需兵力 <= 当前兵力预算

输出不仅给排序用，也作为战斗记忆（BattleMemory）估算剩余资源的依据，
最终决定「已可获取资源是否抢完、该不该撤退」。

本模块不识别屏幕；掩码/城墙坐标由调用方（教练会话、模拟器、API）提供，
保证离线、确定、可单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# 各类型建筑的默认可抢容量估算；有检测等级或模拟器精确值时优先使用精确值。
DEFAULT_CAPACITY_BY_TYPE: Mapping[str, int] = {
    "金矿": 1500,
    "圣水收集器": 1500,
    "储金罐": 4000,
    "圣水瓶": 4000,
    "暗黑重油钻井": 2000,
    "暗黑重油罐": 2000,
}


@dataclass(frozen=True)
class LootConfig:
    """估算/下兵模型参数。"""

    loot_per_troop: int = 1000
    troop_budget: int = 30
    optimal_landing_px: float = 120.0   # 落点距建筑此距离内视为零步行损耗
    max_landing_px: float = 260.0       # 超过该距离默认无法有效获取
    landing_radius: float = 320.0       # 找落点时的最大搜索半径
    wall_breaker_cost: int = 3          # 每层城墙的破墙兵力消耗
    wall_cost_limit: int = 6
    safe_defense_px: float = 200.0
    max_risk: float = 0.80
    min_expected_loot: int = 400        # 剩余太少视为不值得再派一队兵
    min_efficiency: float = 0.35
    walk_px_per_extra_troop: float = 180.0
    capacity_by_type: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_CAPACITY_BY_TYPE))


@dataclass(frozen=True)
class LootEstimate:
    """一个资源建筑的获取估算结果。"""

    name: str
    coords: tuple[int, int]
    accessible: bool
    reason: str
    landing_point: tuple[int, int] | None
    landing_dist_px: float | None
    wall_cost: int
    walk_extra: int
    troops_needed: int
    loot_per_troop: int
    expected_loot: int
    risk_score: float
    loot_factor: float
    efficiently_lootable: bool
    breakdown: dict[str, float] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accessible": self.accessible,
            "reason": self.reason,
            "landing_point": list(self.landing_point) if self.landing_point else None,
            "landing_dist_px": self.landing_dist_px,
            "wall_cost": self.wall_cost,
            "walk_extra": self.walk_extra,
            "troops_needed": self.troops_needed,
            "loot_per_troop": self.loot_per_troop,
            "expected_loot": self.expected_loot,
            "risk_score": self.risk_score,
            "loot_factor": self.loot_factor,
            "efficiently_lootable": self.efficiently_lootable,
            "breakdown": self.breakdown,
        }


def _nearest_allowed(
    coords: tuple[int, int],
    allowed: Any,
    max_radius: float,
) -> tuple[tuple[int, int], float] | None:
    """目标外最近可下兵点（像素）；allowed 与渲染尺寸同为 (h, w) 的 bool 矩阵。"""
    if allowed is None or not getattr(allowed, "any", lambda: False)():
        return None
    try:
        from scipy import ndimage
    except ImportError:
        return None
    h, w = allowed.shape
    x = min(max(int(coords[0]), 0), w - 1)
    y = min(max(int(coords[1]), 0), h - 1)
    dist, idx = ndimage.distance_transform_edt(~allowed, return_indices=True)
    d = float(dist[y, x])
    if d == 0:
        return (x, y), 0.0
    if d <= 0 or d > max_radius:
        return None
    return (int(idx[1, y, x]), int(idx[0, y, x])), d


def _line_pixels(
    start: tuple[int, int],
    end: tuple[int, int],
    step: float = 2.0,
):
    """沿两点直线按像素步长采样，用于检测城墙是否阻挡路线。"""
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist <= 0:
        yield x0, y0
        return
    steps = max(1, int(math.ceil(dist / step)))
    for i in range(steps + 1):
        t = i / steps
        yield int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """判断线段 ab 与 cd 是否相交（含端点触碰与共线重叠）。"""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    if (
        max(ax, bx) < min(cx, dx)
        or max(cx, dx) < min(ax, bx)
        or max(ay, by) < min(cy, dy)
        or max(cy, dy) < min(ay, by)
    ):
        return False

    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r) -> bool:
        return (
            min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    o1 = orient(c, d, a)
    o2 = orient(c, d, b)
    o3 = orient(a, b, c)
    o4 = orient(a, b, d)
    if o1 == 0 and on_segment(c, d, a):
        return True
    if o2 == 0 and on_segment(c, d, b):
        return True
    if o3 == 0 and on_segment(a, b, c):
        return True
    if o4 == 0 and on_segment(a, b, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def wall_crossing_cost(
    landing: tuple[int, int],
    target: tuple[int, int],
    wall_mask: Any = None,
    wall_segments: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
    wall_breaker_cost: int = 3,
) -> int:
    """估算从落点到目标需要穿越的城墙层数，换算成额外兵力。

    支持两种描述：像素掩码（bool 矩阵）或线框（[(起点, 终点), ...]）。
    穿过一层算一层破墙兵力，>= 2 层仍按两层封顶，避免过拟合。
    """
    layers = 0
    if wall_mask is not None:
        crossing_frames: list[bool | None] = []
        in_wall = False
        for x, y in _line_pixels(landing, target):
            px = wall_mask[y, x] if 0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1] else False
            if bool(px) and not in_wall:
                layers += 1
            in_wall = bool(px)
    elif wall_segments:
        for start, end in wall_segments:
            if _segments_intersect(landing, target, tuple(start), tuple(end)):
                layers += 1
    if layers <= 0:
        return 0
    return wall_breaker_cost * min(2, layers)


def defense_risk(
    coords: tuple[int, int],
    defenses: Sequence[tuple[int, int]] = (),
    safe_px: float = 200.0,
) -> float:
    """防御风险 0~1：离最近防御越近越高；没有防御信息时给低风险 0.1。"""
    if not defenses:
        return 0.1
    nearest = min(math.dist(coords, d) for d in defenses)
    if safe_px <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - nearest / safe_px))


def _capacity_from_detections(
    name: str,
    coords: tuple[int, int],
    detections: Sequence[dict] | None,
    match_radius: float = 24.0,
) -> tuple[int | None, int | None]:
    """从检测信息里提取容量/已抢量；检测项是 dict 且含可选字段。"""
    if not detections:
        return None, None
    for det in detections:
        det_coords = det.get("coords")
        if not det_coords or det.get("type") != name:
            continue
        if math.dist(tuple(det_coords), coords) > match_radius:
            continue
        capacity = det.get("capacity")
        looted = det.get("looted")
        return (
            int(capacity) if capacity is not None else None,
            int(looted) if looted is not None else None,
        )
    return None, None


class LootEstimator:
    """纯确定性估算器：输入掩码/防御/容量上下文，输出 LootEstimate。"""

    def __init__(self, config: LootConfig | None = None) -> None:
        self.config = config or LootConfig()

    def estimate(
        self,
        name: str,
        coords: tuple[int, int],
        *,
        screenshot: Any = None,
        detections: Sequence[dict] | None = None,
        no_land_mask: Any = None,
        green_mask: Any = None,
        wall_mask: Any = None,
        wall_segments: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
        defenses: Sequence[tuple[int, int]] = (),
        capacity: int | None = None,
        looted: int = 0,
        level: int | None = None,
        troop_budget: int | None = None,
        landing_requested: tuple[int, int] | None = None,
    ) -> LootEstimate:
        cfg = self.config
        det_capacity, det_looted = _capacity_from_detections(name, coords, detections)
        if capacity is None:
            capacity = det_capacity
        if capacity is None:
            base_capacity = int(cfg.capacity_by_type.get(name, 1000))
            if level is not None:
                base_capacity = int(base_capacity * (0.8 + 0.15 * max(1, level)))
            capacity = base_capacity
        if det_looted is not None:
            looted = det_looted
        budget = int(troop_budget if troop_budget is not None else cfg.troop_budget)

        remaining = max(0, int(capacity) - int(looted))
        no_land = no_land_mask
        if no_land is None and screenshot is not None:
            try:
                from tools.coach_masks import no_land_mask as _no_land
                no_land = _no_land(screenshot)
            except Exception:
                no_land = None
        if green_mask is None and screenshot is not None:
            try:
                from tools.coach_masks import green_mask as _green
                green_mask = _green(screenshot)
            except Exception:
                green_mask = None

        allowed: Any = None
        if no_land is not None:
            land_any = ~no_land
            if green_mask is not None:
                land_any = green_mask & land_any
            allowed = land_any

        landing: tuple[int, int] | None = None
        if allowed is not None:
            land_info = _nearest_allowed(coords, allowed, cfg.landing_radius)
            if land_info is not None:
                landing = land_info[0]
        else:
            landing = landing_requested or coords
        if landing_requested is not None and allowed is not None:
            h, w = allowed.shape
            rx = min(max(int(landing_requested[0]), 0), w - 1)
            ry = min(max(int(landing_requested[1]), 0), h - 1)
            if bool(allowed[ry, rx]) and math.dist((rx, ry), coords) <= cfg.max_landing_px:
                landing = landing_requested

        if landing is None:
            return LootEstimate(
                name, coords, False, "无可下兵落点（红区/UI 包围）",
                None, None, 0, 0, 0, cfg.loot_per_troop, 0, 1.0, 0.0, False,
                {"landing": 0.0, "wall": 0.0, "troops": 0.0, "defense": 0.0},
            )

        dist = math.dist(landing, coords)
        wall_cost = wall_crossing_cost(
            landing, coords, wall_mask=wall_mask, wall_segments=wall_segments,
            wall_breaker_cost=cfg.wall_breaker_cost,
        )
        walk_extra = 0
        if dist > cfg.optimal_landing_px and cfg.walk_px_per_extra_troop > 0:
            walk_extra = int(math.ceil((dist - cfg.optimal_landing_px) / cfg.walk_px_per_extra_troop))
        base_needed = 0 if remaining <= 0 else int(math.ceil(remaining / cfg.loot_per_troop))
        troops_needed = base_needed + wall_cost + walk_extra
        loot_factor = 1.0
        if dist > cfg.optimal_landing_px and cfg.max_landing_px > cfg.optimal_landing_px:
            span = cfg.max_landing_px - cfg.optimal_landing_px
            falloff = min(1.0, (dist - cfg.optimal_landing_px) / span)
            loot_factor = max(0.35, 1.0 - falloff * 0.8)
        risk = defense_risk(coords, defenses, cfg.safe_defense_px)

        expected_loot = min(remaining, int(base_needed * cfg.loot_per_troop * loot_factor))
        if budget > 0:
            lootable_troops = max(0, budget - wall_cost - walk_extra)
            expected_loot = min(expected_loot, int(lootable_troops * cfg.loot_per_troop * loot_factor))

        accessible = landing is not None
        reason = "可获取"
        if dist > cfg.max_landing_px:
            accessible = False
            reason = f"落点距建筑 {dist:.0f}px 过远，无法有效获取"
        elif wall_cost > cfg.wall_cost_limit:
            accessible = False
            reason = f"城墙阻隔过重（需 {wall_cost} 兵力破墙）"
        elif troops_needed > budget:
            accessible = False
            reason = f"估算需 {troops_needed} 兵力，超出预算 {budget}"
        elif risk > cfg.max_risk:
            accessible = False
            reason = f"防御风险 {risk:.2f} 过高"
        elif remaining <= 0:
            accessible = False
            reason = "该建筑已无可抢资源"
        elif expected_loot < cfg.min_expected_loot:
            accessible = False
            reason = f"剩余预计可抢 {expected_loot}，不值得派兵"

        efficiency = 0.0
        if troops_needed > 0 and cfg.loot_per_troop > 0:
            efficiency = expected_loot / float(troops_needed * cfg.loot_per_troop)
        efficiently_lootable = bool(
            accessible and efficiency >= cfg.min_efficiency and risk <= cfg.max_risk * 0.7
        )
        breakdown = {
            "landing_dist": round(dist, 1),
            "wall_cost": float(wall_cost),
            "walk_extra": float(walk_extra),
            "troops_needed": float(troops_needed),
            "risk": round(risk, 3),
            "loot_factor": round(loot_factor, 3),
            "expected_loot": float(expected_loot),
        }
        return LootEstimate(
            name, coords, accessible, reason,
            landing, round(dist, 1), wall_cost, walk_extra, troops_needed,
            cfg.loot_per_troop, expected_loot, round(risk, 3), round(loot_factor, 3),
            efficiently_lootable, breakdown,
        )
