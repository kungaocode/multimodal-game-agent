"""Tests for red-zone closed-area exclusion, landing-point planning, and
landable-aware resource-target ranking.

关键语义：
- 红色区域围成一个面积，面积内部（含红线本身）全部禁止下兵；
- 部署落点：先排除不可下兵位置，再从目标资源建筑就近选可下兵点；
- 资源优先度：可达性 = 目标中心到最近可下兵点的距离。
"""

import numpy as np
import pytest
from PIL import Image

from decision.farm_fsm import FarmSignals
from decision.resource_policy import PolicyConfig, ResourcePolicy
from tools.coach_session import _green_mask, _no_land_mask, _plan_deploy_point


def _synthetic_screenshot(w=300, h=200, draw=None):
    """纯色草地画布，上面画可选的红色矩形区域。"""
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = 60
    a[..., 1] = 170
    a[..., 2] = 60
    if draw is not None:
        draw(a)
    return Image.fromarray(a, "RGB")


def _draw_red_rect(a, x0, y0, x1, y1):
    a[y0:y1, x0:x1, 0] = 230
    a[y0:y1, x0:x1, 1] = 90
    a[y0:y1, x0:x1, 2] = 90


def _draw_red_border(a, x0, y0, x1, y1, width=4):
    """只画红色边框，内部保持（草地/其他）颜色 -> 验证闭合面积排除。"""
    a[y0:y0 + width, x0:x1] = (230, 90, 90)
    a[y1 - width:y1, x0:x1] = (230, 90, 90)
    a[y0:y1, x0:x0 + width] = (230, 90, 90)
    a[y0:y1, x1 - width:x1] = (230, 90, 90)


def test_red_zone_is_enclosed_area_not_just_pixels():
    """红区用边框围成一个面积：边框内部（非红像素）也必须被排除。"""
    im = _synthetic_screenshot(draw=lambda a: _draw_red_border(a, 80, 50, 160, 120))
    no_land = _no_land_mask(im, dilate=0, edge_guard=0)
    # 边框内部中心 (120, 85) 虽然是非红像素，也必须被视为不可下兵
    assert no_land[85, 120] == True
    # 外部草地可下兵
    assert no_land[30, 40] == False


def test_snap_avoids_closed_red_area_with_internal_grass():
    """红区内部即使有绿色草地，落点也必须被吸到红区面积外。"""
    im = _synthetic_screenshot(draw=lambda a: _draw_red_border(a, 60, 40, 140, 110))
    no_land = _no_land_mask(im, dilate=0, edge_guard=0)
    # 确认红区内部有草地像素（绿色判定为 True），但不可下兵掩码必须拦住
    green = _green_mask(im)
    assert green[70, 100] == True
    assert no_land[70, 100] == True


def test_plan_deploy_point_avoids_no_land_and_stays_near_target():
    """目标建筑中心在红区内 -> 落点吸附到红区外、且贴近目标。"""
    im = _synthetic_screenshot(draw=lambda a: _draw_red_rect(a, 60, 40, 140, 110))
    target_center = (100, 75)  # 红区内部
    pt = _plan_deploy_point(target_center, "金矿", None, im)
    no_land = _no_land_mask(im, dilate=0, edge_guard=0)
    assert no_land[pt[1], pt[0]] == False
    # 落点应贴近目标（红区 60..140, 40..110：最短外沿约 20px）
    dx = min(abs(pt[0] - 60), abs(pt[0] - 140))
    dy = min(abs(pt[1] - 40), abs(pt[1] - 110))
    assert min(dx, dy) <= 30


def test_plan_deploy_point_snaps_off_non_grass_building_center():
    """建筑中心不是草地时，也必须吸附到附近可下兵草地，而不是原地中心。"""
    def draw_building(a):
        a[60:120, 80:160] = (150, 130, 90)

    im = _synthetic_screenshot(draw=draw_building)
    target_center = (120, 90)
    pt = _plan_deploy_point(target_center, "金矿", None, im)
    no_land = _no_land_mask(im)
    green = _green_mask(im)
    assert no_land[pt[1], pt[0]] == False
    assert green[pt[1], pt[0]] == True
    assert not (80 <= pt[0] <= 160 and 60 <= pt[1] <= 120)


def test_plan_deploy_trusts_coach_when_already_landable():
    """教练落点本身可下兵且贴目标 -> 直接信任，不做多余吸附。"""
    im = _synthetic_screenshot()
    coach = (150, 160)  # 外部草地
    pt = _plan_deploy_point(coach, "金矿", None, im)
    assert pt == coach


def test_plan_deploy_uses_detected_building_center():
    """有检测框时，目标中心用检测框中心（多实例选离教练落点最近的）。"""
    im = _synthetic_screenshot(draw=lambda a: _draw_red_rect(a, 60, 40, 140, 110))
    detections = [
        {"type": "金矿", "coords": [100, 75], "confidence": 0.9},   # 红区内（目标）
        {"type": "金矿", "coords": [250, 60], "confidence": 0.8},   # 远处
        {"type": "储金罐", "coords": [20, 20], "confidence": 0.7},
    ]
    coach = (105, 80)  # 贴近第一个金矿
    pt = _plan_deploy_point(coach, "金矿", detections, im)
    no_land = _no_land_mask(im, dilate=0, edge_guard=0)
    assert no_land[pt[1], pt[0]] == False
    # 落点应靠近目标金矿 (100,75) 而非远处金矿 (250,60)
    assert abs(pt[0] - 90) < 40


def test_resource_policy_landable_ranks_reachable_first():
    """可达性接入：完全被红区包围的目标（无可下兵点）被降权，
    即使其资源价值更高，也应让位于可下兵的目标。"""
    landable = np.ones((200, 300), dtype=bool)
    # 大块不可下兵区域（红区面积）
    landable[30:170, 60:240] = False
    sig = FarmSignals(
        enemy_resources=[
            ("暗黑重油罐", (150, 100)),  # 红区正中心，距可下兵点 30px（其实很近）
            ("金矿", (40, 190)),
        ]
    )
    cfg = PolicyConfig(image_size=(300, 200), deploy_point=(150, 195))
    policy = ResourcePolicy(cfg, landable_provider=lambda s: landable)
    scored = policy.scored(sig)
    by_name = {t.name: t for t in scored}
    # 暗黑罐中心在红区中心：最近可下兵点 (150,30) 距离 70px
    assert by_name["暗黑重油罐"].breakdown["landable_dist"] >= 60
    # 金矿完全可达
    assert by_name["金矿"].breakdown["landable_dist"] == 0.0


def test_resource_policy_without_mask_keeps_old_behavior():
    """不提供 landable_provider 时行为不变：同坐标下价值高的排第一。"""
    sig = FarmSignals(
        enemy_resources=[("金矿", (640, 360)), ("暗黑重油罐", (640, 360))]
    )
    out = ResourcePolicy()(sig)
    assert out[0][0] == "暗黑重油罐"
