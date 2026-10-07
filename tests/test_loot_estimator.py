"""Tests for the deterministic resource-accessibility/loot estimator."""

from __future__ import annotations

import numpy as np

from decision.loot_estimator import LootConfig, LootEstimator


def _full_allowed(h=200, w=300):
    return np.ones((h, w), dtype=bool)


def test_no_land_mask_makes_target_inaccessible():
    estimator = LootEstimator(LootConfig(min_expected_loot=0))
    no_land = np.ones((200, 300), dtype=bool)
    estimate = estimator.estimate(
        "金矿",
        (100, 100),
        no_land_mask=no_land,
    )
    assert estimate.accessible is False
    assert "无可下兵落点" in estimate.reason
    assert estimate.troops_needed == 0


def test_landing_distance_changes_loot_factor_and_walk_cost():
    config = LootConfig(
        optimal_landing_px=50.0,
        max_landing_px=180.0,
        walk_px_per_extra_troop=20.0,
        min_expected_loot=0,
        min_efficiency=0,
        troop_budget=30,
    )
    estimator = LootEstimator(config)
    allowed = _full_allowed()
    allowed[40:160, 90:210] = False  # 目标中心在禁区内，落点会被吸到边界外
    no_land = ~allowed

    estimate = estimator.estimate(
        "金矿",
        (150, 100),
        no_land_mask=no_land,
        capacity=3000,
    )
    assert estimate.accessible is True
    assert estimate.landing_dist_px > 50
    assert estimate.walk_extra >= 1
    assert estimate.loot_factor < 1.0
    assert estimate.expected_loot < 3000

    close = estimator.estimate(
        "金矿",
        (260, 60),
        no_land_mask=np.zeros((200, 300), dtype=bool),
        capacity=3000,
    )
    assert close.landing_dist_px == 0
    assert close.walk_extra == 0
    assert close.loot_factor == 1.0


def test_wall_segments_add_troop_cost():
    estimator = LootEstimator(LootConfig(min_expected_loot=0))
    estimate = estimator.estimate(
        "储金罐",
        (200, 100),
        no_land_mask=np.zeros((200, 300), dtype=bool),
            wall_segments=[
                ((150, 80), (250, 120)),
                ((400, 80), (500, 120)),
            ],
            capacity=2000,
            # 所有地块都可下兵时落点会吸附到建筑中心；显式给一个墙外侧落点，
            # 让「落点不同 → 城墙穿越不同」路径真实发生。
            landing_requested=(180, 100),
        )
    assert estimate.wall_cost == 3
    assert estimate.troops_needed == 2 + 3


def test_high_defense_risk_blocks_target():
    estimator = LootEstimator(LootConfig(min_expected_loot=0))
    estimate = estimator.estimate(
        "暗黑重油罐",
        (300, 300),
        no_land_mask=np.zeros((600, 600), dtype=bool),
        defenses=[(300, 300)],
    )
    assert estimate.accessible is False
    assert "防御风险" in estimate.reason


def test_troop_budget_blocks_expensive_target():
    estimator = LootEstimator(LootConfig(troop_budget=2, min_expected_loot=0))
    estimate = estimator.estimate(
        "储金罐",
        (100, 100),
        no_land_mask=np.zeros((200, 300), dtype=bool),
        capacity=3000,
    )
    assert estimate.accessible is False
    assert "预算" in estimate.reason


def test_low_remaining_loot_blocks_target():
    estimator = LootEstimator(LootConfig(min_expected_loot=400))
    estimate = estimator.estimate(
        "圣水瓶",
        (100, 100),
        no_land_mask=np.zeros((200, 300), dtype=bool),
        capacity=300,
    )
    assert estimate.accessible is False
    assert "不值得" in estimate.reason


def test_estimate_shares_deterministic_breakdown():
    estimator = LootEstimator()
    first = estimator.estimate(
        "金矿",
        (100, 100),
        no_land_mask=np.zeros((200, 300), dtype=bool),
    )
    second = estimator.estimate(
        "金矿",
        (100, 100),
        no_land_mask=np.zeros((200, 300), dtype=bool),
    )
    assert first.to_dict() == second.to_dict()
    assert set(first.breakdown) >= {"landing_dist", "wall_cost", "expected_loot"}
