"""Tests for battle-time target memory and depletion/retreat logic."""

from __future__ import annotations

from decision.battle_memory import BattleMemory, MemoryConfig
from decision.farm_fsm import FarmSignals


def _sig(resources):
    return FarmSignals(enemy_resources=list(resources))


def test_observe_and_deploy_tracks_remaining():
    memory = BattleMemory()
    memory.observe(_sig([("金矿", (100, 100))]))
    assert memory.capacity("金矿", (100, 100)) == 1500
    assert memory.remaining("金矿", (100, 100)) == 1500

    memory.mark_deploy("金矿", (100, 100))
    assert memory.looted("金矿", (100, 100)) == 1000
    assert memory.remaining("金矿", (100, 100)) == 500


def test_depleted_requires_enough_seen_frames():
    memory = BattleMemory(MemoryConfig(min_seen_to_deplete=2))
    memory.observe(_sig([("储金罐", (200, 200))]))
    for _ in range(4):  # 打满估算兵力
        memory.mark_deploy("储金罐", (200, 200))
    assert memory.is_depleted("储金罐", (200, 200)) is False  # 只看见 1 帧

    memory.observe(_sig([("储金罐", (200, 200))]))
    assert memory.is_depleted("储金罐", (200, 200)) is True


def test_remaining_low_marks_depleted():
    memory = BattleMemory(MemoryConfig(min_seen_to_deplete=1))
    memory.observe(_sig([("金矿", (100, 100))]))
    for _ in range(2):  # 2 次下兵 = 2000 > 1500 容量
        memory.mark_deploy("金矿", (100, 100))
    assert memory.remaining("金矿", (100, 100)) == 0
    assert memory.is_depleted("金矿", (100, 100)) is True


def test_deploy_with_estimate_updates_costs():
    from decision.loot_estimator import LootEstimate

    memory = BattleMemory()
    memory.observe(_sig([("圣水瓶", (300, 300))]))
    estimate = LootEstimate(
        name="圣水瓶",
        coords=(300, 300),
        accessible=True,
        reason="ok",
        landing_point=(300, 330),
        landing_dist_px=30.0,
        wall_cost=3,
        walk_extra=1,
        troops_needed=9,
        loot_per_troop=1000,
        expected_loot=4000,
        risk_score=0.1,
        loot_factor=1.0,
        efficiently_lootable=True,
    )
    memory.mark_deploy("圣水瓶", (300, 300), estimate)
    site = memory.find("圣水瓶", (300, 300))
    assert site is not None
    assert site.troops_needed == 9
    assert site.wall_cost == 3
    assert site.walk_extra == 1


def test_same_site_matches_small_offsets():
    memory = BattleMemory()
    memory.observe(_sig([("金矿", (300, 300))]))
    memory.mark_deploy("金矿", (300, 310))
    assert memory.looted("金矿", (300, 300)) == 1000
    assert len(memory.summary()) == 1


def test_reset_clears_state():
    memory = BattleMemory()
    memory.observe(_sig([("金矿", (100, 100))]))
    memory.mark_deploy("金矿", (100, 100))
    memory.reset()
    assert memory.summary() == []
    assert memory.remaining("金矿", (100, 100)) is None
