"""Tests for api/planner.py — GameState → 打资源计划（规则层）。"""

from api.planner import build_plan, llm_plan_prompt, signals_from_state
from state.game_state import GameState

RESOURCE_NAMES = ("圣水收集器", "金矿", "圣水瓶", "储金罐", "暗黑重油罐", "暗黑重油钻井")


def _state_with(**kwargs):
    return GameState(**kwargs)


def test_build_plan_orders_candidates_by_score():
    state = GameState()
    # 两个金矿：一个远离出兵点、贴近防御；一个靠边、远离防御
    state.add_building("金矿", (60, 60), confidence=0.9)
    state.add_building("金矿", (1220, 60), confidence=0.9)
    plan = build_plan(state)
    candidates = plan["policy"]["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["score"] >= candidates[1]["score"]
    # 无防御信息时不报错，且默认 1280x720
    assert plan["policy"]["defenses_used"] is False
    assert plan["policy"]["image_size"] == [1280, 720]


def test_build_plan_skips_full_resource_buildings():
    state = GameState()
    state.add_building("圣水收集器", (200, 300))
    state.add_building("金矿", (400, 300))
    plan = build_plan(state, full_resources=["elixir"])
    names = [c["name"] for c in plan["policy"]["candidates"]]
    assert "圣水收集器" not in names
    assert "金矿" in names
    assert plan["policy"]["skipped_resource_buildings"] == ["圣水收集器", "圣水瓶"]


def test_build_plan_drops_unknown_positions():
    state = GameState()
    state.add_building("金矿", (0, 0))  # 未知位置
    state.add_building("金矿", (100, 100))
    plan = build_plan(state)
    assert len(plan["policy"]["candidates"]) == 1


def test_build_plan_uses_defenses_for_safety():
    state = GameState()
    state.add_building("金矿", (640, 100))
    state.add_building("加农炮", (640, 110))  # 紧贴
    plan = build_plan(state)
    assert plan["policy"]["defenses_used"] is True
    # 紧贴防御时 safety 很低（0），总分应小于完全安全时的同目标
    state2 = GameState()
    state2.add_building("金矿", (640, 100))
    plan2 = build_plan(state2)
    c1 = plan["policy"]["candidates"][0]
    c2 = plan2["policy"]["candidates"][0]
    assert c1["breakdown"]["safety"] < c2["breakdown"]["safety"]
    assert c1["score"] < c2["score"]


def test_signals_from_state_normalizes_full_aliases():
    state = GameState()
    signals, defenses = signals_from_state(state, full_resources=["金", "dark_elixir"])
    assert signals.resources.gold_full is True
    assert signals.resources.elixir_full is False
    assert signals.resources.dark_full is True


def test_llm_plan_prompt_contains_candidates():
    state = GameState(); state.add_building("金矿", (100, 200)); plan = build_plan(state)
    prompt = llm_plan_prompt(plan["policy"])
    assert "金矿" in prompt
    assert "steps" in prompt
    assert "综合分" in prompt


def test_candidates_from_llm_steps_filters_non_deploy():
    from api.planner import candidates_from_llm_steps

    steps = [
        {"action": "deploy", "target": "金矿", "coords": [370, 580], "reason": "好"},
        {"action": "wait", "target": None, "coords": [], "reason": "等"},
        {"action": "deploy", "target": "圣水收集器", "coords": [540, 420], "reason": "好"},
        {"coords": [1, 2]},  # 无 action 的异常项
    ]
    cands = candidates_from_llm_steps(steps)
    assert len(cands) == 2
    assert cands[0] == {
        "name": "金矿",
        "coords": [370, 580],
        "score": 0.0,
        "breakdown": {},
        "source": "llm",
        "reason": "好",
    }
    assert cands[1]["name"] == "圣水收集器"
