"""Tests for the donation simulator + FSM closed loop."""

from __future__ import annotations

from decision.donation_fsm import DonationAction
from decision.donation_fsm import DonationFSM
from executor.donation_simulator import (
    DonationRequest,
    DonationScreen,
    DonationSimulator,
)
from executor.validator import ActionValidator
from state.game_state import Resources
from tasks.donation import DonationTask


def _simulator():
    return DonationSimulator(
        [
            DonationRequest((500, 300), troop_type="气球兵"),
            DonationRequest((500, 420), troop_type="飞龙"),
        ]
    )


def test_full_donation_closed_loop_success():
    simulator = _simulator()
    result = DonationTask().run(simulator, max_steps=40)
    assert result.status == "SUCCESS"
    assert result.donated == 2
    assert result.steps < 40
    assert simulator.done is True
    assert [r.action.target for r in result.records] == [
        "消息列表按钮",
        "请求条目",
        "增援按钮",
        "兵种选择栏",
        "捐赠确认按钮",
        "关闭按钮",
        "请求条目",
        "增援按钮",
        "兵种选择栏",
        "捐赠确认按钮",
        "关闭按钮",
    ]


def test_each_tap_passes_validator_and_has_coords():
    simulator = _simulator()
    task = DonationTask(validator=ActionValidator())
    result = task.run(simulator, max_steps=40)
    assert result.status == "SUCCESS"
    for record in result.records:
        assert isinstance(record.action, DonationAction)
        if record.action.kind == "tap":
            assert record.action.coords is not None


def test_confirm_without_troop_selection_is_invalid():
    simulator = _simulator()
    simulator.screen = simulator.screen.__class__("donating")

    # 未选兵种时直接点确认会被模拟器拒绝。
    result = simulator.step(DonationAction("tap", "捐赠确认按钮", simulator.confirm_button))
    assert result.info["reason"].startswith("无效点击")

    simulator.step(DonationAction("tap", "兵种选择栏", simulator.troop_bar_coords))
    result = simulator.step(DonationAction("tap", "捐赠确认按钮", simulator.confirm_button))
    assert result.info["reason"] == "捐赠完成"


def test_donation_summary_tracks_donated():
    simulator = _simulator()
    DonationTask().run(simulator, max_steps=40)
    assert simulator.summary()["donated"] == 2
    assert simulator.summary()["done"] is True


def test_blocked_donation_resumes_after_resources_topped_up():
    fsm = DonationFSM()
    simulator = DonationSimulator(
        [
            DonationRequest(
                (500, 300),
                troop_type="气球兵",
                cost=40,
                resource_type="elixir",
            )
        ],
        own_resources=Resources(elixir=10),
    )
    task = DonationTask(fsm=fsm)

    blocked = task.run(simulator, max_steps=5)
    assert blocked.status == "BLOCKED"
    assert blocked.blocked_request["resource_type"] == "elixir"
    assert blocked.blocked_request["required"] == 40
    assert blocked.blocked_request["available"] == 10
    assert blocked.blocked_request["missing"] == 30
    assert simulator.screen is DonationScreen.MESSAGE_LIST

    simulator.own_resources.elixir += 30
    resumed = task.run(simulator, max_steps=40, reset=False)

    assert resumed.status == "SUCCESS"
    assert simulator.summary()["donated"] == 1
    assert simulator.own_resources.elixir == 0
