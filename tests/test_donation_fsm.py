"""Tests for decision/donation_fsm.py — 捐兵完整状态转移与可执行动作。"""

from __future__ import annotations

from decision.donation_fsm import DonationFSM, DonationSignals, DonationState
from state.game_state import Resources


def test_full_donation_cycle_selects_troop_then_confirms():
    fsm = DonationFSM()

    action = fsm.step(DonationSignals(message_list_button=(100, 500)))
    assert action.kind == "tap"
    assert action.target == "消息列表按钮"
    assert action.coords == (100, 500)
    assert fsm.state is DonationState.MESSAGE_LIST

    action = fsm.step(
        DonationSignals(
            request_entry=(640, 300),
            close_button=(100, 100),
            troop_type="气球兵",
        )
    )
    assert action.target == "请求条目"
    assert action.coords == (640, 300)
    assert fsm.state is DonationState.REQUEST_DETAIL

    action = fsm.step(
        DonationSignals(
            reinforce_button=(640, 400),
            close_button=(100, 100),
        )
    )
    assert action.target == "增援按钮"
    assert action.coords == (640, 400)
    assert fsm.state is DonationState.DONATING

    action = fsm.step(DonationSignals(troop_bar_coords=(640, 690)))
    assert action.target == "兵种选择栏"
    assert action.coords == (640, 690)
    assert fsm.troop_selected is True

    action = fsm.step(
        DonationSignals(
            troop_bar_coords=(640, 690),
            confirm_button=(640, 380),
        )
    )
    assert action.target == "捐赠确认按钮"
    assert action.coords == (640, 380)
    assert fsm.state is DonationState.DONE

    action = fsm.step(DonationSignals(close_button=(100, 100)))
    assert action.target == "关闭按钮"
    assert action.coords == (100, 100)
    assert fsm.state is DonationState.VILLAGE


def test_donating_waits_for_troop_selection_when_bar_missing():
    fsm = DonationFSM()
    fsm.state = DonationState.DONATING
    action = fsm.step(DonationSignals(confirm_button=(640, 380)))
    assert action.kind == "wait"
    assert fsm.state is DonationState.DONATING
    assert fsm.troop_selected is False


def test_all_transition_taps_carry_coordinates():
    """白名单能执行的前提：捐兵 FSM 任何 tap 都必须给出坐标。"""
    fsm = DonationFSM()
    script = [
        DonationSignals(message_list_button=(100, 500)),
        DonationSignals(request_entry=(640, 300)),
        DonationSignals(reinforce_button=(640, 400)),
        DonationSignals(troop_bar_coords=(640, 690)),
        DonationSignals(troop_bar_coords=(640, 690), confirm_button=(640, 380)),
        DonationSignals(close_button=(100, 100)),
    ]
    for signals in script:
        action = fsm.step(signals)
        if action.kind == "tap":
            assert action.coords is not None, f"{action.target} 缺少坐标"


def test_close_before_request_returns_to_village():
    fsm = DonationFSM()
    fsm.state = DonationState.MESSAGE_LIST
    action = fsm.step(DonationSignals(close_button=(100, 100)))
    assert action.target == "关闭按钮"
    assert fsm.state is DonationState.VILLAGE


def test_done_returns_to_message_list_when_more_requests_exist():
    fsm = DonationFSM()
    fsm.state = DonationState.DONE
    action = fsm.step(
        DonationSignals(close_button=(100, 100), more_requests=True)
    )
    assert action.target == "关闭按钮"
    assert fsm.state is DonationState.MESSAGE_LIST


def test_blocked_reopens_message_list_via_village_button():
    fsm = DonationFSM()
    fsm.state = DonationState.BLOCKED
    fsm.blocked_request = {
        "resource_type": "elixir",
        "required": 40,
        "available": 10,
        "missing": 30,
    }
    action = fsm.step(
        DonationSignals(
            message_list_button=(400, 500),
            request_cost=40,
            request_resource_type="elixir",
            resources=Resources(elixir=10),
        )
    )
    assert action.target == "消息列表按钮"
    assert action.coords == (400, 500)
    assert fsm.state is DonationState.MESSAGE_LIST
    assert fsm.blocked_request is not None


def test_reset_clears_troop_selection():
    fsm = DonationFSM()
    fsm.state = DonationState.DONATING
    fsm.troop_selected = True
    fsm.reset()
    assert fsm.state is DonationState.VILLAGE
    assert fsm.troop_selected is False
    assert fsm.history == ["村庄待机"]
