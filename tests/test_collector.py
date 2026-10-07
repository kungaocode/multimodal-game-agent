"""采集环 FSM / 确定性模拟器 / 闭环任务测试。"""

from __future__ import annotations

from decision.collector_fsm import CollectorFSM, CollectorSignals
from executor.collector_simulator import CollectorSimulator, CollectorSite
from tasks.collector import CollectorTask
from state.game_state import Resources


def test_collector_fsm_taps_first_icon_then_stops_when_empty():
    fsm = CollectorFSM()

    action = fsm.step(CollectorSignals([("gold", (120, 200))]))
    assert action.kind == "tap"
    assert action.coords == (120, 200)
    assert fsm.state.value == "村庄-采集中"

    action = fsm.step(CollectorSignals([]))
    assert action.kind == "stop"
    assert fsm.state.value == "采集完成"


def test_collector_closed_loop_collects_all_sites():
    simulator = CollectorSimulator(
        [
            CollectorSite("gold", (100, 200), amount=500),
            CollectorSite("elixir", (300, 200), amount=300),
        ],
        own_resources=Resources(gold=0, elixir=0),
    )

    result = CollectorTask().run(simulator, max_steps=20)

    assert result.status == "SUCCESS"
    assert result.collected == {"gold": 500, "elixir": 300}
    assert simulator.own_resources.gold == 500
    assert simulator.own_resources.elixir == 300
    assert all(record.verified for record in result.records)
    assert "金币采集图标" in {r.action.target for r in result.records}
    assert "圣水采集图标" in {r.action.target for r in result.records}


def test_collector_task_summary_serializable():
    simulator = CollectorSimulator([CollectorSite("gold", (100, 200), amount=500)])
    result = CollectorTask().run(simulator)
    data = result.summary()
    assert data["status"] == "SUCCESS"
    assert data["collected"] == {"gold": 500}
    assert isinstance(data["records"], list)
