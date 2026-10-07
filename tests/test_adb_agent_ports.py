"""ADB 三环接线契约测试：组合村庄环 + 阻塞切战斗 + 缓存回归。"""

from __future__ import annotations

from PIL import Image

from decision.arbiter import Ring, VillageSnapshot
from decision.farm_fsm import FarmObjective
from decision.resource_gate import ResourceGate
from executor.adb_executor import AdbExecutor
from state.game_state import GameState, ResourceStatus, Resources
from tasks.adb_agent import AdbAgent
from tasks.agent_loop import AgentLoop, AgentPorts, RingOutcome


class _FakeAdb(AdbExecutor):
    def __init__(self) -> None:
        self.taps: list[tuple[int, int]] = []
        self._frame = Image.new("RGB", (320, 180), "white")

    def screencap(self) -> Image.Image:
        return self._frame.copy()

    def tap(self, x: int, y: int) -> None:
        self.taps.append((int(x), int(y)))


class _FakeCascade:
    def __init__(self, states: list[GameState]) -> None:
        self._states = list(states)

    def extract(self, screenshot: Image.Image):
        state = self._states.pop(0) if self._states else GameState()
        return state, {}


def _agent(states: list[GameState]) -> AdbAgent:
    return AdbAgent(
        adb=_FakeAdb(),
        cascade=_FakeCascade(states),
        step_delay=0.0,
        max_steps=20,
    )


def test_perceive_village_reads_latest_state_cache():
    state = GameState(
        resource_status=ResourceStatus(
            detected=3,
            amounts=Resources(gold=2_000_000, elixir=2_000_000, dark_elixir=20_000),
        ),
    )
    state.add_building("圣水采集图标", (120, 80))
    agent = _agent([state])

    snapshot = agent.perceive_village()

    assert snapshot.collectibles == (("elixir", (120, 80)),)


def test_run_village_returns_blocked_when_donation_cannot_be_paid():
    agent = _agent([GameState()])
    agent.run_collect = lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "采集完")
    agent.gate = ResourceGate()
    agent.resources = Resources(gold=10_000_000, elixir=10_000_000, dark_elixir=100_000)
    agent.run_donate = lambda: RingOutcome(
        Ring.DONATE,
        "BLOCKED",
        "缺圣水",
        blocked_request={
            "resource_type": "elixir",
            "required": 2_500,
            "available": 1_000,
            "missing": 1_500,
        },
    )

    outcome = agent.run_village()

    assert outcome.ring is Ring.DONATE
    assert outcome.status == "BLOCKED"
    assert outcome.blocked_request == {
        "resource_type": "elixir",
        "required": 2_500,
        "available": 1_000,
        "missing": 1_500,
    }


def test_blocked_donation_closes_message_list_before_returning():
    first = GameState()
    first.add_building("消息列表按钮", (160, 120))
    second = GameState(
        screen_text=[{"request_cost": 40, "resource_type": "elixir"}]
    )
    second.add_building("请求条目", (250, 120))
    second.add_building("关闭按钮", (300, 20))
    agent = _agent([first, second])
    agent.resources = Resources(elixir=10)
    agent.resource_detected = True

    outcome = agent.run_donate()

    assert outcome.status == "BLOCKED"
    assert (300, 20) in agent.adb.taps
    assert agent.donation_blocked == {
        "resource_type": "elixir",
        "cost": 40,
        "required": 40,
        "available": 10,
        "missing": 30,
    }


def test_loop_switches_from_blocked_village_to_farm_then_back_to_village():
    blocked = RingOutcome(
        Ring.DONATE,
        "BLOCKED",
        "缺圣水",
        blocked_request={
            "resource_type": "elixir",
            "required": 2_500,
            "available": 1_000,
            "missing": 1_500,
        },
    )
    calls: list[str] = []
    objectives: list[FarmObjective] = []
    village_results = [blocked, RingOutcome(Ring.DONATE, "SUCCESS", "采集与捐兵完成")]

    def village() -> RingOutcome:
        calls.append("village")
        return village_results.pop(0)

    def farm(objective: FarmObjective) -> RingOutcome:
        calls.append("farm")
        objectives.append(objective)
        return RingOutcome(Ring.FARM, "SUCCESS", "补给完成", objective=objective)

    outcomes = AgentLoop(
        AgentPorts(
            perceive_village=lambda: VillageSnapshot(),
            run_collect=lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "ok"),
            run_farm=farm,
            run_donate=lambda: RingOutcome(Ring.DONATE, "SUCCESS", "ok"),
            run_village=village,
        ),
        max_rounds=5,
    ).run()

    assert calls == ["village", "farm", "village"]
    assert [o.status for o in outcomes] == ["BLOCKED", "SUCCESS", "SUCCESS"]
    assert objectives[0].requirements == {"elixir": 1_500}


def test_loop_low_resource_triggers_farm_without_block():
    low = RingOutcome(
        Ring.FARM,
        "LOW",
        "elixir 未回升到 1_500_000",
        objective=FarmObjective(
            requirements={"elixir": 1_500_000},
            target_types=("elixir",),
        ),
    )
    calls: list[str] = []
    farm_objectives: list[FarmObjective] = []
    village_results = [low, RingOutcome(Ring.DONATE, "SUCCESS", "完成")]
    village_calls = 0

    def village() -> RingOutcome:
        nonlocal village_calls
        calls.append("village")
        village_calls += 1
        return village_results[min(village_calls - 1, len(village_results) - 1)]

    def farm(objective: FarmObjective) -> RingOutcome:
        calls.append("farm")
        farm_objectives.append(objective)
        return RingOutcome(Ring.FARM, "SUCCESS", "补给完成", objective=objective)

    outcomes = AgentLoop(
        AgentPorts(
            perceive_village=lambda: VillageSnapshot(),
            run_collect=lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "ok"),
            run_farm=farm,
            run_donate=lambda: RingOutcome(Ring.DONATE, "SUCCESS", "ok"),
            run_village=village,
        ),
        max_rounds=5,
    ).run()

    assert calls == ["village", "farm", "village"]
    assert farm_objectives[0].requirements == {"elixir": 1_500_000}
