"""Tests for structured game state models."""

from state.game_state import Building, GameState, Resources


def test_default_game_state():
    state = GameState()
    assert state.resources.gold == 0
    assert state.builders.idle == 0
    assert state.buildings == []


def test_add_building():
    state = GameState()
    building = state.add_building("gold_mine", (100, 200), level=5, confidence=0.9)
    assert isinstance(building, Building)
    assert building.type == "gold_mine"
    assert building.position == (100, 200)
    assert len(state.buildings) == 1


def test_resources_model():
    resources = Resources(gold=850000, elixir=620000)
    assert resources.gold == 850000
    assert resources.elixir == 620000
