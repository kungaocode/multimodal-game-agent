"""执行器层：结构化动作、动作校验与模拟环境。"""

from .action import ALLOWED_ACTION_KINDS, COLLECTOR_TARGETS, FARM_ALLOWED_TARGETS, AgentAction
from .collector_simulator import CollectorSimulator, CollectorSite
from .donation_simulator import DonationRequest, DonationScreen, DonationSimulator
from .simulator import DefenseSite, ResourceSite, SimVillage, Simulator, default_farm_world
from .validator import ActionValidator, ValidationResult

__all__ = [
    "ALLOWED_ACTION_KINDS",
    "COLLECTOR_TARGETS",
    "FARM_ALLOWED_TARGETS",
    "AgentAction",
    "ActionValidator",
    "ValidationResult",
    "Simulator",
    "SimVillage",
    "ResourceSite",
    "DefenseSite",
    "default_farm_world",
    "DonationSimulator",
    "DonationRequest",
    "DonationScreen",
    "CollectorSimulator",
    "CollectorSite",
]
