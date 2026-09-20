"""Training data recorder for coach-guided sessions.

Each step saves the screenshot, local model output, coach verdict, and
execution result to disk. A session_summary.json is written at the end
with correction-rate statistics for retraining evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from decision.coach import CoachVerdict


class TrainingDataRecorder:
    """Save each step's data for later retraining."""

    def __init__(self, session_dir: Path) -> None:
        self.dir = session_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []

    def record(
        self,
        step: int,
        screenshot: Image.Image,
        local_state: str | None,
        local_action: dict,
        local_detections: list[dict],
        coach_verdict: CoachVerdict | None,
        executed: bool,
        result: dict | None = None,
    ) -> None:
        prefix = f"step_{step:04d}"
        screenshot.save(self.dir / f"{prefix}_screenshot.jpg", format="JPEG", quality=90)
        local_data = {
            "step": step,
            "state": local_state,
            "action": local_action,
            "detections": local_detections,
        }
        (self.dir / f"{prefix}_local.json").write_text(
            json.dumps(local_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if coach_verdict is not None:
            coach_data = {
                "approved": coach_verdict.approved,
                "corrected_state": coach_verdict.corrected_state,
                "corrected_action": {
                    "kind": coach_verdict.corrected_action_kind,
                    "target": coach_verdict.corrected_action_target,
                    "coords": list(coach_verdict.corrected_action_coords) if coach_verdict.corrected_action_coords else None,
                    "troop_coords": list(coach_verdict.corrected_troop_coords) if coach_verdict.corrected_troop_coords else None,
                },
                "missed_objects": coach_verdict.missed_objects,
                "reasoning": coach_verdict.reasoning,
                "error": coach_verdict.error,
            }
            (self.dir / f"{prefix}_coach.json").write_text(
                json.dumps(coach_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        step_record = {
            "step": step,
            "local_state": local_state,
            "local_action": local_action,
            "coach_approved": coach_verdict.approved if coach_verdict else None,
            "coach_corrected": coach_verdict is not None and not coach_verdict.approved,
            "executed": executed,
            "result": result or {},
        }
        self.steps.append(step_record)

    def save_summary(self, module: str, max_steps: int) -> dict:
        total = len(self.steps)
        approved = sum(1 for s in self.steps if s.get("coach_approved") is True)
        corrected = sum(1 for s in self.steps if s.get("coach_corrected") is True)
        executed = sum(1 for s in self.steps if s.get("executed") is True)
        summary = {
            "module": module,
            "max_steps": max_steps,
            "total_steps": total,
            "coach_approved": approved,
            "coach_corrected": corrected,
            "coach_correction_rate": round(corrected / total, 3) if total > 0 else 0.0,
            "executed_actions": executed,
            "steps": self.steps,
        }
        (self.dir / "session_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
