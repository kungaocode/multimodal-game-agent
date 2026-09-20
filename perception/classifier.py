"""Building type + level classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BuildingClassifier(nn.Module):
    """Multi-task ResNet18 classifier: type classification + level regression."""

    def __init__(
        self,
        num_types: int,
        max_level: int,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            from torchvision import models

            backbone = models.resnet18(weights="DEFAULT")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torchvision is required for the classifier backbone. "
                "Run: pip install torchvision"
            ) from exc
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.type_head = nn.Linear(feature_dim, num_types)
        self.level_head = nn.Linear(feature_dim, 1)
        self.max_level = int(max_level)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        type_logits = self.type_head(features)
        level = self.level_head(features).squeeze(-1)
        return type_logits, level

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "num_types": self.type_head.out_features,
            "max_level": self.max_level,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path) -> "BuildingClassifier":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            num_types=checkpoint["num_types"],
            max_level=checkpoint["max_level"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
