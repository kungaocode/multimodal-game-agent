"""CLI to evaluate the trained building classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier import BuildingClassifier
from .crawl_dataset import build_label_maps, load_crawled_items, split_train_val
from .train_classifier import ClassifierDataset, build_transforms, evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate building classifier")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--items-dir", default="dataset/text/items", type=Path)
    parser.add_argument("--picture-root", default="dataset/picture", type=Path)
    parser.add_argument("--sections", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    type_to_id = metadata.get("type_to_id", {})
    max_level = metadata.get("max_level", 0)

    if args.sections is None:
        sections = metadata.get("sections", None)
        sections = set(sections) if sections else None
    else:
        sections = set(args.sections)

    items = load_crawled_items(
        items_dir=args.items_dir,
        picture_root=args.picture_root,
        sections=sections,
    )
    _, _, _ = build_label_maps(items)
    _, val_samples = split_train_val(items, type_to_id, val_ratio=0.2, seed=42)
    _, val_transform = build_transforms()
    val_ds = ClassifierDataset(val_samples, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = BuildingClassifier(num_types=len(type_to_id), max_level=max_level)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    type_criterion = nn.CrossEntropyLoss()
    level_criterion = nn.MSELoss()
    val_loss, val_acc, val_mae = evaluate(
        model,
        val_loader,
        device,
        type_criterion,
        level_criterion,
        level_weight=0.5,
    )
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val type accuracy: {val_acc:.4f}")
    print(f"Val level MAE: {val_mae:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
