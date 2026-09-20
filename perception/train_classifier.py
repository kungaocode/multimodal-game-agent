"""CLI to train the building type + level classifier on crawled images."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier import BuildingClassifier
from .crawl_dataset import build_label_maps, load_crawled_items, split_train_val


def build_transforms() -> tuple[Any, Any]:
    """Return torchvision train and validation transforms."""
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required for training. Run: pip install torchvision"
        ) from exc
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


class ClassifierDataset(torch.utils.data.Dataset):
    """PyTorch dataset that yields image tensor, type_id, level."""

    def __init__(
        self,
        samples: list[tuple[Path, int, int]],
        transform: Any | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int, float]:
        from PIL import Image

        path, type_id, level = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, type_id, float(level)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    type_criterion: nn.Module,
    level_criterion: nn.Module,
    level_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for images, type_ids, levels in loader:
        images = images.to(device)
        type_ids = type_ids.to(device)
        levels = levels.to(device)
        optimizer.zero_grad()
        type_logits, level_pred = model(images)
        loss = type_criterion(type_logits, type_ids) + level_weight * level_criterion(
            level_pred, levels
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    type_criterion: nn.Module,
    level_criterion: nn.Module,
    level_weight: float = 1.0,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    level_errors: list[float] = []
    with torch.no_grad():
        for images, type_ids, levels in loader:
            images = images.to(device)
            type_ids = type_ids.to(device)
            levels = levels.to(device)
            type_logits, level_pred = model(images)
            loss = type_criterion(type_logits, type_ids) + level_weight * level_criterion(
                level_pred, levels
            )
            total_loss += loss.item() * images.size(0)
            preds = type_logits.argmax(dim=1)
            correct += (preds == type_ids).sum().item()
            total += type_ids.size(0)
            level_errors.extend(torch.abs(level_pred - levels).cpu().tolist())
    acc = correct / total if total else 0.0
    mae = sum(level_errors) / len(level_errors) if level_errors else 0.0
    return total_loss / total if total else 0.0, acc, mae


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train building type + level classifier")
    parser.add_argument("--items-dir", default="dataset/text/items", type=Path)
    parser.add_argument("--picture-root", default="dataset/picture", type=Path)
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["资源类建筑", "防御建筑", "军事建筑", "大本及武器", "其他建筑", "守卫"],
    )
    parser.add_argument("--output", default="runs/classifier", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--level-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_crawled_items(
        items_dir=args.items_dir,
        picture_root=args.picture_root,
        sections=set(args.sections),
    )
    type_to_id, id_to_type, max_level = build_label_maps(items)
    train_samples, val_samples = split_train_val(
        items, type_to_id, val_ratio=0.2, seed=args.seed
    )
    print(
        f"Loaded {len(items)} items, {len(train_samples)} train, "
        f"{len(val_samples)} val images"
    )

    train_transform, val_transform = build_transforms()
    train_ds = ClassifierDataset(train_samples, transform=train_transform)
    val_ds = ClassifierDataset(val_samples, transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = BuildingClassifier(
        num_types=len(type_to_id), max_level=max_level
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    type_criterion = nn.CrossEntropyLoss()
    level_criterion = nn.MSELoss()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            type_criterion,
            level_criterion,
            args.level_weight,
        )
        val_loss, val_acc, val_mae = evaluate(
            model,
            val_loader,
            device,
            type_criterion,
            level_criterion,
            args.level_weight,
        )
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_level_mae={val_mae:.2f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metadata = {
                "type_to_id": type_to_id,
                "id_to_type": id_to_type,
                "max_level": max_level,
                "sections": args.sections,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_level_mae": val_mae,
            }
            model.save(output_dir / "best.pt", metadata=metadata)

    summary = {
        "num_types": len(type_to_id),
        "max_level": max_level,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": best_val_loss,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Training complete. Best model saved to {output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
