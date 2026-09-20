"""Dataset loader for crawled Clash of Clans item images."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class CrawledItem:
    item_id: str
    name: str
    section: str
    img_folder: Path
    images: list[tuple[int, Path]]  # (level, image_path)


def parse_level(level_str: str) -> int | None:
    """取等级标签中的第一个整数。

    支持 '12 级' -> 12；区间标签 '1 - 2 级'（外观跨级不变）取起始级 -> 1；
    大本标签 '17 本 1 星' 取大本数 -> 17。
    无数字的特殊标签（如 '建造中'、'废墟'）返回 None。
    """
    if not level_str or level_str == "info":
        return None
    match = re.search(r"\d+", level_str)
    if match:
        return int(match.group())
    return None


def load_crawled_items(
    items_dir: Path | str,
    picture_root: Path | str = Path("dataset/picture"),
    sections: set[str] | None = None,
    require_images: bool = True,
) -> list[CrawledItem]:
    """Load metadata and resolve image paths.

    Args:
        items_dir: directory containing <id>-<Name>.json files.
        picture_root: root directory containing the actual images.
        sections: if provided, only keep items with these section values.
        require_images: skip items whose images are all missing.
    """
    items_dir = Path(items_dir)
    picture_root = Path(picture_root)
    items: list[CrawledItem] = []
    for json_path in sorted(items_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        section = data.get("section", "")
        if sections is not None and section not in sections:
            continue
        item_id = data.get("id", json_path.stem.split("-", 1)[0])
        name = data.get("name", "")
        img_folder = picture_root / data.get("img_folder", "")
        images: list[tuple[int, Path]] = []
        for group in data.get("image_groups", []):
            for img in group.get("images", []):
                level = parse_level(img.get("level", ""))
                if level is None:
                    continue
                src = img.get("src", "")
                img_path = img_folder / src
                if require_images and not img_path.exists():
                    continue
                images.append((level, img_path))
        if require_images and not images:
            continue
        items.append(
            CrawledItem(
                item_id=item_id,
                name=name,
                section=section,
                img_folder=img_folder,
                images=images,
            )
        )
    return items


def build_label_maps(items: list[CrawledItem]) -> tuple[dict[str, int], dict[int, str], int]:
    """Return (type_to_id, id_to_type, max_level)."""
    type_names = sorted({item.name for item in items})
    type_to_id = {name: i for i, name in enumerate(type_names)}
    id_to_type = {i: name for name, i in type_to_id.items()}
    max_level = 0
    for item in items:
        for level, _ in item.images:
            if level > max_level:
                max_level = level
    return type_to_id, id_to_type, max_level


def split_train_val(
    items: list[CrawledItem],
    type_to_id: dict[str, int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int, int]]]:
    """Split images stratified by type.

    Returns train and validation lists of (image_path, type_id, level).
    """
    random.seed(seed)
    train_samples: list[tuple[Path, int, int]] = []
    val_samples: list[tuple[Path, int, int]] = []
    for item in items:
        type_id = type_to_id[item.name]
        shuffled = item.images.copy()
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val = shuffled[:n_val]
        train = shuffled[n_val:]
        for level, path in train:
            train_samples.append((path, type_id, level))
        for level, path in val:
            val_samples.append((path, type_id, level))
    return train_samples, val_samples


def dataset_stats(items: list[CrawledItem]) -> dict[str, Any]:
    """Return summary statistics about the loaded dataset."""
    section_counts = Counter(item.section for item in items)
    type_counts = Counter(item.name for item in items)
    image_count = sum(len(item.images) for item in items)
    level_counter: Counter[int] = Counter()
    for item in items:
        for level, _ in item.images:
            level_counter[level] += 1
    return {
        "num_items": len(items),
        "num_images": image_count,
        "sections": dict(section_counts),
        "types": dict(type_counts),
        "levels": dict(level_counter),
    }


def preview_item(item: CrawledItem) -> dict[str, Any]:
    """Return a human-readable preview dict for one item."""
    return {
        "id": item.item_id,
        "name": item.name,
        "section": item.section,
        "num_images": len(item.images),
        "levels": sorted({level for level, _ in item.images}),
    }
