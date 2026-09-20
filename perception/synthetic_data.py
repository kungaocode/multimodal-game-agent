"""Synthetic detection data generator for YOLO.

爬取的 dataset 是"单建筑裁剪图 + 类型/等级标签"（分类风格），没有场景图和边界框，
不能直接喂给 YOLO 检测器。本模块把这些裁剪图当作透明精灵，按随机位置/尺度贴到
程序生成的草地上，产出"场景图 + YOLO 格式标注"，供 train_detector.py 使用。

用法：
    python -m perception.synthetic_data \
        --items-dir dataset/text/items \
        --picture-root dataset/picture \
        --out dataset/detection \
        --train-scenes 1200 --val-scenes 300

输出结构（与 train_detector.py 的默认参数对齐）：
    <out>/images/train|val/*.png
    <out>/labels/train|val/*.txt   # 每行: class_id cx cy w h（归一化）
    <out>/labels.json              # LabelMap：id -> 类型名
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .crawl_dataset import load_crawled_items
from .dataset import LabelMap

# 默认建筑分区（与 train_classifier.py 保持一致）
DEFAULT_SECTIONS = {"资源类建筑", "防御建筑", "军事建筑", "大本及武器", "其他建筑", "守卫"}

CANVAS_W, CANVAS_H = 1280, 720  # 模拟手机截图横屏比例（1920x1280 等比缩小）
MIN_SPRITE_H, MAX_SPRITE_H = 70, 220  # 建筑目标像素高度范围
MIN_OBJECTS, MAX_OBJECTS = 2, 9
MAX_OVERLAP_IOU = 0.35


@dataclass
class Sprite:
    """单张建筑精灵及其在场景中的放置信息。"""

    type_name: str
    class_id: int
    image: Image.Image
    box: tuple[int, int, int, int]  # x1, y1, x2, y2（像素）


def load_sprites(
    items_dir: Path | str,
    picture_root: Path | str,
    sections: set[str] | None = None,
    label_map: LabelMap | None = None,
) -> list[Sprite]:
    """从爬取数据加载建筑精灵。

    Args:
        items_dir: dataset/text/items 目录。
        picture_root: dataset/picture 根目录。
        sections: 只加载这些分区；None 表示全部。
        label_map: 类型名 -> id；None 时按字母序自动生成。
    """
    items = load_crawled_items(
        items_dir=items_dir,
        picture_root=picture_root,
        sections=sections,
        require_images=True,
    )
    names = sorted({item.name for item in items})
    if label_map is None:
        label_map = LabelMap.from_list(names)
    sprites: list[Sprite] = []
    for item in items:
        class_id = label_map.id_for(item.name)
        for _level, path in item.images:
            image = Image.open(path).convert("RGBA")
            sprites.append(Sprite(type_name=item.name, class_id=class_id, image=image, box=(0, 0, 0, 0)))
    return sprites


def make_grass_background(width: int = CANVAS_W, height: int = CANVAS_H, seed: int | None = None) -> Image.Image:
    """生成带噪点的草地背景（模仿部落冲突村庄的绿色地块）。"""
    rng = np.random.default_rng(seed)
    base = np.array([88, 148, 64], dtype=np.float32).reshape(1, 1, 3)  # 草绿
    noise = rng.normal(0, 8, (height, width, 1))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    bg = Image.fromarray(arr, "RGB").convert("RGBA")
    # 淡色网格线，模拟村庄地块边界
    draw = ImageDraw.Draw(bg)
    grid = 64
    color = (230, 240, 210, 90)
    for x in range(0, width, grid):
        draw.line((x, 0, x, height), fill=color, width=1)
    for y in range(0, height, grid):
        draw.line((0, y, width, y), fill=color, width=1)
    return bg


def _fit_sprite(sprite: Image.Image, rng: np.random.Generator, canvas_w: int, canvas_h: int) -> Image.Image:
    """随机缩放精灵，使其高度落在 [MIN_SPRITE_H, MAX_SPRITE_H]，并限制宽度不超过画布 1/3。"""
    orig_w, orig_h = sprite.size
    target_h = int(rng.integers(MIN_SPRITE_H, MAX_SPRITE_H + 1))
    scale = target_h / orig_h
    new_w, new_h = int(orig_w * scale), target_h
    max_w = canvas_w // 3
    if new_w > max_w:
        new_w = max_w
        new_h = int(new_h * (max_w / new_w))  # 用缩放后的 new_w 计算比例
        scale = new_h / orig_h
    return sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """两个 (x1,y1,x2,y2) 框的交并比。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _place_sprites(
    sprites: list[Sprite],
    rng: np.random.Generator,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> tuple[Image.Image, list[Sprite]]:
    """把若干精灵贴到草地上，返回场景图与已放置精灵（含 bbox）。"""
    bg = make_grass_background(canvas_w, canvas_h)
    canvas = bg.convert("RGBA")
    placed: list[Sprite] = []
    n_objects = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    max_attempts = 120
    for _ in range(n_objects):
        source = rng.choice(sprites)  # 贴图和标注必须来自同一精灵
        sprite_img = _fit_sprite(source.image, rng, canvas_w, canvas_h)
        sw, sh = sprite_img.size
        placed_box: tuple[int, int, int, int] | None = None
        for _ in range(max_attempts):
            x1 = int(rng.integers(0, canvas_w - sw))
            y1 = int(rng.integers(0, canvas_h - sh))
            candidate = (x1, y1, x1 + sw, y1 + sh)
            if all(_iou(candidate, p.box) <= MAX_OVERLAP_IOU for p in placed):
                placed_box = candidate
                break
        if placed_box is None:
            continue  # 实在找不到位置就跳过该对象
        # 小概率水平翻转（建筑大多近似对称，翻转无副作用）
        if rng.random() < 0.5:
            sprite_img = sprite_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas.alpha_composite(sprite_img, dest=(placed_box[0], placed_box[1]))
        placed.append(Sprite(type_name=source.type_name, class_id=source.class_id, image=sprite_img, box=placed_box))
    return canvas.convert("RGB"), placed


def generate_scenes(
    sprites: list[Sprite],
    out_root: Path | str,
    num_train: int = 1200,
    num_val: int = 300,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    seed: int = 42,
) -> LabelMap:
    """生成训练/验证场景图与 YOLO 标注。返回所用 LabelMap。"""
    out_root = Path(out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_map = LabelMap.from_list(sorted({s.type_name for s in sprites}))
    label_map.save(out_root / "labels.json")

    rng = np.random.default_rng(seed)
    for split, n_scenes in (("train", num_train), ("val", num_val)):
        for i in range(n_scenes):
            scene, placed = _place_sprites(sprites, rng, canvas_w, canvas_h)
            img_path = out_root / "images" / split / f"scene_{i:05d}.png"
            label_path = out_root / "labels" / split / f"scene_{i:05d}.txt"
            scene.save(img_path)
            lines = []
            for p in placed:
                x1, y1, x2, y2 = p.box
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2 / canvas_w
                cy = (y1 + y2) / 2 / canvas_h
                lines.append(f"{p.class_id} {cx:.6f} {cy:.6f} {w / canvas_w:.6f} {h / canvas_h:.6f}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: 生成 {n_scenes} 场景")
    return label_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO detection dataset")
    parser.add_argument("--items-dir", default="dataset/text/items", type=Path)
    parser.add_argument("--picture-root", default="dataset/picture", type=Path)
    parser.add_argument("--out", default="dataset/detection", type=Path)
    parser.add_argument("--train-scenes", type=int, default=1200)
    parser.add_argument("--val-scenes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sprites = load_sprites(args.items_dir, args.picture_root, sections=DEFAULT_SECTIONS)
    print(f"加载精灵 {len(sprites)} 张 / {len({s.type_name for s in sprites})} 类")
    if len({s.type_name for s in sprites}) < 2:
        print("类型不足，无法生成有效的检测数据")
        return 1

    label_map = generate_scenes(
        sprites,
        args.out,
        num_train=args.train_scenes,
        num_val=args.val_scenes,
        seed=args.seed,
    )
    print(f"LabelMap: {len(label_map.names)} 类，保存至 {args.out / 'labels.json'}")
    print(f"数据集就绪：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
