"""构建打资源专用数据集（farm 数据集）。

打资源循环只需要一个轻量检测器认 8 类：
    6 个资源建筑：圣水收集器 / 金矿 / 圣水瓶 / 储金罐 / 暗黑重油罐 / 暗黑重油钻井
    2 个 UI 按钮：进攻按钮 / 返回按钮

产出：
    dataset/farm_labels.json       8 类 LabelMap（标注工具 --labels 直接用这个）
    dataset/farm_synthetic/        从 43 类合成数据过滤重映射，只留 6 资源建筑
                                   （场景里其他建筑保留为背景负样本）

用法：
    python -m tools.labeling.build_farm_dataset \
        --source dataset/detection --out dataset/farm_synthetic
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from perception.dataset import LabelMap

# 顺序即 id：0 圣水收集器，1 金矿，2 圣水瓶，3 储金罐，4 暗黑重油罐，5 暗黑重油钻井，6 进攻按钮，7 返回按钮
FARM_ORDER = [
    "圣水收集器",
    "金矿",
    "圣水瓶",
    "储金罐",
    "暗黑重油罐",
    "暗黑重油钻井",
    "进攻按钮",
    "返回按钮",
]

# 合成数据里只有建筑：保留这 6 个资源建筑，id 即它们在 FARM_ORDER 里的下标
_RESOURCE_NAMES = FARM_ORDER[:6]


def filter_synthetic(
    src_root: Path,
    dst_root: Path,
    max_background: float = 1.0,
) -> dict:
    """把 43 类合成数据集过滤成 6 资源建筑（重映射 id）。

    - 源标注里不属于 6 资源建筑的框直接丢弃，场景保留为背景负样本；
    - `max_background` 限制纯背景场景占正样本的比例（默认 1.0 = 不限制）。
    """
    keep = {name: i for i, name in enumerate(_RESOURCE_NAMES)}
    src_lm = LabelMap.load(src_root / "labels.json")
    dst_lm = LabelMap.from_list(FARM_ORDER)

    dst_root.mkdir(parents=True, exist_ok=True)
    dst_lm.save(dst_root / "labels.json")

    stats: dict = {"train": {}, "val": {}}
    for split in ("train", "val"):
        src_img_dir = src_root / "images" / split
        src_lab_dir = src_root / "labels" / split
        dst_img_dir = dst_root / "images" / split
        dst_lab_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lab_dir.mkdir(parents=True, exist_ok=True)

        counts: Counter[str] = Counter()
        n_images = 0
        n_background = 0
        for img in src_img_dir.glob("*.png"):
            lab = src_lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue  # 源数据里没标注的图不参与
            new_lines: list[str] = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid, xc, yc, w, h = map(float, parts)
                name = src_lm.names.get(int(cid))
                if name is None or name not in keep:
                    continue
                new_lines.append(f"{keep[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                counts[name] += 1

            if not new_lines:
                n_background += 1
                if n_background > max_background * max(1, n_images):
                    continue  # 背景负样本过多则跳过该场景
            shutil.copy2(img, dst_img_dir / img.name)
            dst_lab_dir.joinpath(f"{img.stem}.txt").write_text(
                "\n".join(new_lines) + "\n" if new_lines else "", encoding="utf-8"
            )
            n_images += 1

        stats[split] = {
            "images": n_images,
            "background_only": n_background,
            "boxes": sum(counts.values()),
            "per_class": dict(counts),
        }
    stats["farm_classes"] = FARM_ORDER
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建打资源专用 farm 数据集")
    parser.add_argument("--source", default="dataset/detection", type=Path, help="43 类合成数据集根目录")
    parser.add_argument("--out", default="dataset/farm_synthetic", type=Path, help="输出目录")
    parser.add_argument("--max-background", type=float, default=1.0, help="纯背景场景允许的上限（相对正样本比例）")
    args = parser.parse_args(argv)

    stats = filter_synthetic(args.source, args.out, max_background=args.max_background)
    for split, s in stats.items():
        if split == "farm_classes":
            continue
        print(f"[{split}] 图 {s['images']} 张（纯背景 {s['background_only']} 张）框 {s['boxes']} 个")
        for name, n in s["per_class"].items():
            print(f"    {name}: {n}")
    print(f"8 类标签: {args.out / 'labels.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
