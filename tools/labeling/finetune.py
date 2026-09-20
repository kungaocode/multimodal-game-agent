"""用人工标注的真实截图微调检测器。

流程：把真实标注（dataset/real_labels/）并入合成数据集（dataset/detection/），
用已有权重微调，产出能更好适应真实截图的检测器。

用法：
    # 先标注：python -m tools.labeling.app --port 8001
    # 再微调：
    python -m tools.labeling.finetune \
        --raw dataset/game_picture \
        --labels-dir dataset/real_labels \
        --synthetic dataset/detection \
        --base runs/detect/train/weights/best.pt \
        --epochs 30 --output runs/detect/finetune

说明：
    - 真实标注按 --real-val-ratio 划分：一部分进验证集（作为"真实验收"指标），
      其余进训练集与合成数据混合。
    - 类别映射以标注输出的 labels.json 为准：打资源专用流程用 farm 8 类。

打资源（farm）专用流程：
    # 1. 先构建 farm 合成数据集（6 资源建筑）
    python -m tools.labeling.build_farm_dataset
    # 2. 标注工具切到 farm 8 类
    python -m tools.labeling.app --port 8001
    # 3. 微调 farm 检测器
    python -m tools.labeling.finetune \
        --labels-dir dataset/real_labels \
        --synthetic dataset/farm_synthetic \
        --base runs/detect/train/weights/best.pt \
        --out runs/detect/farm --epochs 30
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from perception.dataset import LabelMap, build_yolo_dataset_yaml
from perception.detector import Detector


def _copy_merge(src_img_dir: Path, src_lab_dir: Path, dst_img_dir: Path, dst_lab_dir: Path) -> int:
    """把 src 下的图片+同 stem 标签复制到 dst；返回复制的图片数。"""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img in src_img_dir.glob(ext):
            lab = src_lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue  # 未标注的截图不参与训练
            shutil.copy2(img, dst_img_dir / img.name)
            shutil.copy2(lab, dst_lab_dir / lab.name)
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用真实标注微调检测器")
    parser.add_argument("--raw", default="dataset/game_picture", type=Path, help="真实截图目录")
    parser.add_argument("--labels-dir", default="dataset/real_labels", type=Path, help="标注输出目录")
    parser.add_argument("--synthetic", default="dataset/detection", type=Path, help="合成数据集根目录")
    parser.add_argument("--base", default="runs/detect/train/weights/best.pt", type=Path, help="初始权重")
    parser.add_argument("--out", default="runs/detect/finetune", type=Path, help="输出目录")
    parser.add_argument("--real-val-ratio", type=float, default=0.25, help="真实标注划入验证集的比例")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    real_lab_dir = args.labels_dir / "labels"
    real_label_map = LabelMap.load(args.labels_dir / "labels.json")

    # 真实标注按比例分 train/val
    real_imgs = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg") for p in args.raw.glob(ext)
        if (real_lab_dir / f"{p.stem}.txt").exists()
    )
    if not real_imgs:
        print("没有找到已标注的真实截图。请先在标注工具里保存标注。")
        return 1
    rng_imgs = real_imgs
    import random

    random.Random(args.seed).shuffle(rng_imgs)
    n_val = max(1, int(len(rng_imgs) * args.real_val_ratio))
    real_val, real_train = rng_imgs[:n_val], rng_imgs[n_val:]

    # 逐张复制真实图+标注到独立临时目录，再并入对应 split
    n_train = 0
    for split, img_list in (("train", real_train), ("val", real_val)):
        tmp_img = out_root / f"_real_{split}_img"
        tmp_lab = out_root / f"_real_{split}_lab"
        tmp_img.mkdir(parents=True, exist_ok=True)
        tmp_lab.mkdir(parents=True, exist_ok=True)
        for img in img_list:
            shutil.copy2(img, tmp_img / img.name)
            shutil.copy2(real_lab_dir / f"{img.stem}.txt", tmp_lab / f"{img.stem}.txt")
        n = _copy_merge(tmp_img, tmp_lab, out_root / "images" / split, out_root / "labels" / split)
        shutil.rmtree(tmp_img, ignore_errors=True)
        shutil.rmtree(tmp_lab, ignore_errors=True)
        n_train += n

    # 并入合成数据集
    for split in ("train", "val"):
        n_syn = _copy_merge(
            args.synthetic / "images" / split,
            args.synthetic / "labels" / split,
            out_root / "images" / split,
            out_root / "labels" / split,
        )
        if split == "train":
            n_train += n_syn
        print(f"{split}: 真实 {len(img_list)} 张 + 合成 {n_syn} 张")

    real_label_map.save(out_root / "labels.json")
    yaml_path = build_yolo_dataset_yaml(
        root=out_root,
        train_dir="images/train",
        val_dir="images/val",
        label_map=real_label_map,
        output_path=out_root / "dataset.yaml",
    )
    print(f"合并完成：{n_train} 张训练图，验证含 {len(real_val)} 张真实截图")

    detector = Detector(args.base)
    best = detector.train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(out_root.parent.resolve()),
        name=out_root.name,
        exist_ok=True,
    )
    print(f"微调完成，最佳权重: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
