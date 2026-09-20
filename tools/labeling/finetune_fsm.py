"""FSM 识别器微调：合并 6 类合成 + LLM 回流 + OCR 按钮真实标注 → 12 类本地检测器。

背景：farm_synthetic 只有 6 个资源建筑类（按钮类无数据），导致模型无法识别状态转换
标志。本脚本把三类数据合并成 perception/fsm_labels.py 的 12 类体系并微调：

    0-5   资源建筑（来自 farm_synthetic，1200+260 张合成）
    6-11  按钮类（来自 OCR 半自动标注 fsm_real_buttons / LLM 回流 llm_finetune）

用法：
    # 1. 先转换回流样本与 OCR 按钮标注（可选，已有则跳过）
    python -m tools.labeling.llm_samples_to_yolo --samples dataset/llm_samples --out dataset/llm_finetune
    python -m tools.labeling.ocr_buttons_to_yolo --images final_test_dataset dataset/game_picture \
        --out dataset/fsm_real_buttons

    # 2. 合并微调
    python -m tools.labeling.finetune_fsm \
        --synthetic dataset/farm_synthetic \
        --llm dataset/llm_finetune \
        --buttons dataset/fsm_real_buttons \
        --base runs/detect/farm/weights/best.pt --epochs 20 --out runs/detect/fsm

说明：
    - 按钮类真实标注量少时，按 --buttons-val-ratio 把一部分划入验证集做「真实验收」；
    - 合成集带按钮类 0 正样本，仅作为资源建筑学习的基础（背景负样本）。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from perception.dataset import LabelMap, build_yolo_dataset_yaml
from perception.detector import Detector
from perception.fsm_labels import FSM_ORDER


def _copy(src_root: Path, split: str, dst_img_dir: Path, dst_lab_dir: Path, limit: int | None = None) -> int:
    """把 src_root/images/<split> 图片 + labels/<split> 标签复制到目标目录；返回张数。"""
    src_img_dir = src_root / "images" / split
    src_lab_dir = src_root / "labels" / split
    if not src_img_dir.exists():
        return 0
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in sorted(src_img_dir.glob(ext)):
            lab = src_lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            if limit is not None and n >= limit:
                return n
            dst_img = dst_img_dir / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not (dst_lab_dir / f"{img.stem}.txt").exists():
                shutil.copy2(lab, dst_lab_dir / f"{img.stem}.txt")
            n += 1
    return n


def _copy_flat(src_root: Path, dst_img_dir: Path, dst_lab_dir: Path) -> int:
    """复制扁平结构源（images/ + labels/ 无 train/val 子目录）。"""
    src_img_dir = src_root / "images"
    src_lab_dir = src_root / "labels"
    if not src_img_dir.exists():
        return 0
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in sorted(src_img_dir.glob(ext)):
            lab = src_lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            dst_img = dst_img_dir / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not (dst_lab_dir / f"{img.stem}.txt").exists():
                shutil.copy2(lab, dst_lab_dir / f"{img.stem}.txt")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FSM 识别器 12 类合并微调")
    parser.add_argument("--synthetic", default="dataset/farm_synthetic", type=Path, help="6 类合成集")
    parser.add_argument("--llm", default="dataset/llm_finetune", type=Path, help="LLM 回流转换产物")
    parser.add_argument("--buttons", default="dataset/fsm_real_buttons", type=Path, help="OCR 按钮标注")
    parser.add_argument("--base", default="yolov8n.pt", type=str, help="初始权重")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="runs/detect/fsm", type=Path)
    parser.add_argument("--buttons-val-ratio", type=float, default=0.3, help="按钮真实标注划入验证集的比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    # 1) 合成集：已有 train/val 划分
    n = _copy(args.synthetic, "train", out_root / "images/train", out_root / "labels/train")
    summary["synthetic_train"] = n
    n = _copy(args.synthetic, "val", out_root / "images/val", out_root / "labels/val")
    summary["synthetic_val"] = n

    # 2) LLM 回流：全部进训练
    if args.llm.exists():
        summary["llm_train"] = _copy_flat(args.llm, out_root / "images/train", out_root / "labels/train")

    # 3) OCR 按钮真实标注：按比例分 train/val（真实验收）
    if args.buttons.exists():
        imgs = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in (args.buttons / "images").glob(ext)
            if (args.buttons / "labels" / f"{p.stem}.txt").exists()
        )
        rng = random.Random(args.seed)
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.buttons_val_ratio)) if imgs else 0
        for i, img in enumerate(imgs):
            lab = args.buttons / "labels" / f"{img.stem}.txt"
            split = "val" if i < n_val else "train"
            dst_img = out_root / "images" / split / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not (out_root / "labels" / split / f"{img.stem}.txt").exists():
                shutil.copy2(lab, out_root / "labels" / split / f"{img.stem}.txt")
        summary["buttons_train"] = len(imgs) - n_val
        summary["buttons_val"] = n_val
    else:
        summary["buttons_train"] = summary["buttons_val"] = 0

    # labels.json + dataset.yaml
    LabelMap.from_list(FSM_ORDER).save(out_root / "labels.json")
    yaml_path = build_yolo_dataset_yaml(
        root=out_root,
        train_dir="images/train",
        val_dir="images/val",
        label_map=LabelMap.from_list(FSM_ORDER),
        output_path=out_root / "dataset.yaml",
    )
    print("数据合并统计:", summary)
    print("dataset.yaml:", yaml_path)

    detector = Detector(args.base)
    best = detector.train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(out_root.parent.resolve()),
        name=out_root.name,
        exist_ok=True,
    )
    print(f"微调完成，最佳权重: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
