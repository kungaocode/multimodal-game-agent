"""用 OCR 文本位置半自动生成「按钮类」YOLO 标注（弥补合成数据无按钮的缺口）。

合成数据集（dataset/farm_synthetic）只有 6 个资源建筑类，没有任何按钮标注，
导致训练出的 farm 模型无法识别状态转换信号（进攻按钮/返回按钮/下一个/结束战斗/增援）。
本工具在真实截图目录上跑 PaddleOCR，把识别到的按钮文案（进攻/回营/返回/搜索对手/
下一个/结束战斗/增援）转成 YOLO 框标注，快速扩充按钮类训练数据。

体验：通常按钮文字与按钮本身形状不一致（文字在按钮图案上），OCR bbox 是文字区域，
默认对 bbox 做 --pad 外扩（上下左右各扩 pad 像素）以覆盖整个按钮。之后可配合
tools/labeling/app.py 人工校正。也支持 --prior 模式：改用每类先验框尺寸（宽度从先验
乘以按钮文本宽度推算）。

用法：
    python -m tools.labeling.ocr_buttons_to_yolo \
        --images final_test_dataset dataset/game_picture \
        --out dataset/fsm_real_buttons \
        --pad 12
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from perception.dataset import LabelMap, build_yolo_dataset_yaml
from perception.ocr import OCRReader
from perception.fsm_labels import FSM_NAME_TO_ID, FSM_ORDER, class_id_for_button_text


def ocr_buttons_to_labels(
    image_path: Path,
    ocr: OCRReader,
    pad: int = 12,
    min_conf: float = 0.5,
    prior_mode: bool = False,
) -> list[list[float]] | None:
    """跑 OCR 并转 YOLO 行；返回 None 表示无按钮。"""
    results = ocr.read_text(image_path)
    img = Image.open(image_path)
    w, h = img.size
    lines: list[list[float]] = []
    for r in results:
        if r.confidence < min_conf:
            continue
        cid = class_id_for_button_text(r.text)
        if cid is None:
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if not prior_mode:
            # 文本框外扩整按钮
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
        else:
            # 先验框宽 = 最大(文本宽*2.2, 先验尺寸*图宽)，中心对齐文本
            from perception.fsm_labels import PRIOR_BOX

            pw_norm, _ph_norm = PRIOR_BOX[FSM_ORDER[cid]]
            bw = max((x2 - x1) * 2.2, pw_norm * w)
            bh = (y2 - y1) * 1.4
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x1, y1 = max(0, cx - bw / 2), max(0, cy - bh / 2)
            x2, y2 = min(w, cx + bw / 2), min(h, cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            [
                float(cid),
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            ]
        )
    return lines or None


def convert(images_dirs: list[Path], out_root: Path, pad: int, min_conf: float, prior_mode: bool) -> dict:
    out_root = Path(out_root)
    img_dir = out_root / "images"
    lab_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    ocr = OCRReader(lang="ch")
    per_class: dict[str, int] = {}
    n_images = n_boxes = n_no_button = 0
    for d in images_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in sorted(Path(d).glob(ext)):
                lines = ocr_buttons_to_labels(img, ocr, pad=pad, min_conf=min_conf, prior_mode=prior_mode)
                if lines is None:
                    n_no_button += 1
                    continue
                src = Image.open(img)
                dst = img_dir / f"{img.stem}.jpg"
                src.convert("RGB").save(dst, format="JPEG", quality=92)
                lab_dir.joinpath(f"{img.stem}.txt").write_text(
                    "\n".join(
                        f"{int(line[0])} {line[1]:.6f} {line[2]:.6f} {line[3]:.6f} {line[4]:.6f}"
                        for line in lines
                    )
                    + "\n",
                    encoding="utf-8",
                )
                n_images += 1
                n_boxes += len(lines)
                for line in lines:
                    name = FSM_ORDER[int(line[0])]
                    per_class[name] = per_class.get(name, 0) + 1
    LabelMap.from_list(FSM_ORDER).save(out_root / "labels.json")
    build_yolo_dataset_yaml(
        root=out_root,
        train_dir="images",
        val_dir="images",
        label_map=LabelMap.from_list(FSM_ORDER),
        output_path=out_root / "dataset.yaml",
    )
    return {
        "images_with_buttons": n_images,
        "no_button_images": n_no_button,
        "boxes": n_boxes,
        "per_class": per_class,
        "output": str(out_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR 文本 → 按钮类 YOLO 标注")
    parser.add_argument("--images", nargs="+", required=True, type=Path, help="真实截图目录（可多个）")
    parser.add_argument("--out", default="dataset/fsm_real_buttons", type=Path)
    parser.add_argument("--pad", type=int, default=12, help="文本框外扩像素（--prior 时忽略）")
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--prior", action="store_true", help="用每类先验框尺寸代替简单外扩")
    args = parser.parse_args(argv)

    stats = convert(args.images, args.out, pad=args.pad, min_conf=args.min_conf, prior_mode=args.prior)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
