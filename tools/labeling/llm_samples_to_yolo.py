"""把 LLM 回流样本（dataset/llm_samples/*.json）转成 YOLO 检测标注（阶段 6/14 深化）。

背景：级联感知（perception/cascade.py）在本地置信度不足时调用 qwen3-vl-plus，
把成功样本（截图 + 提示词 + 带像素坐标的 objects 输出）回流到 dataset/llm_samples/。
这些「教师」标注可直接转成 YOLO 检测标注，用于微调本地 FSM 检测器
（大模型蒸馏 → 本地模型成本趋零），即用户方案中的「先调用视觉大模型 → 回流样本 → 常规分类器学习运用」。

坐标约定：
- LLM 提示词以「画面 1280x720」逻辑画布要求输出像素坐标；
- LlmSampleRecorder 保存回流图时把宽压缩到 1280（高按比例）→ 图片实际尺寸可能不是 1280x720；
- 转换时先按 1280x720 归一化，再乘图片实际宽高得到原图像素中心；
- LLM 只输出中心点无 bbox → 用每类先验框尺寸（--prior 指定，默认取自 farm 合成集统计）。

过滤规则（LLM 估算坐标有噪声，防污染训练集）：
- 类型不在 FARM 8 类里（资源 6 类 + 进攻/返回按钮）→ 跳过
- coords 缺失 / 非数字 / 0,0 → 跳过
- confidence < --min-conf（默认 0.5）→ 跳过
- 归一化中心点在图片外 → 跳过
- 框超出图片边界 → 钳制

去重：
- --dedup 按图片字节哈希去重（默认开启）——同一秒重复记录的联调样本只保留一份。

用法：
    python -m tools.labeling.llm_samples_to_yolo \
        --samples dataset/llm_samples --out dataset/llm_finetune \
        --min-conf 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from perception.dataset import LabelMap, build_yolo_dataset_yaml

# 与 dataset/farm_synthetic/labels.json 一致（id 即下标）
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
NAME_TO_ID = {name: i for i, name in enumerate(FARM_ORDER)}

# 逻辑画布（与 CASCADE_PROMPT 一致）
LOGICAL_W, LOGICAL_H = 1280, 720

# 每类先验框尺寸（归一化宽、高）。
# 资源 6 类来自 farm_synthetic 标注统计；按钮类合成集无标注，取自游戏 UI 经验值。
PRIOR_BOX = {
    "圣水收集器": (0.0866, 0.2004),
    "金矿": (0.1308, 0.2036),
    "圣水瓶": (0.1236, 0.1993),
    "储金罐": (0.1070, 0.2018),
    "暗黑重油罐": (0.1038, 0.1949),
    "暗黑重油钻井": (0.0958, 0.2075),
    "进攻按钮": (0.1300, 0.0900),
    "返回按钮": (0.0600, 0.0900),
}


def _parse_coords(coords: list | tuple) -> tuple[float, float] | None:
    """解析 LLM 输出 coords；非法（缺失/非数字/0,0）返回 None。"""
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    try:
        x, y = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    if x <= 0 or y <= 0:
        return None
    return x, y


def _image_digest(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def llm_meta_to_yolo(
    meta: dict,
    image_path: Path,
    min_conf: float,
    box_scale: float,
) -> list[list[float]] | None:
    """把一个回流样本转成 YOLO 标注行列表；无有效框返回 None。"""
    response = meta.get("response")
    if not isinstance(response, dict):
        return None
    objects = response.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return None

    img = Image.open(image_path)
    w, h = img.size
    lines: list[list[float]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("type")
        if name not in NAME_TO_ID:
            continue
        try:
            conf = float(obj.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_conf:
            continue
        coords = _parse_coords(obj.get("coords"))
        if coords is None:
            continue
        lx, ly = coords
        # 逻辑画布 → 归一化 → 原图像素
        xc_n, yc_n = lx / LOGICAL_W, ly / LOGICAL_H
        cx, cy = xc_n * w, yc_n * h
        if not (0 < cx < w and 0 < cy < h):
            continue
        pw, ph = PRIOR_BOX.get(name, (0.10, 0.10))
        bw, bh = pw * box_scale * w, ph * box_scale * h
        x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
        x2, y2 = min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        # 转回归一化 YOLO 格式
        lines.append(
            [
                float(NAME_TO_ID[name]),
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            ]
        )
    return lines or None


def convert(
    samples_dir: Path,
    out_root: Path,
    min_conf: float = 0.5,
    box_scale: float = 1.0,
    dedup: bool = True,
) -> dict:
    """转换所有回流样本；返回统计信息。"""
    samples_dir = Path(samples_dir)
    out_root = Path(out_root)
    img_dir = out_root / "images"
    lab_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_meta, n_unique, n_images, n_boxes, n_skipped = 0, 0, 0, 0, 0
    per_class: dict[str, int] = defaultdict(int)
    skipped_reasons = {"no_objects": 0, "overlap": 0, "mismatch": 0}

    for meta_f in sorted(samples_dir.glob("*.json")):
        n_meta += 1
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            n_skipped += 1
            continue
        img_name = meta.get("image")
        if not img_name:
            n_skipped += 1
            continue
        image_path = samples_dir / img_name
        if not image_path.exists():
            skipped_reasons["mismatch"] += 1
            continue
        if dedup:
            digest = _image_digest(image_path)
            if digest in seen:
                skipped_reasons["overlap"] += 1
                continue
            seen.add(digest)
        n_unique += 1

        lines = llm_meta_to_yolo(meta, image_path, min_conf, box_scale)
        if lines is None:
            skipped_reasons["no_objects"] += 1
            continue
        # 输出图片（YOLO 训练输入不做二次压缩，直接引用副本）
        stem = meta_f.stem
        dst_img = img_dir / f"{stem}.jpg"
        shutil.copy2(image_path, dst_img)
        lab_dir.joinpath(f"{dst_img.stem}.txt").write_text(
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
            per_class[FARM_ORDER[int(line[0])]] += 1

    # 写 labels.json + dataset.yaml
    LabelMap.from_list(FARM_ORDER).save(out_root / "labels.json")
    build_yolo_dataset_yaml(
        root=out_root,
        train_dir="images",
        val_dir="images",
        label_map=LabelMap.from_list(FARM_ORDER),
        output_path=out_root / "dataset.yaml",
    )
    return {
        "meta_files": n_meta,
        "unique_images": n_unique,
        "converted_images": n_images,
        "boxes": n_boxes,
        "skipped": n_skipped,
        "skipped_reasons": dict(skipped_reasons),
        "per_class": dict(per_class),
        "output": str(out_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 回流样本 → YOLO 训练标注")
    parser.add_argument("--samples", default="dataset/llm_samples", type=Path)
    parser.add_argument("--out", default="dataset/llm_finetune", type=Path)
    parser.add_argument("--min-conf", type=float, default=0.5, help="LLM 置信度过滤阈值")
    parser.add_argument("--box-scale", type=float, default=1.0, help="先验框尺寸缩放（中心点无 bbox，用先验框近似）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭按图片去重")
    args = parser.parse_args(argv)

    stats = convert(
        args.samples,
        args.out,
        min_conf=args.min_conf,
        box_scale=args.box_scale,
        dedup=not args.no_dedup,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
