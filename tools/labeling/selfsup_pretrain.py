"""Self-supervised pretraining: extract DINOv2 features + k-means clustering.

Purpose: discover visual states in unlabeled video frames without any labels.
Each cluster likely corresponds to one game state (village, searching, battle, donate...).
Send 1 representative frame per cluster to LLM for labeling, propagate to all frames
in that cluster. This reduces LLM API calls from N frames to ~num_clusters.

Usage:
    # Extract features and cluster:
    python -m tools.labeling.selfsup_pretrain --images dataset/frames --out dataset/clusters

    # Custom clusters:
    python -m tools.labeling.selfsup_pretrain --images dataset/frames --out dataset/clusters --clusters 10

Output:
    dataset/clusters/
      features.npy          # DINOv2 feature vectors
      assignments.npy       # cluster id per frame
      cluster_summary.json  # cluster stats + representative frame index
      representatives/      # representative frame images per cluster
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _collect_images(image_dir: Path, max_images: int = 2000) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if len(paths) > max_images:
        step = len(paths) / max_images
        paths = [paths[int(i * step)] for i in range(max_images)]
    return paths


def extract_dinov2_features(
    image_paths: list[Path],
    model_name: str = "facebook/dinov2-small",
    batch_size: int = 16,
) -> np.ndarray:
    """Extract DINOv2 CLS features for a list of images.

    Returns: (N, D) float32 array. D=384 for dinov2-small, 768 for dinov2-base.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading DINOv2 model: %s (device=%s)", model_name, device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    for module in model.modules():
        module.training = False

    features_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token
            features_list.append(cls_features.cpu().numpy())
            if (i + batch_size) % 100 < batch_size:
                logger.info("  features: %d / %d", min(i + batch_size, len(image_paths)), len(image_paths))
    features = np.concatenate(features_list, axis=0).astype(np.float32)
    logger.info("Extracted features: shape=%s", features.shape)
    return features


def kmeans_cluster(features: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """K-means clustering on L2-normalized features."""
    from sklearn.cluster import KMeans

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = features / norms
    logger.info("K-means clustering: %d frames -> %d clusters", len(features), n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    assignments = km.fit_predict(normalized)
    logger.info("Cluster sizes: %s", np.bincount(assignments).tolist())
    return assignments


def select_representatives(
    features: np.ndarray,
    assignments: np.ndarray,
    image_paths: list[Path],
    out_dir: Path,
) -> dict[int, dict]:
    """Select the frame closest to each cluster centroid as representative."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[int, dict] = {}
    for cid in range(assignments.max() + 1):
        member_indices = np.where(assignments == cid)[0]
        if len(member_indices) == 0:
            continue
        # Centroid = mean of member features
        centroid = features[member_indices].mean(axis=0)
        # Find closest member to centroid
        dists = np.linalg.norm(features[member_indices] - centroid, axis=1)
        rep_idx = member_indices[np.argmin(dists)]
        rep_path = image_paths[rep_idx]
        # Save representative image
        rep_img = Image.open(rep_path).convert("RGB")
        rep_img.save(out_dir / f"cluster_{cid:03d}_rep.jpg", format="JPEG", quality=92)
        summary[cid] = {
            "size": int(len(member_indices)),
            "representative_frame": str(rep_path),
            "representative_index": int(rep_idx),
            "member_indices": member_indices.tolist()[:50],  # first 50 for reference
        }
        logger.info("Cluster %d: %d frames, rep=%s", cid, len(member_indices), rep_path.name)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-supervised pretraining (DINOv2 + clustering)")
    parser.add_argument("--images", type=Path, required=True, help="Directory of unlabeled frames")
    parser.add_argument("--out", type=Path, default=Path("dataset/clusters"))
    parser.add_argument("--clusters", type=int, default=8, help="Number of k-means clusters")
    parser.add_argument("--model", type=str, default="facebook/dinov2-small")
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    image_paths = _collect_images(args.images, args.max_images)
    if not image_paths:
        logger.error("No images found in %s", args.images)
        return 1
    logger.info("Found %d images", len(image_paths))

    features = extract_dinov2_features(image_paths, model_name=args.model, batch_size=args.batch_size)
    assignments = kmeans_cluster(features, args.clusters, seed=args.seed)
    summary = select_representatives(features, assignments, image_paths, args.out / "representatives")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "features.npy", features)
    np.save(args.out / "assignments.npy", assignments)
    (args.out / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== Self-supervised pretraining summary =====")
    print(f"Images:         {len(image_paths)}")
    print(f"Feature dim:    {features.shape[1]}")
    print(f"Clusters:       {args.clusters}")
    print(f"Output:         {args.out}")
    print("Representative images saved to:", args.out / "representatives")
    print("Next step: label representatives with LLM, then propagate to cluster members.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
