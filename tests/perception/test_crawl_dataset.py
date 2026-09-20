"""Tests for crawled dataset loader."""

from pathlib import Path

import pytest

from perception.crawl_dataset import (
    build_label_maps,
    dataset_stats,
    load_crawled_items,
    parse_level,
    split_train_val,
)

DATASET_ITEMS = Path("dataset/text/items")
DATASET_PICTURE = Path("dataset/picture")


def test_parse_level():
    assert parse_level("1 级") == 1
    assert parse_level("12 级") == 12
    assert parse_level("info") is None
    assert parse_level("") is None
    # 区间标签取起始级，而非把所有数字拼接
    assert parse_level("1 - 2 级") == 1
    assert parse_level("3 - 4 级") == 3
    # 大本标签取大本数
    assert parse_level("17 本 1 星") == 17
    # 特殊状态标签无数字 -> None
    assert parse_level("建造中") is None
    assert parse_level("废墟") is None


def test_load_building_items():
    sections = {"资源类建筑", "防御建筑"}
    items = load_crawled_items(
        DATASET_ITEMS,
        picture_root=DATASET_PICTURE,
        sections=sections,
    )
    assert len(items) > 0
    assert all(item.section in sections for item in items)
    assert all(len(item.images) > 0 for item in items)
    for item in items:
        for level, img_path in item.images:
            assert isinstance(level, int)
            assert img_path.exists()


def test_build_label_maps_and_split():
    sections = {"资源类建筑", "防御建筑"}
    items = load_crawled_items(
        DATASET_ITEMS,
        picture_root=DATASET_PICTURE,
        sections=sections,
    )
    type_to_id, id_to_type, max_level = build_label_maps(items)
    assert len(type_to_id) == len(items)
    assert max_level >= 1
    train, val = split_train_val(items, type_to_id, val_ratio=0.2, seed=42)
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == sum(len(i.images) for i in items)


def test_dataset_stats():
    sections = {"资源类建筑"}
    items = load_crawled_items(
        DATASET_ITEMS,
        picture_root=DATASET_PICTURE,
        sections=sections,
    )
    stats = dataset_stats(items)
    assert stats["num_items"] == len(items)
    assert stats["num_images"] > 0
    assert "资源类建筑" in stats["sections"]
