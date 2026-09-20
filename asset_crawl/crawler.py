"""爬虫编排：抓取索引 → 逐个详情 → 下载图片 → 输出 JSON。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import unquote

from .client import CrawlError, HttpClient
from .live import find_detail_chunk, find_index_chunk, parse_detail_chunk, parse_index_chunk
from .models import INDEX_URL, DetailData, IndexData, ItemRef
from .parsers import parse_detail_markdown, parse_index_markdown


class HomeCrawler:
    """家乡（Home Village）数据爬虫。

    source:
      - "live"：爬线上 https://clashpost.com/upgrade/category/home
      - "repo"：解析本地 GitHub 仓库源 markdown（--repo-path 指定 docs/upgrade 目录）
    """

    def __init__(
        self,
        *,
        source: str = "live",
        output_dir: str | Path = "dataset/text",
        images_dir: str | Path = "dataset/picture",
        repo_upgrade_dir: str | Path = ".reference-clashpost/docs/upgrade",
        http: HttpClient | None = None,
        limit: int | None = None,
        download_images: bool = True,
    ) -> None:
        if source not in ("live", "repo"):
            raise ValueError(f"未知数据源: {source}，可选 live / repo")
        self.source = source
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.repo_upgrade_dir = Path(repo_upgrade_dir)
        self._http = http or HttpClient()
        self.limit = limit
        self.download_images = download_images

        self.items_dir = self.output_dir / "items"

    # ---- 对外入口 ----

    def run(self) -> IndexData:
        index = self._load_index()
        index_path = self.output_dir / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(index.model_dump(), fh, ensure_ascii=False, indent=2)

        items = index.iter_items()
        if self.limit is not None:
            items = items[: self.limit]

        success = 0
        failed = 0
        for item in items:
            try:
                detail = self._load_detail(item, index)
            except CrawlError as exc:
                failed += 1
                print(f"[FAIL] {item.link}: {exc}")
                continue
            self._save_detail(detail)
            if self.download_images:
                self._download_detail_images(detail)
            success += 1
            print(f"[ OK ] {item.link}  {item.name}")

        print(
            f"\n完成：共 {len(items)} 条，成功 {success}，失败 {failed}。"
            f"索引: {index_path}"
        )
        return index

    # ---- 索引 ----

    def _load_index(self) -> IndexData:
        if self.source == "repo":
            home_md = self.repo_upgrade_dir / "category" / "home.md"
            if not home_md.exists():
                raise FileNotFoundError(
                    f"未找到仓库索引文件: {home_md}\n"
                    "请先克隆仓库并确认路径，或改用 --source live。"
                )
            with home_md.open("r", encoding="utf-8") as fh:
                return parse_index_markdown(fh.read(), source="repo")
        # live
        html = self._http.get_text(INDEX_URL)
        chunk_path = find_index_chunk(html)
        chunk = self._http.get_text(f"https://clashpost.com{chunk_path}")
        return parse_index_chunk(chunk, source="live")

    # ---- 详情 ----

    def _load_detail(self, item: ItemRef, index: IndexData) -> DetailData:
        tab, section = _locate(item, index)
        if self.source == "repo":
            # 索引里的 link 是 URL 编码形式（如 Builder%27s），文件名是字面量形式
            md_path = self.repo_upgrade_dir / f"{unquote(item.link)}.md"
            if not md_path.exists():
                raise CrawlError(f"仓库中缺少详情文件: {md_path}")
            with md_path.open("r", encoding="utf-8") as fh:
                detail = parse_detail_markdown(
                    fh.read(), item.link, tab=tab, section=section
                )
        else:
            # live
            html = self._http.get_text(item.url)
            chunk_path = find_detail_chunk(html, item.link)
            chunk = self._http.get_text(f"https://clashpost.com{chunk_path}")
            detail = parse_detail_chunk(chunk, item.link, tab=tab, section=section)
        _trim_hero_levels(detail)
        return detail

    def _save_detail(self, detail: DetailData) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        path = self.items_dir / f"{detail.link}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(detail.model_dump(), fh, ensure_ascii=False, indent=2)

    def _download_detail_images(self, detail: DetailData) -> None:
        """下载详情页的全部图片：主图 + 等级图片。"""
        targets: list[tuple[str, str]] = []
        if detail.info_image:
            targets.append((detail.info_image.src, detail.info_image.url))
        for group in detail.image_groups:
            for img in group.images:
                targets.append((img.src, img.url))
        for src, url in targets:
            dest = self.images_dir / detail.img_folder / src
            if dest.exists():
                continue
            try:
                self._http.download(url, dest)
            except CrawlError as exc:
                print(f"      [IMG FAIL] {url}: {exc}")
            time.sleep(0.1)  # 静态 CDN 也保持轻节奏，避免触发限流


def _locate(item: ItemRef, index: IndexData) -> tuple[str, str]:
    """返回条目所属的 (标签页名, 分组名)。"""
    for tab in index.tabs:
        for section in tab.sections:
            if any(it.link == item.link for it in section.items):
                return tab.name, section.title
    return "", ""


def _trim_hero_levels(detail: DetailData) -> None:
    """英雄仅需识别，不需要等级细节（升级后外观不变，视觉模块用不到）。

    保留：名称、描述、主图、静态属性；
    裁剪：等级图片、升级表、时间线、可建造数量、说明。
    """
    if detail.tab != "英雄":
        return
    detail.image_groups = []
    detail.tables = []
    detail.timeline = []
    detail.building_num = []
    detail.notes = []
