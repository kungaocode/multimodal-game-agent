"""命令行入口：python -m asset_crawl --help"""

from __future__ import annotations

import argparse

from .client import HttpClient
from .crawler import HomeCrawler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m asset_crawl",
        description="爬取《部落冲突》家乡（Home Village）的建筑/兵种/法术数据",
    )
    parser.add_argument(
        "--source",
        choices=["live", "repo"],
        default="live",
        help=(
            "数据源：live=爬取线上 clashpost.com；"
            "repo=解析本地 GitHub 仓库 markdown（更稳定，推荐）"
        ),
    )
    parser.add_argument(
        "--repo-path",
        default=".reference-clashpost/docs/upgrade",
        help="repo 源使用的仓库 upgrade 目录（包含 category/home.md 与各详情 .md）",
    )
    parser.add_argument(
        "--output",
        default="dataset/text",
        help="文本 JSON 输出目录（index.json / items/*.json）",
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/picture",
        help="图片输出目录（按 img_folder 分目录）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只抓取前 N 个条目（测试用）",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="跳过图片下载（只抓结构化数据）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="每次请求之间的间隔秒数（对站点礼貌）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="单次 HTTP 请求超时秒数",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    http = HttpClient(timeout=args.timeout, delay=args.delay)
    try:
        crawler = HomeCrawler(
            source=args.source,
            output_dir=args.output,
            images_dir=args.images_dir,
            repo_upgrade_dir=args.repo_path,
            http=http,
            limit=args.limit,
            download_images=not args.no_images,
        )
        crawler.run()
    finally:
        http.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
