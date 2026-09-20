"""asset_crawl：部落冲突家乡（Home Village）游戏数据爬虫。

数据源：
- live：爬取 https://clashpost.com/upgrade/category/home（VitePress 静态站，内容位于懒加载 JS chunk）。
- repo：解析本地克隆的 https://github.com/lemonicy/clashpost 源 markdown（更健壮，推荐）。

两个数据源输出相同的结构化 schema（见 models.py），图片统一从
https://static.clashpost.com/upgrade/{img_folder}/{filename} 下载。

用法：
    python -m asset_crawl --source live --limit 3 --no-images
    python -m asset_crawl --source repo
"""
