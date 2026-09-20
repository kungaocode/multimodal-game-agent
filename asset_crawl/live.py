"""线上站点数据源解析器（--source live）。

数据源：https://clashpost.com/upgrade/category/home（VitePress 静态站）。

VitePress 页面主体 HTML 只有容器骨架，实际内容位于懒加载的 JS chunk
（例如 /assets/upgrade_category_home.md.<hash>.lean.js）。本模块负责：
1. 从 HTML 中找出内容 chunk 的 URL；
2. 解析 chunk 内 Vue 编译后的 hyperscript，还原结构化数据。

chunk 内格式说明（已对照线上真实数据确认）：
- 索引：a(u,{name:`野蛮人之王`,imgSrc:`0200/...`,link:`0200-Barbarian-King`})
- 详情：JSON.parse(`{...frontmatter...}`) + 组件调用
       r(f,{pKey:`占地面积`,pValue:`3×3`})                  -> 属性
       r(f,{imgTitle:`1 级`,imgSrc:`X-Bow1.png`})          -> 等级图片
       a(`table`,{...},[a(`thead`,...),a(`tbody`,...)])    -> 数据表
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .client import CrawlError, HttpClient
from .models import (
    BASE_IMG,
    BASE_SITE,
    BuildingNum,
    DetailData,
    ImageGroup,
    ImageItem,
    IndexData,
    ItemRef,
    PropertyItem,
    Section,
    Tab,
    TimelineEntry,
    UpgradeTable,
)
from .parsers import TAB_NAMES, _clean_cell

# 页面 HTML 中内容 chunk 的引用
_CHUNK_RE = re.compile(r"/assets/upgrade_[^\s\"']+\.lean\.js")
_INDEX_CHUNK_RE = re.compile(r"/assets/upgrade_category_home[^\s\"']*\.lean\.js")

# ---- 索引 chunk 解析 ----
_TAB_BLOCK_RE = re.compile(r"id:`cp-upgrade-([a-z-]+)`,class:`cp-upgrade-item`")
_SECTION_RE = re.compile(r"title:`([^`]+)`,imgFolder:`([^`]+)`")
_ITEM_RE = re.compile(r"name:`([^`]+)`,imgSrc:`([^`]+)`,link:`([^`]+)`")

# ---- 详情 chunk 解析 ----
_FRONTMATTER_JSON_RE = re.compile(r"JSON\.parse\(`([^`]*)`\)")
_INFO_IMG_RE = re.compile(r"imgSrc:`([^`]+)`,imgAlt:")
_PROP_RE = re.compile(r"pKey:`([^`]*)`,pValue:`([^`]*)`")
_LEVEL_IMG_RE = re.compile(r"imgTitle:`([^`]+)`,imgSrc:`([^`]+)`")
_GROUP_TITLE_RE = re.compile(r"title:`([^`]+)`,folder:")
_BUILDING_NUM_RE = re.compile(r"title:`([^`]+)`,num:`([^`]+)`")
_TABLE_RE = re.compile(r"a\(`table`")
_TEXT_RE = re.compile(r"t\(`([^`]*?)`(?=[,)])")
_DATE_RE = re.compile(r"date:`([^`]+)`")


def find_index_chunk(html: str) -> str:
    """从首页 HTML 中找出分类页内容 chunk 的路径。"""
    m = _INDEX_CHUNK_RE.search(html)
    if m:
        return m.group(0)
    raise CrawlError("未在首页 HTML 中找到分类内容 chunk（upgrade_category_home）")


def find_detail_chunk(html: str, link: str) -> str:
    """从详情页 HTML 中找出该页内容 chunk 的路径。"""
    candidates = list(_CHUNK_RE.findall(html))
    for c in candidates:
        if link in c:
            return c
    if candidates:
        return candidates[-1]
    raise CrawlError(f"未在详情页 HTML 中找到内容 chunk: {link}")


# ---- 索引解析 ----

def parse_index_chunk(chunk: str, source: str = "live") -> IndexData:
    """解析分类页内容 chunk，构建 IndexData。"""
    tab_positions = list(_TAB_BLOCK_RE.finditer(chunk))
    tabs: list[Tab] = []
    for i, m in enumerate(tab_positions):
        tab_id = m.group(1)
        block_start = m.end()
        block_end = tab_positions[i + 1].start() if i + 1 < len(tab_positions) else len(chunk)
        block = chunk[block_start:block_end]
        tabs.append(_parse_tab_block(tab_id, block))
    index = IndexData(
        source=source,
        crawled_at=datetime.now(timezone.utc).isoformat(),
        tabs=tabs,
    )
    index.total_items = len(index.iter_items())
    return index


def _parse_tab_block(tab_id: str, block: str) -> Tab:
    sections: list[Section] = []
    sec_positions = list(_SECTION_RE.finditer(block))
    for i, m in enumerate(sec_positions):
        title, img_folder = m.group(1), m.group(2)
        sub_start = m.end()
        sub_end = sec_positions[i + 1].start() if i + 1 < len(sec_positions) else len(block)
        sub = block[sub_start:sub_end]
        items: list[ItemRef] = []
        for im in _ITEM_RE.finditer(sub):
            name, img_src, link = im.group(1), im.group(2), im.group(3)
            item_id = link.split("-", 1)[0]
            items.append(
                ItemRef(
                    id=item_id,
                    name=name,
                    link=link,
                    img_src=img_src,
                    url=f"{BASE_SITE}/upgrade/{link}",
                    thumb_url=f"{BASE_IMG}/{img_folder}/{img_src}",
                )
            )
        sections.append(Section(title=title, img_folder=img_folder, items=items))
    return Tab(id=tab_id, name=TAB_NAMES.get(tab_id, tab_id), sections=sections)


# ---- 详情解析 ----

def parse_detail_chunk(
    chunk: str,
    link: str,
    *,
    tab: str = "",
    section: str = "",
) -> DetailData:
    """解析一个详情页的内容 chunk。"""
    frontmatter = _extract_frontmatter(chunk)
    item_id = link.split("-", 1)[0]
    img_folder = frontmatter.get("imgFolder", "")
    name = frontmatter.get("navTitle") or frontmatter.get("shownTitle") or frontmatter.get("title", "")

    info_src = _first_match(_INFO_IMG_RE, chunk)
    info_image = None
    if info_src:
        info_image = ImageItem(level="info", src=info_src, url=f"{BASE_IMG}/{img_folder}/{info_src}")

    return DetailData(
        id=item_id,
        link=link,
        tab=tab,
        section=section,
        name=name,
        title=frontmatter.get("title", ""),
        description=frontmatter.get("description", ""),
        module=frontmatter.get("module", ""),
        img_folder=img_folder,
        canonical=frontmatter.get("canonical", ""),
        wiki=frontmatter.get("wiki", ""),
        info_image=info_image,
        image_groups=_extract_image_groups(chunk, img_folder),
        building_num=_extract_building_num(chunk),
        properties=_extract_properties(chunk),
        tables=_extract_tables(chunk),
        timeline=_extract_timeline(chunk),
        notes=_extract_notes(chunk),
    )


def _extract_frontmatter(chunk: str) -> dict:
    m = _FRONTMATTER_JSON_RE.search(chunk)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    frontmatter = data.get("frontmatter", {})
    return frontmatter if isinstance(frontmatter, dict) else {}


def _extract_image_groups(chunk: str, img_folder: str) -> list[ImageGroup]:
    """等级图片分组：图片归属于它之前最近的 'title:...,folder:' 组。"""
    boundaries: list[tuple[int, ImageGroup]] = []
    groups: list[ImageGroup] = []
    for m in _GROUP_TITLE_RE.finditer(chunk):
        g = ImageGroup(group=m.group(1), images=[])
        boundaries.append((m.start(), g))
        groups.append(g)

    fallback: ImageGroup | None = None

    def _owner(pos: int) -> ImageGroup:
        owner = None
        for bpos, g in boundaries:
            if bpos < pos:
                owner = g
            else:
                break
        if owner is None:
            nonlocal fallback
            if fallback is None:
                fallback = ImageGroup(group="", images=[])
                groups.append(fallback)
            owner = fallback
        return owner

    for m in _LEVEL_IMG_RE.finditer(chunk):
        level, src = m.group(1), m.group(2)
        _owner(m.start()).images.append(
            ImageItem(level=level, src=src, url=f"{BASE_IMG}/{img_folder}/{src}")
        )
    return groups


def _extract_building_num(chunk: str) -> list[BuildingNum]:
    return [
        BuildingNum(title=_clean_cell(m.group(1)), num=m.group(2))
        for m in _BUILDING_NUM_RE.finditer(chunk)
    ]


def _extract_properties(chunk: str) -> list[PropertyItem]:
    return [
        PropertyItem(key=m.group(1), value=_clean_cell(m.group(2)))
        for m in _PROP_RE.finditer(chunk)
    ]


def _extract_tables(chunk: str) -> list[UpgradeTable]:
    """解析所有 a(`table`) 子树，并标注所属小节标题。"""
    tables: list[UpgradeTable] = []
    text_positions = [(m.start(), m.group(1)) for m in _TEXT_RE.finditer(chunk)]
    for tm in _TABLE_RE.finditer(chunk):
        open_paren = chunk.find("(", tm.start())
        end = _match_paren(chunk, open_paren)
        if end < 0:
            continue
        subtree = chunk[tm.start() : end]
        label = ""
        for tpos, text in text_positions:
            if tpos < tm.start():
                label = text
            else:
                break
        headers, rows = _parse_table_subtree(subtree)
        tables.append(UpgradeTable(label=label, headers=headers, rows=rows))
    return tables


def _parse_table_subtree(subtree: str) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []

    thead = _find_call_subtree(subtree, "thead")
    if thead is not None:
        headers = _extract_cells(thead, "th")

    tbody = _find_call_subtree(subtree, "tbody")
    if tbody is not None:
        for tr in _iter_call_subtrees(tbody, "tr"):
            cells = _extract_cells(tr, "td")
            if cells:
                rows.append(cells)
    return headers, rows


def _find_call_subtree(text: str, tag: str) -> str | None:
    """找到 text 内 a(`tag`,...) 的第一个子树文本。"""
    for m in re.finditer(rf"a\(`{tag}`", text):
        open_paren = text.find("(", m.start())
        end = _match_paren(text, open_paren)
        if end >= 0:
            return text[m.start() : end]
    return None


def _iter_call_subtrees(text: str, tag: str):
    """迭代 text 内所有 a(`tag`,...) 子树文本。"""
    for m in re.finditer(rf"a\(`{tag}`", text):
        open_paren = text.find("(", m.start())
        end = _match_paren(text, open_paren)
        if end >= 0:
            yield text[m.start() : end]


def _extract_cells(node: str, tag: str) -> list[str]:
    """从 thead/tr 子树中提取所有 th/td 单元格文本。"""
    cells: list[str] = []
    for m in re.finditer(rf"a\(`{tag}`", node):
        open_paren = node.find("(", m.start())
        end = _match_paren(node, open_paren)
        if end < 0:
            continue
        args = node[open_paren + 1 : end]
        arg = args
        if arg.lstrip().startswith("null,"):
            arg = arg.split(",", 1)[1]
        cells.append(_cell_text(arg))
    return cells


def _cell_text(arg: str) -> str:
    """单元格参数还原为文本：`x` 或 [t(`a`),...,t(`b`)] -> 'ab'。"""
    arg = arg.strip()
    if arg.startswith("`"):
        end = arg.find("`", 1)
        if end > 0:
            return _clean_cell(arg[1:end])
        return ""
    if arg.startswith("["):
        parts = [m.group(1) for m in _TEXT_RE.finditer(arg)]
        return _clean_cell("".join(parts))
    return ""


def _extract_timeline(chunk: str) -> list[TimelineEntry]:
    """更新历史：date:`...` 后的 t(`...`) 文本即该条目的行。"""
    dates = list(_DATE_RE.finditer(chunk))
    entries: list[TimelineEntry] = []
    for i, m in enumerate(dates):
        date = m.group(1)
        seg_start = m.end()
        seg_end = dates[i + 1].start() if i + 1 < len(dates) else len(chunk)
        seg = chunk[seg_start:seg_end]
        rows = [m.group(1) for m in _TEXT_RE.finditer(seg)]
        entries.append(TimelineEntry(date=date, rows=[_clean_cell(r) for r in rows]))
    return entries


def _extract_notes(chunk: str) -> list[str]:
    """重要说明：'重要说明' 标题后的 <ol> 子树文本（尽力而为）。"""
    label_pos = None
    for m in _TEXT_RE.finditer(chunk):
        if m.group(1) == "重要说明":
            label_pos = m.start()
            break
    if label_pos is None:
        return []
    tail = chunk[label_pos:]
    for om in re.finditer(r"a\(`ol`", tail):
        open_paren = tail.find("(", om.start())
        end = _match_paren(tail, open_paren)
        if end < 0:
            continue
        ol = tail[om.start() : end]
        notes = [m.group(1) for m in _TEXT_RE.finditer(ol)]
        return [_clean_cell(n) for n in notes]
    return []


def _match_paren(text: str, open_idx: int) -> int:
    """text[open_idx] 必须是 '('，返回配对 ')' 的位置（跳过反引号字符串）。"""
    depth = 0
    in_str = False
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "`":
            in_str = not in_str
        elif not in_str:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None
