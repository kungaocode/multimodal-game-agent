"""GitHub 仓库 markdown 数据源解析器（--source repo）。

数据源：本地克隆的 https://github.com/lemonicy/clashpost
- 索引：docs/upgrade/category/home.md（ListItems/ListItem 自定义标签）
- 详情：docs/upgrade/{link}.md（frontmatter + 自定义标签 + markdown 表格）

相比线上 chunk 解析（live.py），markdown 是网站的源文件，结构清晰、稳定，
是推荐的数据源。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

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

# 标签页 ID → 中文名（cp-upgrade- 之后的部分）
TAB_NAMES = {
    "heroes": "英雄",
    "techniques": "科技",
    "town-hall": "大本",
    "buildings": "建筑",
}

# ---- 索引解析（category/home.md） ----

_TAB_GROUP_RE = re.compile(r'<SwitchTabGroup\s+id="cp-upgrade-([a-z-]+)"', re.IGNORECASE)
_LISTITEMS_RE = re.compile(r"<ListItems\s+title=\"([^\"]*)\"\s+imgFolder=\"([^\"]*)\"", re.IGNORECASE)
_LISTITEM_RE = re.compile(
    r"<[lL]istItem\s+name=\"([^\"]*)\"\s+imgSrc=\"([^\"]*)\"\s+link=\"([^\"]*)\""
)

# ---- 详情解析共用 ----

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SMALL_TITLE_RE = re.compile(r"<SmallTitle>\s*([^<\n]+?)\s*</SmallTitle>", re.IGNORECASE)
_UNIT_INFO_RE = re.compile(r"<UnitInfo\b[^>]*imgSrc=\"([^\"]*)\"", re.IGNORECASE)
_UNIT_IMG_GROUP_RE = re.compile(r"<UnitImgGroup\b([^>]*)>", re.IGNORECASE)
_UNIT_IMG_RE = re.compile(
    r"<UnitImg\b[^>]*imgTitle=\"([^\"]*)\"\s+imgSrc=\"([^\"]*)\"", re.IGNORECASE
)
_UNIT_PROP_RE = re.compile(
    r"<UnitProperty\b[^>]*pKey=\"([^\"]*)\"\s+pValue=\"([^\"]*)\"", re.IGNORECASE
)
_BUILDING_NUM_RE = re.compile(
    r"<BuildingNumRow\s+title=\"([^\"]*)\"\s+num=\"([^\"]*)\"", re.IGNORECASE
)
_TIMELINE_ITEM_RE = re.compile(
    r"<TimelineItem\s+date=\"([^\"]*)\">(.*?)</TimelineItem>", re.IGNORECASE | re.DOTALL
)
_TIMELINE_ROW_RE = re.compile(r"<TimelineRow>(.*?)</TimelineRow>", re.IGNORECASE | re.DOTALL)
_EXTRA_INFO_RE = re.compile(r"tableExtraInfo\s*=\s*(\[[\s\S]*?\])")
_TABLE_SEP_RE = re.compile(r"^\|[\s:\-|]+\|$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def parse_index_markdown(text: str, source: str = "repo") -> IndexData:
    """解析 category/home.md，构建 IndexData。"""
    text = text.lstrip("﻿")  # 部分源文件带 BOM
    tab_positions = list(_TAB_GROUP_RE.finditer(text))
    tabs: list[Tab] = []
    for i, m in enumerate(tab_positions):
        tab_id = m.group(1)
        block_start = m.end()
        block_end = tab_positions[i + 1].start() if i + 1 < len(tab_positions) else len(text)
        block = text[block_start:block_end]
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
    group_positions = list(_LISTITEMS_RE.finditer(block))
    for i, m in enumerate(group_positions):
        title, img_folder = m.group(1), m.group(2)
        sub_start = m.end()
        sub_end = group_positions[i + 1].start() if i + 1 < len(group_positions) else len(block)
        sub = block[sub_start:sub_end]
        items: list[ItemRef] = []
        for im in _LISTITEM_RE.finditer(sub):
            name, img_src, link = im.group(1), im.group(2), im.group(3)
            items.append(_make_item(name, img_src, link, img_folder))
        sections.append(Section(title=title, img_folder=img_folder, items=items))
    return Tab(id=tab_id, name=TAB_NAMES.get(tab_id, tab_id), sections=sections)


def _make_item(name: str, img_src: str, link: str, img_folder: str) -> ItemRef:
    item_id = link.split("-", 1)[0]
    return ItemRef(
        id=item_id,
        name=name,
        link=link,
        img_src=img_src,
        url=f"{BASE_SITE}/upgrade/{link}",
        thumb_url=f"{BASE_IMG}/{img_folder}/{img_src}",
    )


# ---- 详情解析（{link}.md） ----

def parse_detail_markdown(
    text: str,
    link: str,
    *,
    tab: str = "",
    section: str = "",
) -> DetailData:
    """解析一个详情页的源 markdown。"""
    text = text.lstrip("﻿")  # 部分源文件带 BOM
    frontmatter = parse_frontmatter(text)
    item_id = link.split("-", 1)[0]
    img_folder = frontmatter.get("imgFolder", "")
    name = frontmatter.get("navTitle") or frontmatter.get("shownTitle") or frontmatter.get("title", "")
    title = frontmatter.get("title", "")

    info_src = _extract_info_image(text)
    info_image = None
    if info_src:
        info_image = ImageItem(level="info", src=info_src, url=f"{BASE_IMG}/{img_folder}/{info_src}")

    return DetailData(
        id=item_id,
        link=link,
        tab=tab,
        section=section,
        name=name,
        title=title,
        description=frontmatter.get("description", ""),
        module=frontmatter.get("module", ""),
        img_folder=img_folder,
        canonical=frontmatter.get("canonical", ""),
        wiki=frontmatter.get("wiki", ""),
        info_image=info_image,
        image_groups=_extract_image_groups(text, img_folder),
        building_num=_extract_building_num(text),
        properties=_extract_properties(text),
        tables=_extract_tables(text),
        timeline=_extract_timeline(text),
        notes=_extract_notes(text),
    )


def parse_frontmatter(text: str) -> dict:
    """解析 markdown 文件头部的 --- ... --- frontmatter（简单键值）。"""
    text = text.lstrip("﻿")  # 部分源文件带 BOM
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    data: dict = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z_][\w]*):\s*(.*)$", line)
        if not km:
            continue
        key, val = km.group(1), km.group(2).strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        data[key] = val
    return data


def _extract_info_image(text: str) -> str | None:
    m = _UNIT_INFO_RE.search(text)
    return m.group(1) if m else None


def _extract_image_groups(text: str, img_folder: str) -> list[ImageGroup]:
    """提取 UnitImgGroup 分组及每个组内的 UnitImg 等级图片。

    图片归属 = 出现在它之前且距离最近的 UnitImgGroup；若全页没有分组标签，
    则所有图片归入一个共享的兜底组（group 为空字符串）。
    """
    boundaries: list[tuple[int, ImageGroup]] = []
    groups: list[ImageGroup] = []
    for m in _UNIT_IMG_GROUP_RE.finditer(text):
        attrs = m.group(1)
        tm = re.search(r'title="([^"]*)"', attrs)
        g = ImageGroup(group=tm.group(1) if tm else "", images=[])
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

    for m in _UNIT_IMG_RE.finditer(text):
        level, src = m.group(1), m.group(2)
        _owner(m.start()).images.append(
            ImageItem(level=level, src=src, url=f"{BASE_IMG}/{img_folder}/{src}")
        )
    return groups


def _extract_building_num(text: str) -> list[BuildingNum]:
    return [
        BuildingNum(title=m.group(1), num=m.group(2)) for m in _BUILDING_NUM_RE.finditer(text)
    ]


def _extract_properties(text: str) -> list[PropertyItem]:
    return [PropertyItem(key=m.group(1), value=m.group(2)) for m in _UNIT_PROP_RE.finditer(text)]


def _extract_tables(text: str) -> list[UpgradeTable]:
    """提取 markdown 表格，并为每个表标注所属小节标题。"""
    small_titles = [(m.start(), m.group(1)) for m in _SMALL_TITLE_RE.finditer(text)]
    extra = _extract_extra_info(text)
    tables: list[UpgradeTable] = []
    for mt in _iter_markdown_tables(text):
        label = ""
        for pos, st in small_titles:
            if pos < mt["pos"]:
                label = st
            else:
                break
        headers = [_clean_cell(h) for h in mt["headers"]]
        rows = [[_clean_cell(c) for c in row] for row in mt["rows"]]
        tables.append(UpgradeTable(label=label, headers=headers, rows=rows, extra_info=extra))
    return tables


def _iter_markdown_tables(text: str):
    """按行扫描标准 GFM markdown 表格，产出 {pos, headers, rows}。

    pos 为表格起始行的字符偏移（用于与 SmallTitle 的位置比较）。
    """
    lines = text.splitlines()
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1  # +1 换行符
    i = 0
    while i < len(lines) - 1:
        cur = lines[i].strip()
        nxt = lines[i + 1].strip()
        if cur.startswith("|") and _TABLE_SEP_RE.match(nxt):
            headers = [c.strip() for c in cur.strip("|").split("|")]
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                rows.append(cells)
                j += 1
            yield {"pos": line_offsets[i], "headers": headers, "rows": rows}
            i = j
        else:
            i += 1


def _clean_cell(value: str) -> str:
    """清理单元格：去 HTML 标签、合并空白。"""
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", value)).strip()


def _extract_extra_info(text: str) -> list[dict]:
    m = _EXTRA_INFO_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _extract_timeline(text: str) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    for m in _TIMELINE_ITEM_RE.finditer(text):
        date = m.group(1).strip()
        body = m.group(2)
        rows = [_clean_cell(r) for r in _TIMELINE_ROW_RE.findall(body) if r.strip()]
        if date:
            entries.append(TimelineEntry(date=date, rows=rows))
    return entries


def _extract_notes(text: str) -> list[str]:
    """提取 '重要说明' 小节下的有序/无序列表内容。"""
    title_pos = None
    for tm in _SMALL_TITLE_RE.finditer(text):
        if tm.group(1).strip() == "重要说明":
            title_pos = tm.end()
            break
    if title_pos is None:
        return []
    tail = text[title_pos:]
    nxt = _SMALL_TITLE_RE.search(tail)
    if nxt:
        tail = tail[: nxt.start()]
    notes: list[str] = []
    for line in tail.splitlines():
        s = line.strip()
        lm = re.match(r"^(\d+\.|[-*])\s+(.*)$", s)
        if lm:
            notes.append(_clean_cell(lm.group(2)))
    return notes
