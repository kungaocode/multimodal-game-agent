"""爬虫输出的结构化数据模型（pydantic）。

索引（IndexData）描述家乡页面有哪些分类 / 分组 / 条目；
详情（DetailData）描述单个建筑 / 兵种 / 法术等的完整数据。

字段命名统一使用小写 snake_case；原始站点字段名保留在注释中。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 站点常量
BASE_SITE = "https://clashpost.com"
BASE_IMG = "https://static.clashpost.com/upgrade"
INDEX_URL = f"{BASE_SITE}/upgrade/category/home"


class ItemRef(BaseModel):
    """索引条目：指向一个详情页的引用。"""

    id: str = Field(description="物品短 ID（link 中 '-' 前的部分），如 0309")
    name: str = Field(description="中文名（原站点 ListItem.name）")
    link: str = Field(description="详情页链接片段，如 0309-X-Bow")
    img_src: str = Field(description="缩略图相对路径，相对 /upgrade/ 目录")
    url: str = Field(description="详情页完整 URL")
    thumb_url: str = Field(description="缩略图完整 URL")


class Section(BaseModel):
    """一个分组（原站点 ListItems.title），如：防御建筑 / 圣水兵 / 法术。"""

    title: str
    img_folder: str = Field(description="该分组图片所在文件夹名，如 home_buildings")
    items: list[ItemRef] = Field(default_factory=list)


class Tab(BaseModel):
    """一级标签页（原站点 SwitchTabGroup），如：英雄 / 科技 / 大本 / 建筑。"""

    id: str = Field(description="标签页 ID，如 home-buildings")
    name: str = Field(description="标签页中文名")
    sections: list[Section] = Field(default_factory=list)


class IndexData(BaseModel):
    """家乡索引页解析结果。"""

    village: str = "home"
    source: str = Field(description="数据来源（live / repo）")
    source_url: str = INDEX_URL
    crawled_at: str = Field(default="", description="抓取时间 ISO 格式")
    total_items: int = 0
    tabs: list[Tab] = Field(default_factory=list)

    def iter_items(self) -> list[ItemRef]:
        """平铺所有条目。"""
        return [item for tab in self.tabs for sec in tab.sections for item in sec.items]


class PropertyItem(BaseModel):
    """一条属性（原站点 UnitProperty.pKey / pValue）。"""

    key: str
    value: str


class ImageItem(BaseModel):
    """一张图片（原站点 UnitImg / UnitInfo）。"""

    level: str = Field(description="图片标题，如 '1 级'；info 图为 'info'")
    src: str = Field(description="文件名，相对 img_folder 目录")
    url: str = Field(description="完整图片 URL")


class ImageGroup(BaseModel):
    """一组同主题的等级图片（原站点 UnitImgGroup）。"""

    group: str = Field(description="组标题，如 '地面模式'；无标题时为空字符串")
    images: list[ImageItem] = Field(default_factory=list)


class BuildingNum(BaseModel):
    """建筑数量对照表行（原站点 BuildingNumRow）。"""

    title: str
    num: str


class UpgradeTable(BaseModel):
    """一张数据表（markdown 表格 → 结构化行列）。"""

    label: str = Field(default="升级数据", description="该表所属小节标题，如 '升级数据' / '技能相关的数据'")
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    extra_info: list[dict] = Field(
        default_factory=list,
        description="表格列附加信息（升级费用/时间的资源图标等，仅 repo 源解析）",
    )


class TimelineEntry(BaseModel):
    """更新历史条目（原站点 TimelineItem）。"""

    date: str
    rows: list[str] = Field(default_factory=list)


class DetailData(BaseModel):
    """单个建筑 / 兵种 / 法术的完整详情数据。"""

    id: str
    link: str
    tab: str = Field(default="", description="所属标签页名（英雄 / 科技 / 大本 / 建筑）")
    section: str = Field(default="", description="所属分组名（防御建筑 / 圣水兵 ...）")
    name: str = Field(description="名称（原站点 frontmatter.navTitle / shownTitle）")
    title: str = Field(description="页面标题（原站点 frontmatter.title）")
    description: str = Field(default="", description="描述（原站点 frontmatter.description）")
    module: str = Field(default="", description="模块标识，如 upgrade-home")
    img_folder: str = Field(default="", description="详情图片文件夹，如 home_buildings/0309")
    canonical: str = Field(default="", description="规范 URL 路径")
    wiki: str = Field(default="", description="Clash of Clans Wiki 链接")
    info_image: ImageItem | None = Field(default=None, description="主展示图（UnitInfo）")
    image_groups: list[ImageGroup] = Field(default_factory=list)
    building_num: list[BuildingNum] = Field(default_factory=list)
    properties: list[PropertyItem] = Field(default_factory=list)
    tables: list[UpgradeTable] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="重要说明（按行）")
