# 家乡数据爬虫（image.code）

为「多模态游戏 AI Agent 实验平台」抓取《部落冲突》家乡（Home Village）建筑、兵种、
英雄、法术、装备的结构化数据（名称、描述、等级图片、升级数据、更新历史等），
输出文本 JSON 与图片数据，供后续视觉识别与行为决策模块使用。

## 数据源

两个数据源产出相同 schema，可互为校验：

| 数据源 | 说明 | 优点 |
| ------ | ---- | ---- |
| `live` | 爬线上 https://clashpost.com/upgrade/category/home | 始终最新，但依赖站点结构、受 Cloudflare 防护 |
| `repo` | 解析本地 GitHub 克隆 `lemonicy/clashpost` 的源 markdown | 结构稳定，离线可跑，**推荐** |

VitePress 站点的页面内容在懒加载 JS chunk 里（`/assets/upgrade_*.lean.js`），
`live` 源通过解析 Vue 编译后的 hyperscript 还原数据。

## 环境

```bash
conda create -n multimodal-agent python=3.12
conda activate multimodal-agent
pip install -r requirements.txt   # 含 httpx[socks,brotli]
```

爬虫通过环境代理自动走代理（`HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`，
`socks://` 会自动规范化为 `socks5://`），无代理时直连。

## 使用

```bash
# repo 源（推荐）：解析本地克隆仓库的源 markdown，全量下载图片与 JSON
python -m asset_crawl --source repo

# 只抓 JSON，不下载图片
python -m asset_crawl --source repo --no-images

# live 源（爬线上站点），先抓少量验证
python -m asset_crawl --source live --limit 3 --no-images

# 完整参数
python -m asset_crawl --source repo --repo-path .reference-clashpost/docs/upgrade \
                      --output dataset/text --images-dir dataset/picture \
                      --limit 10 --delay 0.6 --timeout 30
```

常用参数：

| 参数 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--source` | `live` | `live`（爬线上）或 `repo`（解析本地仓库） |
| `--repo-path` | `.reference-clashpost/docs/upgrade` | repo 源的仓库 docs 目录 |
| `--output` | `dataset/text` | 文本 JSON 输出目录 |
| `--images-dir` | `dataset/picture` | 图片输出目录 |
| `--limit` | 无 | 只抓前 N 条（调试用） |
| `--no-images` | 关 | 不下载图片，只输出 JSON |
| `--delay` | `0.6` | 页面请求间隔秒数 |
| `--timeout` | `30` | 单请求超时秒数 |

## 输出结构

```
data/
├── text/                    # 文本 JSON 数据
│   ├── index.json           # 索引：4 个标签页 → 分组 → 条目引用
│   └── items/{link}.json    # 每个条目的详情
└── picture/                 # 图片数据（按 img_folder 分目录）
    ├── home_buildings/…
    ├── home_tech/…
    ├── home_th/…
    └── home_heroes/…
```

`dataset/text/index.json`：

```jsonc
{
  "source": "repo",
  "crawled_at": "2026-09-01T...+00:00",
  "total_items": 191,
  "tabs": [                   // 英雄 / 科技 / 大本 / 建筑
    { "id": "heroes", "name": "英雄", "sections": [
        { "title": "英雄", "imgFolder": "home_heroes", "items": [
            { "id": "0200", "name": "野蛮人之王", "link": "0200-Barbarian-King",
              "imgSrc": "0200/Barbarian_King.png",
              "url": "https://clashpost.com/upgrade/0200-Barbarian-King",
              "thumb_url": "https://static.clashpost.com/upgrade/home_heroes/0200/Barbarian_King.png" }
        ]}
    ]}
  ]
}
```

`dataset/text/items/{link}.json`（以野蛮人为例）：

```jsonc
{
  "id": "0000", "link": "0000-Barbarian", "tab": "科技", "section": "部队",
  "name": "野蛮人", "title": "...", "description": "...",
  "img_folder": "home_troops",
  "info_image": { "level": "info", "src": "...", "url": "..." },   // 详情主图
  "image_groups": [ { "group": "", "images": [ { "level": "1 级", "src": "...", "url": "..." } ] } ],  // 各等级外观图
  "building_num": [ { "title": "可建造数量", "num": "4" } ],       // 建筑专用
  "properties": [ { "key": "占地面积", "value": "3×3" } ],         // 静态属性
  "tables": [ { "label": "升级数据", "headers": ["等级", "升级花费", ...],
                "rows": [["1", "\\", "\\", ...], ...], "extra_info": [] } ],
  "timeline": [ { "date": "2024-06-18", "rows": ["新增 1 级。"] } ],
  "notes": ["野蛮人皮肤不影响战斗属性。"]
}
```

图片在 `dataset/picture/{img_folder}/{level图名}`，例如
`dataset/picture/home_buildings/0401/Gold_Mine1.png`。

字段约定：

- **`\` 占位符**：源数据用 `\` 表示「无此数据 / 不适用」（类似 `—`），解析器保留原值。
- **图片归属**：等级图片按最近的 `UnitImgGroup` 分组；无分组标签时全部归入
  一个 `group: ""` 的兜底组。
- **英雄栏裁剪**：英雄标签页（英雄 / 战宠 / 装备）只需识别，不需要等级细节，
  其 `image_groups`、`tables`、`timeline`、`building_num`、`notes` 统一置空，
  仅保留名称、描述、主图与静态属性。

## 模块结构

| 文件 | 职责 |
| ---- | ---- |
| `models.py` | pydantic 数据模型（IndexData / DetailData 等） |
| `client.py` | HTTP 客户端：浏览器请求头、代理、重试、限速 |
| `parsers.py` | repo 源 markdown 解析 |
| `live.py` | live 源 JS chunk（Vue hyperscript）解析 |
| `crawler.py` | 编排：抓索引 → 逐条详情 → 下载图片 → 输出 JSON |
| `cli.py` | 命令行入口 |

## 校验与安全

```bash
# 安全审计（项目强制：所有 Python 代码合并前须通过）
python -m security.audit --path image/code

# 校验输出（示例：对比两源同一条目）
python -c "from image.code.crawler import HomeCrawler; ..."
```
