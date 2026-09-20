"""FSM 识别器类别定义（本地模型只做有限状态机的识别器，不用太强）。

打资源 + 捐兵两个状态机需要的 UI 信号类别：
    0-5   资源建筑（与 farm_synthetic 一致）
    6     进攻按钮   （村庄界面 → 进入常规战搜索）
    7     返回按钮   （回营 / 战斗结束返回）
    8     搜索对手按钮（联机模式-常规战「搜索对手」）
    9     下一个按钮 （对手不值得 → 继续搜索）
    10    结束战斗按钮（战斗中主动结束）
    11    增援按钮   （捐兵模块）
"""

from __future__ import annotations

FSM_ORDER = [
    "圣水收集器",
    "金矿",
    "圣水瓶",
    "储金罐",
    "暗黑重油罐",
    "暗黑重油钻井",
    "进攻按钮",
    "返回按钮",
    "搜索对手按钮",
    "下一个按钮",
    "结束战斗按钮",
    "增援按钮",
]
FSM_NAME_TO_ID = {name: i for i, name in enumerate(FSM_ORDER)}
FSM_ID_TO_NAME = {i: name for i, name in enumerate(FSM_ORDER)}

# ---- 扩展类别：捐兵流程 + 兵种选择栏（id 从 12 起，兼容旧 12 类模型）----
DONATION_EXTRA_ORDER = [
    "消息列表按钮",
    "请求条目",
    "关闭按钮",
    "捐赠确认按钮",
    "兵种选择栏",
]
EXTENDED_FSM_ORDER = FSM_ORDER + DONATION_EXTRA_ORDER
EXTENDED_NAME_TO_ID = {name: i for i, name in enumerate(EXTENDED_FSM_ORDER)}
EXTENDED_ID_TO_NAME = {i: name for i, name in enumerate(EXTENDED_FSM_ORDER)}

# 按钮类 OCR 文本 → 类别（含英文文案）
BUTTON_TEXT_RULES: dict[str, tuple[int, ...]] = {
    "进攻": (FSM_NAME_TO_ID["进攻按钮"],),
    "attack": (FSM_NAME_TO_ID["进攻按钮"],),
    "回营": (FSM_NAME_TO_ID["返回按钮"],),
    "返回": (FSM_NAME_TO_ID["返回按钮"],),
    "back": (FSM_NAME_TO_ID["返回按钮"],),
    "搜索对手": (FSM_NAME_TO_ID["搜索对手按钮"],),
    "search": (FSM_NAME_TO_ID["搜索对手按钮"],),
    "下一个": (FSM_NAME_TO_ID["下一个按钮"],),
    "next": (FSM_NAME_TO_ID["下一个按钮"],),
    "结束战斗": (FSM_NAME_TO_ID["结束战斗按钮"],),
    "增援": (FSM_NAME_TO_ID["增援按钮"],),
    "donate": (FSM_NAME_TO_ID["增援按钮"],),
    "reinforce": (FSM_NAME_TO_ID["增援按钮"],),
}

# 每类先验框尺寸（归一化宽、高）——OCR 只给文本 bbox，按钮框比文本略大
PRIOR_BOX = {
    "圣水收集器": (0.0866, 0.2004),
    "金矿": (0.1308, 0.2036),
    "圣水瓶": (0.1236, 0.1993),
    "储金罐": (0.1070, 0.2018),
    "暗黑重油罐": (0.1038, 0.1949),
    "暗黑重油钻井": (0.0958, 0.2075),
    "进攻按钮": (0.1300, 0.0900),
    "返回按钮": (0.0700, 0.1000),
    "搜索对手按钮": (0.1600, 0.0800),
    "下一个按钮": (0.1200, 0.0800),
    "结束战斗按钮": (0.1500, 0.1000),
    "增援按钮": (0.1400, 0.0900),
}

# 扩展类别的先验框尺寸
EXTENDED_PRIOR_BOX = dict(PRIOR_BOX)
EXTENDED_PRIOR_BOX.update({
    "消息列表按钮": (0.0800, 0.1000),
    "请求条目": (0.3000, 0.0800),
    "关闭按钮": (0.0600, 0.0600),
    "捐赠确认按钮": (0.1200, 0.0800),
    "兵种选择栏": (0.6000, 0.1000),
})


def class_id_for_button_text(text: str) -> int | None:
    """按 OCR 文本找按钮类别；返回第一个匹配或 None。"""
    for key, ids in BUTTON_TEXT_RULES.items():
        if key in text:
            return ids[0]
    return None


def extended_class_id_for_button_text(text: str) -> int | None:
    """按 OCR 文本找扩展按钮类别（含捐兵流程）；返回第一个匹配或 None。"""
    extended_rules: dict[str, tuple[int, ...]] = {
        "消息": (EXTENDED_NAME_TO_ID["消息列表按钮"],),
        "消息列表": (EXTENDED_NAME_TO_ID["消息列表按钮"],),
        "关闭": (EXTENDED_NAME_TO_ID["关闭按钮"],),
        "close": (EXTENDED_NAME_TO_ID["关闭按钮"],),
        "确认": (EXTENDED_NAME_TO_ID["捐赠确认按钮"],),
        "确定": (EXTENDED_NAME_TO_ID["捐赠确认按钮"],),
        "donate": (EXTENDED_NAME_TO_ID["捐赠确认按钮"],),
    }
    # 先查基础按钮规则
    base_id = class_id_for_button_text(text)
    if base_id is not None:
        return base_id
    for key, ids in extended_rules.items():
        if key in text:
            return ids[0]
    return None


def extended_id_for_name(name: str) -> int | None:
    return EXTENDED_NAME_TO_ID.get(name)


def extended_name_for_id(class_id: int) -> str | None:
    return EXTENDED_ID_TO_NAME.get(class_id)


# ---- LLM 批量标注提示词模板 ----
LLM_LABEL_PROMPT_TEMPLATE = (
    "你是部落冲突游戏界面标注专家。请仔细识别截图中的所有可交互 UI 元素和游戏内建筑。\n"
    "\n"
    "必须识别的类别（type 字段必须精确使用以下名称）：\n"
    "资源建筑：金矿、圣水收集器、储金罐、圣水瓶、暗黑重油罐、暗黑重油钻井\n"
    "打资源按钮：进攻按钮、搜索对手按钮、下一个按钮、结束战斗按钮、返回按钮\n"
    "捐兵按钮：增援按钮、消息列表按钮、请求条目、关闭按钮、捐赠确认按钮\n"
    "面板：兵种选择栏\n"
    "\n"
    "请输出 JSON（不要输出其他文字）：\n"
    '{{\n'
    '  "state": "<当前游戏状态>",\n'
    '  "objects": [\n'
    '    {{"type": "<类别名>", "bbox": [左, 上, 右, 下], "confidence": <0.0~1.0>}}\n'
    "  ]\n"
    "}}\n"
    "\n"
    "bbox 是像素坐标。画面尺寸为 {width}x{height}。\n"
    "state 取值：村庄待机 / 搜索中 / 战斗中 / 战斗结束 / 捐兵-消息列表 / 捐兵-增援确认\n"
    "如果看不到某类元素，不要编造，直接不输出该条。\n"
)


def build_llm_label_prompt(width: int, height: int) -> str:
    """生成 LLM 标注提示词。"""
    return LLM_LABEL_PROMPT_TEMPLATE.format(width=width, height=height)


def id_for_name(name: str) -> int | None:
    """类别名 → id；未知返回 None。"""
    return FSM_NAME_TO_ID.get(name)


def name_for_id(class_id: int) -> str | None:
    """id → 类别名；未知返回 None。"""
    return FSM_ID_TO_NAME.get(class_id)
