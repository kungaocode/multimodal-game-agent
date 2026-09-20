"""Multimodal vision model wrapper (API-first)."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


def _extract_json(content: str) -> dict[str, Any]:
    """从模型输出中解析 JSON，容忍 ```json 代码围栏及前后多余文本。"""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：取第一个 { 到最后一个 } 之间的内容
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass
class VisionModelResponse:
    description: str
    objects: list[dict[str, Any]]
    raw: dict[str, Any]


class VisionModel:
    """Call a Qwen-VL / OpenAI-compatible vision API for high-level understanding."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url or os.getenv("VISION_MODEL_BASE_URL", "")
        self.api_key = api_key or os.getenv("VISION_MODEL_API_KEY", "")
        self.model = model or os.getenv("VISION_MODEL_NAME", "qwen-vl-max")
        self.timeout = timeout
        # 默认不信任环境代理变量：本机常见 socks5:// 代理会让 httpx 直接崩溃。
        # 需要走系统代理时设置 VISION_TRUST_ENV=1。
        self.trust_env = os.getenv("VISION_TRUST_ENV", "0") == "1"

    def _encode_image(self, image: Image.Image | Path | str) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def describe(self, image: Image.Image | Path | str, prompt: str | None = None) -> VisionModelResponse:
        """Send the image to the vision model and return structured JSON.

        优先请求 JSON Output（response_format=json_object）；部分模型/网关不支持该参数时，
        自动降级为普通输出并本地解析 JSON，保证链路不中断。
        """
        if not self.base_url:
            raise RuntimeError(
                "Vision model base_url is not set. Configure VISION_MODEL_BASE_URL or pass base_url."
            )
        system_prompt = (
            "You are a game UI analyst. Describe the screenshot in detail. "
            "Identify key UI elements and in-game objects as a JSON list under 'objects'. "
            "Each object should have fields: type, position_estimate, description, confidence."
        )
        user_prompt = prompt or "Analyze this game screenshot and return structured JSON."
        image_b64 = self._encode_image(image)
        data_url = f"data:image/jpeg;base64,{image_b64}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            return self._chat(payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 422) and payload.get("response_format"):
                # 模型不支持 JSON Output → 去掉 response_format 重试
                payload.pop("response_format", None)
                return self._chat(payload)
            raise

    def chat(
        self,
        image: Image.Image | Path | str,
        prompt: str,
        include_usage: bool = False,
    ) -> dict[str, Any]:
        """发送图片 + 自定义提示词，返回模型输出的完整 JSON 对象。

        用于高层任务规划（如 /plan）：提示词可要求模型输出任意结构化 JSON
        （summary / steps / coords 等），不做 describe 的字段级过滤。
        include_usage=True 时在结果中附加 "_usage"（token 用量，成本统计用）。
        """
        if not self.base_url:
            raise RuntimeError(
                "Vision model base_url is not set. Configure VISION_MODEL_BASE_URL or pass base_url."
            )
        image_b64 = self._encode_image(image)
        data_url = f"data:image/jpeg;base64,{image_b64}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        def _run() -> tuple[dict[str, Any], dict[str, Any]]:
            raw = self._chat(payload).raw
            return self.parse_content(raw), raw.get("usage", {})

        parsed, usage = _run()
        if include_usage:
            parsed["_usage"] = usage
        return parsed

    @staticmethod
    def parse_content(raw: dict[str, Any]) -> dict[str, Any]:
        """从 chat completions 原始响应中提取 content 并解析 JSON。"""
        raw_message = raw["choices"][0]["message"]
        content = raw_message.get("content") or ""
        return _extract_json(content)

    def _chat(self, payload: dict[str, Any]) -> VisionModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
        data = resp.json()
        raw_message = data["choices"][0]["message"]
        # qwen3 系列思考模型可能同时返回 reasoning_content，正文仍在 content 字段
        content = raw_message.get("content") or ""
        parsed = _extract_json(content)
        return VisionModelResponse(
            description=parsed.get("description", ""),
            objects=parsed.get("objects", []),
            raw=data,
        )
