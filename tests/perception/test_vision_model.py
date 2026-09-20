"""Tests for vision model wrapper."""

import pytest

from perception.vision_model import VisionModel


def test_vision_model_requires_base_url():
    model = VisionModel()
    with pytest.raises(RuntimeError):
        model.describe("nonexistent.jpg")


def test_vision_model_uses_env_vars():
    import os

    os.environ["VISION_MODEL_BASE_URL"] = "http://localhost:8000/v1"
    os.environ["VISION_MODEL_API_KEY"] = "test-key"
    try:
        model = VisionModel()
        assert model.base_url == "http://localhost:8000/v1"
        assert model.api_key == "test-key"
    finally:
        os.environ.pop("VISION_MODEL_BASE_URL", None)
        os.environ.pop("VISION_MODEL_API_KEY", None)


def test_vision_model_model_name_from_env():
    import os

    os.environ["VISION_MODEL_NAME"] = "qwen-vl-plus"
    try:
        model = VisionModel()
        assert model.model == "qwen-vl-plus"
    finally:
        os.environ.pop("VISION_MODEL_NAME", None)


def test_vision_model_model_name_default():
    assert VisionModel().model == "qwen-vl-max"


def test_vision_model_retries_without_response_format_on_400(monkeypatch):
    """400/422 时去掉 response_format 自动重试，保证链路不中断。"""
    import httpx
    from PIL import Image

    from perception.vision_model import VisionModel

    requests_seen: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json, headers):
            requests_seen.append(dict(json))  # 深拷贝：重试时同 payload 会被原地 pop
            if len(requests_seen) == 1:
                req = httpx.Request("POST", url)
                httpx.Response(400, request=req).raise_for_status()
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"description": "a village", '
                                    '"objects": [{"type": "gold_mine", '
                                    '"position_estimate": [100, 200], '
                                    '"description": "a gold mine", "confidence": 0.9}]}'
                                )
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)
    model = VisionModel(base_url="http://fake/v1", api_key="k", model="qwen3-vl-plus")

    result = model.describe(Image.new("RGB", (64, 64), "white"))

    assert len(requests_seen) == 2
    # 第一次请求带 JSON Output；第二次降级为普通输出
    assert "response_format" in requests_seen[0]
    assert "response_format" not in requests_seen[1]
    assert result.description == "a village"
    assert result.objects[0]["type"] == "gold_mine"


def test_vision_model_chat_returns_full_json(monkeypatch):
    """chat() 返回模型输出的完整 JSON（summary + steps），不做 describe 字段过滤。"""
    import httpx
    from PIL import Image

    from perception.vision_model import VisionModel

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary": "先打金矿", '
                                    '"steps": [{"action": "deploy", "target": "金矿", '
                                    '"coords": [370, 580], "reason": "外围无防御"}]}'
                                )
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)
    model = VisionModel(base_url="http://fake/v1", api_key="k")
    result = model.chat(Image.new("RGB", (64, 64), "white"), "制定计划")
    assert result["summary"] == "先打金矿"
    assert result["steps"][0]["target"] == "金矿"
    assert result["steps"][0]["coords"] == [370, 580]
