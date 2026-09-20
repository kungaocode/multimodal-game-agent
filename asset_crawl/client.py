"""HTTP 客户端：负责页面请求与图片下载。

要点：
- 带浏览器请求头，规避 Cloudflare 对裸 curl 的 403。
- 兼容环境代理：读取 http(s)_proxy / all_proxy，并把 socks:// 规范化为
  socks5://（httpx 不识别不带版本号的 socks scheme）。
- 带重试与限速，保持对目标站点礼貌（不暴力请求）。
- 统一异常类型 CrawlError，便于上层处理。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx


def _accept_encoding() -> str:
    """只在解码器可用时声明 br，否则服务器返回 brotli 而 httpx 无法解压。"""
    encoding = ["gzip", "deflate"]
    try:
        import brotli  # noqa: F401
    except ImportError:
        pass
    else:
        encoding.append("br")
    return ", ".join(encoding)


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": _accept_encoding(),
    "Referer": "https://clashpost.com/",
    "Connection": "keep-alive",
}


class CrawlError(Exception):
    """爬取过程中的统一异常。"""


def _resolve_proxy() -> str | None:
    """从环境变量解析出单个代理 URL（或 None 表示直连）。

    优先级 https > http > all。httpx 只接受 http/https/socks4/socks5 等 scheme，
    常见代理软件导出的是 `socks://`，这里统一规范化为 socks5://。
    本爬虫只访问 https 站点，单个代理即可覆盖全部请求。
    """
    url = None
    for env_names in (
        ("HTTPS_PROXY", "https_proxy"),
        ("HTTP_PROXY", "http_proxy"),
        ("ALL_PROXY", "all_proxy"),
    ):
        url = next((os.environ.get(name) for name in env_names if os.environ.get(name)), None)
        if url:
            break
    if not url:
        return None
    if url.startswith("socks://"):
        url = "socks5://" + url[len("socks://") :]
    return url


class HttpClient:
    """带重试与限速的同步 HTTP 客户端。"""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        delay: float = 0.6,
        max_retries: int = 3,
    ) -> None:
        self._delay = delay
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
            trust_env=False,
            proxy=_resolve_proxy(),
        )

    def close(self) -> None:
        """关闭底层连接池。"""
        self._client.close()

    def get_text(self, url: str) -> str:
        """GET 并返回文本内容。"""
        resp = self._get(url)
        return resp.text

    def get_bytes(self, url: str) -> bytes:
        """GET 并返回二进制内容（用于图片）。"""
        resp = self._get(url)
        return resp.content

    def download(self, url: str, dest: Path) -> Path:
        """下载 url 内容到 dest（自动创建父目录）。"""
        data = self.get_bytes(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            fh.write(data)
        return dest

    def _get(self, url: str) -> httpx.Response:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            if attempt > 0:
                time.sleep(self._delay * attempt)  # 递增退避
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                last_err = exc
                continue
            if resp.status_code == 200:
                return resp
            last_err = CrawlError(f"HTTP {resp.status_code}: {url}")
        raise CrawlError(f"请求失败（重试 {self._max_retries} 次）: {url}") from last_err
