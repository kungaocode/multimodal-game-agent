#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export VISION_MODEL_BASE_URL="${VISION_MODEL_BASE_URL:-https://ws-avxkjb2tq5lq1gwm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
export VISION_MODEL_NAME="${VISION_MODEL_NAME:-qwen3.8-omni-flash}"
export VISION_MODEL_TIMEOUT="${VISION_MODEL_TIMEOUT:-120}"
if [ -z "${VISION_MODEL_API_KEY:-}" ] && [ -r api.txt ]; then
  export VISION_MODEL_API_KEY="$(sed -n '4p' api.txt)"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/multimodal-agent/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

exec "$PYTHON_BIN" -m uvicorn api.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
