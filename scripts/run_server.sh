#!/usr/bin/env bash
# 天文融合 Web 界面（图像处理页）启动脚本。
# 同样在导入 numpy 前锁定单线程，保证页面触发的离线任务与命令行
# 走同一套锁定环境；依赖版本由运行中的离线任务按 requirements.txt 核对。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

PORT="${1:-8000}"
exec "$PY" -m astrostack.server --port "$PORT" --host 127.0.0.1
