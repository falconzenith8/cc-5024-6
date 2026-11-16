#!/usr/bin/env bash
# 离线配准-融合任务的启动脚本。
#
# 关键职责：在 Python（numpy/scipy）启动**之前**导出线程环境变量，
# 把 BLAS/FFT/OMP 后端全部锁定为单线程；任务内部还会再次核对这些变量，
# 不一致即拒绝运行，避免换环境后数值/性能悄悄变化。
#
# 用法：
#   scripts/run_offline.sh <job_dir> "<frames_glob>" [reference] [requirements]
# 例：
#   scripts/run_offline.sh runs/demo "data/demo/frame_*.npy" 0 requirements.txt
set -euo pipefail

JOB_DIR="${1:?用法: run_offline.sh <job_dir> <frames_glob> [reference] [requirements]}"
FRAMES_GLOB="${2:?缺少 frames_glob}"
REFERENCE="${3:-0}"
REQUIREMENTS="${4:-requirements.txt}"

# ---- 线程锁定：必须在任何数值库导入前生效 ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- 解释器：优先仓库内 .venv（离线发布按 requirements.txt 建的环境）----
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
  echo "警告: 未找到 .venv，使用 $PY；请确认依赖与 $REQUIREMENTS 锁定版本一致" >&2
fi

# ---- 样例原帧不存在时，从模拟器实际生成（不下载任何外部数据）----
if ! compgen -G "$FRAMES_GLOB" > /dev/null; then
  case "$FRAMES_GLOB" in
    *"/demo/"*)
      DEMO_DIR="$(dirname "$FRAMES_GLOB")"
      echo "未找到样例原帧，调用模拟器生成到 $DEMO_DIR ..."
      "$PY" -m astrostack.cli simulate --out "$DEMO_DIR"
      ;;
  esac
fi

TRUTH_ARG=()
DEMO_DIR="$(dirname "$FRAMES_GLOB")"
if [[ -f "$DEMO_DIR/truth.json" ]]; then
  TRUTH_ARG=(--truth-json "$DEMO_DIR/truth.json")
fi

exec "$PY" -m astrostack.cli offline run \
  --job-dir "$JOB_DIR" \
  --frames "$FRAMES_GLOB" \
  --ref "$REFERENCE" \
  --requirements "$REQUIREMENTS" \
  --threads 1 \
  "${TRUTH_ARG[@]}"
