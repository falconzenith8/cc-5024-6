"""加权叠加：按实际贡献归一化，输出融合图与覆盖图。

方法（可解释，逐步可复算）：
  每帧 k 有标量权重 w_k（默认 1/背景噪声方差，可由调用方覆盖），
  重采样得到对齐图 A_k 与覆盖掩膜 c_k（0/1）。
  像素级有效权重 W_k = w_k * c_k，于是

      fused[p]   = sum_k W_k[p] * A_k[p] / sum_k W_k[p]
      coverage[p]= sum_k c_k[p]            （覆盖帧数）
      wsum[p]    = sum_k W_k[p]            （权重和）

  sum_k W_k[p] == 0 的像素没有任何帧贡献，fused 记为 NaN，
  绝不写入 0（否则边缘会被零值稀释）。

注意：这是**原生分辨率上的加权平均**。不放大、不重建高频，
本模块不做、也不声称做任何"超分辨率"处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .resample import shift_bilinear


@dataclass
class StackResult:
    fused: np.ndarray        # (H,W) float64，无覆盖处为 NaN
    coverage: np.ndarray     # (H,W) float64，覆盖帧数
    weight_sum: np.ndarray   # (H,W) float64，权重和
    used: list               # 实际参与叠加的帧索引
    aligned: dict = field(default_factory=dict)   # k -> 对齐图（供残差用）
    footprints: dict = field(default_factory=dict)  # k -> 覆盖掩膜


def stack_frames(frames, estimates, enabled=None, weights=None,
                 keep_aligned=True, progress_cb=None, is_cancelled=None):
    """frames: 帧序列；estimates: estimate_shifts 的结果。

    enabled: 可选布尔序列，用户逐帧开关；ok=False 的帧无论开关都不参与。
    weights: 可选每帧标量权重覆盖（None 用估计给出的默认权重）。
    progress_cb(fraction) / is_cancelled() 为离线任务的可选钩子。
    """
    n = len(frames)
    if enabled is None:
        enabled = [True] * n
    if weights is None:
        weights = [e.weight for e in estimates]

    acc = None
    wsum = None
    cov = None
    used, aligned, footprints = [], {}, {}

    for k in range(n):
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("任务已取消")
        est = estimates[k]
        if not enabled[k] or not est.ok:
            if progress_cb is not None:
                progress_cb((k + 1) / n)
            continue
        w_k = float(weights[k])
        if not np.isfinite(w_k) or w_k <= 0:
            if progress_cb is not None:
                progress_cb((k + 1) / n)
            continue
        a, c = shift_bilinear(frames[k], est.dx, est.dy)
        eff = w_k * c
        if acc is None:
            acc = eff * a
            wsum = eff.copy()
            cov = c.copy()
        else:
            acc += eff * a
            wsum += eff
            cov += c
        used.append(k)
        if keep_aligned:
            aligned[k] = a
            footprints[k] = c
        if progress_cb is not None:
            progress_cb((k + 1) / n)

    if acc is None:
        raise ValueError("没有可参与叠加的帧（全部被剔除或禁用）")

    with np.errstate(invalid="ignore", divide="ignore"):
        fused = np.where(wsum > 0, acc / wsum, np.nan)
    return StackResult(fused=fused, coverage=cov, weight_sum=wsum,
                       used=used, aligned=aligned, footprints=footprints)


def frame_residuals(result: StackResult, fused=None):
    """每帧残差：R_k = A_k - fused（仅在该帧覆盖区域内统计）。

    返回 {k: {"rms": ..., "max_abs": ..., "map": R_k}}。
    """
    fused = result.fused if fused is None else fused
    out = {}
    for k, a in result.aligned.items():
        c = result.footprints[k]
        r = (a - fused) * c
        valid = c > 0
        vals = r[valid]
        out[k] = {
            "rms": float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan"),
            "max_abs": float(np.max(np.abs(vals))) if vals.size else float("nan"),
            "map": r,
        }
    return out
