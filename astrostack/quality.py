"""离线产物的实测质量指标：输出信噪比（SNR）与帧清单（frame manifest）。

所有指标都在**实际融合产物与配准估计**上计算，不使用任何占位默认值；
缺失或无意义的量返回 None，由调用方按失败处理而不是悄悄替换成默认值。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .estimate import background_stats


def output_snr(fused, cx, cy, r_ap=6.0, r_in=10.0, r_out=14.0):
    """在融合图上实测输出信噪比。

    信号取孔径 r_ap 内的背景扣除流量总和；噪声按孔径像素数折算，
    包含融合图背景像素的实测 MAD 噪声与信号本身的泊松项：

        S   = sum_ap (fused - bg)
        N   = sqrt(n_ap * sigma_bg^2 + max(S, 0))
        SNR = S / N

    另给"峰值 SNR"= 星点峰值 / sigma_bg，便于与单帧 SNR 口径对照。
    无有效像素或无正信号时返回 None（调用方不得用 0 等默认值顶替）。
    """
    fused = np.asarray(fused, dtype=np.float64)
    h, w = fused.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.hypot(xx - cx, yy - cy)
    valid = np.isfinite(fused)
    ann = (d >= r_in) & (d < r_out) & valid
    ap = (d <= r_ap) & valid
    if ann.sum() < 8 or ap.sum() < 4:
        return None
    bg = float(np.median(fused[ann]))
    _, sigma_bg = background_stats(fused[ann])
    aperture = fused[ap] - bg
    signal = float(aperture.sum())
    n_ap = int(ap.sum())
    if signal <= 0 or not np.isfinite(signal) or sigma_bg <= 0:
        return None
    noise = float(np.sqrt(n_ap * sigma_bg**2 + max(signal, 0.0)))
    peak = float(np.nanmax(fused[ap]) - bg)
    return {
        "signal_aperture": signal,
        "noise_aperture": noise,
        "snr_aperture": float(signal / noise),
        "snr_peak": float(peak / sigma_bg),
        "bg_per_pix": bg,
        "sigma_bg": float(sigma_bg),
        "r_ap": r_ap,
        "n_ap_pix": n_ap,
    }


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_frame_manifest(inputs, estimates, used, residuals=None):
    """构造帧清单：每帧的输入指纹、配准结果、门控状态与残差实测。

    inputs: list[dict]，每项含 path/size/sha256（坏帧可含 error 字段）。
    estimates: estimate_shifts 返回的 FrameEstimate 列表。
    used: 实际参与叠加的帧索引列表。
    residuals: stack.frame_residuals 的结果（可选）。
    """
    residuals = residuals or {}
    used_set = set(used)
    manifest = []
    for k, inp in enumerate(inputs):
        item = {
            "index": k,
            "name": inp.get("name"),
            "path": inp.get("path"),
            "sha256": inp.get("sha256"),
            "bytes": inp.get("bytes"),
            "shape": inp.get("shape"),
            "load_error": inp.get("error"),
            "gated": True,
            "reasons": [],
            "dx": None,
            "dy": None,
            "snr": None,
            "weight": None,
            "used": False,
            "residual_rms": None,
            "residual_max": None,
        }
        if k < len(estimates):
            e = estimates[k]
            item["gated"] = not e.ok
            item["reasons"] = list(e.reasons)
            if e.ok:
                item["dx"] = float(e.dx)
                item["dy"] = float(e.dy)
                item["snr"] = float(e.snr)
                item["weight"] = float(e.weight)
                item["used"] = k in used_set
        if k in residuals:
            item["residual_rms"] = float(residuals[k]["rms"])
            item["residual_max"] = float(residuals[k]["max_abs"])
        manifest.append(item)
    return manifest


def summarize_manifest(manifest):
    """帧清单汇总数；任何一帧缺少输入指纹都视为清单不完整。"""
    n = len(manifest)
    n_used = sum(1 for m in manifest if m["used"])
    n_gated = sum(1 for m in manifest if m["gated"])
    n_bad_input = sum(1 for m in manifest if m["load_error"])
    missing_hash = [m["index"] for m in manifest if not m["sha256"]]
    return {
        "n_frames": n,
        "n_used": n_used,
        "n_gated": n_gated,
        "n_bad_input": n_bad_input,
        "missing_hash": missing_hash,
        "complete": n > 0 and not missing_hash,
    }


def hash_array(arr):
    """对数组内容取 sha256（NaN 掩膜也纳入，保证产物可复验）。"""
    a = np.asarray(arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(np.where(np.isfinite(a), a, 0.0).tobytes())
    h.update(np.isfinite(a).astype(np.uint8).tobytes())
    return h.hexdigest()


def write_manifest_file(manifest, path):
    import json

    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
