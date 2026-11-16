#!/usr/bin/env python3
"""独立比较脚本：对比两份融合输出（浮点 .npy/.fits），不依赖 astrostack 包。

自行实现质心、矩法 FWHM、孔径测光，对两份结果重新测量后比较，
避免"用被测代码验证被测代码"。

用法：
  python3 scripts/compare.py out_a/fused.npy out_b/fused.npy \
      [--cov-a out_a/coverage.npy] [--cov-b out_b/coverage.npy] \
      [--tol-diff 1e-6] [--tol-flux 0.01] [--tol-fwhm 0.05]
退出码 0 = 一致；1 = 超出容差；2 = 用法/读取错误。
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load(path):
    p = str(path).lower()
    if p.endswith(".npy"):
        return np.load(path).astype(np.float64)
    if p.endswith((".fits", ".fit", ".fts")):
        return _read_fits(path)
    raise SystemExit(f"不支持的格式: {path}（支持 .npy / .fits）")


def _read_fits(path):
    raw = open(path, "rb").read()
    cards, pos = {}, 0
    while True:
        card = raw[pos:pos + 80].decode("ascii", "replace")
        pos += 80
        key = card[:8].strip()
        if key == "END":
            break
        if "=" in card[:10]:
            val = card[10:30].strip()
            cards[key] = int(float(val)) if val not in ("T", "F") else val == "T"
    pos = (pos + 2879) // 2880 * 2880
    w, h = cards["NAXIS1"], cards["NAXIS2"]
    dtype = { -32: ">f4", -64: ">f8" }[cards["BITPIX"]]
    n = w * h * np.dtype(dtype).itemsize
    return np.frombuffer(raw[pos:pos + n], dtype=dtype).reshape(h, w).astype(np.float64)


def centroid_moment(img, x0, y0, r=8):
    """矩法质心 + 二阶矩 FWHM（独立实现）。"""
    h, w = img.shape
    xs = np.arange(max(0, int(x0) - r), min(w, int(x0) + r + 1))
    ys = np.arange(max(0, int(y0) - r), min(h, int(y0) + r + 1))
    xx, yy = np.meshgrid(xs, ys)
    sub = img[np.ix_(ys, xs)]
    valid = np.isfinite(sub)
    bg = np.median(sub[valid])
    wgt = np.where(valid, np.clip(sub - bg, 0, None), 0.0)
    tot = wgt.sum()
    cx = (wgt * xx).sum() / tot
    cy = (wgt * yy).sum() / tot
    vx = (wgt * (xx - cx) ** 2).sum() / tot
    vy = (wgt * (yy - cy) ** 2).sum() / tot
    f = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return cx, cy, f * np.sqrt(vx), f * np.sqrt(vy)


def aperture_flux(img, x, y, r_ap=6.0, r_in=10.0, r_out=14.0):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.hypot(xx - x, yy - y)
    valid = np.isfinite(img)
    ann = (d >= r_in) & (d < r_out) & valid
    bg = np.median(img[ann]) if ann.any() else 0.0
    ap = (d <= r_ap)
    apv = ap & valid
    frac = apv.sum() / max(ap.sum(), 1)
    return float((img[apv] - bg).sum() / max(frac, 1e-9))


def find_peak(img):
    valid = np.isfinite(img)
    idx = np.argmax(np.where(valid, img, -np.inf))
    return np.unravel_index(idx, img.shape)[::-1]  # (x, y)


def main(argv=None):
    ap = argparse.ArgumentParser(description="独立比较两份融合输出")
    ap.add_argument("fused_a")
    ap.add_argument("fused_b")
    ap.add_argument("--cov-a")
    ap.add_argument("--cov-b")
    ap.add_argument("--tol-diff", type=float, default=1e-6,
                    help="共同有效区最大绝对差容差（adu）")
    ap.add_argument("--tol-flux", type=float, default=0.01,
                    help="星点流量相对差容差")
    ap.add_argument("--tol-fwhm", type=float, default=0.05,
                    help="FWHM 差容差（像素）")
    args = ap.parse_args(argv)

    a, b = load(args.fused_a), load(args.fused_b)
    ok = True

    def report(name, cond, detail):
        nonlocal ok
        ok = ok and cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    report("形状一致", a.shape == b.shape, f"{a.shape} vs {b.shape}")
    if a.shape != b.shape:
        return 1

    va, vb = np.isfinite(a), np.isfinite(b)
    both = va & vb
    report("有效区（NaN 掩膜）一致", np.array_equal(va, vb),
           f"仅A有效 {(va & ~vb).sum()} 像素，仅B有效 {(vb & ~va).sum()} 像素")
    if both.any():
        d = np.abs(a[both] - b[both])
        report("共同有效区数值一致", d.max() <= args.tol_diff,
               f"max={d.max():.3e} rms={np.sqrt((d ** 2).mean()):.3e} "
               f"（容差 {args.tol_diff:g}）")

    if args.cov_a and args.cov_b:
        ca, cb = load(args.cov_a), load(args.cov_b)
        dc = np.abs(ca - cb).max()
        report("覆盖图一致", dc <= 1e-9, f"max|Δcoverage|={dc:.3e}")

    xa, ya = find_peak(a)
    cxa, cya, fx_a, fy_a = centroid_moment(a, xa, ya)
    cxb, cyb, fx_b, fy_b = centroid_moment(b, cxa, cya)
    fa, fb = aperture_flux(a, cxa, cya), aperture_flux(b, cxb, cyb)
    report("星点位置一致", abs(cxa - cxb) < 0.5 and abs(cya - cyb) < 0.5,
           f"A=({cxa:.2f},{cya:.2f}) B=({cxb:.2f},{cyb:.2f})")
    report("FWHM 一致", abs(fx_a - fx_b) < args.tol_fwhm
           and abs(fy_a - fy_b) < args.tol_fwhm,
           f"A=({fx_a:.3f},{fy_a:.3f})px B=({fx_b:.3f},{fy_b:.3f})px")
    rel = abs(fa / fb - 1.0) if fb else float("inf")
    report("星点流量一致", rel < args.tol_flux,
           f"A={fa:.1f} B={fb:.1f} 相对差={rel * 100:.3f}%")

    print(json.dumps({"fwhm_a": [fx_a, fy_a], "fwhm_b": [fx_b, fy_b],
                      "flux_a": fa, "flux_b": fb,
                      "centroid_a": [cxa, cya], "centroid_b": [cxb, cyb]},
                     ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
