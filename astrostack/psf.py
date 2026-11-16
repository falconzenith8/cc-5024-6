"""PSF 与流量测量：二维高斯拟合（含旋转）、FWHM、孔径测光、矩法交叉核对。

所有测量在实际数据上进行；坐标单位为像素，像素中心在整数坐标。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


def moment_params(img, cx, cy, radius):
    """矩法估计：(cx, cy, amp, sigma_x, sigma_y, bg)。用作拟合初值与独立核对。"""
    h, w = img.shape
    r = int(np.ceil(radius))
    xs = np.arange(max(0, int(cx) - r), min(w, int(cx) + r + 1))
    ys = np.arange(max(0, int(cy) - r), min(h, int(cy) + r + 1))
    xx, yy = np.meshgrid(xs, ys)
    sub = img[np.ix_(ys, xs)].astype(np.float64)
    bg = float(np.median(sub))
    wgt = np.clip(sub - bg, 0, None)
    total = wgt.sum()
    if total <= 0:
        return None
    mx = float((wgt * xx).sum() / total)
    my = float((wgt * yy).sum() / total)
    vx = float((wgt * (xx - mx) ** 2).sum() / total)
    vy = float((wgt * (yy - my) ** 2).sum() / total)
    amp = float(sub[np.argmin(np.abs(ys - my)), np.argmin(np.abs(xs - mx))] - bg)
    return mx, my, max(amp, 1e-6), np.sqrt(max(vx, 1e-6)), np.sqrt(max(vy, 1e-6)), bg


def fit_gaussian2d(img, cx, cy, radius=8.0):
    """在 (cx, cy) 附近窗口内拟合二维旋转高斯 + 常数背景。

    返回 dict：x0, y0, amp, sigma_x, sigma_y, theta, bg,
    fwhm_x, fwhm_y, fwhm_mean, flux（高斯积分流量）, residual_rms。
    """
    init = moment_params(img, cx, cy, radius)
    if init is None:
        return None
    mx, my, amp, sx, sy, bg = init
    h, w = img.shape
    r = int(np.ceil(radius))
    xs = np.arange(max(0, int(mx) - r), min(w, int(mx) + r + 1))
    ys = np.arange(max(0, int(my) - r), min(h, int(my) + r + 1))
    xx, yy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    data = img[np.ix_(ys, xs)].astype(np.float64)

    def model(p):
        x0, y0, a, sx_, sy_, th, b = p
        ct, st = np.cos(th), np.sin(th)
        u = ct * (xx - x0) + st * (yy - y0)
        v = -st * (xx - x0) + ct * (yy - y0)
        return b + a * np.exp(-0.5 * ((u / sx_) ** 2 + (v / sy_) ** 2))

    def resid(p):
        return (model(p) - data).ravel()

    p0 = [mx, my, amp, sx, sy, 0.0, bg]
    lo = [xs[0], ys[0], 0, 0.2, 0.2, -np.pi / 2, -np.inf]
    hi = [xs[-1], ys[-1], np.inf, radius, radius, np.pi / 2, np.inf]
    sol = least_squares(resid, p0, bounds=(lo, hi))
    x0, y0, a, sx_, sy_, th, b = sol.x
    fx, fy = FWHM_FACTOR * sx_, FWHM_FACTOR * sy_
    return {
        "x0": float(x0), "y0": float(y0), "amp": float(a),
        "sigma_x": float(sx_), "sigma_y": float(sy_), "theta": float(th),
        "bg": float(b),
        "fwhm_x": float(fx), "fwhm_y": float(fy),
        "fwhm_mean": float(0.5 * (fx + fy)),
        "flux": float(2.0 * np.pi * a * sx_ * sy_),
        "residual_rms": float(np.sqrt(np.mean(sol.fun**2))),
    }


def aperture_photometry(img, x, y, r_ap=6.0, r_in=10.0, r_out=14.0):
    """孔径测光：r_ap 内求和减去环带 [r_in, r_out) 估计的背景。

    NaN 像素（无覆盖）被跳过，并按有效像素占比折算，避免把无覆盖当 0。
    返回 dict(flux, bg_per_pix, n_pix)。
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.hypot(xx - x, yy - y)
    ap = d <= r_ap
    ann = (d >= r_in) & (d < r_out)
    valid = np.isfinite(img)
    ann_vals = img[ann & valid]
    bg = float(np.median(ann_vals)) if ann_vals.size else 0.0
    ap_valid = ap & valid
    frac = ap_valid.sum() / max(ap.sum(), 1)
    flux = float((img[ap_valid] - bg).sum() / max(frac, 1e-9))
    return {"flux": flux, "bg_per_pix": bg, "n_pix": int(ap.sum())}
