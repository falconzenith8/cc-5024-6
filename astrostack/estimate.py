"""平移估计：主用强度质心，相位相关做交叉校验；质量不足的帧打回不参与叠加。

约定见 astrostack.conventions：
  dx_k = 帧k星点质心 x - 参考帧星点质心 x（像素，+dx 表示内容向 +x 移动）。

质量门限（任一不满足即剔除，给出原因，绝不强行参与）：
  * 未检测到星点（显著性不足）；
  * 星点距边缘不足 edge_margin（配准后星点会落到覆盖之外）；
  * 质心不确定度 sigma_c 超过 max_sigma_c；
  * 质心位移与相位相关位移不一致（> pc_tol），说明估计不可信。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class GateConfig:
    snr_min: float = 8.0        # 星点信噪比下限
    edge_margin: float = 10.0   # 星心距图像边缘的最小像素数
    max_sigma_c: float = 0.25   # 质心不确定度上限（像素）
    pc_tol: float = 1.0         # 质心与相位相关结果允许的最大分歧（像素）
    detect_sigma: float = 5.0   # 检测阈值（背景噪声倍数）
    centroid_radius: float = 6.0  # 质心窗口半径（像素）


@dataclass
class FrameEstimate:
    index: int
    ok: bool
    reasons: list = field(default_factory=list)
    dx: float = 0.0             # 相对参考帧的位移（像素）
    dy: float = 0.0
    cx: float = float("nan")    # 本帧星点质心（像素）
    cy: float = float("nan")
    sigma_cx: float = float("nan")
    sigma_cy: float = float("nan")
    snr: float = float("nan")
    flux: float = float("nan")
    background: float = float("nan")
    noise: float = float("nan")
    weight: float = 0.0         # 默认权重 = 1/背景噪声方差
    pc_dx: float = float("nan")  # 相位相关交叉校验位移
    pc_dy: float = float("nan")


def background_stats(img):
    """中位数背景 + MAD 噪声估计。"""
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    return med, 1.4826 * mad


def detect_star(img, bg, noise, k=5.0):
    """阈值分割取含全局最大值的连通域，返回该域掩膜；失败返回 None。"""
    mask = img > bg + k * noise
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    peak = np.unravel_index(np.argmax(img), img.shape)
    star_lab = lab[peak]
    if star_lab == 0:
        return None
    comp = lab == star_lab
    if comp.sum() < 9:
        return None
    return comp


def centroid(img, cx0, cy0, radius, bg, noise):
    """在 (cx0, cy0) 附近 radius 窗口内做背景扣除强度质心。

    返回 (cx, cy, sigma_cx, sigma_cy, flux, snr)。
    不确定度按误差传播：cx = sum(w_i x_i)/W，w_i = max(I_i - bg, 0)，
    像素噪声 sigma_i = sqrt(max(I_i,0) + noise^2)（泊松+读出）。
    """
    h, w = img.shape
    x0 = int(round(cx0))
    y0 = int(round(cy0))
    r = int(np.ceil(radius))
    xs = slice(max(0, x0 - r), min(w, x0 + r + 1))
    ys = slice(max(0, y0 - r), min(h, y0 + r + 1))
    sub = img[ys, xs].astype(np.float64) - bg
    yy, xx = np.mgrid[ys, xs].astype(np.float64)
    wgt = np.clip(sub, 0, None)
    total = float(wgt.sum())
    if total <= 0:
        return None
    cx = float((wgt * xx).sum() / total)
    cy = float((wgt * yy).sum() / total)
    pix_noise = np.sqrt(np.clip(img[ys, xs], 0, None) + noise**2)
    var_cx = float((((xx - cx) * pix_noise / total) ** 2).sum())
    var_cy = float((((yy - cy) * pix_noise / total) ** 2).sum())
    flux = total
    n_pix = wgt.size
    snr = flux / np.sqrt(flux + n_pix * noise**2)
    return cx, cy, np.sqrt(var_cx), np.sqrt(var_cy), flux, snr


def phase_correlation(ref, img):
    """相位相关求 img 相对 ref 的位移（抛物线拟合亚像素峰）。

    返回 (dx, dy, peak_ratio)。约定：img(x) = ref(x - d) 时返回 d，
    即与质心位移同号同向。
    """
    ref0 = ref - np.median(ref)
    img0 = img - np.median(img)
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    F = np.fft.fft2(ref0 * win)
    G = np.fft.fft2(img0 * win)
    cross = G * np.conj(F)
    mag = np.abs(cross)
    mag[mag == 0] = 1e-12
    corr = np.real(np.fft.ifft2(cross / mag))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape

    def _parab(a, b, c):
        denom = a - 2.0 * b + c
        return 0.0 if denom == 0 else 0.5 * (a - c) / denom

    dx = px + _parab(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    dy = py + _parab(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    peak_ratio = float(corr[py, px] / (np.sqrt((corr**2).mean()) + 1e-30))
    return dx, dy, peak_ratio


def measure_frame(img, index, cfg: GateConfig):
    """对单帧做检测+质心（不做帧间比较），供估计与参考帧测量复用。"""
    bg, noise = background_stats(img)
    comp = detect_star(img, bg, noise, cfg.detect_sigma)
    est = FrameEstimate(index=index, ok=False, background=bg, noise=noise)
    if comp is None:
        est.reasons.append("未检测到显著星点（低于 %.1fσ 阈值）" % cfg.detect_sigma)
        return est
    ys, xs = np.nonzero(comp)
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    for _ in range(2):  # 迭代重新定心
        out = centroid(img, cx0, cy0, cfg.centroid_radius, bg, noise)
        if out is None:
            est.reasons.append("质心窗口内无正流量")
            return est
        cx0, cy0 = out[0], out[1]
    cx, cy, scx, scy, flux, snr = out
    est.cx, est.cy = cx, cy
    est.sigma_cx, est.sigma_cy = scx, scy
    est.flux, est.snr = flux, snr
    est.weight = 1.0 / max(noise, 1e-9) ** 2
    return est


def estimate_shifts(frames, reference=0, cfg: GateConfig = GateConfig()):
    """估计所有帧相对 reference 帧的位移，返回 list[FrameEstimate]。

    参考帧自身的位移恒为 (0, 0)。未通过门限的帧 ok=False 且 weight=0。
    """
    frames = [np.asarray(f, dtype=np.float64) for f in frames]
    h, w = frames[reference].shape
    ref_est = measure_frame(frames[reference], reference, cfg)
    if ref_est.reasons:
        # 参考帧本身不可用：所有估计都失去基准，全部打回并说明。
        out = []
        for i in range(len(frames)):
            e = FrameEstimate(index=i, ok=False)
            e.reasons.append(f"参考帧 {reference} 测量失败："
                             + "；".join(ref_est.reasons))
            out.append(e)
        return out
    results = []

    ref_pc = frames[reference]
    for i, img in enumerate(frames):
        est = measure_frame(img, i, cfg)
        if est.reasons:
            results.append(est)
            continue
        m = cfg.edge_margin
        if not (m <= est.cx <= w - 1 - m and m <= est.cy <= h - 1 - m):
            est.reasons.append(
                f"星心 ({est.cx:.1f}, {est.cy:.1f}) 距边缘不足 {m:.0f}px")
            results.append(est)
            continue
        if est.snr < cfg.snr_min:
            est.reasons.append(f"信噪比 {est.snr:.1f} 低于下限 {cfg.snr_min:.1f}")
            results.append(est)
            continue
        if max(est.sigma_cx, est.sigma_cy) > cfg.max_sigma_c:
            est.reasons.append(
                f"质心不确定度 {max(est.sigma_cx, est.sigma_cy):.3f}px "
                f"超过上限 {cfg.max_sigma_c:.3f}px")
            results.append(est)
            continue
        est.dx = est.cx - ref_est.cx
        est.dy = est.cy - ref_est.cy
        pdx, pdy, _ = phase_correlation(ref_pc, img)
        est.pc_dx, est.pc_dy = pdx, pdy
        if abs(pdx - est.dx) > cfg.pc_tol or abs(pdy - est.dy) > cfg.pc_tol:
            est.reasons.append(
                f"质心位移 ({est.dx:+.2f},{est.dy:+.2f}) 与相位相关 "
                f"({pdx:+.2f},{pdy:+.2f}) 分歧超过 {cfg.pc_tol:.1f}px")
            results.append(est)
            continue
        est.ok = True
        results.append(est)

    if reference < len(results):
        results[reference].dx = 0.0
        results[reference].dy = 0.0
    return results
