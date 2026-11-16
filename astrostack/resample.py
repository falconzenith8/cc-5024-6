"""亚像素重采样：双线性插值 + 覆盖掩膜。

约定（见 astrostack.conventions）：
  aligned[y, x] = img(y + dy, x + dx)
即把帧内容逆着位移方向"拉回"参考网格。采样点落在源图有效域之外的
输出像素 coverage=0，其值不参与后续加权平均（不是按 0 参与）。

双线性是显式的四点加权平均，权重即面积权重，可逐像素解释；
它保持总流量（权重和恒为 1），代价是轻微的 PSF 展宽
（等效与宽度 2px 的三角核卷积），测试中会量化这一效应。
"""

from __future__ import annotations

import numpy as np


def shift_bilinear(img, dx, dy):
    """按 (dx, dy) 把 img 对齐到参考网格，返回 (aligned, coverage)。

    dx, dy 单位：像素；方向约定见 conventions。
    aligned 与 img 同形状 float64；coverage 为 float64 0/1。
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    ys = np.arange(h, dtype=np.float64) + dy
    xs = np.arange(w, dtype=np.float64) + dx

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0

    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)

    Ia = img[np.ix_(y0c, x0c)]
    Ib = img[np.ix_(y0c, x1c)]
    Ic = img[np.ix_(y1c, x0c)]
    Id = img[np.ix_(y1c, x1c)]

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    aligned = (wy0[:, None] * wx0[None, :] * Ia
               + wy0[:, None] * fx[None, :] * Ib
               + fy[:, None] * wx0[None, :] * Ic
               + fy[:, None] * fx[None, :] * Id)

    ok_x = (xs >= 0.0) & (xs <= w - 1)
    ok_y = (ys >= 0.0) & (ys <= h - 1)
    coverage = (ok_y[:, None] & ok_x[None, :]).astype(np.float64)
    return aligned, coverage
