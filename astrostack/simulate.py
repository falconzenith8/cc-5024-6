"""生成带已知小数位移的模拟短曝光序列，用于验证与演示。

星点用**解析高斯**直接在各帧自身坐标系下求值（中心在 x0+dx, y0+dy），
而不是先画一幅真值图再平移插值——避免验证时被同一个插值器"自我印证"。

噪声模型：泊松（光子）+ 高斯（读出）；帧间亮度按比例因子变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import COORD_UNIT  # noqa: F401  (单位约定见 conventions)


@dataclass
class Star:
    x0: float       # 参考帧中的中心 x（像素，中心在整数坐标）
    y0: float
    peak: float     # 峰值（adu，参考亮度下）
    sigma: float    # 高斯 sigma（像素，各向同性）

    @property
    def flux(self):
        return 2.0 * np.pi * self.sigma**2 * self.peak


@dataclass
class SimConfig:
    size: int = 128
    n_frames: int = 16
    stars: tuple = (
        Star(64.2, 63.7, 4000.0, 1.5),
        Star(89.5, 45.3, 1200.0, 1.5),
    )
    sky: float = 100.0            # 天空背景（adu）
    read_noise: float = 5.0       # 读出噪声（adu）
    shift_range: float = 3.0      # 位移幅值上限（像素，各轴均匀采样）
    brightness: tuple = (0.8, 1.2)  # 帧间亮度因子范围
    seed: int = 20261116
    n_noise_frames: int = 0       # 追加的纯噪声帧（应被剔除）
    n_edge_frames: int = 0        # 追加的星点贴近边缘帧（应被剔除）


def _gaussian_grid(size, cx, cy, peak, sigma):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))


def make_sequence(cfg: SimConfig = SimConfig()):
    """返回 dict：
    frames (N,H,W) float64、shifts (N,2) 真值位移、brightness (N,)、
    truth 参考帧无噪声图像、stars、sky、read_noise、bad_frames 索引集。
    """
    rng = np.random.default_rng(cfg.seed)
    size = cfg.size
    truth = np.full((size, size), cfg.sky, dtype=np.float64)
    for s in cfg.stars:
        truth += _gaussian_grid(size, s.x0, s.y0, s.peak, s.sigma)

    n = cfg.n_frames
    shifts = rng.uniform(-cfg.shift_range, cfg.shift_range, size=(n, 2))
    bright = rng.uniform(*cfg.brightness, size=n)

    frames = []
    all_shifts = [tuple(shifts[i]) for i in range(n)]
    all_bright = list(bright)
    bad = set()

    for i in range(n):
        dx, dy = shifts[i]
        clean = np.full((size, size), cfg.sky)
        for s in cfg.stars:
            clean += _gaussian_grid(size, s.x0 + dx, s.y0 + dy,
                                    s.peak * bright[i], s.sigma)
        noisy = rng.poisson(np.clip(clean, 0, None)).astype(np.float64)
        noisy += rng.normal(0.0, cfg.read_noise, size=(size, size))
        frames.append(noisy)

    for _ in range(cfg.n_noise_frames):  # 纯噪声帧：无星点
        idx = len(frames)
        bad.add(idx)
        all_shifts.append((0.0, 0.0))
        all_bright.append(1.0)
        frames.append(rng.poisson(cfg.sky, size=(size, size)).astype(np.float64)
                      + rng.normal(0, cfg.read_noise, size=(size, size)))

    for _ in range(cfg.n_edge_frames):  # 星点被推到边缘的帧
        idx = len(frames)
        bad.add(idx)
        dx = size / 2 - 4.0   # 星心距边缘约 4px，应被边缘门限剔除
        dy = 0.0
        all_shifts.append((dx, dy))
        all_bright.append(1.0)
        clean = np.full((size, size), cfg.sky)
        for s in cfg.stars:
            clean += _gaussian_grid(size, s.x0 + dx, s.y0 + dy, s.peak, s.sigma)
        noisy = rng.poisson(np.clip(clean, 0, None)).astype(np.float64)
        noisy += rng.normal(0.0, cfg.read_noise, size=(size, size))
        frames.append(noisy)

    return {
        "frames": np.stack(frames),
        "shifts": np.array(all_shifts, dtype=np.float64),
        "brightness": np.array(all_bright, dtype=np.float64),
        "truth": truth,
        "stars": list(cfg.stars),
        "sky": cfg.sky,
        "read_noise": cfg.read_noise,
        "bad_frames": bad,
        "coord_unit": COORD_UNIT,
    }
