"""已知真值验证套件：小数位移恢复、PSF 形状、总能量、噪声与亮度变化、剔除行为。

运行：python3 tests/test_registration.py
退出码 0 = 全部通过。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrostack.estimate import GateConfig, estimate_shifts
from astrostack.pipeline import run_pipeline
from astrostack.resample import shift_bilinear
from astrostack.simulate import SimConfig, Star, make_sequence
from astrostack.stack import stack_frames

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{PASS if cond else FAIL}] {name}  {detail}")


def _rel_shifts(sim, reference=0):
    """真值位移换算到"相对参考帧"基准（与估计值同基准）。"""
    return sim["shifts"] - sim["shifts"][reference]


def test_shift_recovery_high_snr():
    sim = make_sequence(SimConfig())
    est = estimate_shifts(sim["frames"], reference=0)
    truth = _rel_shifts(sim)
    errs = []
    for i, e in enumerate(est):
        assert e.ok, f"帧 {i} 应通过门控: {e.reasons}"
        errs.append([e.dx - truth[i][0], e.dy - truth[i][1]])
    errs = np.abs(errs)
    check("高信噪比位移恢复 max<0.05px", errs.max() < 0.05,
          f"max={errs.max():.4f}px rms={errs.std():.4f}px")


def test_shift_recovery_low_snr():
    cfg = SimConfig(stars=(Star(64.2, 63.7, 150.0, 1.5),), seed=7)
    sim = make_sequence(cfg)
    gates = GateConfig(snr_min=5.0, max_sigma_c=0.5)  # 放宽门限专测恢复精度
    est = estimate_shifts(sim["frames"], reference=0, cfg=gates)
    truth = _rel_shifts(sim)
    errs, zs = [], []
    for i, e in enumerate(est):
        if not e.ok:
            continue
        ex, ey = e.dx - truth[i][0], e.dy - truth[i][1]
        errs.append([ex, ey])
        # 相对位移的不确定度 = 本帧与参考帧质心不确定度的合成
        sig = np.hypot(e.sigma_cx, est[0].sigma_cx)
        zs.append(abs(ex) / max(sig, 1e-9))
    errs = np.abs(errs)
    check("低信噪比位移恢复与不确定度自洽 (max z<3.5)", max(zs) < 3.5,
          f"max z={max(zs):.2f}，max|err|={errs.max():.4f}px，"
          f"{len(errs)}/{len(est)} 帧通过门控")
    est_strict = estimate_shifts(sim["frames"], reference=0)
    n_ok = sum(1 for e in est_strict if e.ok)
    check("默认门限会拦截部分低质帧（不强行参与）", n_ok < len(est_strict),
          f"默认门限通过 {n_ok}/{len(est_strict)}")


def test_brightness_variation():
    cfg = SimConfig(brightness=(0.5, 1.5), seed=11)
    sim = make_sequence(cfg)
    est = estimate_shifts(sim["frames"], reference=0)
    truth = _rel_shifts(sim)
    errs = np.abs([[e.dx - s[0], e.dy - s[1]]
                   for e, s in zip(est, truth) if e.ok])
    check("亮度变化下位移恢复 max<0.08px", errs.max() < 0.08,
          f"max={errs.max():.4f}px")


def test_weight_reflects_noise():
    from astrostack.estimate import measure_frame
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    star = 3000 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 1.5**2))
    quiet = star + 100 + rng.normal(0, 5, (64, 64))
    noisy = star + 100 + rng.normal(0, 20, (64, 64))
    eq = measure_frame(quiet, 0, GateConfig())
    en = measure_frame(noisy, 1, GateConfig())
    ratio = eq.weight / en.weight
    check("权重按 1/噪声方差 区分帧质量", 10 < ratio < 25,
          f"权重比={ratio:.1f}（噪声 5 vs 20，期望≈16）")


def test_frame_rejection():
    cfg = SimConfig(n_noise_frames=1, n_edge_frames=1, seed=3)
    sim = make_sequence(cfg)
    res, report = run_pipeline(sim["frames"], reference=0,
                               truth_shifts=sim["shifts"],
                               truth_star=sim["stars"][0])
    rejected = {e["index"] for e in report["estimates"] if not e["ok"]}
    check("纯噪声帧与贴边帧被剔除", sim["bad_frames"] <= rejected,
          f"剔除={sorted(rejected)} 期望包含={sorted(sim['bad_frames'])}")
    check("被剔除帧未参与叠加", not (sim["bad_frames"] & set(res.used)),
          f"参与={res.used}")
    check("剔除原因写入报告",
          any("剔除" in (a["message"] if isinstance(a, dict) else a)
              for a in report["anomalies"]),
          f"anomalies={report['anomalies']}")


def test_no_zero_dilution():
    rng = np.random.default_rng(0)
    base = rng.normal(100.0, 3.0, size=(32, 32))
    f1 = base + rng.normal(0, 1, size=(32, 32))
    f2 = base + rng.normal(0, 1, size=(32, 32))
    frames = [f1, f2]

    class E:  # 手工构造位移 (+2.0,+2.0) 与 (+3.0,+3.0)
        def __init__(self, i, dx, dy):
            self.index, self.dx, self.dy = i, dx, dy
            self.ok, self.weight = True, 1.0

    res = stack_frames(frames, [E(0, 2.0, 2.0), E(1, 3.0, 3.0)])
    # +dx 对齐时在 x+dx 处采样：输出右下角的采样点越界，无覆盖。
    check("无覆盖区域为 NaN 而非 0",
          np.isnan(res.fused[30:, 30:]).all()
          and not np.isnan(res.fused[:29, :29]).any(),
          f"fused[31,31]={res.fused[31, 31]} fused[0,0]={res.fused[0, 0]:.1f}")
    a1, c1 = shift_bilinear(f1, 2.0, 2.0)
    a2, c2 = shift_bilinear(f2, 3.0, 3.0)
    num = c1 * a1 + c2 * a2
    den = c1 + c2
    with np.errstate(invalid="ignore", divide="ignore"):
        manual = np.where(den > 0, num / den, np.nan)
    diff = np.nanmax(np.abs(res.fused - manual))
    check("部分覆盖像素按实际贡献归一化", diff < 1e-12, f"max diff={diff:.2e}")
    # (29,29) 只有帧0覆盖：值必须等于帧0对齐值，而不是被帧1的"0"稀释。
    check("单帧覆盖像素不被零稀释",
          abs(res.fused[29, 29] - a1[29, 29]) < 1e-12,
          f"fused={res.fused[29, 29]:.3f} 帧0对齐值={a1[29, 29]:.3f} "
          f"（若被零稀释应≈{a1[29, 29] / 2:.3f}）")


def test_psf_and_flux():
    cfg = SimConfig()
    sim = make_sequence(cfg)
    star = sim["stars"][0]
    res, report = run_pipeline(sim["frames"], reference=0,
                               truth_shifts=sim["shifts"], truth_star=star)
    psf = report["psf"]
    fwhm_t = 2.354820045 * star.sigma
    check("融合 PSF 的 FWHM 接近真值（允许双线性轻微展宽）",
          0.9 * fwhm_t < psf["fwhm_mean"] < 1.10 * fwhm_t,
          f"测得={psf['fwhm_mean']:.3f}px 真值={fwhm_t:.3f}px")
    check("融合 PSF 保持近圆", abs(psf["fwhm_x"] / psf["fwhm_y"] - 1) < 0.1,
          f"x/y={psf['fwhm_x'] / psf['fwhm_y']:.3f}")
    e = report["energy"]
    check("总能量误差 <3%", abs(e["rel_error"]) < 0.03,
          f"rel_err={e['rel_error'] * 100:+.2f}%")


def test_reference_change():
    sim = make_sequence(SimConfig(seed=23))
    est0 = estimate_shifts(sim["frames"], reference=0)
    est3 = estimate_shifts(sim["frames"], reference=3)
    diffs = []
    for a, b in zip(est0, est3):
        if a.ok and b.ok:
            diffs.append(abs(b.dx - (a.dx - est0[3].dx)))
            diffs.append(abs(b.dy - (a.dy - est0[3].dy)))
    check("更换参考帧后位移自洽 (dx'_k = dx_k - dx_ref)",
          max(diffs) < 1e-9, f"max diff={max(diffs):.2e}px")


def main():
    test_shift_recovery_high_snr()
    test_shift_recovery_low_snr()
    test_brightness_variation()
    test_frame_rejection()
    test_no_zero_dilution()
    test_psf_and_flux()
    test_reference_change()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} 通过")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
