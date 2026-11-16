"""端到端流水线：估计 -> 门控 -> 对齐 -> 加权叠加 -> 测量，产出报告 dict。

报告同时供 CLI、Web 界面、离线任务与测试使用；所有数值都来自实际运算。
progress_cb(phase, fraction) 与 is_cancelled() 为可选钩子，供离线任务
上报进度与响应取消；普通调用不传时行为与原来完全一致。
"""

from __future__ import annotations

import numpy as np

from .estimate import GateConfig, estimate_shifts
from .psf import aperture_photometry, fit_gaussian2d, moment_params
from .quality import output_snr
from .stack import frame_residuals, stack_frames

# 异常分类：star=星点细节相关，registration=配准残差相关，coverage=覆盖/权重相关
KIND_STAR = "star"
KIND_REGISTRATION = "registration"
KIND_COVERAGE = "coverage"


def _anomaly(kind, message):
    return {"kind": kind, "message": message}


def build_report(result, estimates, reference, truth_shifts=None,
                 truth_star=None):
    """对 StackResult 与配准估计做实测测量，构造可 JSON 化报告。"""
    residuals = frame_residuals(result)

    ref_est = estimates[reference]
    sx, sy = ref_est.cx, ref_est.cy
    psf_fit = fit_gaussian2d(result.fused, sx, sy)
    phot = aperture_photometry(result.fused, sx, sy)
    mom = moment_params(result.fused, sx, sy, 8.0)

    anomalies = []
    for e in estimates:
        if not e.ok:
            kind = KIND_REGISTRATION
            if any("星点" in r for r in e.reasons):
                kind = KIND_STAR
            anomalies.append(_anomaly(
                kind, f"帧 {e.index} 被剔除：" + "；".join(e.reasons)))
    n = len(estimates)
    if result.coverage.size:
        low = result.coverage < 0.5 * max(1, len(result.used))
        frac = float(low.mean())
        if frac > 0.01:
            anomalies.append(_anomaly(
                KIND_COVERAGE,
                f"{frac * 100:.1f}% 的像素覆盖不足可用帧的一半"))
    ok_weights = [estimates[k].weight for k in result.used]
    if ok_weights and max(ok_weights) / min(ok_weights) > 10:
        anomalies.append(_anomaly(
            KIND_COVERAGE,
            "帧间权重差异超过 10 倍（亮度/噪声差异大），请确认加权合理"))

    # 配准残差异常：残差 RMS 相对背景噪声明显偏高的帧单独标出
    for k, r in residuals.items():
        noise = estimates[k].noise
        if np.isfinite(noise) and noise > 0 and r["rms"] > 3.0 * noise:
            anomalies.append(_anomaly(
                KIND_REGISTRATION,
                f"帧 {k} 配准残差 RMS={r['rms']:.2f} 超过本帧背景噪声 "
                f"{noise:.2f} 的 3 倍，配准可能失准"))
    # 星点细节异常：拟合失败或 FWHM 明显拉宽
    if psf_fit is None:
        anomalies.append(_anomaly(KIND_STAR, "融合图星点高斯拟合失败，无法给出 FWHM/流量"))
    else:
        if max(psf_fit["fwhm_x"], psf_fit["fwhm_y"]) > 2.5 * min(
                psf_fit["fwhm_x"], psf_fit["fwhm_y"]):
            anomalies.append(_anomaly(
                KIND_STAR,
                f"星点 FWHM 各向异性过大 "
                f"({psf_fit['fwhm_x']:.2f},{psf_fit['fwhm_y']:.2f})px，可能未配准好"))

    est_list = []
    for e in estimates:
        item = {
            "index": e.index, "ok": e.ok, "reasons": e.reasons,
            "dx": e.dx, "dy": e.dy, "cx": e.cx, "cy": e.cy,
            "sigma_cx": e.sigma_cx, "sigma_cy": e.sigma_cy,
            "snr": e.snr, "flux": e.flux, "background": e.background,
            "noise": e.noise, "weight": e.weight,
            "pc_dx": e.pc_dx, "pc_dy": e.pc_dy,
            "used": e.index in result.used,
        }
        if e.index in residuals:
            item["residual_rms"] = residuals[e.index]["rms"]
            item["residual_max"] = residuals[e.index]["max_abs"]
        if truth_shifts is not None:
            # 真值位移是相对无位移真值图的；估计值是相对参考帧的。
            # 比较前先把真值也换算到"相对参考帧"基准。
            t_ref = truth_shifts[reference]
            t = truth_shifts[e.index]
            item["dx_truth"] = float(t[0] - t_ref[0])
            item["dy_truth"] = float(t[1] - t_ref[1])
            if e.ok:
                item["dx_err"] = float(e.dx - item["dx_truth"])
                item["dy_err"] = float(e.dy - item["dy_truth"])
        est_list.append(item)

    snr = output_snr(result.fused, sx, sy)
    if snr is None:
        anomalies.append(_anomaly(KIND_STAR, "融合图输出信噪比无法实测（无有效星点信号）"))

    report = {
        "reference": reference,
        "n_frames": n,
        "n_used": len(result.used),
        "used": result.used,
        "estimates": est_list,
        "star_position": {"x": sx, "y": sy},
        "psf": psf_fit,
        "psf_moment": ({"sigma_x": mom[3], "sigma_y": mom[4],
                        "fwhm_x": 2.354820045 * mom[3],
                        "fwhm_y": 2.354820045 * mom[4]} if mom else None),
        "photometry": phot,
        "output_snr": snr,
        "coverage_stats": {
            "max": float(result.coverage.max()),
            "min": float(result.coverage.min()),
            "nan_fraction": float((result.coverage == 0).mean()),
        },
        "anomalies": anomalies,
    }

    if truth_star is not None and psf_fit is not None:
        tflux = truth_star.flux
        report["energy"] = {
            "flux_stacked": psf_fit["flux"],
            "flux_truth": float(tflux),
            "rel_error": float(psf_fit["flux"] / tflux - 1.0),
            "fwhm_truth": float(2.354820045 * truth_star.sigma),
        }
    return report, residuals


def run_pipeline(frames, reference=0, enabled=None, weights=None,
                 gates: GateConfig = GateConfig(), truth_shifts=None,
                 truth_star=None, keep_aligned=True,
                 progress_cb=None, is_cancelled=None):
    """返回 (result, report)。result 为 StackResult；report 为可 JSON 化的 dict。"""
    def _progress(phase, fraction):
        if progress_cb is not None:
            progress_cb(phase, float(fraction))

    def _cancel():
        return bool(is_cancelled and is_cancelled())

    frames = [np.asarray(f, dtype=np.float64) for f in frames]
    if _cancel():
        raise RuntimeError("任务已取消")
    _progress("estimate", 0.0)
    estimates = estimate_shifts(frames, reference=reference, cfg=gates)
    if _cancel():
        raise RuntimeError("任务已取消")
    _progress("stack", 0.0)
    result = stack_frames(frames, estimates, enabled=enabled, weights=weights,
                          keep_aligned=keep_aligned,
                          progress_cb=lambda f: _progress("stack", f),
                          is_cancelled=is_cancelled)
    _progress("measure", 0.0)
    report, _ = build_report(result, estimates, reference,
                             truth_shifts=truth_shifts,
                             truth_star=truth_star)
    _progress("done", 1.0)
    return result, report
