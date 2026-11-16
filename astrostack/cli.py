"""辅助命令行（仅作天文融合的辅助手段；图形界面为主入口，见 docs/README.md）。

用法：
  python3 -m astrostack.cli simulate --out data/demo [--n 16]
  python3 -m astrostack.cli run --frames "data/demo/frame_*.npy" --out data/out
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from . import io
from .pipeline import run_pipeline
from .simulate import SimConfig, make_sequence


def cmd_simulate(args):
    sim = make_sequence(SimConfig(n_frames=args.n, seed=args.seed,
                                  n_noise_frames=1, n_edge_frames=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(sim["frames"]):
        io.save_npy(out / f"frame_{i:03d}.npy", f)
    io.save_npy(out / "truth.npy", sim["truth"])
    io.save_json(out / "truth.json", {
        "shifts": sim["shifts"].tolist(),
        "brightness": sim["brightness"].tolist(),
        "bad_frames": sorted(sim["bad_frames"]),
        "stars": [dict(x0=s.x0, y0=s.y0, peak=s.peak, sigma=s.sigma)
                  for s in sim["stars"]],
        "note": "shifts 为相对无位移真值图的位移（像素，+dx 内容向 +x 移动）",
    })
    print(f"已生成 {len(sim['frames'])} 帧到 {out}/（含 1 纯噪声帧、1 贴边帧，应被剔除）")


def cmd_run(args):
    files = sorted(glob.glob(args.frames))
    if not files:
        sys.exit(f"未找到帧文件: {args.frames}")
    frames, names = io.sniff_frames(files)
    print(f"载入 {len(frames)} 帧，参考帧 #{args.ref}")

    truth_shifts = None
    truth_star = None
    truth_json = Path(files[0]).parent / "truth.json"
    if truth_json.exists():
        import json
        meta = json.loads(truth_json.read_text())
        truth_shifts = np.array(meta["shifts"])
        from .simulate import Star
        s = meta["stars"][0]
        truth_star = Star(s["x0"], s["y0"], s["peak"], s["sigma"])

    result, report = run_pipeline(frames, reference=args.ref,
                                  truth_shifts=truth_shifts,
                                  truth_star=truth_star)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    io.save_npy(out / "fused.npy", result.fused)
    io.save_npy(out / "coverage.npy", result.coverage)
    io.save_npy(out / "weight_sum.npy", result.weight_sum)
    io.write_fits(out / "fused.fits", result.fused.astype(np.float32))
    io.write_fits(out / "coverage.fits", result.coverage.astype(np.float32))
    io.save_json(out / "report.json", report)

    print(f"参与叠加 {report['n_used']}/{report['n_frames']} 帧")
    for e in report["estimates"]:
        tag = "OK " if e["ok"] else "剔除"
        line = (f"  帧{e['index']:3d} [{tag}] dx={e['dx']:+8.4f} dy={e['dy']:+8.4f} "
                f"snr={e['snr']:6.1f} w={e['weight']:.5f}")
        if "dx_err" in e:
            line += f"  真值误差=({e['dx_err']:+.4f},{e['dy_err']:+.4f})"
        if e["reasons"]:
            line += "  " + "；".join(e["reasons"])
        print(line)
    if report["psf"]:
        p = report["psf"]
        print(f"PSF: FWHM=({p['fwhm_x']:.3f},{p['fwhm_y']:.3f})px 峰值={p['amp']:.1f}")
    if "energy" in report:
        en = report["energy"]
        print(f"能量误差: {en['rel_error'] * 100:+.2f}%  "
              f"(FWHM 真值 {en['fwhm_truth']:.3f}px)")
    for a in report["anomalies"]:
        print("异常提示:", a)
    print(f"输出: {out}/fused.npy, coverage.npy, fused.fits, report.json")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="astrostack-cli",
        description="天文短曝光叠加（辅助命令行；图形界面见 docs/README.md）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="生成带已知位移的模拟曝光组")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261116)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("run", help="估计+配准+加权融合")
    p.add_argument("--frames", required=True, help="帧文件 glob，如 'data/demo/frame_*.npy'")
    p.add_argument("--ref", type=int, default=0, help="参考帧索引")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
