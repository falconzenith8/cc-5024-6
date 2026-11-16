"""辅助命令行（仅作天文融合的辅助手段；图形界面为主入口，见 docs/README.md）。

用法：
  python3 -m astrostack.cli simulate --out data/demo [--n 16]
  python3 -m astrostack.cli run --frames "data/demo/frame_*.npy" --out data/out
  python3 -m astrostack.cli offline run    --job-dir runs/demo --frames "data/demo/frame_*.npy"
  python3 -m astrostack.cli offline status --job-dir runs/demo
  python3 -m astrostack.cli offline cancel --job-dir runs/demo
  python3 -m astrostack.cli offline resume --job-dir runs/demo
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from . import io
from .offline import (STATUS_CANCELLED, STATUS_FAILED, STATUS_SUCCEEDED,
                      JobConfig, OfflineJob)
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
        msg = a["message"] if isinstance(a, dict) else a
        print("异常提示:", msg)
    print(f"输出: {out}/fused.npy, coverage.npy, fused.fits, report.json")


def _print_job_footer(state, job):
    print(f"任务状态: {state['status']}（阶段 {state.get('stage')}，"
          f"进度 {state.get('progress', 0) * 100:.0f}%）")
    if state["status"] == STATUS_SUCCEEDED:
        ev = json.loads(job.evidence_path.read_text())
        snr = ev["output_snr"]
        ms = ev["frame_manifest_summary"]
        print(f"输出信噪比: 孔径 SNR={snr['snr_aperture']:.2f} "
              f"峰值 SNR={snr['snr_peak']:.2f}")
        print(f"帧清单: 共 {ms['n_frames']} 帧，参与 {ms['n_used']}，"
              f"门控剔除 {ms['n_gated']}，坏输入 {ms['n_bad_input']}")
        print(f"产物目录: {job.out_dir}")
    elif state["status"] in (STATUS_FAILED, STATUS_CANCELLED):
        if job.error_path.exists():
            err = json.loads(job.error_path.read_text())
            print(f"退出信息: [{err['error_type']}] {err['error']}")
        print(f"必要输入保留在: {job.inputs_dir}")


def cmd_offline(args):
    if args.offline_cmd == "run":
        cfg = JobConfig(
            job_dir=args.job_dir, frames_glob=args.frames, reference=args.ref,
            requirements=args.requirements, threads=args.threads,
            truth_json=args.truth_json, name=args.name)
        job = OfflineJob.create(cfg)
        state = job.run()
        _print_job_footer(state, job)
        sys.exit({STATUS_SUCCEEDED: 0, STATUS_CANCELLED: 130,
                  STATUS_FAILED: 1}[state["status"]])
    job = OfflineJob(args.job_dir)
    if args.offline_cmd == "status":
        state = job._read_state()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        if job.evidence_path.exists():
            print("--- evidence.json ---")
            print(job.evidence_path.read_text())
        sys.exit(0 if state["status"] == STATUS_SUCCEEDED else 1)
    if args.offline_cmd == "cancel":
        job.request_cancel()
        print(f"已写入 {job.cancel_path}；运行中的任务会在阶段/帧边界退出")
        sys.exit(0)
    if args.offline_cmd == "resume":
        state = job.run()
        _print_job_footer(state, job)
        sys.exit({STATUS_SUCCEEDED: 0, STATUS_CANCELLED: 130,
                  STATUS_FAILED: 1}[state["status"]])
    sys.exit(2)


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

    p = sub.add_parser("offline", help="可恢复离线任务（run/status/cancel/resume）")
    osub = p.add_subparsers(dest="offline_cmd", required=True)

    pr = osub.add_parser("run", help="创建并执行离线任务")
    pr.add_argument("--job-dir", required=True)
    pr.add_argument("--frames", required=True)
    pr.add_argument("--ref", type=int, default=0)
    pr.add_argument("--requirements", default="requirements.txt")
    pr.add_argument("--threads", type=int, default=1)
    pr.add_argument("--truth-json", default=None)
    pr.add_argument("--name", default="offline-job")
    pr.set_defaults(fn=cmd_offline)

    ps = osub.add_parser("status", help="查看任务状态与执行证据")
    ps.add_argument("--job-dir", required=True)
    ps.set_defaults(fn=cmd_offline)

    pc = osub.add_parser("cancel", help="请求取消运行中的任务")
    pc.add_argument("--job-dir", required=True)
    pc.set_defaults(fn=cmd_offline)

    pm = osub.add_parser("resume", help="恢复（或重试）离线任务")
    pm.add_argument("--job-dir", required=True)
    pm.set_defaults(fn=cmd_offline)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
