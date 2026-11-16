"""离线可恢复任务的验证套件。

覆盖：依赖锁/线程核对、坏输入帧、失败留痕且禁产物、取消与退出码、
检查点恢复、输出 SNR 与帧清单实测、缓存成片指纹拦截、
以及图像处理页后台任务的状态机（用 Flask 测试客户端，不依赖外部服务）。

运行：OMP_NUM_THREADS=1 ... .venv/bin/python tests/test_offline.py
退出码 0 = 全部通过。
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrostack import io as aio
from astrostack.offline import (STATUS_CANCELLED, STATUS_FAILED,
                                STATUS_SUCCEEDED, JobConfig, JobError,
                                OfflineJob, verify_dependency_lock,
                                verify_thread_env)
from astrostack.quality import (build_frame_manifest, hash_array,
                                output_snr, summarize_manifest)
from astrostack.simulate import SimConfig, make_sequence

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{PASS if cond else FAIL}] {name}  {detail}")


REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="astrostack-offline-"))


def _need_thread_env():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(var) != "1":
            print("跳过需要单线程环境的测试：请用 scripts/run_server.sh 同样的"
                  "环境变量运行本测试")
            return False
    return True


def _make_demo_frames(d):
    sim = make_sequence(SimConfig(n_frames=16, n_noise_frames=1,
                                  n_edge_frames=1))
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(sim["frames"]):
        aio.save_npy(d / f"frame_{i:03d}.npy", f)
    (d / "truth.json").write_text(json.dumps({
        "shifts": sim["shifts"].tolist(),
        "stars": [dict(x0=s.x0, y0=s.y0, peak=s.peak, sigma=s.sigma)
                  for s in sim["stars"]]}), encoding="utf-8")
    return d


def test_dependency_lock():
    ev = verify_dependency_lock(REPO / "requirements.txt")
    check("依赖锁核对通过且记录实际版本",
          ev["actual"] == ev["locked"] and ev["locked"]["numpy"] == "2.1.3",
          str(ev["actual"]))
    bad_req = TMP / "bad-req.txt"
    bad_req.write_text("numpy==0.0.1\n")
    try:
        verify_dependency_lock(bad_req)
        ok = False
    except JobError:
        ok = True
    check("依赖版本不符时拒绝运行", ok)


def test_lock_failure_retains_inputs_no_artifacts():
    if not _need_thread_env():
        return
    bad_req = TMP / "wrong-req.txt"
    bad_req.write_text("numpy==2.1.2\nscipy==1.14.1\nflask==3.0.3\n"
                       "pillow==11.0.0\n")
    frames_dir = _make_demo_frames(TMP / "demo_lock")
    job_dir = TMP / "job_lockfail"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(bad_req))
    state = OfflineJob.create(cfg).run()
    check("依赖锁不符时任务失败", state["status"] == STATUS_FAILED)
    check("锁失败仍保留全部输入快照",
          len(list((job_dir / "inputs").glob("*frame_*.npy"))) == 18)
    check("锁失败无任何产物文件",
          len(list((job_dir / "outputs").glob("*"))) == 0
          and (job_dir / "error.json").exists())


def test_thread_env():
    if not _need_thread_env():
        return
    ev = verify_thread_env(1)
    check("线程设置锁定为 1 且记录环境变量",
          ev["limit"] == 1 and all(v == "1" for v in ev["env"].values()),
          str(ev["env"]))


def test_full_job_and_evidence():
    if not _need_thread_env():
        return
    frames_dir = _make_demo_frames(TMP / "demo")
    job_dir = TMP / "job_ok"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"),
                    truth_json=str(frames_dir / "truth.json"),
                    name="unit-job")
    state = OfflineJob.create(cfg).run()
    check("离线任务成功", state["status"] == STATUS_SUCCEEDED, state.get("error"))
    ev = json.loads((job_dir / "evidence.json").read_text())
    snr = ev["output_snr"]
    check("输出信噪比为实测有限正值",
          snr and np.isfinite(snr["snr_aperture"]) and snr["snr_aperture"] > 10,
          f"孔径SNR={snr and round(snr['snr_aperture'], 2)}")
    ms = ev["frame_manifest_summary"]
    check("帧清单完整：18 帧、16 参与、2 门控、0 坏输入",
          ms == {"n_frames": 18, "n_used": 16, "n_gated": 2,
                 "n_bad_input": 0, "missing_hash": [], "complete": True},
          str(ms))
    manifest = json.loads((job_dir / "outputs/frame_manifest.json").read_text())
    check("每帧都有输入指纹与配准记录",
          all(m["sha256"] and m["path"] for m in manifest)
          and {m["index"] for m in manifest} == set(range(18)))
    check("被剔除帧在清单中标注原因且 used=False",
          any((not m["used"]) and m["gated"] and m["reasons"]
              for m in manifest))
    # 产物指纹与回读一致
    fused = aio.load_npy(job_dir / "outputs/fused.npy")
    check("融合产物内容指纹与证据一致",
          hash_array(fused.astype(np.float32))
          == ev["artifact_content_hash"]["fused"])
    check("产物文件 sha256 与证据一致",
          all(__import__("astrostack.quality", fromlist=["file_sha256"])
              .file_sha256(job_dir / "outputs" / n) == h
              for n, h in ev["artifacts"].items()))
    coverage = aio.load_npy(job_dir / "outputs/coverage.npy")
    nan_mask = ~np.isfinite(fused)
    masks_agree = np.array_equal(nan_mask, coverage == 0)
    no_zero_dilution = (not nan_mask.any()) or not (fused[nan_mask] == 0).any()
    check("无覆盖像素为 NaN（而非 0），NaN 掩膜与覆盖图一致",
          masks_agree and no_zero_dilution,
          f"NaN 像素 {nan_mask.sum()} 个")
    report = json.loads((job_dir / "outputs/report.json").read_text())
    check("报告异常带分类标签",
          all(isinstance(a, dict) and a["kind"]
              in ("star", "registration", "coverage")
              for a in report["anomalies"]))


def test_bad_input_fails_without_artifacts():
    if not _need_thread_env():
        return
    bd = TMP / "bad_frames"
    bd.mkdir(exist_ok=True)
    good = make_sequence(SimConfig(n_frames=3))["frames"]
    aio.save_npy(bd / "frame_000.npy", good[0])
    bad = good[1].copy()
    bad[0, 0] = np.nan
    aio.save_npy(bd / "frame_001.npy", bad)
    aio.save_npy(bd / "frame_002.npy", good[2])
    job_dir = TMP / "job_bad"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(bd / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"))
    state = OfflineJob.create(cfg).run()
    check("坏输入帧导致任务失败", state["status"] == STATUS_FAILED)
    err = json.loads((job_dir / "error.json").read_text())
    check("error.json 记录退出信息", "坏输入帧" in err["error"]
          and err["error_type"] == "JobError", err["error"])
    check("坏帧保留 .error.txt 说明",
          (job_dir / "inputs").glob("*.error.txt") and
          any(job_dir.glob("inputs/*.error.txt")))
    check("必要输入已快照保留",
          len(list((job_dir / "inputs").glob("*_frame_*.npy"))) == 3)
    check("失败任务不产出 fused.npy（禁止采用产物）",
          not (job_dir / "outputs/fused.npy").exists())


def test_cancel_and_resume():
    if not _need_thread_env():
        return
    frames_dir = TMP / "demo"
    job_dir = TMP / "job_cancel"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"))
    job = OfflineJob.create(cfg)
    job.request_cancel()
    state = job.run()
    check("预置 cancel.flag 的任务以 cancelled 退出",
          state["status"] == STATUS_CANCELLED)
    check("取消退出码信息为 JobCancelled",
          json.loads((job_dir / "error.json").read_text())
          ["error_type"] == "JobCancelled")
    check("取消后无产物但输入保留",
          not (job_dir / "outputs/fused.npy").exists()
          and len(list((job_dir / "inputs").glob("*_frame_*.npy"))) == 18)
    # resume：清除 cancel.flag 后续跑成功
    state2 = job.run()
    check("取消后 resume 可续跑成功", state2["status"] == STATUS_SUCCEEDED)
    check("resume 后产物齐备",
          (job_dir / "outputs/fused.npy").exists()
          and (job_dir / "outputs/frame_manifest.json").exists())


def test_checkpoint_resume_skips_stages():
    if not _need_thread_env():
        return
    frames_dir = TMP / "demo"
    job_dir = TMP / "job_resume"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"))
    job = OfflineJob.create(cfg)
    job.run()
    first = json.loads((job_dir / "evidence.json").read_text())
    # 模拟只丢产物、检查点仍在：resume 应复用检查点并给出相同内容指纹
    shutil.rmtree(job_dir / "outputs")
    (job_dir / "state.json").write_text(json.dumps(
        {"status": "failed", "stage": "publish", "progress": 0.9}))
    job2 = OfflineJob(job_dir)
    state = job2.run()
    second = json.loads((job_dir / "evidence.json").read_text())
    check("丢产物后靠检查点恢复成功", state["status"] == STATUS_SUCCEEDED)
    check("恢复后内容指纹与首次一致",
          second["artifact_content_hash"] == first["artifact_content_hash"])
    check("恢复后 SNR 实测值一致",
          second["output_snr"]["snr_aperture"]
          == first["output_snr"]["snr_aperture"])


def test_succeeded_job_with_lost_outputs_rebuilds():
    if not _need_thread_env():
        return
    frames_dir = TMP / "demo"
    job_dir = TMP / "job_lost_outputs"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"))
    job = OfflineJob.create(cfg)
    job.run()
    assert (job_dir / "outputs/fused.npy").exists()
    # 外部只删产物、保留检查点与 succeeded 状态
    shutil.rmtree(job_dir / "outputs")
    state = OfflineJob(job_dir).run()
    check("成功状态但产物缺失时不允许短路，重新发布成功",
          state["status"] == STATUS_SUCCEEDED
          and (job_dir / "outputs/fused.npy").exists()
          and (job_dir / "outputs/frame_manifest.json").exists())
    ev = json.loads((job_dir / "evidence.json").read_text())
    check("重建后 SNR 仍为实测值", ev["output_snr"]["snr_aperture"] > 10)


def test_cached_artifacts_rejected():
    if not _need_thread_env():
        return
    frames_dir = TMP / "demo"
    job_dir = TMP / "job_cache"
    cfg = JobConfig(job_dir=str(job_dir),
                    frames_glob=str(frames_dir / "frame_*.npy"),
                    requirements=str(REPO / "requirements.txt"))
    job = OfflineJob.create(cfg)
    job.run()
    result = job._load_stack_ckpt(
        [aio.load_npy(p) for p in sorted((job_dir / "inputs").glob("*.npy"))],
        job._load_estimate_ckpt())
    expected = hash_array(result.fused.astype(np.float32))
    # 模拟写盘后被掉包：首次回读 fused.npy 返回污染数组
    real_load = aio.load_npy
    state = {"n": 0}

    def poisoned(path):
        arr = real_load(path)
        if str(path).endswith("fused.npy"):
            state["n"] += 1
            if state["n"] == 1:
                return arr + 9000.0
        return arr

    aio.load_npy = poisoned
    try:
        try:
            job._publish(result, {"output_snr": {"snr_aperture": 50.0},
                                 "n_used": 16, "psf": {"fwhm_mean": 3.5},
                                 "anomalies": []},
                         [{"index": 0, "sha256": "x"}],
                         {"complete": True, "n_used": 16},
                         {"actual": {}, "locked": {}},
                         {"limit": 1, "env": {}}, {}, [])
            ok = False
        except JobError as e:
            ok = "指纹" in str(e)
    finally:
        aio.load_npy = real_load
    check("指纹校验拦截掉包的缓存成片", ok)
    check("被拒后真实 outputs 未被污染",
          hash_array(real_load(job_dir / "outputs/fused.npy")
                     .astype(np.float32)) == expected)


def test_output_snr_measure():
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 2.0, (64, 64))
    yy, xx = np.mgrid[0:64, 0:64]
    img += 3000 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 1.5 ** 2))
    snr = output_snr(img, 32, 32)
    check("output_snr 在仿真星点上给出高信噪比",
          snr and snr["snr_aperture"] > 50 and snr["snr_peak"] > 50,
          f"孔径={snr and round(snr['snr_aperture'], 1)}")
    flat = np.full((64, 64), 100.0)
    check("无星点信号时 output_snr 返回 None 而非默认值",
          output_snr(flat, 32, 32) is None)


def test_manifest_helpers():
    class E:
        def __init__(self, i, ok, reasons=()):
            self.index, self.ok, self.reasons = i, ok, list(reasons)
            self.dx = self.dy = 0.1
            self.snr, self.weight = 20.0, 0.01
    inputs = [{"index": 0, "name": "a", "path": "a", "sha256": "h0",
               "bytes": 1, "shape": [2, 2]},
              {"index": 1, "name": "b", "path": "b", "sha256": None,
               "bytes": 1, "shape": [2, 2], "error": "bad"}]
    manifest = build_frame_manifest(inputs, [E(0, True), E(1, False, ["x"])],
                                    [0])
    summary = summarize_manifest(manifest)
    check("帧清单汇总反映坏输入与缺指纹",
          summary["n_bad_input"] == 1 and summary["complete"] is False
          and summary["n_used"] == 1 and summary["n_gated"] == 1,
          str(summary))


def test_web_offline_endpoints():
    if not _need_thread_env():
        return
    from astrostack.server import app

    c = app.test_client()
    r = c.post("/api/offline/start", json={})
    check("无曝光组时页面离线任务被拒(400)", r.status_code == 400)
    assert c.post("/api/demo").status_code == 200
    r = c.post("/api/offline/start", json={"reference": 0})
    check("页面可发起离线任务", r.status_code == 200)
    jid = r.get_json()["job_id"]
    for _ in range(120):
        snap = c.get(f"/api/offline/status/{jid}").get_json()
        if snap["status"] in (STATUS_SUCCEEDED, STATUS_FAILED,
                              STATUS_CANCELLED):
            break
        time.sleep(0.5)
    check("页面离线任务成功且状态可见",
          snap["status"] == STATUS_SUCCEEDED
          and snap["evidence"]["output_snr"]["snr_aperture"] > 10,
          str(snap.get("status")))
    st = c.get("/api/state").get_json()
    check("成功后真实产物载入会话供页面视图展示",
          st["has_stack"] and st["report"]["output_snr"] is not None)
    for view in ("fused", "coverage", "residual"):
        rr = c.get(f"/api/render?view={view}&frame=0")
        check(f"页面视图 {view} 可渲染",
              rr.status_code == 200 and rr.data[:4] == b"\x89PNG")
    rr = c.get("/api/cutout?view=fused&half=12")
    check("星点细节切块可载入", rr.status_code == 200
          and len(rr.get_json()["data"]) > 0)
    rr = c.get("/api/offline/status/does-not-exist")
    check("未知任务返回 404 而非默认值遮错", rr.status_code == 404)


def main():
    test_dependency_lock()
    test_lock_failure_retains_inputs_no_artifacts()
    test_thread_env()
    test_output_snr_measure()
    test_manifest_helpers()
    test_full_job_and_evidence()
    test_bad_input_fails_without_artifacts()
    test_cancel_and_resume()
    test_checkpoint_resume_skips_stages()
    test_succeeded_job_with_lost_outputs_rebuilds()
    test_cached_artifacts_rejected()
    test_web_offline_endpoints()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} 通过")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
