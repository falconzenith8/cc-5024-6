"""天文融合 Web 界面（图形主入口）。

启动：python3 -m astrostack.server [--port 8000]
然后浏览器打开 http://127.0.0.1:8000/

设计要点：
* 所有科学数据在服务端以 float64 保存；显示拉伸（百分位/对数）只影响
  /api/render 的 PNG 渲染，绝不写回数据。
* 修改参考帧 → /api/estimate 重新估计，并使旧融合结果失效（界面提示重算）。
* 星点细节与配准残差全部由服务端实际运算返回；异常在界面消息区提示。
"""

from __future__ import annotations

import argparse
import io as _io
import json
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, request, send_file
from PIL import Image

from . import io as aio
from .estimate import GateConfig, estimate_shifts
from .offline import STATUS_SUCCEEDED, JobConfig, OfflineJob
from .pipeline import run_pipeline
from .simulate import SimConfig, make_sequence
from .stack import frame_residuals

app = Flask(__name__, static_folder="web", static_url_path="/static")

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_JOBS_ROOT = REPO_ROOT / "runs" / "web"


def _sanitize(o):
    """NaN/Inf 不是合法 JSON，浏览器会拒解析；统一换成 None。"""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return v if (not isinstance(v, float) or np.isfinite(v)) else None
    return o


def safe_jsonify(obj):
    return app.response_class(
        response=json.dumps(_sanitize(obj), ensure_ascii=False),
        mimetype="application/json")

STATE = {
    "frames": None,        # list[np.ndarray float64 (H,W)]
    "names": [],
    "truth_shifts": None,  # 仅模拟数据有
    "truth_star": None,
    "reference": 0,
    "estimates": None,     # list[FrameEstimate]
    "stack": None,         # StackResult
    "report": None,
    "stack_key": None,     # (ref, enabled, weights) 用于判断结果是否过期
    "group_id": 0,         # 每载入一组曝光自增，客户端据此重置逐帧开关
}

# 图像处理页发起的离线任务：job_id -> {thread, dir, started_at}
WEB_JOBS = {}
WEB_JOBS_LOCK = threading.Lock()


# ---------------- 工具 ----------------

def _est_to_dict(e, used):
    return {
        "index": e.index, "ok": e.ok, "reasons": e.reasons,
        "dx": e.dx, "dy": e.dy, "cx": e.cx, "cy": e.cy,
        "sigma_cx": e.sigma_cx, "sigma_cy": e.sigma_cy,
        "snr": e.snr, "flux": e.flux, "background": e.background,
        "noise": e.noise, "weight": e.weight,
        "pc_dx": e.pc_dx, "pc_dy": e.pc_dy,
        "used": e.index in used,
    }


def _summary():
    s = {
        "n_frames": len(STATE["frames"]) if STATE["frames"] else 0,
        "names": STATE["names"],
        "reference": STATE["reference"],
        "has_estimates": STATE["estimates"] is not None,
        "has_stack": STATE["stack"] is not None,
        "has_truth": STATE["truth_shifts"] is not None,
        "group_id": STATE["group_id"],
    }
    if STATE["report"] is not None:
        s["report"] = STATE["report"]
        # 报告中的估计含残差与真值误差，优先用于展示
        s["estimates"] = STATE["report"]["estimates"]
    elif STATE["estimates"] is not None:
        used = STATE["stack"].used if STATE["stack"] else []
        s["estimates"] = [_est_to_dict(e, used) for e in STATE["estimates"]]
    if STATE["truth_shifts"] is not None:
        ref = STATE["reference"]
        t = STATE["truth_shifts"] - STATE["truth_shifts"][ref]
        s["truth_rel"] = t.tolist()
    return s


def _invalidate_stack():
    STATE["stack"] = None
    STATE["report"] = None
    STATE["stack_key"] = None


def _set_frames(frames, names):
    STATE["frames"] = [np.asarray(f, dtype=np.float64) for f in frames]
    STATE["names"] = names
    STATE["reference"] = 0
    STATE["estimates"] = None
    STATE["group_id"] += 1
    _invalidate_stack()


def _render_png(arr, lo=1.0, hi=99.0, log=False):
    """百分位拉伸渲染为 PNG（仅显示用，不改动科学数据）。NaN 显示为暗紫。"""
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        rgb = np.zeros(arr.shape + (3,), np.uint8)
    else:
        vals = arr[valid]
        if log:
            vals = np.log10(np.clip(vals, 1e-3, None))
        vlo, vhi = np.percentile(vals, [lo, hi])
        if vhi <= vlo:
            # 恒定图像（如全 1 的覆盖足迹）：非零常数显示为白，零显示为黑
            g = np.zeros_like(arr)
            g[valid] = 1.0 if vhi > 0 else 0.0
            gray = (g * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
            rgb[~valid] = (70, 20, 70)
            img = Image.fromarray(rgb)
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf
        if log:
            g = np.zeros_like(arr)
            g[valid] = (np.log10(np.clip(arr[valid], 1e-3, None)) - vlo) / (vhi - vlo)
        else:
            g = np.zeros_like(arr)
            g[valid] = (arr[valid] - vlo) / (vhi - vlo)
        g = np.clip(g, 0, 1)
        gray = (g * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[~valid] = (70, 20, 70)
    img = Image.fromarray(rgb)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------------- 页面与静态资源 ----------------

@app.get("/")
def index():
    return app.send_static_file("index.html")


# ---------------- API ----------------

@app.post("/api/demo")
def api_demo():
    sim = make_sequence(SimConfig(n_frames=16, n_noise_frames=1, n_edge_frames=1))
    _set_frames(list(sim["frames"]),
                [f"demo_{i:02d}" for i in range(len(sim["frames"]))])
    STATE["truth_shifts"] = sim["shifts"]
    STATE["truth_star"] = sim["stars"][0]
    return _do_estimate(0)


@app.post("/api/upload")
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return safe_jsonify({"error": "未选择文件"}), 400
    frames, names = [], []
    for f in files:
        try:
            arr = aio.read_image(f.stream)
        except Exception as exc:
            return safe_jsonify({"error": f"{f.filename}: {exc}"}), 400
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                frames.append(arr[i])
                names.append(f"{f.filename}#{i}")
        else:
            frames.append(arr)
            names.append(f.filename)
    if len(frames) < 2:
        return safe_jsonify({"error": "至少需要 2 帧"}), 400
    _set_frames(frames, names)
    STATE["truth_shifts"] = None
    STATE["truth_star"] = None
    return _do_estimate(0)


def _do_estimate(reference):
    STATE["reference"] = int(reference)
    STATE["estimates"] = estimate_shifts(
        STATE["frames"], reference=STATE["reference"], cfg=GateConfig())
    _invalidate_stack()  # 参考帧变更后旧融合结果失效，必须重算
    return safe_jsonify(_summary())


@app.post("/api/estimate")
def api_estimate():
    if STATE["frames"] is None:
        return safe_jsonify({"error": "尚未载入曝光组"}), 400
    ref = int(request.get_json(force=True).get("reference", 0))
    if not 0 <= ref < len(STATE["frames"]):
        return safe_jsonify({"error": "参考帧索引越界"}), 400
    return _do_estimate(ref)


@app.post("/api/stack")
def api_stack():
    if STATE["estimates"] is None:
        return safe_jsonify({"error": "请先估计位移"}), 400
    body = request.get_json(force=True)
    n = len(STATE["frames"])
    enabled = list(body.get("enabled", [True] * n))[:n]
    weights_in = body.get("weights", [None] * n)
    weights = []
    for w, e in zip(weights_in, STATE["estimates"]):
        weights.append(float(w) if w not in (None, "", 0, "0") else e.weight)
    try:
        result, report = run_pipeline(
            STATE["frames"], reference=STATE["reference"],
            enabled=enabled, weights=weights,
            truth_shifts=STATE["truth_shifts"],
            truth_star=STATE["truth_star"])
    except ValueError as exc:
        return safe_jsonify({"error": str(exc)}), 400
    STATE["stack"] = result
    STATE["report"] = report
    STATE["stack_key"] = (STATE["reference"], tuple(enabled), tuple(weights))
    return safe_jsonify(_summary())


@app.get("/api/state")
def api_state():
    return safe_jsonify(_summary())


@app.get("/api/render")
def api_render():
    """渲染视图为 PNG。lo/hi/log 仅为显示拉伸，不写回科学数据。"""
    view = request.args.get("view", "fused")
    frame = int(request.args.get("frame", 0))
    lo = float(request.args.get("lo", 1.0))
    hi = float(request.args.get("hi", 99.0))
    log = request.args.get("log", "0") == "1"

    arr = None
    if view == "frame" and STATE["frames"] is not None:
        arr = STATE["frames"][frame]
    elif view == "fused" and STATE["stack"] is not None:
        arr = STATE["stack"].fused
    elif view == "coverage" and STATE["stack"] is not None:
        arr = STATE["stack"].coverage
    elif view == "footprint" and STATE["stack"] is not None:
        arr = STATE["stack"].footprints.get(frame)
    elif view == "residual" and STATE["stack"] is not None:
        res = frame_residuals(STATE["stack"])
        if frame in res:
            arr = res[frame]["map"]
    if arr is None:
        arr = np.full((64, 64), np.nan)
    return send_file(_render_png(arr, lo, hi, log), mimetype="image/png")


@app.get("/api/cutout")
def api_cutout():
    """星点细节：返回以星点为中心的原始浮点切块（JSON），由前端放大显示。"""
    if STATE["report"] is None:
        return safe_jsonify({"error": "尚未融合"}), 400
    half = int(request.args.get("half", 12))
    view = request.args.get("view", "fused")
    frame = int(request.args.get("frame", 0))
    sx = STATE["report"]["star_position"]["x"]
    sy = STATE["report"]["star_position"]["y"]
    if view == "fused":
        src = STATE["stack"].fused
    elif view == "frame":
        src = STATE["frames"][frame]
    else:
        return safe_jsonify({"error": "未知视图"}), 400
    x0, y0 = int(round(sx)), int(round(sy))
    h, w = src.shape
    xs = slice(max(0, x0 - half), min(w, x0 + half + 1))
    ys = slice(max(0, y0 - half), min(h, y0 + half + 1))
    cut = src[ys, xs]
    data = [[(None if not np.isfinite(v) else round(float(v), 3)) for v in row]
            for row in cut]
    return safe_jsonify({
        "data": data, "cx": sx - xs.start, "cy": sy - ys.start,
        "x0": xs.start, "y0": ys.start, "view": view, "frame": frame,
    })


@app.get("/api/download/<kind>")
def api_download(kind):
    if STATE["stack"] is None:
        return safe_jsonify({"error": "尚未融合"}), 400
    buf = _io.BytesIO()
    if kind == "fused-npy":
        np.save(buf, STATE["stack"].fused.astype(np.float32))
        name = "fused.npy"
    elif kind == "coverage-npy":
        np.save(buf, STATE["stack"].coverage.astype(np.float32))
        name = "coverage.npy"
    elif kind == "fused-fits":
        aio.write_fits(buf, STATE["stack"].fused.astype(np.float32))
        name = "fused.fits"
    elif kind == "report-json":
        buf.write(json.dumps(_sanitize(STATE["report"]), ensure_ascii=False,
                             indent=2).encode("utf-8"))
        name = "report.json"
    else:
        return safe_jsonify({"error": "未知类型"}), 404
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=name)


# ---------------- 离线任务（图像处理页内交互，非 CLI） ----------------

def _job_snapshot(job_id, job_dir):
    state_path = Path(job_dir) / "state.json"
    if not state_path.exists():
        return {"job_id": job_id, "status": "unknown"}
    snap = json.loads(state_path.read_text())
    snap["job_id"] = job_id
    err_path = Path(job_dir) / "error.json"
    if err_path.exists():
        snap["error_detail"] = json.loads(err_path.read_text())
    ev_path = Path(job_dir) / "evidence.json"
    if ev_path.exists():
        snap["evidence"] = json.loads(ev_path.read_text())
    with WEB_JOBS_LOCK:
        info = WEB_JOBS.get(job_id)
        snap["alive"] = bool(info and info["thread"].is_alive())
    return snap


def _stage_web_frames(job_id):
    """把当前会话已载入的曝光组落盘为任务输入，返回 (frames_glob, truth_json)。"""
    if not STATE["frames"]:
        raise RuntimeError("尚未载入曝光组，无法发起离线任务")
    job_dir = OFFLINE_JOBS_ROOT / job_id
    src_dir = job_dir / "source_frames"
    src_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(STATE["frames"]):
        aio.save_npy(src_dir / f"frame_{i:03d}.npy", frame)
    truth_json = None
    if STATE["truth_shifts"] is not None and STATE["truth_star"] is not None:
        meta = {
            "shifts": np.asarray(STATE["truth_shifts"]).tolist(),
            "stars": [dict(x0=STATE["truth_star"].x0, y0=STATE["truth_star"].y0,
                           peak=STATE["truth_star"].peak,
                           sigma=STATE["truth_star"].sigma)],
        }
        tj = src_dir / "truth.json"
        tj.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        truth_json = str(tj)
    return str(src_dir / "frame_*.npy"), truth_json


def _load_job_into_state(job):
    """任务成功后把真实产物载入会话，使页面各视图展示本次离线结果。"""
    out = Path(job.dir) / "outputs"
    fused = aio.load_npy(out / "fused.npy")
    coverage = aio.load_npy(out / "coverage.npy")
    weight_sum = aio.load_npy(out / "weight_sum.npy")
    report = json.loads((out / "report.json").read_text())
    used = report["used"]
    from .stack import StackResult

    STATE["stack"] = StackResult(fused=fused, coverage=coverage,
                                 weight_sum=weight_sum, used=used,
                                 aligned={}, footprints={})
    STATE["report"] = report
    # 补算对齐图供残差视图使用
    if STATE["estimates"] is not None and STATE["frames"] is not None:
        from .resample import shift_bilinear
        for k in used:
            if k < len(STATE["frames"]) and k < len(STATE["estimates"]):
                a, c = shift_bilinear(STATE["frames"][k],
                                      STATE["estimates"][k].dx,
                                      STATE["estimates"][k].dy)
                STATE["stack"].aligned[k] = a
                STATE["stack"].footprints[k] = c
    STATE["stack_key"] = ("offline", job.dir)


def _run_web_job(job_id, cfg):
    job = OfflineJob.create(cfg)
    state = job.run()
    if state["status"] == STATUS_SUCCEEDED:
        # 仅当会话仍是同一曝光组时载入，避免覆盖用户后续操作
        _load_job_into_state(job)


@app.post("/api/offline/start")
def api_offline_start():
    body = request.get_json(force=True) or {}
    with WEB_JOBS_LOCK:
        for jid, info in WEB_JOBS.items():
            if info["thread"].is_alive():
                return safe_jsonify(
                    {"error": f"已有离线任务 {jid} 在运行，请先取消或等待完成"}), 409
    job_id = time.strftime("webjob-%Y%m%d-%H%M%S")
    job_dir = OFFLINE_JOBS_ROOT / job_id
    try:
        frames_glob, truth_json = _stage_web_frames(job_id)
    except RuntimeError as exc:
        return safe_jsonify({"error": str(exc)}), 400
    cfg = JobConfig(
        job_dir=str(job_dir), frames_glob=frames_glob,
        reference=int(body.get("reference", STATE["reference"])),
        requirements=str(REPO_ROOT / "requirements.txt"),
        threads=1, truth_json=truth_json, name=job_id)
    t = threading.Thread(target=_run_web_job, args=(job_id, cfg),
                         name=job_id, daemon=True)
    with WEB_JOBS_LOCK:
        WEB_JOBS[job_id] = {"thread": t, "dir": str(job_dir),
                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t.start()
    return safe_jsonify({"job_id": job_id, **_job_snapshot(job_id, job_dir)})


@app.get("/api/offline/status/<job_id>")
def api_offline_status(job_id):
    job_dir = OFFLINE_JOBS_ROOT / job_id
    if not job_dir.exists():
        return safe_jsonify({"error": "任务不存在"}), 404
    return safe_jsonify(_job_snapshot(job_id, job_dir))


@app.post("/api/offline/cancel/<job_id>")
def api_offline_cancel(job_id):
    job_dir = OFFLINE_JOBS_ROOT / job_id
    if not job_dir.exists():
        return safe_jsonify({"error": "任务不存在"}), 404
    OfflineJob(job_dir).request_cancel()
    return safe_jsonify(_job_snapshot(job_id, job_dir))


@app.post("/api/offline/resume/<job_id>")
def api_offline_resume(job_id):
    job_dir = OFFLINE_JOBS_ROOT / job_id
    if not (job_dir / "config.json").exists():
        return safe_jsonify({"error": "任务不存在"}), 404
    with WEB_JOBS_LOCK:
        for jid, info in WEB_JOBS.items():
            if info["thread"].is_alive():
                return safe_jsonify({"error": f"任务 {jid} 正在运行"}), 409
    cfg = JobConfig.from_json(json.loads((job_dir / "config.json").read_text()))
    t = threading.Thread(target=_run_web_job, args=(job_id, cfg),
                         name=job_id, daemon=True)
    with WEB_JOBS_LOCK:
        WEB_JOBS[job_id] = {"thread": t, "dir": str(job_dir),
                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t.start()
    return safe_jsonify({"job_id": job_id, **_job_snapshot(job_id, job_dir)})


def main():
    ap = argparse.ArgumentParser(description="天文融合 Web 界面")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
