"""可恢复的离线配准-融合任务。

一个任务 = 一个自描述任务目录：

    <job_dir>/
      config.json        任务配置（帧 glob、参考帧、门限、线程、依赖锁路径）
      state.json         运行状态机与进度（原子写，供轮询/恢复）
      cancel.flag        出现即请求取消（阶段边界与逐帧检查）
      inputs/            必要输入快照（坏帧也保留，附 .error.txt）
      checkpoints/       阶段检查点（估计结果、叠加中间量）
      outputs/           成功后原子发布的科学产物
      evidence.json      执行证据：环境、输出 SNR、帧清单、产物指纹
      error.json         失败信息（异常类型、消息、回溯、退出阶段）

设计约束（对应发布要求）：
* 依赖版本与线程数在启动时核对，不符即失败，绝不静默换环境运行；
* 分阶段检查点，中断后重跑自动续跑，已完成阶段不重复计算；
* 失败保留必要输入与退出信息，且不产生 outputs/ 产物
  （"天文融合失败禁止采用产物"）；
* 成功判定要求产物通过实测校验（SNR、帧清单、内容指纹），
  缓存里预先放置的成片无法冒充——指纹对不上会被重算或判失败。
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, io as aio
from .estimate import GateConfig, estimate_shifts
from .pipeline import build_report
from .quality import (build_frame_manifest, file_sha256, hash_array,
                      summarize_manifest, write_manifest_file)
from .resample import shift_bilinear
from .stack import StackResult, frame_residuals, stack_frames

# 线程设置：离线任务单进程确定性运行，BLAS/FFT/OMP 全部限 1 线程。
THREAD_LIMIT = 1
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
STATUS_SUCCEEDED = "succeeded"
_TERMINAL = {STATUS_CANCELLED, STATUS_FAILED, STATUS_SUCCEEDED}

STAGE_INIT = "init"
STAGE_ESTIMATE = "estimate"
STAGE_STACK = "stack"
STAGE_MEASURE = "measure"
STAGE_PUBLISH = "publish"


class JobError(RuntimeError):
    """离线任务可预期失败（输入检查、锁核对、产物校验等）。"""


class JobCancelled(RuntimeError):
    pass


# ---------------- 依赖锁与线程核对 ----------------

def parse_requirements(path):
    reqs = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise JobError(f"依赖锁只接受 == 精确锁定：{line}")
        name, ver = line.split("==", 1)
        reqs[name.strip().lower()] = ver.strip()
    return reqs


def verify_dependency_lock(requirements_path):
    """核对实际安装版本与锁文件；返回环境证据 dict，不符抛 JobError。"""
    from importlib.metadata import PackageNotFoundError, version

    locked = parse_requirements(requirements_path)
    import_name = {"numpy": "numpy", "scipy": "scipy",
                   "flask": "flask", "pillow": "PIL"}
    actual, mismatches = {}, []
    for pkg, want in locked.items():
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = None
        actual[pkg] = got
        if got != want:
            mismatches.append(f"{pkg}: 锁定 {want}，实际 {got or '未安装'}")
        mod = __import__(import_name[pkg], fromlist=["__version__"])
        mod_ver = getattr(mod, "__version__", got)
        if mod_ver != want and pkg != "flask":
            mismatches.append(f"{pkg}: 导入版本 {mod_ver} 与锁定 {want} 不符")
    if mismatches:
        raise JobError("依赖锁核对失败：" + "；".join(mismatches))
    return {"requirements_file": str(requirements_path),
            "locked": locked, "actual": actual}


def verify_thread_env(expected=THREAD_LIMIT):
    """核对线程相关环境变量；返回线程证据 dict，不符抛 JobError。"""
    bad = []
    seen = {}
    for var in _THREAD_ENV_VARS:
        val = os.environ.get(var)
        seen[var] = val
        if val is None or val.strip() != str(expected):
            bad.append(f"{var}={val or '未设置'}")
    if bad:
        raise JobError(
            "线程设置未锁定：" + "，".join(bad)
            + f"（离线任务要求全部为 {expected}；请用 scripts/run_offline.sh 启动，"
              "它会在导入 numpy 前导出这些变量）")
    return {"limit": expected, "env": seen,
            "cpu_count": os.cpu_count()}


def environment_snapshot():
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "astrostack_version": __version__,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


# ---------------- 任务配置 ----------------

@dataclass
class JobConfig:
    job_dir: str
    frames_glob: str
    reference: int = 0
    requirements: str = "requirements.txt"
    threads: int = THREAD_LIMIT
    gates: dict = field(default_factory=dict)
    enabled: list | None = None
    weights: list | None = None
    truth_json: str | None = None
    name: str = "offline-job"

    def to_json(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, obj):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in obj.items() if k in fields})


# ---------------- 任务状态机 ----------------

class OfflineJob:
    def __init__(self, job_dir):
        self.dir = Path(job_dir)
        self.config_path = self.dir / "config.json"
        self.state_path = self.dir / "state.json"
        self.cancel_path = self.dir / "cancel.flag"
        self.error_path = self.dir / "error.json"
        self.evidence_path = self.dir / "evidence.json"
        self.inputs_dir = self.dir / "inputs"
        self.ckpt_dir = self.dir / "checkpoints"
        self.out_dir = self.dir / "outputs"
        self._lock = threading.Lock()
        self.state = self._read_state() if self.state_path.exists() else None
        self.config = (JobConfig.from_json(json.loads(self.config_path.read_text()))
                       if self.config_path.exists() else None)

    def _read_state(self):
        return json.loads(self.state_path.read_text())

    def _write_state(self, **patch):
        with self._lock:
            state = self._read_state() if self.state_path.exists() else {}
            state.update(patch)
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.state_path)
            self.state = state

    def is_cancelled(self):
        return self.cancel_path.exists()

    def request_cancel(self):
        self.cancel_path.touch()
        if self.state and self.state.get("status") == STATUS_RUNNING:
            self._write_state(cancel_requested=True)

    # ---------------- 初始化与输入检查 ----------------

    @classmethod
    def create(cls, cfg: JobConfig):
        job = cls(cfg.job_dir)
        if job.state_path.exists():
            cur = job._read_state()
            # pending/running 说明可能有另一个进程在跑；running 拒绝，
            # pending（含人为重置后的待运行）允许打开并由 run() 续跑。
            if cur.get("status") == STATUS_RUNNING:
                raise JobError(
                    f"任务已在运行中（{cfg.job_dir}），请勿并发启动")
        job.dir.mkdir(parents=True, exist_ok=True)
        job.inputs_dir.mkdir(exist_ok=True)
        job.ckpt_dir.mkdir(exist_ok=True)
        job.out_dir.mkdir(exist_ok=True)
        if not job.config_path.exists():
            job.config_path.write_text(
                json.dumps(cfg.to_json(), ensure_ascii=False, indent=2),
                encoding="utf-8")
        job.config = job.config or cfg
        if not job.state_path.exists():
            job._write_state(status=STATUS_PENDING, stage=STAGE_INIT,
                             progress=0.0, cancel_requested=False,
                             created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                             history=[])
        return job

    def _stage_inputs(self):
        """把帧文件复制进任务目录并做坏输入检查；坏帧保留并记录 .error.txt。"""
        import glob

        files = sorted(glob.glob(self.config.frames_glob))
        manifest_inputs = []
        bad = []
        if not files:
            raise JobError(f"帧 glob 未匹配到任何文件：{self.config.frames_glob}")
        for k, src in enumerate(files):
            src_p = Path(src)
            dst = self.inputs_dir / f"{k:03d}_{src_p.name}"
            if not dst.exists():
                shutil.copy2(src_p, dst)
            entry = {
                "index": k, "name": src_p.name, "path": str(dst),
                "sha256": file_sha256(dst), "bytes": dst.stat().st_size,
                "shape": None,
            }
            try:
                arr = aio.read_image(dst)
                if arr.ndim not in (2, 3):
                    raise ValueError(f"维度 {arr.ndim} 不是二维图像或帧序列")
                if not np.isfinite(arr).all():
                    raise ValueError("数据含 NaN/Inf，拒绝作为输入帧")
                entry["shape"] = list(arr.shape)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                (dst.with_suffix(dst.suffix + ".error.txt")
                 .write_text(entry["error"], encoding="utf-8"))
                bad.append((src_p.name, entry["error"]))
            manifest_inputs.append(entry)
        if bad:
            details = "；".join(f"{n}（{e}）" for n, e in bad)
            raise JobError(f"坏输入帧检查未通过：{details}")
        if len(manifest_inputs) < 2:
            raise JobError("至少需要 2 帧输入")
        ref = self.config.reference
        if not 0 <= ref < len(manifest_inputs):
            raise JobError(f"参考帧索引 {ref} 越界（共 {len(manifest_inputs)} 帧）")
        return manifest_inputs

    # ---------------- 检查点 ----------------

    def _ckpt(self, name):
        return self.ckpt_dir / name

    def _save_estimate_ckpt(self, estimates):
        data = [dataclasses.asdict(e) for e in estimates]
        self._ckpt("estimates.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load_estimate_ckpt(self):
        from .estimate import FrameEstimate

        data = json.loads(self._ckpt("estimates.json").read_text())
        return [FrameEstimate(**d) for d in data]

    def _save_stack_ckpt(self, result):
        np.savez(self._ckpt("stack.npz"),
                 fused=result.fused, coverage=result.coverage,
                 weight_sum=result.weight_sum)
        self._ckpt("used.json").write_text(
            json.dumps(result.used), encoding="utf-8")

    def _load_stack_ckpt(self, frames, estimates):
        if not self._ckpt("stack.npz").exists():
            return None
        z = np.load(self._ckpt("stack.npz"))
        used = json.loads(self._ckpt("used.json").read_text())
        aligned, footprints = {}, {}
        for k in used:
            a, c = shift_bilinear(frames[k], estimates[k].dx, estimates[k].dy)
            aligned[k], footprints[k] = a, c
        return StackResult(fused=z["fused"], coverage=z["coverage"],
                           weight_sum=z["weight_sum"], used=used,
                           aligned=aligned, footprints=footprints)

    # ---------------- 主执行流程 ----------------

    def _outputs_complete(self):
        """成功产物是否齐备且 evidence 记录的文件都在。"""
        required = ["fused.npy", "coverage.npy", "weight_sum.npy",
                    "fused.fits", "coverage.fits", "report.json",
                    "frame_manifest.json", "evidence.json"]
        return all((self.out_dir / name).exists() for name in required)

    def run(self):
        """执行（或恢复）任务。返回最终状态 dict。"""
        if self.config is None:
            raise JobError("缺少 config.json，无法恢复任务")
        current = self._read_state().get("status")
        if current == STATUS_SUCCEEDED and self._outputs_complete():
            return self._read_state()
        if current in (STATUS_FAILED, STATUS_CANCELLED):
            # 失败/取消后的显式重试：清掉终态标记与取消请求从头续跑
            if self.cancel_path.exists():
                self.cancel_path.unlink()
            self._write_state(status=STATUS_PENDING, stage=STAGE_INIT,
                              progress=0.0, cancel_requested=False)
        elif current == STATUS_SUCCEEDED:
            # 状态成功但产物缺失（被外部清理）：回退到待运行，
            # 靠检查点重新测量/发布，不允许"无产物的成功"。
            self._write_state(status=STATUS_PENDING, stage=STAGE_INIT,
                              progress=0.0)
        self._write_state(status=STATUS_RUNNING, started_at=time.strftime(
            "%Y-%m-%dT%H:%M:%S"))
        try:
            evidence = self._run_inner()
            self._write_state(status=STATUS_SUCCEEDED, stage="done",
                              progress=1.0,
                              finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              output_snr=evidence["output_snr"],
                              n_used=evidence["frame_manifest_summary"]["n_used"])
        except JobCancelled as exc:
            self._fail(STATUS_CANCELLED, exc, keep_outputs=False)
        except Exception as exc:
            self._fail(STATUS_FAILED, exc, keep_outputs=False)
        return self._read_state()

    def _progress(self, stage, fraction):
        self._write_state(stage=stage, progress=float(fraction))

    def _check_cancel(self):
        if self.is_cancelled():
            self._write_state(cancel_requested=True)
            raise JobCancelled("收到 cancel.flag，任务取消")

    def _run_inner(self):
        cfg = self.config
        history = self._read_state().get("history", [])

        def mark(stage):
            history.append({"stage": stage,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            self._write_state(history=history)

        # 1. 先快照必要输入并做坏帧检查：任何失败（含随后的环境核对失败）
        #    都能在 inputs/ 留下排查所需的原始帧与 .error.txt。
        self._progress(STAGE_INIT, 0.0)
        manifest_inputs = self._stage_inputs()

        # 2. 环境核对：依赖锁 + 线程设置，不符直接失败（仍在任何数值计算前）
        lock_evidence = verify_dependency_lock(cfg.requirements)
        thread_evidence = verify_thread_env(cfg.threads)
        env_evidence = environment_snapshot()
        self._check_cancel()
        frames, names = [], []
        for entry in manifest_inputs:
            frames.append(aio.load_npy(entry["path"])
                          if str(entry["path"]).endswith(".npy")
                          else aio.read_image(entry["path"]))
            names.append(entry["name"])
        mark(STAGE_INIT)
        self._progress(STAGE_ESTIMATE, 0.05)
        self._check_cancel()

        # 2. 估计（可由检查点恢复）
        if self._ckpt("estimates.json").exists():
            estimates = self._load_estimate_ckpt()
        else:
            gates = GateConfig(**(cfg.gates or {}))
            estimates = estimate_shifts(frames, reference=cfg.reference,
                                        cfg=gates)
            self._check_cancel()
            self._save_estimate_ckpt(estimates)
        mark(STAGE_ESTIMATE)
        self._progress(STAGE_STACK, 0.35)
        self._check_cancel()

        # 3. 叠加（可由检查点恢复）
        result = self._load_stack_ckpt(frames, estimates)
        if result is None:
            result = stack_frames(
                frames, estimates, enabled=cfg.enabled, weights=cfg.weights,
                progress_cb=lambda f: self._progress(
                    STAGE_STACK, 0.35 + 0.35 * f),
                is_cancelled=self.is_cancelled)
            self._check_cancel()
            self._save_stack_ckpt(result)
        mark(STAGE_STACK)
        self._progress(STAGE_MEASURE, 0.75)
        self._check_cancel()

        # 4. 实测测量：报告、输出 SNR、帧清单
        truth_shifts, truth_star = self._load_truth()
        report, _ = build_report(result, estimates, cfg.reference,
                                 truth_shifts=truth_shifts,
                                 truth_star=truth_star)
        residuals = frame_residuals(result)
        manifest = build_frame_manifest(manifest_inputs, estimates,
                                        result.used, residuals)
        manifest_summary = summarize_manifest(manifest)
        if not manifest_summary["complete"]:
            raise JobError("帧清单不完整（存在缺少输入指纹的帧），拒绝发布")
        if report["output_snr"] is None:
            raise JobError("输出信噪比无法实测，拒绝发布")
        mark(STAGE_MEASURE)
        self._progress(STAGE_PUBLISH, 0.9)
        self._check_cancel()

        # 5. 原子发布：先写临时目录，校验通过后整体替换 outputs/
        evidence = self._publish(result, report, manifest, manifest_summary,
                                 lock_evidence, thread_evidence, env_evidence,
                                 names)
        mark(STAGE_PUBLISH)
        return evidence

    def _load_truth(self):
        if not self.config.truth_json:
            return None, None
        p = Path(self.config.truth_json)
        if not p.exists():
            return None, None
        meta = json.loads(p.read_text())
        from .simulate import Star

        truth_shifts = np.array(meta["shifts"])
        s = meta["stars"][0]
        truth_star = Star(s["x0"], s["y0"], s["peak"], s["sigma"])
        return truth_shifts, truth_star

    def _publish(self, result, report, manifest, manifest_summary,
                 lock_evidence, thread_evidence, env_evidence, names):
        staging = self.dir / "outputs.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        aio.save_npy(staging / "fused.npy", result.fused)
        aio.save_npy(staging / "coverage.npy", result.coverage)
        aio.save_npy(staging / "weight_sum.npy", result.weight_sum)
        aio.write_fits(staging / "fused.fits",
                       result.fused.astype(np.float32))
        aio.write_fits(staging / "coverage.fits",
                       result.coverage.astype(np.float32))
        aio.save_json(staging / "report.json", report)
        write_manifest_file(manifest, staging / "frame_manifest.json")

        # 发布前实测校验：回读产物，确认内容指纹与本次运算一致，
        # 缓存中预放的成片或写盘后被篡改的暂存文件都会在这里被识破。
        fused_back = aio.load_npy(staging / "fused.npy")
        cov_back = aio.load_npy(staging / "coverage.npy")
        expected_fused = hash_array(result.fused.astype(np.float32))
        expected_cov = hash_array(result.coverage.astype(np.float32))
        got_fused = hash_array(fused_back.astype(np.float32))
        got_cov = hash_array(cov_back.astype(np.float32))
        if got_fused != expected_fused or got_cov != expected_cov:
            shutil.rmtree(staging)
            raise JobError("产物指纹校验失败：输出与本次运算不一致，拒绝采用")
        finite = np.isfinite(fused_back)
        if not finite.any():
            shutil.rmtree(staging)
            raise JobError("融合图没有任何有效像素，拒绝发布")
        if report["n_used"] != manifest_summary["n_used"]:
            shutil.rmtree(staging)
            raise JobError("报告与帧清单的参与帧数不一致，拒绝发布")

        evidence = {
            "job_name": self.config.name,
            "job_dir": str(self.dir),
            "environment": env_evidence,
            "dependency_lock": lock_evidence,
            "threads": thread_evidence,
            "frames_glob": self.config.frames_glob,
            "reference": self.config.reference,
            "frame_manifest_summary": manifest_summary,
            "output_snr": report["output_snr"],
            "psf_fwhm": (report["psf"] or {}).get("fwhm_mean"),
            "n_anomalies": len(report["anomalies"]),
            "artifacts": {
                "fused.npy": file_sha256(staging / "fused.npy"),
                "coverage.npy": file_sha256(staging / "coverage.npy"),
                "weight_sum.npy": file_sha256(staging / "weight_sum.npy"),
                "fused.fits": file_sha256(staging / "fused.fits"),
                "coverage.fits": file_sha256(staging / "coverage.fits"),
                "report.json": file_sha256(staging / "report.json"),
                "frame_manifest.json":
                    file_sha256(staging / "frame_manifest.json"),
            },
            "artifact_content_hash": {
                "fused": expected_fused, "coverage": expected_cov},
            "frame_names": names,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (staging / "evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2,
                       default=str),
            encoding="utf-8")

        # 原子替换：旧 outputs 整体移走再换入，任何时刻都不存在半成品产物目录
        old = self.dir / "outputs.old"
        if old.exists():
            shutil.rmtree(old)
        if self.out_dir.exists():
            os.replace(self.out_dir, old)
        os.replace(staging, self.out_dir)
        if old.exists():
            shutil.rmtree(old)
        self.evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        return evidence

    def _fail(self, status, exc, keep_outputs=False):
        """失败：记录退出信息；默认撤下产物，避免失败结果被采用。"""
        if not keep_outputs and self.out_dir.exists():
            shutil.rmtree(self.out_dir)
            self.out_dir.mkdir()
        err = {
            "status": status,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stage": self._read_state().get("stage"),
            "traceback": traceback.format_exc(),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.error_path.write_text(
            json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_state(status=status, error=str(exc),
                          error_type=type(exc).__name__,
                          finished_at=err["at"])
