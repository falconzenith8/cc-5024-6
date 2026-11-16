#!/usr/bin/env bash
# 离线计算发布流水线（本地，无外部云服务）。
#
# 流程：
#   1. 核对 .venv 中依赖与 requirements.txt 锁定版本完全一致；
#   2. 从样例原帧实际生成叠加图（全新任务目录，不沿用任何缓存成片）；
#   3. 用硬判据核对执行证据：输出 SNR、帧清单、产物指纹、环境/线程锁定；
#   4. 缓存冒充检查：把伪造 fused.npy 塞进任务的发布暂存路径，
#      重跑必须识破并失败（"不得以默认值/缓存遮错"）；
#   5. 取消检查：对一个新任务写 cancel.flag，运行必须以 130 退出且无产物；
#   6. 坏输入帧检查：放入含 NaN 的坏帧，任务必须失败并保留输入与退出信息；
#   7. 恢复检查：删掉检查点重跑（resume），结果证据必须与首次一致。
#
# 任一检查不通过即整体失败（set -e + 显式断言，没有默认值兜底）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "缺少 .venv，请先 python3 -m venv .venv && pip install -r requirements.txt" >&2; exit 2; }

DEMO_DIR="$REPO_ROOT/data/demo"
RUN_ROOT="$REPO_ROOT/runs/release"
JOB="$RUN_ROOT/demo_job"
JOB_CANCEL="$RUN_ROOT/cancel_job"
JOB_BAD="$RUN_ROOT/bad_job"

echo "== 1. 依赖锁核对 =="
"$PY" - <<'PYCHECK'
from pathlib import Path
from astrostack.offline import verify_dependency_lock, verify_thread_env
ev = verify_dependency_lock("requirements.txt")
print("依赖锁核对通过:", ev["actual"])
PYCHECK

# 样例原帧：实际由模拟器生成（仓库不跟踪 data/）
if ! ls "$DEMO_DIR"/frame_*.npy >/dev/null 2>&1; then
  echo "== 生成样例原帧 =="
  "$PY" -m astrostack.cli simulate --out "$DEMO_DIR"
fi

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

echo "== 2. 全新离线任务（实际运算）=="
scripts/run_offline.sh "$JOB" "$DEMO_DIR/frame_*.npy" 0 requirements.txt

echo "== 3. 执行证据硬判据（SNR / 帧清单 / 指纹 / 线程）=="
"$PY" - <<'PYCHECK'
import json, os
from pathlib import Path
from astrostack.offline import verify_thread_env
from astrostack.quality import file_sha256, hash_array
from astrostack import io as aio

job = Path("runs/release/demo_job")
state = json.loads((job / "state.json").read_text())
assert state["status"] == "succeeded", state
ev = json.loads((job / "evidence.json").read_text())

# 输出信噪比必须是实测的有限正数，不接受 None/默认值
snr = ev["output_snr"]
assert snr is not None and snr["snr_aperture"] > 10, snr
assert snr["snr_peak"] > 10, snr
print(f"输出 SNR: 孔径={snr['snr_aperture']:.2f} 峰值={snr['snr_peak']:.2f}")

# 帧清单必须完整，且 18 帧样例中 16 帧参与、2 帧被门控剔除
ms = ev["frame_manifest_summary"]
assert ms["complete"] and ms["n_frames"] == 18, ms
assert ms["n_used"] == 16 and ms["n_gated"] == 2 and ms["n_bad_input"] == 0, ms
manifest = json.loads((job / "outputs/frame_manifest.json").read_text())
assert len(manifest) == 18
assert all(m["sha256"] and m["path"] for m in manifest)
print(f"帧清单: {ms}")

# 产物指纹：evidence 记录的 sha256 必须与磁盘实际文件一致
for name, want in ev["artifacts"].items():
    got = file_sha256(job / "outputs" / name)
    assert got == want, f"{name} 指纹不符"
# 内容指纹与回读数组一致（防缓存成片冒充）
fused = aio.load_npy(job / "outputs/fused.npy")
assert hash_array(fused.astype("float32")) == ev["artifact_content_hash"]["fused"]
import numpy as np
assert np.isfinite(fused).any()

# 线程与依赖锁定证据
assert ev["threads"]["limit"] == 1
assert all(v == "1" for k, v in ev["threads"]["env"].items()), ev["threads"]
assert ev["dependency_lock"]["actual"] == ev["dependency_lock"]["locked"]
print("线程/依赖锁定证据:", ev["threads"]["env"])
print("证据核对全部通过")
PYCHECK

echo "== 4. 缓存冒充检查：伪造成片必须被实测/指纹校验识破 =="
# 4a. 已成功任务重跑：直接短路成功，不重算、不被外部改动带偏
OUT=$(scripts/run_offline.sh "$JOB" "$DEMO_DIR/frame_*.npy" 0 requirements.txt)
echo "$OUT" | grep -q "任务状态: succeeded"
# 4b. 攻击向量一：预放缓存成片到 outputs/，再删检查点强制重算。
#     真实重算结果与缓存片不同，发布采用原子替换，缓存片必须被丢弃。
CACHE_JOB="$RUN_ROOT/cache_job"
rm -rf "$CACHE_JOB"
cp -r "$JOB" "$CACHE_JOB"
rm -rf "$CACHE_JOB/checkpoints" "$CACHE_JOB/state.json"
# 重建为待运行状态（保留 config/inputs，outputs 里放一份被篡改的缓存片）
"$PY" - <<'PYCHECK'
import json, os
import numpy as np
from pathlib import Path
cj = Path("runs/release/cache_job")
fused = cj / "outputs/fused.npy"
a = np.load(fused)
np.save(fused, a + 5000.0)  # 篡改缓存成片
state = {"status": "pending", "stage": "init", "progress": 0.0,
         "cancel_requested": False, "history": []}
(cj / "state.json").write_text(json.dumps(state))
print("已在 cache_job 植入篡改缓存片")
PYCHECK
scripts/run_offline.sh "$CACHE_JOB" "$DEMO_DIR/frame_*.npy" 0 requirements.txt
"$PY" - <<'PYCHECK'
import json
import numpy as np
from astrostack import io as aio
from astrostack.quality import hash_array
job_ev = json.load(open("runs/release/demo_job/evidence.json"))
cache_ev = json.load(open("runs/release/cache_job/evidence.json"))
# 重算后的内容指纹必须等于真实任务，而不是被 +5000 的缓存片
assert cache_ev["artifact_content_hash"]["fused"] == \
       job_ev["artifact_content_hash"]["fused"]
a = aio.load_npy("runs/release/cache_job/outputs/fused.npy")
b = aio.load_npy("runs/release/demo_job/outputs/fused.npy")
assert np.nanmax(np.abs(a - b)) < 1e-6
print("缓存成片被原子替换丢弃，实际运算结果已发布")
PYCHECK
# 4c. 攻击向量二：发布暂存文件写盘后、指纹校验前被外部篡改。
#     monkeypatch read_image（load_npy 底层读取）使其首次回读 fused.npy 时
#     返回被污染的数组，模拟文件在写盘后被掉包；_publish 必须拒绝发布。
set +e
"$PY" - <<'PYCHECK'
import shutil
import numpy as np
import astrostack.io as aio
from astrostack.offline import OfflineJob, JobError
from astrostack.quality import hash_array

shutil.rmtree("runs/release/tamper_job", ignore_errors=True)
shutil.copytree("runs/release/demo_job", "runs/release/tamper_job")
tamper = OfflineJob("runs/release/tamper_job")
est = tamper._load_estimate_ckpt()
files = sorted((tamper.dir / "inputs").glob("*.npy"))
frames = [aio.load_npy(p) for p in files]
result = tamper._load_stack_ckpt(frames, est)
expected = hash_array(result.fused.astype("float32"))
real_load = aio.load_npy
calls = {"n": 0}

def poisoned_load(path):
    arr = real_load(path)
    if str(path).endswith("fused.npy"):
        calls["n"] += 1
        if calls["n"] == 1:  # _publish 内首次回读 fused.npy 即掉包
            return arr + 3000.0
    return arr

aio.load_npy = poisoned_load
try:
    tamper._publish(result, {"output_snr": {"snr_aperture": 99.0},
                             "n_used": 16, "psf": {"fwhm_mean": 3.5},
                             "anomalies": []},
                    [{"index": 0, "sha256": "x"}],
                    {"complete": True, "n_used": 16},
                    {"actual": {}, "locked": {}}, {"limit": 1, "env": {}},
                    {}, [p.name for p in files])
    raise SystemExit("指纹校验未拦截污染暂存文件")
except JobError as e:
    assert "指纹" in str(e), str(e)
    print("污染暂存文件被拦截:", e)
finally:
    aio.load_npy = real_load
assert not (tamper.dir / "outputs.staging").exists(), "失败后暂存目录必须清理"
back = real_load(tamper.dir / "outputs/fused.npy")
assert hash_array(back.astype("float32")) == expected
print("真实 outputs 未被污染，缓存冒充防护全部通过")
PYCHECK
RC_FAKE=$?
set -e
[[ "$RC_FAKE" == "0" ]] || { echo "缓存冒充检查失败" >&2; exit 1; }

echo "== 5. 取消检查 =="
mkdir -p "$JOB_CANCEL"
touch "$JOB_CANCEL/cancel.flag"
set +e
scripts/run_offline.sh "$JOB_CANCEL" "$DEMO_DIR/frame_*.npy" 0 requirements.txt
RC_CANCEL=$?
set -e
echo "取消任务退出码=$RC_CANCEL"
[[ "$RC_CANCEL" == "130" ]] || { echo "取消未按 130 退出" >&2; exit 1; }
[[ ! -f "$JOB_CANCEL/outputs/fused.npy" ]] || { echo "取消任务不得产出 fused.npy" >&2; exit 1; }
[[ -f "$JOB_CANCEL/error.json" ]] || { echo "取消必须保留退出信息 error.json" >&2; exit 1; }
[[ -d "$JOB_CANCEL/inputs" && $(ls "$JOB_CANCEL/inputs" | wc -l) -gt 0 ]] \
  || { echo "取消必须保留必要输入" >&2; exit 1; }

echo "== 6. 坏输入帧检查 =="
mkdir -p "$RUN_ROOT/bad_frames"
cp "$DEMO_DIR"/frame_000.npy "$RUN_ROOT/bad_frames/frame_000.npy"
"$PY" - <<'PYCHECK'
import numpy as np
a = np.load("runs/release/bad_frames/frame_000.npy")
b = a.copy()
b[0, 0] = np.nan
np.save("runs/release/bad_frames/frame_001.npy", b)
PYCHECK
set +e
scripts/run_offline.sh "$JOB_BAD" "$RUN_ROOT/bad_frames/frame_*.npy" 0 requirements.txt
RC_BAD=$?
set -e
echo "坏输入任务退出码=$RC_BAD"
[[ "$RC_BAD" != "0" ]] || { echo "坏输入帧必须导致失败" >&2; exit 1; }
[[ -f "$JOB_BAD/inputs/001_frame_001.npy.error.txt" ]] \
  || { echo "坏帧必须保留 .error.txt 说明" >&2; exit 1; }
[[ -f "$JOB_BAD/error.json" ]] || { echo "坏输入失败必须写 error.json" >&2; exit 1; }
[[ ! -f "$JOB_BAD/outputs/fused.npy" ]] || { echo "失败任务禁止产出产物" >&2; exit 1; }

echo "== 7. 恢复检查：resume 后证据一致 =="
FIRST_SNR=$(python3 -c "import json;print(json.load(open('$JOB/evidence.json'))['output_snr']['snr_aperture'])")
# 已成功任务 resume 应短路；删掉检查点与产物后 resume 必须完整重算并再次成功
rm -rf "$JOB/outputs" "$JOB/checkpoints"
scripts/run_offline.sh "$JOB" "$DEMO_DIR/frame_*.npy" 0 requirements.txt
SECOND_SNR=$("$PY" -c "import json;print(json.load(open('$JOB/evidence.json'))['output_snr']['snr_aperture'])")
[[ "$FIRST_SNR" == "$SECOND_SNR" ]] || { echo "恢复后 SNR 不一致: $FIRST_SNR != $SECOND_SNR" >&2; exit 1; }
echo "恢复后输出 SNR 一致: $SECOND_SNR"

echo ""
echo "发布流水线全部通过。执行证据: $JOB/evidence.json"
