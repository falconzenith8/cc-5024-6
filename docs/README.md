# astrostack — 天文短曝光平移叠加工具

对一组小幅曝光图像（自制星点或公开许可图像）估计相对参考帧的平移，
以亚像素精度配准，按实际贡献加权融合，输出**浮点**融合图与覆盖图。

本工具做的是**原生分辨率下的加权平均融合**：不放大、不重建高频。
它不把"简单平均后放大"称为超分辨率——这里根本没有放大步骤。

## 图形界面（主入口）

```bash
python3 -m astrostack.server --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

界面功能：

* **载入曝光组**：点"生成示例曝光组"（18 帧模拟，含 1 纯噪声帧、
  1 贴边帧，用于演示剔除），或"载入曝光组"上传 `.npy/.fits/.png` 等文件。
* **逐帧启用**：每帧复选框开关；权重可逐帧覆盖（默认 1/背景噪声方差）。
* **参考帧**：单选钮切换；切换后服务端重新估计全部位移，旧融合结果
  立即失效，界面提示"请重新融合"。
* **融合**：输出区展示融合图、覆盖图、单帧、残差图、帧覆盖足迹。
* **星点细节**：融合图上星点切块放大显示，高斯拟合 FWHM、峰值、
  流量、能量误差（模拟数据相对真值）、**融合图实测输出信噪比**
  ——全部为服务端实际运算结果。
* **配准残差**：每帧 dx/dy 散点图与残差 RMS 表格；残差异常帧在图、表中
  红色高亮。
* **离线任务面板**：把当前曝光组作为可恢复离线任务在服务端后台运行，
  实时显示状态/阶段/进度，可取消、可恢复；成功后真实产物自动载入本页
  各视图。任务状态与退出信息全程可见。
* **异常提示**：按**星点细节 / 配准残差 / 覆盖权重**分类展示被剔除帧及
  原因、低覆盖区域、权重悬殊、残差超阈等，不用默认值掩盖错误。
* **显示拉伸**：百分位低/高与对数开关**只影响屏幕显示**，
  不写回科学数据；下载的浮点文件不受拉伸影响。
* **下载**：`fused.npy`、`fused.fits`、`coverage.npy`、`report.json`，
  均为浮点科学数据（无覆盖处为 NaN）。

## 坐标与位移约定

* 图像 `img[y, x]`，像素中心在整数坐标 `(x, y)`，单位为**像素**；
  x 向右、y 向下。
* 帧 k 位移 `(dx, dy)`：参考帧中 `(x0, y0)` 的目标在帧 k 中位于
  `(x0+dx, y0+dy)`（+dx 表示内容向 +x 移动）。
* 配准：`aligned[y,x] = frame_k(y+dy, x+dx)`（双线性插值）。
* 详见 `astrostack/conventions.py`。

## 方法（可解释）

1. **估计**：背景扣除后的强度质心为主，相位相关交叉校验；
   质心不确定度按误差传播计算。
2. **门控**：未检测到星点 / 星点贴边 / 信噪比不足 / 质心不确定度超限 /
   两种估计分歧超限的帧**被剔除并给出原因，绝不强行参与叠加**。
3. **重采样**：双线性插值（四点面积加权，保持总流量；
   轻微 PSF 展宽已在测试中量化）。
4. **累积**：`fused = Σ w_k·c_k·A_k / Σ w_k·c_k`，其中 `c_k` 为覆盖掩膜。
   无覆盖像素输出 **NaN 而非 0**，边缘不按零值稀释；
   覆盖图记录每像素的实际贡献帧数。

## 辅助命令行

命令行仅为辅助手段，完整功能请用上面的图形界面。页面中的"离线任务"面板
与本页交互共用同一套配准/融合代码，CLI 不能替代图像处理页中的融合交互。

```bash
# 生成带已知位移真值的模拟曝光组
python3 -m astrostack.cli simulate --out data/demo

# 估计 + 配准 + 加权融合，输出浮点结果与报告（一次性进程，非离线任务）
python3 -m astrostack.cli run --frames "data/demo/frame_*.npy" --ref 0 --out data/out

# 可恢复离线任务（推荐用启动脚本，它会在导入 numpy 前锁定单线程）
scripts/run_offline.sh runs/demo "data/demo/frame_*.npy" 0 requirements.txt
python3 -m astrostack.cli offline status --job-dir runs/demo   # 查状态/证据
python3 -m astrostack.cli offline cancel --job-dir runs/demo   # 请求取消
python3 -m astrostack.cli offline resume --job-dir runs/demo   # 恢复/重试
```

## 可恢复离线任务

离线任务把"配准 + 融合"打包成一个自描述任务目录（`runs/<job>/`）：

```
config.json        任务配置（帧 glob、参考帧、门限、线程、依赖锁路径）
state.json         状态机与进度（原子写；pending/running/succeeded/failed/cancelled）
cancel.flag        出现即请求取消，在阶段边界与逐帧循环处响应
inputs/            必要输入快照（坏帧也保留，附 *.error.txt 说明）
checkpoints/       估计与叠加阶段检查点（中断后 resume 不重复计算）
outputs/           成功后原子发布的科学产物（失败/取消时该目录无成片）
evidence.json      执行证据：环境、依赖锁、线程、输出 SNR、帧清单、产物指纹
error.json         失败/取消的退出信息（异常类型、消息、回溯、阶段）
```

关键约束：

* **依赖锁定**：`requirements.txt` 精确锁定 numpy/scipy/flask/pillow 版本，
  任务启动时逐项核对实际安装版本，不符即失败，绝不静默换环境运行。
* **线程锁定**：`scripts/run_offline.sh` 与 `scripts/run_server.sh` 在导入
  numpy 前导出 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_*_NUM_THREADS=1`；
  任务内部再次核对，未锁定即拒绝运行。
* **可恢复**：分阶段检查点。已成功任务重跑短路返回；失败/取消后
  `resume` 清除 `cancel.flag` 并从检查点续跑；丢失产物时按检查点重算，
  恢复后的内容指纹与 SNR 与首次一致。
* **失败禁产物**：失败或取消会撤下 `outputs/` 成片，但保留 `inputs/`
  与 `error.json`，天文融合失败时不会有任何产物被采用。
* **坏输入检查**：含 NaN/Inf、维度非法或无法解码的帧在初始化阶段即判失败，
  坏帧原样保留并写 `.error.txt`。
* **成功判定不靠缓存**：发布在临时目录写盘后回读，校验内容指纹与本次运算
  一致、SNR 可实测、帧清单完整，再原子替换 `outputs/`。预放的缓存成片
  指纹对不上会被拒绝（`scripts/release_offline.sh` 含冒充攻击演练）。
* **无外部云服务**：全部为本地进程与本地文件。

### 输出信噪比与帧清单（执行证据）

* `evidence.json::output_snr`：在融合图上实测的孔径 SNR 与峰值 SNR
  （信号为背景扣除孔径流量，噪声含背景 MAD 与泊松项），无法实测时为
  `null` 且任务判失败，不用默认值遮错。
* `outputs/frame_manifest.json`：逐帧记录输入文件 sha256、形状、配准
  dx/dy、单帧 SNR、权重、门控原因、是否参与、残差 RMS/max；
  `frame_manifest_summary` 给出总帧数/参与/门控/坏输入计数。

## 离线计算发布流水线

```bash
scripts/release_offline.sh
```

该脚本在全新任务目录上实际跑通整条流水线，并硬性核对：依赖锁、线程设置、
输出 SNR 与帧清单、产物 sha256/内容指纹、缓存成片冒充拦截、取消（退出码
130 且无产物但留输入）、坏输入帧失败留痕、检查点恢复后证据一致。
任一检查不通过即整体失败，没有默认值兜底。

## 环境准备

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # 安装锁定版本
```

无 `ensurepip` 的环境可先用 `get-pip.py` 引导 pip。运行任务与服务请使用
`scripts/run_offline.sh`、`scripts/run_server.sh`，它们会完成线程锁定并
优先选用 `.venv` 解释器。

## 验证

```bash
.venv/bin/python tests/test_registration.py   # 配准/融合数值正确性（14 项）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python tests/test_offline.py      # 离线任务与页面后台任务（39 项）
scripts/release_offline.sh                    # 发布流水线端到端核对
```

用已知小数位移的模拟序列检验：位移恢复精度（高/低信噪比）、
亮度变化鲁棒性、剔除行为、无零值稀释、PSF 的 FWHM 与圆度、
总能量误差、更换参考帧的自洽性；离线套件另外检验依赖/线程锁定、
取消与恢复、坏输入、缓存冒充拦截、输出 SNR 与帧清单、页面 API 状态机。

## 独立比较脚本

`scripts/compare.py` 不依赖 astrostack，自行重算质心/矩法 FWHM/孔径流量，
用于对比两份浮点输出（如 CLI 与界面结果、或两次运行）：

```bash
python3 scripts/compare.py out_a/fused.npy out_b/fused.npy \
    --cov-a out_a/coverage.npy --cov-b out_b/coverage.npy
```

注意：不同参考帧产生的融合图锚定在不同坐标原点，星点位置会整体平移
（FWHM 与流量仍应一致），比较脚本会如实报告这一差异。

## 依赖

Python 3.11+，精确版本锁定在 `requirements.txt`（离线任务启动时强制核对）：
numpy 2.1.3、scipy 1.14.1、flask 3.0.3、pillow 11.0.0。
线程相关环境变量由 `scripts/run_offline.sh`、`scripts/run_server.sh`
在解释器启动前统一锁定为单线程。
