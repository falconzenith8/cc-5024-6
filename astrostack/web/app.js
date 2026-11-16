/* 天文融合界面逻辑：所有科学数值来自服务端实际运算，前端只做展示。 */
"use strict";

let S = { n_frames: 0 };          // 服务端状态摘要
let groupId = -1;                 // 曝光组编号（服务端每次载入自增）
let enabled = [];                 // 用户逐帧开关
let weightOverride = [];          // 用户逐帧权重覆盖（null=默认）
let view = "fused";
let pickFrame = 0;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  const text = await r.text();
  let j;
  try { j = JSON.parse(text); }
  catch { alert("服务返回无法解析: " + text.slice(0, 200)); throw new Error("bad json"); }
  if (!r.ok) { alert(j.error || ("请求失败 " + r.status)); throw new Error(j.error); }
  return j;
}
const post = (p, body) => api(p, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

async function refresh() {
  S = await api("/api/state");
  if (S.group_id !== groupId) {   // 新曝光组：重置逐帧开关与权重覆盖
    groupId = S.group_id;
    enabled = Array(S.n_frames).fill(true);
    weightOverride = Array(S.n_frames).fill(null);
  }
  renderFrameList();
  renderAnomalies();
  renderResults();
  $("btn-restimate").disabled = !S.n_frames;
  $("btn-stack").disabled = !S.has_estimates;
  $("stale").classList.toggle("hidden", !(S.has_estimates && !S.has_stack && S.n_frames));
}

function renderFrameList() {
  const el = $("frame-list");
  el.innerHTML = "";
  if (!S.n_frames) { el.innerHTML = "<div class='hint'>尚未载入曝光组</div>"; return; }
  const ests = S.estimates || [];
  for (let i = 0; i < S.n_frames; i++) {
    const e = ests[i];
    const div = document.createElement("div");
    div.className = "frame-item";
    const chk = document.createElement("input");
    chk.type = "checkbox"; chk.checked = enabled[i];
    chk.title = "启用/禁用该帧";
    chk.onchange = () => { enabled[i] = chk.checked; markStale(); };
    const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "ref"; radio.checked = S.reference === i;
    radio.title = "设为参考帧";
    radio.onchange = () => changeReference(i);
    const body = document.createElement("div");
    let html = `<div class="name">#${i} ${S.names[i] || ""}</div>`;
    if (e) {
      if (e.ok) {
        html += `<div class="meta">dx=${fmt(e.dx)} dy=${fmt(e.dy)} σc=${fmt(e.sigma_cx, 3)}`
              + ` SNR=${fmt(e.snr, 1)} 权重=${fmt(e.weight, 5)}</div>`;
        if (e.residual_rms !== undefined)
          html += `<div class="meta">残差RMS=${fmt(e.residual_rms, 3)} max=${fmt(e.residual_max, 2)}</div>`;
        if (S.has_truth && S.truth_rel)
          html += `<div class="meta">真值位移=(${fmt(S.truth_rel[i][0])},${fmt(S.truth_rel[i][1])})</div>`;
        html += `<span class="badge ok">正常</span>`;
      } else {
        html += `<div class="reason">${e.reasons.join("；")}</div>`
              + `<span class="badge bad">已剔除</span>`;
      }
      html += ` <label class="meta">权重覆盖<input type="number" step="0.0001" `
            + `style="width:80px" data-w="${i}" value="${weightOverride[i] ?? ""}" `
            + `placeholder="${fmt(e.weight, 5)}"></label>`;
    }
    body.innerHTML = html;
    div.append(chk, radio, body);
    el.append(div);
  }
  el.querySelectorAll("input[data-w]").forEach((inp) => {
    inp.onchange = () => {
      const k = +inp.dataset.w;
      weightOverride[k] = inp.value === "" ? null : +inp.value;
      markStale();
    };
  });
}

function fmt(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return (+v).toFixed(d);
}

function markStale() {
  // 本地参数变更后，旧融合结果不再对应当前参数
  if (S.has_stack) $("stale").classList.remove("hidden");
}

async function changeReference(i) {
  // 修改参考帧：服务端重新估计并使旧融合失效
  await post("/api/estimate", { reference: i });
  await refresh();
}

function renderAnomalies() {
  const el = $("anomalies");
  const a = (S.report && S.report.anomalies) || [];
  el.textContent = a.length ? a.join("\n") : "（无）";
}

function renderResults() {
  const hasStack = !!S.has_stack;
  document.querySelectorAll("#downloads a").forEach((a) => {
    a.style.pointerEvents = hasStack ? "auto" : "none";
    a.style.opacity = hasStack ? 1 : 0.4;
  });
  refreshImage();
  renderStarDetail();
  renderResiduals();
}

function stretchQuery() {
  return `lo=${$("lo").value}&hi=${$("hi").value}&log=${$("logstretch").checked ? 1 : 0}`;
}

function refreshImage() {
  const needFrame = ["frame", "residual", "footprint"].includes(view);
  $("frame-picker-wrap").style.display = needFrame ? "" : "none";
  const img = $("main-img");
  img.src = `/api/render?view=${view}&frame=${pickFrame}&${stretchQuery()}&t=${Date.now()}`;
}

async function renderStarDetail() {
  const info = $("star-info");
  if (!S.has_stack || !S.report || !S.report.psf) {
    info.textContent = "（融合后显示星点实测细节）";
    return;
  }
  const j = await api(`/api/cutout?view=fused&half=12`);
  drawCutout(j);
  const p = S.report.psf;
  const ph = S.report.photometry;
  let txt = `高斯拟合（融合图实测）\n`
    + `  中心      : (${fmt(p.x0, 2)}, ${fmt(p.y0, 2)}) px\n`
    + `  FWHM      : ${fmt(p.fwhm_x, 3)} × ${fmt(p.fwhm_y, 3)} px\n`
    + `  峰值      : ${fmt(p.amp, 1)} adu\n`
    + `  拟合流量  : ${fmt(p.flux, 0)} adu\n`
    + `  孔径流量  : ${fmt(ph.flux, 0)} adu (r=6px, 背景 ${fmt(ph.bg_per_pix, 1)})\n`
    + `  拟合残差RMS: ${fmt(p.residual_rms, 2)} adu\n`;
  if (S.report.energy)
    txt += `  能量误差  : ${fmt(S.report.energy.rel_error * 100, 2)} %（相对模拟真值）\n`;
  txt += `\n覆盖: 最大 ${S.report.coverage_stats.max} 帧, `
    + `无覆盖像素占比 ${fmt(S.report.coverage_stats.nan_fraction * 100, 1)}%`;
  info.textContent = txt;
}

function drawCutout(j) {
  const cv = $("cutout");
  const ctx = cv.getContext("2d");
  const d = j.data, h = d.length, w = d[0].length;
  const scale = Math.floor(Math.min(cv.width / w, cv.height / h));
  const vals = d.flat().filter((v) => v !== null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  ctx.fillStyle = "#0d0f13";
  ctx.fillRect(0, 0, cv.width, cv.height);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = d[y][x];
      if (v === null) { ctx.fillStyle = "#461a46"; }
      else {
        const g = Math.round(255 * (v - lo) / (hi - lo || 1));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
      }
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  // 实测中心位置十字线
  ctx.strokeStyle = "#4fc3f7";
  ctx.beginPath();
  ctx.moveTo(j.cx * scale + scale / 2, j.cy * scale - 6);
  ctx.lineTo(j.cx * scale + scale / 2, j.cy * scale + scale + 6);
  ctx.moveTo(j.cx * scale - 6, j.cy * scale + scale / 2);
  ctx.lineTo(j.cx * scale + scale + 6, j.cy * scale + scale / 2);
  ctx.stroke();
}

function renderResiduals() {
  const tbl = $("resid-table");
  const cv = $("shift-plot");
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d0f13";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!S.has_estimates) { tbl.innerHTML = ""; return; }
  let html = `<tr><th>帧</th><th>dx(px)</th><th>dy(px)</th><th>σc(px)</th>`
    + `<th>残差RMS</th><th>状态</th></tr>`;
  const pts = [];
  for (const e of S.estimates) {
    html += `<tr><td>${e.index}</td><td>${fmt(e.dx, 3)}</td><td>${fmt(e.dy, 3)}</td>`
      + `<td>${fmt(Math.max(e.sigma_cx, e.sigma_cy), 3)}</td>`
      + `<td>${e.residual_rms !== undefined ? fmt(e.residual_rms, 3) : "—"}</td>`
      + `<td>${e.ok ? (e.used ? "参与" : "禁用") : "剔除"}</td></tr>`;
    if (e.ok) pts.push([e.dx, e.dy, e.used]);
  }
  tbl.innerHTML = html;
  if (!pts.length) return;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs) - 0.5, x1 = Math.max(...xs) + 0.5;
  const y0 = Math.min(...ys) - 0.5, y1 = Math.max(...ys) + 0.5;
  const mx = (v) => 20 + (v - x0) / (x1 - x0 || 1) * (cv.width - 40);
  const my = (v) => cv.height - 20 - (v - y0) / (y1 - y0 || 1) * (cv.height - 40);
  ctx.strokeStyle = "#3a4350";
  ctx.strokeRect(20, 20, cv.width - 40, cv.height - 40);
  ctx.fillStyle = "#8b96a5";
  ctx.font = "10px sans-serif";
  ctx.fillText(`dx ${fmt(x0, 1)}~${fmt(x1, 1)} px`, 22, 14);
  ctx.fillText(`dy ${fmt(y0, 1)}~${fmt(y1, 1)} px`, 22, 26);
  for (const [x, y, used] of pts) {
    ctx.fillStyle = used ? "#4fc3f7" : "#777";
    ctx.beginPath();
    ctx.arc(mx(x), my(y), 4, 0, 7);
    ctx.fill();
  }
}

// ---------------- 事件 ----------------

$("btn-demo").onclick = async () => { await post("/api/demo"); await refresh(); };

$("file-input").onchange = async (ev) => {
  const fd = new FormData();
  for (const f of ev.target.files) fd.append("files", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { alert(j.error); return; }
  await refresh();
};

$("btn-restimate").onclick = async () => {
  await post("/api/estimate", { reference: S.reference });
  await refresh();
};

$("btn-stack").onclick = async () => {
  await post("/api/stack", { enabled, weights: weightOverride });
  $("stale").classList.add("hidden");
  await refresh();
};

$("view").onchange = (e) => { view = e.target.value; refreshImage(); };
$("frame-picker").onchange = (e) => { pickFrame = +e.target.value; refreshImage(); };
for (const id of ["lo", "hi", "logstretch"])
  $(id).onchange = refreshImage;

async function boot() {
  await refresh();
  const fp = $("frame-picker");
  fp.innerHTML = "";
  for (let i = 0; i < S.n_frames; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = `#${i}`;
    fp.append(o);
  }
}
setInterval(async () => {  // 帧选择器随状态更新
  const fp = $("frame-picker");
  if (fp.options.length !== S.n_frames) {
    fp.innerHTML = "";
    for (let i = 0; i < S.n_frames; i++) {
      const o = document.createElement("option");
      o.value = i; o.textContent = `#${i}`;
      fp.append(o);
    }
  }
}, 1000);
boot();
