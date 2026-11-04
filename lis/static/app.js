/* OpenImage Studio editor. Plain script, no libraries, nothing fetched from the internet. */
"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = { target: [255, 138, 31], protected: [46, 196, 255], candidate: [255, 225, 77] };
const ICON = {
  pass: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.01"/><path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  fail: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/></svg>',
  shield: '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
};

const S = {
  status: { state: "starting" },
  project: null,          // project view from the server
  tool: "select",
  view: { s: 1, tx: 0, ty: 0 },
  pending: null,          // {object_id, points: [[x,y,l]], box, candidates, picked, title}
  proposal: null,
  showOverlays: true,
  compare: false, compareX: 0.5, holdBefore: false,
  showMap: false,
  job: null,
  selectedObj: null,      // mask_id painted by the brush
  brushTarget: "new-target",
  pointer: null,          // image coords of the mouse
  drag: null,
  spaceDown: false,
  busy: false,
  lastRevision: null,
};

const canvas = $("view");
const ctx = canvas.getContext("2d");
const stage = $("stage");
let dirty = true;
const redraw = () => { dirty = true; };

// ---- helpers ------------------------------------------------------------------------------

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; }, kind === "error" ? 6500 : 3500);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 4000);
}

async function api(method, url, body, opts = {}) {
  const init = { method, headers: {} };
  if (body instanceof Blob || body instanceof ArrayBuffer) { init.body = body; init.headers["Content-Type"] = "application/octet-stream"; }
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers["Content-Type"] = "application/json"; }
  let r;
  try { r = await fetch(url, init); } catch (e) { throw new Error("The editor server is not reachable. Is it still running?"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) {}
    const err = new Error(msg); err.status = r.status; throw err;
  }
  return opts.raw ? r : r.json();
}

// An upload with progress (fetch cannot report it): over a slow link a large photo takes a
// while, and a moving bar with the share sent says so instead of looking stuck.
function uploadWithProgress(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.upload.onload = () => onProgress(1);
    x.onerror = () => reject(new Error("The editor server is not reachable. Is it still running?"));
    x.onload = () => {
      let j = null;
      try { j = JSON.parse(x.responseText); } catch (_) {}
      if (x.status >= 200 && x.status < 300) resolve(j);
      else { const err = new Error((j && j.detail) || x.statusText || "Upload failed"); err.status = x.status; reject(err); }
    };
    x.send(file);
  });
}

async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, "error"); console.error(e); }
}

// The inspector's tabs: Edit (words, selection, result), Create, Adjust, History.
function showTab(name) {
  S.tab = name;
  document.querySelectorAll("#inspector .tab").forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle("on", on); b.setAttribute("aria-selected", on ? "true" : "false"); });
  document.querySelectorAll("#inspector .tab-page").forEach((pg) => { pg.hidden = pg.dataset.page !== name; });
  if (name === "create" && typeof extSizeText === "function") $("extSize").textContent = extSizeText();
}
document.querySelectorAll("#inspector .tab").forEach((b) => b.onclick = () => showTab(b.dataset.tab));

// Light or dark: the system setting until the button is used, then the choice is remembered in this browser.
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("lis-theme", next); } catch (_) {}
  redraw();
}

// Caches are bounded (most recently used kept): a long session of clicks and renders otherwise
// holds every full-size mask and version in memory until the tab crashes.
const IMG_MAX = 36, LAYER_MAX = 24;
function lru(map, key, max) {
  if (map.has(key)) { const v = map.get(key); map.delete(key); map.set(key, v); }
  while (map.size > max) map.delete(map.keys().next().value);
}
const imgCache = new Map();
function loadImage(url) {
  if (imgCache.has(url)) lru(imgCache, url, IMG_MAX);
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((res, rej) => {
      const im = new Image();
      im.decoding = "async";
      im.onload = () => { res(im); redraw(); };
      im.onerror = () => { imgCache.delete(url); rej(new Error("could not load " + url)); };
      im.src = url;
    }));
    lru(imgCache, url, IMG_MAX);
  }
  return imgCache.get(url);
}
const ready = new Map(); // url -> image once loaded
function img(url) {
  if (!url) return null;
  if (ready.has(url)) { lru(ready, url, IMG_MAX); return ready.get(url); }
  loadImage(url).then((im) => { ready.set(url, im); lru(ready, url, IMG_MAX); }).catch(() => {});
  return null;
}
// The photo, or its thumbnail while the full image is still arriving. Zoomed in past what the
// fit copy (VIEW_FIT pixels) resolves, the sharp copy is fetched and used once it is here.
const VIEW_FIT = 2048; // matches lis/web.py
function wantSharp() {
  if (!P()) return false;
  const long = Math.max(P().width, P().height);
  return long > VIEW_FIT && long * S.view.s * (window.devicePixelRatio || 1) > VIEW_FIT * 1.15;
}
function pic(url) {
  if (!url) return null;
  const have = ready.get(url);
  if (have && have.naturalWidth >= P().width) return have; // already full resolution (the file the user dropped)
  if (url.includes("/view/") && wantSharp()) { const f = img(url.replace("/view/", "/full/")); if (f) return f; }
  if (have) return have;
  const t = img(url.replace("/view/", "/thumb/")); // ask for the small one first
  return img(url) || t;
}

// A mask PNG (0 or 255) becomes a tinted layer with a solid outline, built once per url and colour.
const layerCache = new Map();
function layer(url, rgb, fill = 0.34) {
  const key = url + "|" + rgb.join(",") + "|" + fill;
  if (layerCache.has(key)) { lru(layerCache, key, LAYER_MAX); return layerCache.get(key); }
  layerCache.set(key, null);
  lru(layerCache, key, LAYER_MAX);
  loadImage(url).then((im) => {
    const ds = Math.min(1, VIEW_MAX / Math.max(im.naturalWidth, im.naturalHeight));
    const w = Math.max(1, Math.round(im.naturalWidth * ds)), h = Math.max(1, Math.round(im.naturalHeight * ds));
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.drawImage(im, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    const px = d.data;
    const n = w * h;
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) m[i] = px[i * 4] >= 128 ? 1 : 0;
    // outline: inside pixels within t of an outside pixel
    const t = Math.max(1, Math.round(Math.max(w, h) / 700));
    let er = m;
    for (let k = 0; k < t; k++) {
      const nx = new Uint8Array(n);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          const i = row + x;
          if (!er[i]) continue;
          if (x > 0 && x < w - 1 && y > 0 && y < h - 1 && er[i - 1] && er[i + 1] && er[i - w] && er[i + w]) nx[i] = 1;
        }
      }
      er = nx;
    }
    const fa = Math.round(fill * 255);
    for (let i = 0; i < n; i++) {
      const j = i * 4;
      if (!m[i]) { px[j + 3] = 0; continue; }
      px[j] = rgb[0]; px[j + 1] = rgb[1]; px[j + 2] = rgb[2];
      px[j + 3] = er[i] ? fa : 255;
    }
    g.putImageData(d, 0, 0);
    layerCache.set(key, c); lru(layerCache, key, LAYER_MAX);
    redraw();
  }).catch(() => layerCache.delete(key));
  return null;
}

const P = () => S.project;
const VIEW_MAX = 4096; // matches lis/web.py: overlays and strokes are kept at most this size in the browser
const DS = () => (P() ? Math.min(1, VIEW_MAX / Math.max(P().width, P().height)) : 1);
const pid = () => S.project && S.project.project_id;
const headRev = () => P() && P().revisions.find((r) => r.revision_id === P().head);
const hasTargets = () => P() && P().objects.some((o) => o.kind === "target");
const newObjectId = () => "obj-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

// ---- view transform -----------------------------------------------------------------------

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = stage.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  if (typeof autoGrow === "function" && ta) autoGrow();
  redraw();
}
function stageSize() { const r = stage.getBoundingClientRect(); return { w: r.width, h: r.height }; }

function fit() {
  if (!P()) return;
  const { w, h } = stageSize();
  const dock = $("dock");
  const top = 56, side = w < 600 ? 8 : 24;
  const bottom = (dock && !dock.hidden ? dock.offsetHeight : 90) + 34;
  const s = Math.min((w - side * 2) / P().width, (h - top - bottom) / P().height);
  S.view.s = Math.max(0.02, s);
  S.view.tx = (w - P().width * S.view.s) / 2;
  S.view.ty = top + (h - top - bottom - P().height * S.view.s) / 2;
  updateZoom();
  redraw();
}
function zoomAt(f, cx, cy) {
  const v = S.view;
  const ns = Math.min(16, Math.max(0.02, v.s * f));
  v.tx = cx - (cx - v.tx) * (ns / v.s);
  v.ty = cy - (cy - v.ty) * (ns / v.s);
  v.s = ns;
  updateZoom();
  redraw();
}
function updateZoom() { $("zoomOut").textContent = Math.round(S.view.s * 100) + "%"; }
function toImage(ev) {
  const r = canvas.getBoundingClientRect();
  const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
  return { x: (sx - S.view.tx) / S.view.s, y: (sy - S.view.ty) / S.view.s, sx, sy };
}
function inside(p) { return P() && p.x >= 0 && p.y >= 0 && p.x < P().width && p.y < P().height; }

// ---- drawing ------------------------------------------------------------------------------

let strokeCanvas = null; // brush strokes at image resolution, sent on pointer up

function draw() {
  dirty = false;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!P()) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const v = S.view, W = P().width, H = P().height;
  const x = v.tx, y = v.ty, w = W * v.s, h = H * v.s;
  ctx.imageSmoothingEnabled = v.s < 2;
  ctx.imageSmoothingQuality = "high";
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,.6)"; ctx.shadowBlur = 24;
  ctx.fillStyle = "#000"; ctx.fillRect(x, y, w, h);
  ctx.restore();

  const head = headRev();
  const cur = adjustPreview() || pic(P().image_url);  // the Adjust sliders' live preview, while one is moved
  const before = head ? pic(head.parent_url) : null;
  if (S.showMap && head) {
    const m = img(head.change_map_url);
    if (m) ctx.drawImage(m, x, y, w, h); else if (cur) ctx.drawImage(cur, x, y, w, h);
  } else if (S.holdBefore && before) {
    ctx.drawImage(before, x, y, w, h);
  } else if (S.compare && head && before && cur && cur !== before) {
    const split = x + w * S.compareX;
    ctx.drawImage(before, x, y, w, h);
    ctx.save();
    ctx.beginPath(); ctx.rect(split, y, x + w - split, h); ctx.clip();
    ctx.drawImage(cur, x, y, w, h);
    ctx.restore();
    ctx.fillStyle = "#fff"; ctx.fillRect(split - 1, y, 2, h);
    ctx.beginPath(); ctx.arc(split, y + h / 2, 15, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.strokeStyle = "#333"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(split - 5, y + h / 2 - 5); ctx.lineTo(split - 9, y + h / 2); ctx.lineTo(split - 5, y + h / 2 + 5);
    ctx.moveTo(split + 5, y + h / 2 - 5); ctx.lineTo(split + 9, y + h / 2); ctx.lineTo(split + 5, y + h / 2 + 5); ctx.stroke();
    label("Before", x + 10, y + 10); label("After", x + w - 10, y + 10, true);
  } else if (cur || before) {
    ctx.drawImage(cur || before, x, y, w, h);
  } else {
    ctx.fillStyle = "#777"; ctx.font = "13px system-ui"; ctx.textAlign = "center"; ctx.fillText("Loading photo", x + w / 2, y + h / 2); ctx.textAlign = "left";
  }

  const overlays = S.showOverlays && !S.showMap && !S.compare && !S.holdBefore;
  if (overlays) {
    const replaced = new Set(S.proposal ? S.proposal.before || [] : []);  // the proposal will replace these: do not show them as if they stayed
    for (const o of P().objects) {
      if (replaced.has(o.mask_id)) continue;
      const l = layer(o.url, COLORS[o.kind], S.selectedObj === o.mask_id ? 0.42 : 0.3);
      if (l) ctx.drawImage(l, x, y, w, h);
    }
  }
  if (S.proposal && !S.compare && !S.showMap) {
    for (const e of S.proposal.protect) if (e.url) { const l = layer(e.url, COLORS.protected, 0.45); if (l) ctx.drawImage(l, x, y, w, h); }
    if (S.proposal.target && S.proposal.target.url) { const l = layer(S.proposal.target.url, COLORS.target, 0.45); if (l) ctx.drawImage(l, x, y, w, h); }
    for (const o of P().objects) if (o.kind === "protected") { const l = layer(o.url, COLORS.protected, 0.45); if (l) ctx.drawImage(l, x, y, w, h); }  // still kept, on top
  }
  const pend = S.pending;
  if (pend && pend.candidates && pend.candidates.length && !S.compare && !S.showMap) {
    const c = pend.candidates[pend.picked];
    const l = layer(c.url, COLORS.candidate, 0.42);
    if (l) ctx.drawImage(l, x, y, w, h);
  }
  if (pend && pend.box) drawBox(pend.box, "rgba(255,225,77,.9)", [6, 4]);
  if (pend) for (const [px, py, l] of pend.points) {
    const sx = x + (px + 0.5) * v.s, sy = y + (py + 0.5) * v.s;
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = l ? "#22c55e" : "#ef4444"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(sx - 3.5, sy); ctx.lineTo(sx + 3.5, sy); if (l) { ctx.moveTo(sx, sy - 3.5); ctx.lineTo(sx, sy + 3.5); } ctx.stroke();
  }
  if (S.drag && S.drag.kind === "box" && S.drag.moved) {
    const a = S.drag.start, b = S.drag.now;
    drawBox([Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)], "#ffe14d", [6, 4]);
  }
  if (strokeCanvas && S.drag && S.drag.kind === "paint") {
    ctx.save(); ctx.globalAlpha = 0.55; ctx.drawImage(strokeCanvas, x, y, w, h); ctx.restore();
  }
  if ((S.tool === "brush" || S.tool === "eraser") && S.pointer && !S.spaceDown) {
    const r = +$("brushSize").value / 2;
    ctx.beginPath(); ctx.arc(S.pointer.sx, S.pointer.sy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,0,0,.7)"; ctx.lineWidth = 3; ctx.stroke();
    ctx.strokeStyle = S.tool === "eraser" ? "#ff8080" : "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  if (S.crop) drawCrop(x, y, w, h);
  if (S.move) drawMove(x, y, w, h);
}
function drawBox(b, color, dash) {
  const v = S.view;
  ctx.save(); ctx.setLineDash(dash || []); ctx.lineWidth = 2; ctx.strokeStyle = "rgba(0,0,0,.6)";
  const r = [v.tx + b[0] * v.s, v.ty + b[1] * v.s, (b[2] - b[0]) * v.s, (b[3] - b[1]) * v.s];
  ctx.strokeRect(r[0] + 1, r[1] + 1, r[2], r[3]); ctx.strokeStyle = color; ctx.strokeRect(...r); ctx.restore();
}
function label(t, x, y, right) {
  ctx.save(); ctx.font = "600 12px system-ui"; const w = ctx.measureText(t).width + 16;
  const lx = right ? x - w : x;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.beginPath(); ctx.roundRect(lx, y, w, 22, 11); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.fillText(t, lx + 8, y + 15); ctx.restore();
}
function loop() { if (dirty) draw(); requestAnimationFrame(loop); }

// ---- status ---------------------------------------------------------------------------------

async function pollStatus() {
  try {
    const s = await api("GET", "/api/status");
    S.status = s;
  } catch (e) { S.status = { state: "error", message: e.message }; }
  renderStatus();
  if (S.status.state === "starting") setTimeout(pollStatus, 1500);
  else if (S.status.state === "ready") { loadRecent(); loadSamples(); if (!S.project) restoreFromHash(); }
}
function renderStatus() {
  const pill = $("statusPill"), st = S.status.state;
  pill.className = "pill " + (st === "ready" && !S.job ? "ready" : st === "error" ? "error" : "busy");
  $("statusText").textContent = S.job ? "Editing on this computer" : st === "ready" ? "Offline, on this computer" : st === "error" ? "Not running" : "Loading models";
  pill.title = S.status.message || "";
  if (st === "error") toast(S.status.message, "error");
  updateButtons();
}

// ---- projects -------------------------------------------------------------------------------

async function loadSamples() {
  const list = await guard(() => api("GET", "/api/samples"));
  if (!list) return;
  $("samplesWrap").hidden = !list.length;
  const grid = $("samples");
  grid.innerHTML = "";
  for (const x of list) {
    const b = document.createElement("button");
    b.className = "recent-card sample-card";
    b.innerHTML = `<img loading="lazy" src="${esc(x.thumb_url)}" alt=""><div><span>"${esc(x.say)}"</span></div>`;
    b.onclick = async () => {
      if (S.opening) return;
      S.opening = true; S.typeAhead = ""; S.enterAhead = false; showOpening("the sample");
      const p = await guard(() => api("POST", "/api/samples/" + x.id));
      S.opening = false; hideOpening();
      if (p) { setProject(p, true); $("instruction").focus(); $("dockNote").innerHTML = "The sentence is filled in. Press <b>Enter</b> to see what it selects."; $("dockNote").dataset.locked = "1"; setTimeout(() => delete $("dockNote").dataset.locked, 8000); applyTypeAhead(); }
      else { S.typeAhead = ""; S.enterAhead = false; }
    };
    grid.appendChild(b);
  }
}

async function loadRecent() {
  const list = await guard(() => api("GET", "/api/projects"));
  if (!list) return;
  const grid = $("recent");
  grid.innerHTML = "";
  $("recentWrap").hidden = !list.length;
  for (const r of list) {
    const b = document.createElement("button");
    b.className = "recent-card";
    b.innerHTML = `<img loading="lazy" src="${esc(r.thumb_url)}" alt=""><div><b>${esc(r.name)}</b>${r.width}x${r.height} · ${r.edits} edit${r.edits === 1 ? "" : "s"}</div><span class="del" title="Delete this project (the photo you opened is not touched)">${ICON.x}</span>`;
    b.onclick = (e) => {
      if (e.target.closest(".del")) {
        e.stopPropagation();
        if (b.dataset.confirm) { guard(async () => { await api("DELETE", "/api/projects/" + r.project_id); loadRecent(); toast("Project deleted."); }); }
        else { b.dataset.confirm = "1"; b.querySelector(".del").style.background = "var(--fail)"; toast("Press the red button again to delete this project and its versions."); setTimeout(() => { delete b.dataset.confirm; b.querySelector(".del").style.background = ""; }, 4000); }
        return;
      }
      openProject(r.project_id);
    };
    grid.appendChild(b);
  }
}

// ---- new pictures from words --------------------------------------------------------------------

const MAKE_AR = { square: 1, landscape: 1216 / 832, portrait: 832 / 1216, wide: 1344 / 768 };
S.make = { size: "square", job: null, cards: [] };
function makeSize(v) {
  S.make.size = v;
  document.querySelectorAll("#makeSize button").forEach((b) => b.classList.toggle("on", b.dataset.size === v));
}
async function makePictures() {
  const prompt = $("makePrompt").value.trim();
  if (!prompt) { toast("Describe the picture first."); $("makePrompt").focus(); return; }
  if (S.status.state !== "ready") { toast("The models are still loading. Try again in a moment."); return; }
  if (S.make.job) return;
  const r = await guard(() => api("POST", "/api/generate", { prompt, size: S.make.size, count: 4, seed: window.LIS_MAKE_SEED }));  // a fixed seed only when a script sets one
  if (!r) return;
  S.make.job = { id: r.job_id, n: 0, count: r.count, size: S.make.size, start: Date.now(), shown: 0 };
  const grid = $("makeGrid");
  grid.hidden = false; grid.innerHTML = "";
  grid.style.setProperty("--ar", MAKE_AR[S.make.size]);
  for (let k = 0; k < r.count; k++) {
    const c = document.createElement("div");
    c.className = "make-card wait" + (k ? "" : " now");
    c.style.setProperty("--ar", MAKE_AR[S.make.size]);
    c.textContent = k ? "Waiting" : "Starting";
    grid.appendChild(c);
  }
  $("makeBtn").disabled = true; $("makeBtn").textContent = "Making";
  $("makeNote").textContent = "Making the pictures on this computer. Each one appears when it is ready.";
  pollPictures(S.make.job, grid, () => S.make.job, endMake, "Edit this one");
}
// ---- extending the photo: versions appear in the panel; one opens as a new project ----------
S.ext = { job: null };
function extSizeText() {
  if (!P()) return "";
  const W = P().width, H = P().height, sh = $("extShape").value;
  let L = 0, R = 0, T = 0, B = 0;
  if (sh === "wider") { L = R = Math.round(W * 0.25); }
  else if (sh === "taller") { T = B = Math.round(H * 0.25); }
  else if (sh === "all") { L = R = Math.round(W * 0.15); T = B = Math.round(H * 0.15); }
  else {
    const r = { square: 1, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4 }[sh];
    if (W / H < r) { const a = Math.round(H * r) - W; L = a >> 1; R = a - L; } else { const a = Math.round(W / r) - H; T = a >> 1; B = a - T; }
  }
  return L + R + T + B > 0 ? `${W}x${H} to ${W + L + R}x${H + T + B}` : "Already this shape";
}
async function extendPhoto() {
  if (!P() || S.ext.job || S.job) return;
  const r = await guard(() => api("POST", `/api/projects/${pid()}/extend`, { shape: $("extShape").value, hint: $("extHint").value, count: 2 }));
  if (!r) return;
  S.ext.job = { id: r.job_id, n: 0, count: r.count, start: Date.now(), shown: 0, pid: pid(), w: r.size[0], h: r.size[1] };
  const grid = $("extGrid");
  grid.hidden = false; grid.innerHTML = "";
  for (let k = 0; k < r.count; k++) {
    const c = document.createElement("div");
    c.className = "make-card wait" + (k ? "" : " now");
    c.style.setProperty("--ar", r.size[0] / r.size[1]);
    c.textContent = k ? "Waiting" : "Starting";
    grid.appendChild(c);
  }
  $("extBtn").disabled = true; $("extBtn").textContent = "Making";
  $("extNote").textContent = "Making the new edges on this computer.";
  document.body.classList.add("rendering");
  pollPictures(S.ext.job, grid, () => S.ext.job, (msg, err) => {
    S.ext.job = null; document.body.classList.remove("rendering");
    $("extBtn").disabled = false; $("extBtn").textContent = "Make 2 versions";
    $("extNote").textContent = msg; if (err) toast(msg, "error");
  }, "Open this one");
}
$("extBtn").onclick = extendPhoto;

// ---- new pictures from this photo (a pose, a scene, a style) or with another photo ------------
S.cr = { job: null, ref: null };
const CR_IDEAS = [
  ["New pose", "the same person, same clothes, now sitting on a bench with legs crossed, photo"],
  ["Waving", "the same person, same clothes, waving at the camera with a big smile, same place, photo"],
  ["New place", "the same person, same clothes, standing on a beach at sunset, photo"],
  ["Studio portrait", "the same person, a professional studio portrait, soft light, plain grey background"],
  ["Painting", "the same scene as an oil painting with visible brush strokes"],
  ["Cartoon", "the same scene as a colourful 3D animated film still"],
  ["Put them here", "the person from the second photo standing next to the person in the first photo, same place, photo"],
];
(function renderCrIdeas() {
  const box = $("crIdeas");
  for (const [label, text] of CR_IDEAS) {
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.textContent = label;
    b.onclick = () => { $("crPrompt").value = text; $("crPrompt").focus(); if (/second photo/.test(text) && !S.cr.ref) toast("Add the other photo with Add another photo."); };
    box.appendChild(b);
  }
})();
$("crRefBtn").onclick = () => $("crRefInput").click();
$("crRefInput").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  $("crNote").textContent = "Adding the photo";
  const r = await guard(() => uploadWithProgress("/api/refs?name=" + encodeURIComponent(f.name), f, (x) => { $("crNote").textContent = `Adding the photo · ${Math.round(x * 100)}%`; }));
  $("crNote").textContent = "";
  if (!r) return;
  S.cr.ref = r.ref_id;
  const u = URL.createObjectURL(f);
  $("crRefImg").src = u; $("crRefImg").hidden = false; $("crRefClear").hidden = false;
};
$("crRefClear").onclick = () => { S.cr.ref = null; $("crRefImg").hidden = true; $("crRefClear").hidden = true; };
$("crBtn").onclick = async () => {
  if (!P() || S.cr.job || S.job) return;
  const prompt = $("crPrompt").value.trim();
  if (!prompt) { toast("Describe the picture first."); $("crPrompt").focus(); return; }
  const useThis = $("crUseThis").checked;
  let size = $("crSize").value;
  if (size === "photo" && !useThis && !S.cr.ref) size = "square";
  const body = { prompt, size, count: 4, ...(useThis ? { ref_project: pid() } : {}), ...(S.cr.ref ? { ref_uploads: [S.cr.ref] } : {}) };
  const r = await guard(() => api("POST", "/api/generate", body));
  if (!r) return;
  const ar = size === "photo" ? P().width / P().height : MAKE_AR[size];
  S.cr.job = { id: r.job_id, n: 0, count: r.count, start: Date.now(), shown: 0 };
  const grid = $("crGrid");
  grid.hidden = false; grid.innerHTML = "";
  for (let k = 0; k < r.count; k++) {
    const c = document.createElement("div");
    c.className = "make-card wait" + (k ? "" : " now");
    c.style.setProperty("--ar", ar);
    c.textContent = k ? "Waiting" : "Starting";
    grid.appendChild(c);
  }
  $("crBtn").disabled = true; $("crBtn").textContent = "Making";
  $("crNote").textContent = "Making the pictures on this computer. Each one appears when it is ready.";
  document.body.classList.add("rendering");
  pollPictures(S.cr.job, grid, () => S.cr.job, (msg, err) => {
    S.cr.job = null; document.body.classList.remove("rendering");
    $("crBtn").disabled = false; $("crBtn").textContent = "Make 4 pictures";
    $("crNote").textContent = msg; if (err) toast(msg, "error");
  }, "Open this one");
};

// ---- generate something new inside a selected area ----------------------------------------------
$("genHereBtn").onclick = async () => {
  if (!S.pending || !S.pending.candidates.length) return;
  await dropUsedTargets();
  await keepPending("target");
  fillSentence("add ...");
  $("dockNote").innerHTML = "Say what to make in the orange area (for example <b>a framed painting of a lake</b>), then press <b>Enter</b>. Only that area and its edge can change.";
};
$("extShape").onchange = () => { $("extSize").textContent = extSizeText(); };
document.getElementById("extendPanel").addEventListener("toggle", () => { $("extSize").textContent = extSizeText(); });

// Shared by "make pictures" and "extend": fills cards as versions finish.
async function pollPictures(j, grid, alive, end, label) {
  if (alive() !== j) return;
  let r;
  try { r = await api("GET", `/api/jobs/${j.id}?after=${j.n}`); }
  catch (e) { if (e.status === 404) { end("The job was lost (the editor restarted?). Try again.", true); return; } setTimeout(() => pollPictures(j, grid, alive, end, label), 1000); return; }
  j.n = r.n;
  const cards = grid.children;
  for (const e of r.events) { const c = cards[e.variant - 1]; if (c && c.classList.contains("wait")) { c.classList.add("now"); c.textContent = `Step ${e.step} of ${e.of}`; } }
  for (const im of r.images.slice(j.shown)) {
    const b = document.createElement("button");
    b.className = "make-card";
    b.style.setProperty("--ar", im.width / im.height);
    b.title = `${label} (seed ${im.seed}, ${(im.timing_ms.all / 1000).toFixed(1)} s)`;
    b.innerHTML = `<img alt="" src="${esc(im.url)}"><span>${esc(label)}</span>`;
    b.onclick = () => openMade(j.id, im.i);
    grid.replaceChild(b, cards[im.i]);
    j.shown++;
  }
  if (!r.done) { setTimeout(() => pollPictures(j, grid, alive, end, label), 400); return; }
  for (const c of [...cards]) if (c.classList.contains("wait")) c.remove();
  const secs = Math.round((Date.now() - j.start) / 1000);
  end(r.status === "ok" ? `${r.images.length} version${r.images.length === 1 ? "" : "s"} in ${secs} s. Click one to open it as a new project.` + (r.message ? " " + r.message : "")
    : r.status === "cancelled" ? "Stopped." : (r.message || "It could not be made."), r.status !== "ok" && r.status !== "cancelled");
}

function endMake(msg, error = false) {
  S.make.job = null;
  $("makeBtn").disabled = false; $("makeBtn").textContent = "Make 4 pictures";
  $("makeNote").textContent = msg;
  if (error) toast(msg, "error");
}
async function openMade(jid, i) {
  if (S.opening) return;
  S.opening = true; showOpening("the picture");
  const p = await guard(() => api("POST", `/api/generated/${jid}/${i}/open`));
  S.opening = false; hideOpening();
  if (!p) return;
  setProject(p, true);
  $("instruction").focus();
  toast("Opened as a new project. Say what to change, as with any photo.", "ok");
}
function goHome() {
  if (!S.project) return;
  if (S.move) endMove();
  if (S.crop && typeof cancelCrop === "function") cancelCrop();
  S.project = null; S.pending = null; S.proposal = null; S.variants = null;
  history.replaceState(null, "", location.pathname + location.search);
  renderAll(); redraw(); loadRecent();
}
document.querySelectorAll("#makeSize button").forEach((b) => b.onclick = () => makeSize(b.dataset.size));
$("makeBtn").onclick = makePictures;
$("makePrompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); e.stopPropagation(); makePictures(); } });

async function uploadFile(file) {
  if (!file) return;
  if (S.status.state !== "ready") { toast("The models are still loading. Try again in a moment."); return; }
  if (!/^image\//.test(file.type) && !/\.(png|jpe?g|webp|bmp|tiff?|heic|heif|gif)$/i.test(file.name)) { toast("Open a JPEG, PNG, HEIC, WebP, TIFF or BMP photo.", "error"); return; }
  if (file.size > 80 * 1024 * 1024) { toast("The photo is larger than 80 MB. Save a smaller copy and open that.", "error"); return; }
  if (S.opening) return;
  S.opening = true;
  const localUrl = URL.createObjectURL(file);
  const preview = new Image();
  preview.src = localUrl;
  showOpening(file.name);
  const t0 = Date.now();
  const p = await guard(() => uploadWithProgress("/api/projects?name=" + encodeURIComponent(file.name), file, (f) => {
    $("progressBar").style.width = Math.round(5 + f * 85) + "%";
    if (Date.now() - t0 > 600) $("progressText").textContent = f < 1 ? `Sending the photo to the editor · ${Math.round(f * 100)}% of ${(file.size / 1048576).toFixed(1)} MB` : "Copying the photo into a new project";
  }));
  S.opening = false;
  hideOpening();
  if (!p) { URL.revokeObjectURL(localUrl); return; }
  try { await preview.decode(); } catch (_) {}
  // The browser already has the photo: show it at once while the server's copy arrives.
  if (preview.naturalWidth === p.width && preview.naturalHeight === p.height) { ready.set(p.image_url, preview); imgCache.set(p.image_url, Promise.resolve(preview)); }
  setProject(p, true);
  $("instruction").focus();
  applyTypeAhead();
  if (p.import_notes && p.import_notes.length) toast("Opened: " + p.import_notes.join("; ") + ".");
}
function showOpening(name) {
  $("progress").hidden = false;
  $("progressTitle").textContent = "Opening " + name;
  $("progressText").textContent = "Copying the photo into a new project";
  $("progressBar").style.width = "30%";
  $("cancelBtn").hidden = true;
  document.body.classList.add("opening");
}
function hideOpening() {
  if (!S.job) $("progress").hidden = true;
  $("cancelBtn").hidden = false;
  document.body.classList.remove("opening");
}

async function openProject(id) {
  const p = await guard(() => api("GET", "/api/projects/" + encodeURIComponent(id)));
  if (p) setProject(p, true);
}
function restoreFromHash() {
  const m = location.hash.match(/p=([A-Za-z0-9_.-]+)/);
  if (m) openProject(m[1]);
}

function setProject(p, fresh) {
  const changed = !S.project || S.project.project_id !== p.project_id;
  if (changed && S.move) endMove();  // a move belongs to the photo it was started on
  S.project = p;
  if (changed || fresh) {
    S.pending = null; S.proposal = null; S.compare = false; S.showMap = false; S.showOverlays = true; S.selectedObj = null;
    $("instruction").value = p.instruction || "";
    S.selFor = p.sel_for || null;
    showTab("edit");
    if (changed) window.scrollTo(0, 0);  // on a phone the start page may have been scrolled down to a sample
    $("seed").value = p.seed || 1234;
    autoGrow();
    history.replaceState(null, "", "#p=" + p.project_id);
    loadImage(p.image_url).then(() => fit());
    fit();
    if (S.job && S.job.project_id !== p.project_id) { $("progress").hidden = true; }
    if (p.running_job && (!S.job || S.job.id !== p.running_job)) attachJob(p.running_job);
  }
  img(p.image_url);
  const head = headRev();
  if (head) { img(head.parent_url); }
  renderAll();
}

// ---- selection: clicks, boxes, words -----------------------------------------------------

async function runSelect() {
  const pend = S.pending;
  const r = await guard(() => api("POST", `/api/projects/${pid()}/select`, { object_id: pend.object_id, points: pend.points, box: pend.box }));
  if (!r || S.pending !== pend) return;
  pend.candidates = r.candidates;
  pend.picked = 0;
  showTab("edit");
  if (!r.candidates.length) toast("Nothing found there. Click inside the object or draw a box around it.");
  renderPanels(); redraw();
}

function startPending(title) {
  S.pending = { object_id: newObjectId(), points: [], box: null, candidates: [], picked: 0, title: title || "New selection" };
}

function clickSelect(p, negative) {
  if (!S.pending || S.pending.fromWords && !S.pending.box) startPending();
  S.pending.points.push([Math.round(p.x), Math.round(p.y), negative ? 0 : 1]);
  S.showOverlays = true;
  runSelect();
}
function boxSelect(b) {
  if (!S.pending) startPending();
  S.pending.box = b.map(Math.round);
  S.showOverlays = true;
  runSelect();
}

async function findByName(text) {
  if (!text.trim()) return;
  const id = newObjectId();
  const r = await guard(() => api("POST", `/api/projects/${pid()}/find`, { object_id: id, text }));
  if (!r) return;
  if (!r.candidates.length) { toast(`Nothing matching "${text}" was found. Try other words, or click the object.`); return; }
  S.pending = { object_id: id, points: [], box: r.candidates[0].box, candidates: r.candidates, picked: 0, title: `Found: ${r.phrase}`, fromWords: true };
  showTab("edit");
  S.showOverlays = true;
  renderPanels(); redraw();
}

// A new edit area after a finished edit starts a new edit: the area the last render used is
// dropped instead of being edited again together with the new one. Kept areas stay.
async function dropUsedTargets() {
  const head = headRev();
  if (!head || S.lastRevision !== head.revision_id || !S.usedTargets) return;
  const used = P().objects.filter((o) => o.kind === "target" && S.usedTargets.has(o.mask_id));
  for (const o of used) { const q = await guard(() => api("DELETE", `/api/projects/${pid()}/objects/${o.mask_id}`)); if (q) S.project = q; }
  S.usedTargets = null;
  if (used.length) toast("New edit area. The one the last edit used was cleared; kept areas stay.");
}

async function keepPending(kind, invert = false) {
  const pend = S.pending;
  if (!pend || !pend.candidates.length) return;
  S.subtract = false; $("subBtn").classList.remove("on"); $("subBtn").textContent = "Take away a part";
  if (kind === "target") await dropUsedTargets();
  const c = pend.candidates[pend.picked];
  let name = (c.label || "").replace(/\s*\(weak match, check it\)/, "").replace(/^candidate$/, "");
  if (!name) name = kind === "target" ? (invert ? "everything else" : "edit area") : "kept area";
  if (invert) name = name === "everything else" ? name : "all except " + name;
  const p = await guard(() => api("POST", `/api/projects/${pid()}/objects`, { items: [{ object_id: pend.object_id, candidate_id: c.candidate_id, kind, name, invert }] }));
  if (!p) return;
  S.pending = null;
  S.project = p;
  setSelFor(null);  // changed by hand: keep it whatever the sentence says
  renderAll(); redraw();
}

async function propose(useFields = false) {
  const instruction = $("instruction").value.trim();
  if (!instruction) { toast("Describe the edit first."); $("instruction").focus(); return null; }
  const body = { instruction };
  if (useFields) { body.target_text = $("targetField").value; body.protect_text = $("protectField").value; }
  setBusy(true, "Finding what the sentence names");
  const r = await guard(() => api("POST", `/api/projects/${pid()}/propose`, body));
  setBusy(false);
  if (!r) return null;
  // A proposal replaces the earlier edit areas; keeps the user marked stay, whatever the sentence says.
  S.proposal = { ...r, instruction, before: P().objects.filter((o) => o.kind === "target").map((o) => o.mask_id) };
  S.pending = null;
  S.showOverlays = true; S.compare = false; S.showMap = false;
  showTab("edit");
  $("targetField").value = r.fields.target;
  $("protectField").value = r.fields.protect;
  renderAll(); redraw();
  return r;
}

async function useProposal() {
  const pr = S.proposal;
  if (!pr) return false;
  if (!pr.target) { toast("Nothing to edit was found. Correct the Edit field and press Find again, or click the object."); return false; }
  const items = [];
  for (const e of pr.protect) if (e.candidate_id) items.push({ candidate_id: e.candidate_id, kind: "protected", name: e.phrase, object_id: newObjectId() });
  items.push({ candidate_id: pr.target.candidate_id, kind: "target", name: pr.target.name, object_id: newObjectId() });
  // Replace the selection that was there when the proposal was made, not what the user added while reading it.
  const p = await guard(() => api("POST", `/api/projects/${pid()}/objects`, { items, replace_ids: pr.before || [] }));
  if (!p) return false;
  setSelFor(pr.instruction);  // this selection was found from this sentence
  S.proposal = null;
  S.project = p;
  renderAll(); redraw();
  return true;
}

// ---- brush ------------------------------------------------------------------------------------

function ensureStroke() {
  const w = Math.round(P().width * DS()), h = Math.round(P().height * DS());
  if (!strokeCanvas || strokeCanvas.width !== w || strokeCanvas.height !== h) {
    strokeCanvas = document.createElement("canvas");
    strokeCanvas.width = w; strokeCanvas.height = h;
  }
  const g = strokeCanvas.getContext("2d");
  g.clearRect(0, 0, strokeCanvas.width, strokeCanvas.height);
  return g;
}
function paintSeg(a, b) {
  const g = strokeCanvas.getContext("2d");
  g.strokeStyle = S.tool === "eraser" ? "#ff5050" : (brushKind() === "protected" ? "rgb(46,196,255)" : "rgb(255,138,31)");
  g.lineCap = "round"; g.lineJoin = "round";
  const d = DS();
  g.lineWidth = Math.max(1, +$("brushSize").value / S.view.s * d);
  g.beginPath(); g.moveTo(a.x * d, a.y * d); g.lineTo(b.x * d + 0.01, b.y * d); g.stroke();
  redraw();
}
function brushKind() {
  const t = $("brushTarget").value;
  if (t === "new-protected") return "protected";
  if (t === "new-target") return "target";
  const o = P().objects.find((o) => o.mask_id === t);
  return o ? o.kind : "target";
}
async function commitStroke() {
  const png = strokeCanvas.toDataURL("image/png");
  const t = $("brushTarget").value;
  let p;
  if (t.startsWith("new-")) {
    if (S.tool === "eraser") { toast("Choose an area in Paint into to erase from."); return; }
    const kind = t === "new-protected" ? "protected" : "target";
    if (kind === "target") await dropUsedTargets();
    p = await guard(() => api("POST", `/api/projects/${pid()}/objects`, { items: [{ candidate_id: null, kind, name: kind === "target" ? "painted area" : "painted keep", brush_png: png, object_id: newObjectId() }] }));
    if (p) { S.project = p; const last = p.objects[p.objects.length - 1]; if (last) { S.selectedObj = last.mask_id; S.brushTarget = last.mask_id; } }
  } else {
    p = await guard(() => api("POST", `/api/projects/${pid()}/objects/${t}/brush`, { op: S.tool === "eraser" ? "subtract" : "add", png }));
    if (p) S.project = p;
  }
  ensureStroke();
  setSelFor(null);
  renderAll(); redraw();
}

// ---- render -------------------------------------------------------------------------------------

function settings() {
  return { seed: +$("seed").value || 0, steps: +$("steps").value, blend: $("blend").value, band: +$("band").value,
           strategy: $("strategy").value, strength: +$("strength").value };
}

// Which sentence the current selection was found from; kept on the server so a reload remembers it
// (otherwise a new sentence after a reload would render on the old sentence's objects).
function setSelFor(v) {
  S.selFor = v;
  if (P()) api("PUT", `/api/projects/${pid()}/prefs`, { sel_for: v }).catch(() => {});
}

async function primaryAction() {
  if (!P() || S.job || S.acting) return;  // a double click or a second Enter must not add the selection twice
  S.acting = true;
  try { await primaryActionOnce(); } finally { S.acting = false; }
}
async function primaryActionOnce() {
  if (S.proposal && S.proposal.instruction !== $("instruction").value.trim()) {  // the sentence changed since: find again
    await propose();
    return;
  }
  if (S.proposal) { if (await useProposal()) startRender(); return; }
  if (S.pending && S.pending.candidates.length) { await keepPending("target"); }
  const said = $("instruction").value.trim();
  if (hasTargets() && S.selFor && S.selFor !== said) {
    // The selection was found from an earlier sentence: find what the new one names first.
    const r = await propose();
    if (r) $("dockNote").innerHTML = "New sentence, so this is what it names. Press <b>Use and render</b>, or <b>Escape</b> to keep the previous selection and render with it.";
    return;
  }
  if (!hasTargets()) {
    const r = await propose();
    if (r) $("dockNote").innerHTML = r.target ? "Check the selection: <b>orange</b> is edited, <b>blue</b> is kept exactly. Press <b>Use and render</b>, or fix it." : "Nothing to edit was found. Click the object, or correct the Edit field.";
    return;
  }
  startRender();
}

async function startRender(extra = {}) {
  const instruction = $("instruction").value.trim();
  if (!instruction) { toast("Describe the edit first."); $("instruction").focus(); return; }
  const body = { instruction, ...settings(), ...extra };
  const r = await guard(() => api("POST", `/api/projects/${pid()}/render`, body));
  if (!r) return;
  attachJob(r.job_id, extra.pickFrom || null);
}

function attachJob(jobId, pickFrom = null) {
  S.job = { id: jobId, project_id: pid(), n: 0, step: 0, of: +$("steps").value, start: Date.now(), pickFrom };
  $("progress").hidden = false;
  $("progressTitle").textContent = "Editing on this computer";
  $("progressText").textContent = "Preparing the edit area";
  $("progressBar").style.width = "4%";
  bringStageIntoView();
  renderStatus();
  updateButtons();
  pollJob();
}

async function pollJob() {
  const j = S.job;
  if (!j) return;
  let r;
  try { r = await api("GET", `/api/jobs/${j.id}?after=${j.n}`); }
  catch (e) { if (e.status === 404) { finishJob(); return; } setTimeout(pollJob, 1000); return; }
  j.n = r.n;
  for (const e of r.events) { j.step = e.step || 0; j.of = e.of || j.of; if (e.note) j.note = e.note; if (e.variants > 1) { j.variant = e.variant; j.variants = e.variants; } }
  $("progress").hidden = j.project_id !== pid();
  const part = j.of ? Math.min(1, j.step / j.of) : 0;
  const frac = j.variants ? ((j.variant - 1) + part) / j.variants : part;
  $("progressBar").style.width = Math.max(4, Math.round(5 + frac * 85)) + "%";
  $("progressText").textContent = (j.variants ? `Version ${j.variant} of ${j.variants} · ` : "") + (j.step ? `Step ${j.step} of ${j.of}` : "Preparing the edit area") + ` · ${Math.round(r.elapsed)} s` + (j.note ? ` · ${j.note}` : "");
  if (!r.done) { setTimeout(pollJob, 400); return; }
  $("progressBar").style.width = "100%";
  finishJob();
  if (j.project_id !== pid()) {  // the user opened another photo meanwhile: leave the screen alone
    toast(r.status === "ok" ? "An edit on another photo finished. Open it from Recent to see it." : (r.message || "An edit on another photo did not complete."));
    return;
  }
  if (r.project) S.project = r.project;
  if (r.status === "ok") {
    S.variants = r.variants ? [...(j.pickFrom ? [j.pickFrom] : []), ...r.variants.map((v) => v.revision_id)] : null;
    S.lastRevision = r.revision.revision_id;
    S.usedTargets = new Set(P().objects.filter((o) => o.kind === "target").map((o) => o.mask_id));
    S.expansionHint = r.expansion_hint;
    S.showOverlays = false; S.pending = null; S.showMap = false;
    S.compare = true; S.compareX = 0.5;
    showTab("edit"); S.freshResult = true;
    pic(r.revision.image_url); pic(r.revision.parent_url);
    loadImage(r.revision.image_url).then(() => img(r.revision.change_map_url)).catch(() => {});
    const ov = r.revision.checks.overall;
    if (S.variants) toast(`${S.variants.length} versions to choose from: click one under Result.`, "ok"); else
    toast(ov === "pass" ? "Done. Checked on the saved file." : ov === "warn" ? "Done, but the edit was not clearly detected. Try another seed." : "The check failed.", ov === "pass" ? "ok" : "");
    $("dockNote").innerHTML = "Drag the slider to compare. Change the sentence and render again to keep editing this result.";
  } else if (r.status === "cancelled") {
    toast("Cancelled.");
  } else {
    toast(r.message || "The edit did not complete.", "error");
  }
  renderAll(); redraw();
}
function finishJob() { S.job = null; $("progress").hidden = true; renderStatus(); updateButtons(); }

// ---- panels -------------------------------------------------------------------------------------

function renderAll() { renderPanels(); renderHistory(); renderResult(); renderCrumb(); updateButtons(); updateHint(); renderBrushTargets(); if (typeof renderIdeas === "function") renderIdeas(); if (typeof renderAdjust === "function") renderAdjust(); }

function renderCrumb() {
  const p = P();
  $("crumb").innerHTML = p ? `<b>${esc(p.name)}</b> · ${p.width}x${p.height}` : "";
  $("empty").hidden = !!p;
  document.body.classList.toggle("no-project", !p);
  $("dock").hidden = !p; $("inspector").hidden = !p; $("viewBar").hidden = !p; $("projectButtons").hidden = !p;
  if (p && typeof ta !== "undefined" && ta) autoGrow();
}

function renderPanels() {
  if (!P()) return;
  // proposal
  const pr = S.proposal;
  $("proposalPanel").hidden = !pr;
  if (pr) {
    let rows = "";
    if (pr.target) rows += `<div class="prop edit"><span class="tag">Edit</span><span class="what">${esc(pr.target.name)}</span>${pr.target.weak ? '<span class="warn">weak match, check it</span>' : ""}</div>`;
    else rows += `<div class="prop edit"><span class="tag">Edit</span><span class="what">${esc(pr.target_phrases.join(", ") || "nothing named")}</span><span class="miss">not found</span></div>`;
    for (const e of pr.protect) {
      rows += `<div class="prop keep"><span class="tag">Keep</span><span class="what">${esc(e.phrase)}</span>${!e.candidate_id ? '<span class="miss">not found</span>' : e.weak ? '<span class="warn">weak match</span>' : e.used > 1 ? `<span class="muted small">${e.used} found</span>` : ""}</div>`;
    }
    $("propRows").innerHTML = rows;
    const notes = [...(pr.notes || [])];
    if (!pr.target && pr.target_phrases.length) {
      notes.push(`Nothing matching "${pr.target_phrases.join(", ")}" was found in this photo. Try another word for it in the Edit field and press Find again, or click the object on the photo, or paint it with the Brush (B).`);
    } else {
      if (pr.edit && pr.edit !== pr.instruction) notes.push(`The model is asked: "${pr.edit}". Keep clauses are enforced by the selection.`);
      const marked = P().objects.filter((o) => o.kind === "protected");
      if (marked.length) notes.push(`Also kept, as you marked: ${marked.map((o) => o.name).join(", ")}.`);
      else if (!pr.protect.length) notes.push("Nothing is marked to keep. Add \"keep the ... unchanged\" to the sentence, or click an object on the photo and choose Keep this exactly.");
    }
    for (const e of pr.protect) if (!e.candidate_id) notes.push(`"${e.phrase}" to keep was not found, so it is not protected. Click it and press Shift+Enter to keep it.`);
    $("propNote").textContent = notes.join(" ");
    $("useBtn").disabled = $("useRenderBtn").disabled = !pr.target;
  }
  // first steps, until anything is selected or rendered
  $("editIntro").hidden = !!(pr || S.pending || P().objects.length || P().revisions.length);
  // candidates
  const pend = S.pending;
  $("candPanel").hidden = !(pend && (pend.candidates.length || pend.points.length || pend.box));
  if (pend) {
    $("candTitle").textContent = pend.title || "New selection";
    const list = $("candList");
    list.innerHTML = "";
    pend.candidates.forEach((c, i) => {
      const b = document.createElement("button");
      b.className = "cand" + (i === pend.picked ? " on" : "");
      let lbl = (c.label || "").replace(/\s*\(weak match, check it\)/, "");
      if (!lbl && c.share != null) {  // click candidates: say how big each one is, so "the whole object" is easy to find
        const shares = pend.candidates.map((x) => x.share || 0).sort((a, b) => a - b);
        const rank = shares.indexOf(c.share);
        const word = shares.length === 1 ? "This" : rank === shares.length - 1 ? "Largest" : rank === 0 ? "Smallest" : "Middle";
        lbl = `${word} · ${c.share >= 0.01 ? Math.round(c.share * 100) : "<1"}% of the photo`;
      }
      if (!lbl) lbl = `Option ${i + 1}`;
      b.innerHTML = `<span>${esc(lbl)}</span>${c.confident === false ? '<span class="weak">weak match</span>' : ""}<span class="score">${c.score != null ? Math.round(c.score * 100) + "%" : ""}</span>`;
      b.onclick = () => { pend.picked = i; if (c.box && !pend.points.length) pend.box = c.box; renderPanels(); redraw(); };
      list.appendChild(b);
    });
    if (!pend.candidates.length) list.innerHTML = '<div class="obj-empty">Finding the object</div>';
    else if (!pend.fromWords && pend.candidates.length > 1 && pend.points.length === 1) {
      const h = document.createElement("div"); h.className = "muted small"; h.style.marginTop = "6px";
      h.textContent = "Only part of it? Pick a larger option, click the missing part too, or drag a box around the whole object.";
      list.appendChild(h);
    }
  }
  // objects
  const ol = $("objList");
  ol.innerHTML = "";
  if (!P().objects.length) {
    ol.innerHTML = '<div class="obj-empty">Nothing selected yet. Say what to change under the photo and press <b>Enter</b>, or click an object on the photo.</div>';
  }
  for (const o of P().objects) {
    const row = document.createElement("div");
    const willGo = S.proposal && (S.proposal.before || []).includes(o.mask_id);
    row.className = `obj ${o.kind}` + (S.selectedObj === o.mask_id ? " on" : "") + (willGo ? " replaced" : "");
    if (willGo) row.title = "Using the proposal replaces this selection; Escape keeps it.";
    row.innerHTML = `<span class="swatch" title="Show on the photo"></span><input class="name" value="${esc(o.name)}" title="Rename">
      <span class="seg"><button class="${o.kind === "target" ? "on target" : ""}" data-k="target" title="Edit this area">Edit</button><button class="${o.kind === "protected" ? "on protected" : ""}" data-k="protected" title="Keep this area exactly">Keep</button></span>
      <button class="icon-btn sm" title="Remove from the selection">${ICON.x}</button>`;
    row.querySelector(".swatch").onclick = () => { S.selectedObj = S.selectedObj === o.mask_id ? null : o.mask_id; S.showOverlays = true; S.compare = false; S.showMap = false; renderAll(); redraw(); };
    const nm = row.querySelector(".name");
    nm.onchange = () => guard(async () => { S.project = await api("PATCH", `/api/projects/${pid()}/objects/${o.mask_id}`, { name: nm.value }); renderAll(); });
    nm.onfocus = () => {
      S.selectedObj = o.mask_id; S.showOverlays = true; S.compare = false; S.showMap = false;
      document.querySelectorAll(".obj").forEach((r) => r.classList.toggle("on", r === row));
      $("objTools").hidden = false; $("objToolsName").textContent = o.name;
      renderBrushTargets(); updateButtons(); redraw();
    };
    row.querySelectorAll(".seg button").forEach((b) => b.onclick = () => {
      if (b.dataset.k === o.kind) return;
      guard(async () => { S.project = await api("PATCH", `/api/projects/${pid()}/objects/${o.mask_id}`, { kind: b.dataset.k }); S.selectedObj = null; renderAll(); redraw(); });
    });
    row.querySelector(".icon-btn").onclick = () => guard(async () => { S.project = await api("DELETE", `/api/projects/${pid()}/objects/${o.mask_id}`); if (S.selectedObj === o.mask_id) S.selectedObj = null; renderAll(); redraw(); });
    ol.appendChild(row);
  }
  $("objClear").hidden = !P().objects.length;
  const selObj = P().objects.find((o) => o.mask_id === S.selectedObj);
  $("objTools").hidden = !selObj;
  if (selObj) $("objToolsName").textContent = selObj.name;
  renderBrushTargets();
  updateButtons();
}

function renderBrushTargets() {
  if (!P()) return;
  const sel = $("brushTarget");
  const opts = [["new-target", "New edit area"], ["new-protected", "New keep area"], ...P().objects.map((o) => [o.mask_id, `${o.name} (${o.kind === "target" ? "edit" : "keep"})`])];
  const want = S.selectedObj && P().objects.some((o) => o.mask_id === S.selectedObj) ? S.selectedObj : (opts.some((o) => o[0] === S.brushTarget) ? S.brushTarget : "new-target");
  sel.innerHTML = opts.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  sel.value = want;
  S.brushTarget = want;
}

function renderResult() {
  const head = headRev();
  $("resultPanel").hidden = !head;
  $("proofChip").hidden = !head;
  if (!head) return;
  const c = head.checks;
  renderProofChip(c);
  if (S.freshResult) { S.freshResult = false; const rp = $("resultPanel"); rp.classList.remove("fresh"); void rp.offsetWidth; rp.classList.add("fresh"); }
  // "Done as asked" only when the edit itself was verified; otherwise the file check passed and the look is the user's call.
  const verified = (c.rows.find((r) => r.key === "edit") || {}).status === "pass";
  const title = { pass: verified ? "Checked: done as asked" : "Checked: safe, now judge the look", warn: "Done, edit not clearly detected", fail: "Check failed" }[c.overall];
  const sub = { pass: verified ? "Kept areas and everything outside the edit are unchanged on the saved file." : "The file check passed. Whether the change matches your words cannot be measured for this kind of edit: compare and decide.", warn: "Nothing outside the edit area changed. Try another seed or reword the sentence.", fail: "This version did not pass the file check." }[c.overall];
  $("verdict").className = "verdict " + c.overall;
  $("verdict").innerHTML = `${c.overall === "pass" ? ICON.shield : ICON[c.overall]}<div>${esc(title)}<small>${esc(sub)}</small></div>`;
  $("checks").innerHTML = c.rows.map((r) => `<li class="${r.status}">${ICON[r.status] || ICON.info}<div>${esc(r.title)}${r.detail ? `<small>${esc(r.detail)}</small>` : ""}</div></li>`).join("");
  const vs = (S.variants || []).filter((id) => P().revisions.some((x) => x.revision_id === id));
  $("variants").hidden = !(vs.length > 1 && vs.includes(head.revision_id));
  if (!$("variants").hidden) {
    $("variants").innerHTML = `<div class="muted small">Pick a version</div><div class="vstrip">` + vs.map((id, i) => {
      const v = P().revisions.find((x) => x.revision_id === id);
      return `<button class="vpick${id === head.revision_id ? " on" : ""} ${v.checks.overall}" data-id="${esc(id)}" title="Seed ${v.seed}: ${esc(v.checks.overall === "pass" ? "checks passed" : v.checks.overall === "warn" ? "edit not clearly detected" : "check failed")}"><img src="${esc(v.thumb_url)}" alt=""><span>${i + 1}</span></button>`;
    }).join("") + `</div>`;
    $("variants").querySelectorAll(".vpick").forEach((b) => b.onclick = async () => { await setHead(b.dataset.id); S.compare = true; S.compareX = 0.5; S.showOverlays = false; renderAll(); redraw(); });
  }
  $("resultMeta").textContent = `seed ${head.seed} · ${head.seconds} s` + (head.worker_size ? ` · ${head.worker_size[0]}x${head.worker_size[1]}` : "");
  $("growBtn").hidden = !(S.expansionHint && S.lastRevision === head.revision_id);
  adjustResultCopy(head);
}

// A small badge on the photo with the strongest claim the checks support, in their own words.
function renderProofChip(c) {
  const chip = $("proofChip");
  const prot = c.rows.find((r) => r.key === "protected");
  let cls, text;
  if (c.overall === "fail") { cls = "fail"; text = "Check failed"; }
  else if (prot && prot.status === "pass") { cls = "pass"; text = "Kept areas unchanged"; }
  else if (c.overall === "pass") { cls = "pass"; text = "File checked"; }
  else { cls = "warn"; text = "Edit not clearly detected"; }
  chip.className = "proof-chip " + cls;
  chip.innerHTML = (cls === "pass" ? ICON.shield : ICON[cls]) + `<span>${esc(text)}</span>`;
  chip.title = (prot && prot.detail ? prot.detail + ". " : "") + "Checked on the saved file. Click to see every check.";
}
$("proofChip").onclick = () => { showTab("edit"); $("resultPanel").scrollIntoView({ block: "nearest", behavior: "smooth" }); };

function renderHistory() {
  if (!P()) return;
  const h = $("history");
  const revs = P().revisions;
  const onPath = new Set();
  let cur = P().head;
  while (cur) { onPath.add(cur); const r = revs.find((x) => x.revision_id === cur); cur = r ? r.parent : null; }
  const rows = [{ id: "source", t: "Original photo", m: `${P().width}x${P().height}`, thumb: `/api/projects/${pid()}/thumb/source.jpg`, on: P().head === null, path: true }];
  for (const r of revs) rows.push({ id: r.revision_id, t: r.instruction || "edit", m: r.kind === "adjust" ? `no model · ${r.seconds} s` : `seed ${r.seed} · ${r.seconds} s`, thumb: r.thumb_url, on: r.revision_id === P().head, path: onPath.has(r.revision_id), badge: r.checks.overall });
  h.innerHTML = "";
  for (const r of rows.reverse()) {
    const b = document.createElement("button");
    b.className = "hist" + (r.on ? " on" : "") + (r.path ? "" : " off-path");
    b.innerHTML = `<img src="${esc(r.thumb)}" alt="" loading="lazy"><div><div class="t">${esc(r.t)}</div><div class="m">${esc(r.m)}</div></div>${r.badge ? `<span class="badge ${r.badge}">${ICON[r.badge]}</span>` : "<span></span>"}`;
    b.onclick = () => setHead(r.id);
    h.appendChild(b);
  }
  $("historyCount").textContent = revs.length ? `${revs.length} edit${revs.length === 1 ? "" : "s"}` : "";
  $("historyTabCount").textContent = revs.length ? String(revs.length) : "";
}

async function setHead(id) {
  const p = await guard(() => api("POST", `/api/projects/${pid()}/head`, { revision_id: id }));
  if (!p) return;
  S.project = p; S.pending = null; S.proposal = null; S.showMap = false;
  S.compare = false; S.showOverlays = id === "source" || !headRev();
  img(p.image_url);
  renderAll(); redraw();
}

function updateButtons() {
  const p = P(), running = !!S.job, readyNow = S.status.state === "ready";
  document.body.classList.toggle("rendering", running && !!p && S.job.project_id === p.project_id);
  if (running && p && S.job.project_id !== p.project_id && !$("dockNote").dataset.locked)
    $("dockNote").textContent = "An edit on another photo is running. Render here when it finishes; selecting already works.";
  const head = headRev();
  $("renderBtn").disabled = !p || running || !readyNow;
  if ($("moreBtn")) $("moreBtn").disabled = !p || running || !readyNow || !head;
  $("findBtn").disabled = !p || running || !readyNow;
  $("seedBtn").disabled = running || !head;
  $("undoBtn").disabled = !p || running || !p.head;
  $("redoBtn").disabled = !p || running || !p.revisions.some((r) => r.parent === p.head);
  $("compareBtn").disabled = !head; $("mapBtn").disabled = !head;
  $("compareBtn").classList.toggle("on", S.compare && !!head);
  $("mapBtn").classList.toggle("on", S.showMap && !!head);
  $("overlayBtn").classList.toggle("on", S.showOverlays && !S.compare && !S.showMap);
  $("mapLegend").hidden = !(S.showMap && head);
  if (typeof updateAdjustButtons === "function") updateAdjustButtons();
  if (p) {
    const label = S.proposal ? "Use and render" : (hasTargets() || (S.pending && S.pending.candidates.length)) ? "Render" : "Find and check";
    $("renderBtn").textContent = label;
    $("findBtn").hidden = !hasTargets() && !S.proposal;
    $("findBtn").textContent = S.proposal ? "Find again" : "Find again";
  }
  document.querySelectorAll(".tool[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === S.tool));
  $("brushBar").hidden = !p || !(S.tool === "brush" || S.tool === "eraser");
  canvas.style.cursor = S.spaceDown || S.tool === "hand" ? (S.drag && S.drag.kind === "pan" ? "grabbing" : "grab") : S.tool === "select" ? "crosshair" : "none";
  if (S.compare && head && S.tool !== "hand") canvas.style.cursor = "ew-resize";
  if (p && !S.proposal && !hasTargets() && !S.job && !$("dockNote").dataset.locked) {
    $("dockNote").innerHTML = "Say what to change and what to keep. Press <b>Enter</b> to see the selection before anything is edited.";
  }
}

function updateHint() {
  const h = $("toolHint");
  $("moveToolBtn").classList.toggle("on", !!S.move);
  document.body.classList.toggle("moving", !!S.move);    // the move and crop bars take the sentence box's place
  document.body.classList.toggle("cropping", !!S.crop);
  if (!P() || S.job) { h.hidden = true; return; }
  let t = "";
  if (S.compare && headRev()) t = "<b>Drag</b> to compare before and after. Hold <b>\\</b> to flash the version before";
  else if (S.showMap) t = "<b>What changed:</b> orange changed, blue kept exactly, white line is the edit area boundary";
  else if (S.tool === "select") t = S.pending ? "<b>Click</b> to add, <b>Alt+click</b> to remove a part, <b>Enter</b> adds to edit, <b>Shift+Enter</b> to keep" : "<b>Click</b> an object to select it, or <b>drag</b> a box around it";
  else if (S.tool === "brush") t = "<b>Paint</b> to add to the area chosen in Paint into";
  else if (S.tool === "eraser") t = "<b>Paint</b> to remove from the area chosen in Paint into";
  else t = "<b>Drag</b> to move, <b>scroll</b> to zoom";
  h.innerHTML = t;
  h.hidden = !!(S.tool === "brush" || S.tool === "eraser");
  if (S.crop || S.move) h.hidden = true;  // the crop and move bars say what to do
}

function setBusy(on, msg) {
  S.busy = on;
  document.body.style.cursor = on ? "progress" : "";
  if (on && msg) $("dockNote").textContent = msg + "...";
}

function setTool(t) { S.tool = t; if ((t === "brush" || t === "eraser")) { S.compare = false; S.showMap = false; S.showOverlays = true; } updateButtons(); updateHint(); redraw(); }

// ---- events -------------------------------------------------------------------------------------

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
// Two fingers: pinch to zoom and move. One finger: the current tool.
const touches = new Map();
let pinch = null;
function pinchState() {
  const [a, b] = [...touches.values()];
  const r = canvas.getBoundingClientRect();
  return { d: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2 - r.left, cy: (a.y + b.y) / 2 - r.top };
}
canvas.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "touch") {
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (touches.size === 2) { S.drag = null; if (strokeCanvas) ensureStroke(); pinch = pinchState(); redraw(); return; }
  }
}, true);
canvas.addEventListener("pointermove", (e) => {
  if (e.pointerType !== "touch" || !touches.has(e.pointerId)) return;
  touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (touches.size === 2 && pinch) {
    e.stopImmediatePropagation();
    const n = pinchState();
    S.view.tx += n.cx - pinch.cx; S.view.ty += n.cy - pinch.cy;
    zoomAt(n.d / Math.max(1, pinch.d), n.cx, n.cy);
    pinch = n;
  }
}, true);
["pointerup", "pointercancel"].forEach((t) => canvas.addEventListener(t, (e) => {
  if (e.pointerType !== "touch") return;
  touches.delete(e.pointerId);
  if (pinch) { e.stopImmediatePropagation(); if (touches.size === 0) pinch = null; S.drag = null; }
}, true));
canvas.addEventListener("pointerdown", (e) => {
  if (!P() || pinch) return;
  canvas.setPointerCapture(e.pointerId);
  const p = toImage(e);
  const pan = S.spaceDown || S.tool === "hand" || e.button === 1;
  if (pan) { S.drag = { kind: "pan", sx: e.clientX, sy: e.clientY, tx: S.view.tx, ty: S.view.ty }; updateButtons(); return; }
  if (S.compare && headRev()) { S.drag = { kind: "compare" }; moveCompare(p); return; }
  if (S.showMap) return;
  if (S.job) return;
  if (S.tool === "select") {
    S.drag = { kind: "box", start: p, now: p, moved: false, negative: e.altKey || e.button === 2 || e.ctrlKey || !!S.subtract, sx: e.clientX, sy: e.clientY };
  } else if (S.tool === "brush" || S.tool === "eraser") {
    if (S.tool === "eraser" && $("brushTarget").value.startsWith("new-")) { toast("Choose an area in Paint into to erase from."); return; }
    ensureStroke();
    S.drag = { kind: "paint", last: p };
    paintSeg(p, p);
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (!P()) return;
  const p = toImage(e);
  S.pointer = p;
  const d = S.drag;
  if (d && d.kind === "pan") { S.view.tx = d.tx + e.clientX - d.sx; S.view.ty = d.ty + e.clientY - d.sy; }
  else if (d && d.kind === "compare") moveCompare(p);
  else if (d && d.kind === "box") { d.now = p; if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) > 5) d.moved = true; }
  else if (d && d.kind === "paint") { paintSeg(d.last, p); d.last = p; }
  redraw();
});
canvas.addEventListener("pointerup", (e) => {
  const d = S.drag;
  S.drag = null;
  if (!d) return;
  if (d.kind === "box") {
    if (d.moved) {
      const a = d.start, b = d.now, W = P().width, H = P().height;
      const box = [Math.max(0, Math.min(a.x, b.x)), Math.max(0, Math.min(a.y, b.y)), Math.min(W, Math.max(a.x, b.x)), Math.min(H, Math.max(a.y, b.y))];
      if (box[2] - box[0] > 3 && box[3] - box[1] > 3) boxSelect(box);
    } else if (inside(d.start)) clickSelect(d.start, d.negative);
  } else if (d.kind === "paint") commitStroke();
  updateButtons(); redraw();
});
canvas.addEventListener("pointerleave", () => { S.pointer = null; redraw(); });
canvas.addEventListener("wheel", (e) => {
  if (!P()) return;
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  if (e.ctrlKey || !e.shiftKey) zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), e.clientX - r.left, e.clientY - r.top);
  else { S.view.tx -= e.deltaX || e.deltaY; redraw(); }
}, { passive: false });
function moveCompare(p) { S.compareX = Math.min(1, Math.max(0, p.x / P().width)); redraw(); }

window.addEventListener("resize", () => { resize(); });
new ResizeObserver(() => resize()).observe(stage);

// drag and drop anywhere on the stage
["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.add("drag-over"); }));
["dragleave", "drop"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); if (t === "dragleave" && stage.contains(e.relatedTarget)) return; stage.classList.remove("drag-over"); }));
stage.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (f) uploadFile(f);
});

$("openBtn").onclick = () => $("fileInput").click();
$("newBtn").onclick = goHome;
document.querySelector(".brand").onclick = goHome;
$("fileInput").onchange = (e) => { uploadFile(e.target.files[0]); e.target.value = ""; };

document.querySelectorAll(".tool[data-tool]").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
$("fitBtn").onclick = fit;
$("zoom100Btn").onclick = () => { const { w, h } = stageSize(); zoomAt(1 / S.view.s, w / 2, h / 2); };
$("helpBtn").onclick = () => { $("helpModal").hidden = false; };
$("helpModal").onclick = (e) => { if (e.target === $("helpModal") || e.target.closest("[data-close]")) $("helpModal").hidden = true; };

$("overlayBtn").onclick = () => { S.showOverlays = !(S.showOverlays && !S.compare && !S.showMap); S.compare = false; S.showMap = false; updateButtons(); updateHint(); redraw(); };
$("compareBtn").onclick = () => { S.compare = !S.compare; S.showMap = false; updateButtons(); updateHint(); redraw(); };
$("mapBtn").onclick = () => { S.showMap = !S.showMap; S.compare = false; updateButtons(); updateHint(); redraw(); };

$("brushSize").oninput = () => { $("brushSizeOut").textContent = $("brushSize").value; redraw(); };
$("brushTarget").onchange = () => { S.brushTarget = $("brushTarget").value; S.selectedObj = S.brushTarget.startsWith("new-") ? null : S.brushTarget; renderPanels(); redraw(); $("brushTarget").blur(); };  // give the keys back to B, E, [ and ]

$("findBtn").onclick = () => propose(!!S.proposal);
$("renderBtn").onclick = primaryAction;
$("refindBtn").onclick = () => propose(true);
$("useBtn").onclick = async () => { if (await useProposal()) $("dockNote").innerHTML = "Selection in use. Press <b>Render</b> when ready, or refine it with clicks and the brush."; };
$("useRenderBtn").onclick = () => { if (S.proposal) primaryAction(); };  // same path as Enter, so a double click cannot add it twice
["targetField", "protectField"].forEach((id) => $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); propose(true); } }));
$("candClear").onclick = () => { S.pending = null; renderPanels(); redraw(); };
$("addEditBtn").onclick = () => keepPending("target");
$("addKeepBtn").onclick = () => keepPending("protected");
$("addInvertBtn").onclick = () => keepPending("target", true);
function pendingName() {
  const c = S.pending && S.pending.candidates[S.pending.picked];
  const n = ((c && c.label) || "").replace(/\s*\(weak match, check it\)/, "").replace(/\s+\d+$/, "");
  return /^(option|candidate|all \d+|new selection)/i.test(n) || !n ? "" : n;
}
$("subBtn").onclick = () => { S.subtract = !S.subtract; $("subBtn").classList.toggle("on", S.subtract); $("subBtn").textContent = S.subtract ? "Taking away: click parts" : "Take away a part"; };
$("removeItBtn").onclick = async () => {
  const name = pendingName();
  await keepPending("target");
  fillSentence(name ? `remove the ${name}` : "remove the ...");
  $("dockNote").innerHTML = name ? "Press <b>Enter</b> to remove it." : "Type what it is (for example <b>lamp</b>), then press <b>Enter</b>. The model needs the word; the selection says where.";
};
// ---- moving an object: drag its cut-out; the model fills the old place, the pixels are its own ----
$("moveItBtn").onclick = async () => {
  if (!S.pending || !S.pending.candidates.length) return;
  const name = pendingName();
  await keepPending("target");
  const targets = P().objects.filter((o) => o.kind === "target");
  if (targets.length !== 1) { toast("Move one object at a time: clear the other edit areas first."); return; }
  startMove(targets[0], name);
};
$("moveObjBtn").onclick = () => {
  const obj = P() && P().objects.find((o) => o.mask_id === S.selectedObj);
  if (!obj) return;
  if (obj.kind !== "target") { toast("Only an edit area (orange) can be moved. Switch it to Edit first."); return; }
  if (P().objects.filter((o) => o.kind === "target").length !== 1) { toast("Move one object at a time: remove the other edit areas first."); return; }
  startMove(obj, obj.name);
};
function startMove(obj, name) {
  if (S.crop) cancelCrop();
  if (S.tool !== "select") setTool("select");
  S.move = { mask_id: obj.mask_id, name: name || obj.name || "object", dx: 0, dy: 0, drag: null, cut: null, box: null };
  S.compare = false; S.showMap = false; S.showOverlays = true;
  $("moveBar").hidden = false; $("moveApplyBtn").disabled = true;
  bringStageIntoView();
  const named = !/^(edit area|kept area|object|background)$/i.test(S.move.name);
  $("moveText").textContent = named ? `Drag the ${S.move.name} to its new place. Arrow keys nudge it.` : "Drag it to its new place. Arrow keys nudge it.";
  $("moveName").value = named ? S.move.name : "";
  // The cut-out: the photo on screen where the mask is, transparent elsewhere, at view size.
  Promise.all([loadImage(P().image_url), loadImage(obj.url)]).then(([im, mk]) => {
    if (!S.move || S.move.mask_id !== obj.mask_id) return;
    const w = im.naturalWidth, h = im.naturalHeight;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(mk, 0, 0, w, h);
    const m = g.getImageData(0, 0, w, h).data;
    g.clearRect(0, 0, w, h); g.drawImage(im, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i += 4) {
      const a = m[i] >= 128 ? 255 : 0;
      d.data[i + 3] = a;
      if (a) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    g.putImageData(d, 0, 0);
    const k = P().width / w;
    S.move.cut = c; S.move.box = x1 >= 0 ? [x0 * k, y0 * k, (x1 + 1) * k, (y1 + 1) * k] : null;
    redraw();
  }).catch(() => toast("Could not prepare the object to move. Try again.", "error"));
  updateHint(); redraw();
}
// On a phone the panels are below the photo: a tool that works on the photo scrolls back up to it.
function bringStageIntoView() { if (matchMedia("(max-width: 860px)").matches && window.scrollY > 0) window.scrollTo({ top: 0, behavior: "smooth" }); }
function endMove() { S.move = null; $("moveBar").hidden = true; redraw(); updateHint(); }
// The rail's Move: moves the new selection, or the one orange area; otherwise says what to do first.
$("moveToolBtn").onclick = () => {
  if (!P() || S.job) return;
  if (S.move) { endMove(); return; }
  if (S.pending && S.pending.candidates.length) { $("moveItBtn").click(); return; }
  const targets = P().objects.filter((o) => o.kind === "target");
  if (targets.length === 1) { startMove(targets[0], targets[0].name); return; }
  toast(targets.length ? "Move one object at a time: keep one orange area, then press Move." : "Click the object to move first, then press Move.");
};
$("themeBtn").onclick = toggleTheme;
function drawMove(x, y, w, h) {
  const m = S.move, v = S.view;
  if (!m.cut) return;
  const ox = m.dx * v.s, oy = m.dy * v.s;
  ctx.save();
  if (m.box && (m.dx || m.dy)) {  // where it was: dimmed, to be filled in
    const b = m.box;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "rgba(255,138,31,.9)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(x + b[0] * v.s, y + b[1] * v.s, (b[2] - b[0]) * v.s, (b[3] - b[1]) * v.s);
    ctx.setLineDash([]);
  }
  ctx.shadowColor = "rgba(0,0,0,.55)"; ctx.shadowBlur = 14;
  ctx.drawImage(m.cut, x + ox, y + oy, w, h);
  ctx.restore();
}
function applyMove() {
  const m = S.move;
  if (!m || (!m.dx && !m.dy)) return;
  const what = $("moveName").value.trim();
  if (!what) { toast("Say what it is (for example: cup). The model needs the word to fill in its old place."); $("moveName").focus(); return; }
  m.name = what;
  $("instruction").value = `move the ${m.name}`;
  autoGrow();
  const move = { dx: Math.round(m.dx), dy: Math.round(m.dy), name: m.name, shadow: $("moveShadow").checked };
  endMove();
  startRender({ move });
}
$("moveCancelBtn").onclick = endMove;
$("moveApplyBtn").onclick = applyMove;
canvas.addEventListener("pointerdown", (e) => {
  if (!S.move || !P() || pinch || e.button === 1 || e.button === 2 || S.spaceDown || S.tool === "hand") return;
  e.stopImmediatePropagation();
  canvas.setPointerCapture(e.pointerId);
  S.move.drag = { start: toImage(e), dx0: S.move.dx, dy0: S.move.dy };
}, true);
canvas.addEventListener("pointermove", (e) => {
  if (!S.move || !S.move.drag) return;
  if (pinch) { S.move.drag = null; return; }
  e.stopImmediatePropagation();
  const p = toImage(e), d = S.move.drag;
  S.move.dx = Math.round(d.dx0 + p.x - d.start.x); S.move.dy = Math.round(d.dy0 + p.y - d.start.y);
  $("moveApplyBtn").disabled = !S.move.dx && !S.move.dy;
  redraw();
}, true);
["pointerup", "pointercancel"].forEach((t) => canvas.addEventListener(t, (e) => {
  if (!S.move || !S.move.drag) return;
  e.stopImmediatePropagation();
  S.move.drag = null; redraw();
}, true));
document.addEventListener("keydown", (e) => {
  if (!S.move) return;
  const tag = document.activeElement?.tagName || "";
  if (/INPUT|TEXTAREA|SELECT/.test(tag)) { if (e.key === "Enter" && document.activeElement.id === "moveName") { e.preventDefault(); e.stopImmediatePropagation(); applyMove(); } return; }
  const step = e.shiftKey ? 10 : 1;
  const nudge = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
  if (nudge) { e.preventDefault(); e.stopImmediatePropagation(); S.move.dx += nudge[0]; S.move.dy += nudge[1]; $("moveApplyBtn").disabled = !S.move.dx && !S.move.dy; redraw(); }
  else if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); endMove(); }
  else if (e.key === "Enter" && !/BUTTON/.test(tag)) { e.preventDefault(); e.stopImmediatePropagation(); applyMove(); }
}, true);

$("bgItBtn").onclick = async () => {
  const pend = S.pending;
  if (!pend || !pend.candidates.length) return;
  const name = pendingName();
  const c = pend.candidates[pend.picked];
  await dropUsedTargets();
  // The clicked object is kept exactly and everything else is the edit area, in one step.
  const p = await guard(() => api("POST", `/api/projects/${pid()}/objects`, { items: [
    { object_id: pend.object_id, candidate_id: c.candidate_id, kind: "protected", name: name || "kept area" },
    { candidate_id: c.candidate_id, kind: "target", name: "background", invert: true }] }));
  if (!p) return;
  S.pending = null; S.project = p; setSelFor(null);
  renderAll(); redraw();
  fillSentence(`replace the background with ..., keep the ${name || "subject"} unchanged`);
  $("dockNote").innerHTML = "Describe the new background, then press <b>Enter</b>. What you kept stays exact.";
};
$("objClear").onclick = () => guard(async () => { S.project = await api("DELETE", `/api/projects/${pid()}/objects`); S.selectedObj = null; renderAll(); redraw(); });
document.querySelectorAll("[data-grow]").forEach((b) => b.onclick = () => {
  if (!S.selectedObj) return;
  guard(async () => { S.project = await api("POST", `/api/projects/${pid()}/objects/${S.selectedObj}/grow`, { px: +b.dataset.grow }); renderAll(); redraw(); });
});
$("findForm").onsubmit = (e) => { e.preventDefault(); findByName($("findText").value); };
$("seedBtn").onclick = () => {
  const head = headRev();
  if (!head) return;
  $("seed").value = (+$("seed").value || 0) + 1;
  startRender({ retry_of: head.revision_id });
};
$("moreBtn").onclick = () => {
  const head = headRev();
  if (!head) return;
  const seeds = [0, 1, 2].map(() => Math.floor(Math.random() * 100000));
  startRender({ retry_of: head.revision_id, seeds, pickFrom: head.revision_id });
};
$("growBtn").onclick = () => { const head = headRev(); if (head) startRender({ grow_from: head.revision_id }); };
$("diceBtn").onclick = () => { $("seed").value = Math.floor(Math.random() * 100000); };
$("cancelBtn").onclick = () => { if (S.job) guard(() => api("POST", `/api/jobs/${S.job.id}/cancel`)); $("progressText").textContent = "Cancelling after this step"; };
$("undoBtn").onclick = () => guard(async () => { if (S.move) endMove(); const p = await api("POST", `/api/projects/${pid()}/undo`); S.project = p; S.compare = false; S.showMap = false; renderAll(); redraw(); });
$("redoBtn").onclick = () => guard(async () => { if (S.move) endMove(); const p = await api("POST", `/api/projects/${pid()}/redo`); S.project = p; S.compare = false; S.showMap = false; renderAll(); redraw(); });

$("exportBtn").onclick = (e) => { e.stopPropagation(); $("exportMenu").hidden = !$("exportMenu").hidden; };
document.addEventListener("click", (e) => { if (!e.target.closest(".menu-wrap")) $("exportMenu").hidden = true; });
$("exportMenu").querySelectorAll("button").forEach((b) => b.onclick = async () => {
  $("exportMenu").hidden = true;
  const fmt = b.dataset.fmt;
  const r = await guard(() => api("GET", `/api/projects/${pid()}/export?fmt=${fmt === "copy" ? "png" : fmt}`, undefined, { raw: true }));
  if (!r) return;
  const blob = await r.blob();
  if (fmt === "copy") {
    try {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      toast("Copied. Paste it anywhere with Ctrl+V.", "ok");
    } catch (_) {
      toast("This browser does not allow copying images here (it needs the page at 127.0.0.1). Use Export, PNG instead.", "error");
    }
    return;
  }
  const name = (r.headers.get("content-disposition") || "").match(/filename="?([^";]+)"?/)?.[1] || `photo.${fmt === "jpeg" ? "jpg" : "png"}`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  toast(fmt === "png" ? `Saved ${name}. Checked on the file before export.` : `Saved ${name}. JPEG is lossy, so kept areas are close but not exact.`, "ok");
});

var ta = $("instruction");
function autoGrow() { ta.style.height = "auto"; ta.style.height = Math.min(120, ta.scrollHeight) + "px"; renderIdeas(); }
// Ready-made sentences for people who do not know what to type; "..." is selected so typing replaces it.
var IDEAS = [
  ["Remove something", "remove the ..."],
  ["Change a colour", "make the ... red"],
  ["New background", "replace the background with ..., keep the ... unchanged"],
  ["Change the material", "make the ... leather"],
  ["Winter", "make it look like a snowy winter day"],
  ["Golden hour", "make it look like golden hour"],
  ["Blur the background", "make the background blurry, keep the ... unchanged"],
];
function fillSentence(text) {
  ta.value = text; autoGrow(); ta.focus();
  const i = text.indexOf("...");
  if (i >= 0) ta.setSelectionRange(i, i + 3); else ta.setSelectionRange(text.length, text.length);
  ta.dispatchEvent(new Event("input"));
}
function renderIdeas() {
  if (typeof ta === "undefined" || !ta || !IDEAS) return;  // called by renderAll before this part of the script has run
  const show = !!P() && !ta.value.trim() && !S.job && !S.proposal;
  $("ideas").hidden = !show;
  if (show && !$("ideas").childElementCount) {
    for (const [label, text] of IDEAS) {
      const b = document.createElement("button");
      b.className = "idea"; b.textContent = label; b.title = text;
      b.onclick = () => fillSentence(text);
      $("ideas").appendChild(b);
    }
  }
}
ta.addEventListener("input", () => {
  autoGrow();
  clearTimeout(ta._t);
  ta._t = setTimeout(() => P() && api("PUT", `/api/projects/${pid()}/prefs`, { instruction: ta.value }).catch(() => {}), 600);
});
ta.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); primaryAction(); }
});

// Typing while a photo opens goes into the sentence once it is open, instead of being lost.
document.addEventListener("keydown", (e) => {
  if (!S.opening || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key.length === 1) S.typeAhead = (S.typeAhead || "") + e.key;
  else if (e.key === "Backspace") S.typeAhead = (S.typeAhead || "").slice(0, -1);
  else if (e.key === "Enter") S.enterAhead = true;
  else return;
  e.preventDefault(); e.stopImmediatePropagation();
}, true);
function applyTypeAhead() {
  const t = S.typeAhead, enter = S.enterAhead;
  S.typeAhead = ""; S.enterAhead = false;
  if (!t) return;
  $("instruction").value = t; autoGrow(); $("instruction").focus();
  if (enter) primaryAction();
}
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); primaryAction(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "o") { e.preventDefault(); $("fileInput").click(); return; }
  if (typing) { if (e.key === "Escape") document.activeElement.blur(); return; }
  if (!$("helpModal").hidden && e.key === "Escape") { $("helpModal").hidden = true; return; }
  if (!P()) return;
  const k = e.key;
  // A button reached with Tab works with Enter and Space as usual (one clicked with the mouse keeps
  // focus too, so only keyboard focus counts; otherwise Space would press Render again).
  const ae = document.activeElement;
  if ((k === "Enter" || k === " ") && ae && /BUTTON|A|SUMMARY/.test(ae.tagName) && ae.matches(":focus-visible")) return;
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.preventDefault(); (e.shiftKey ? $("redoBtn") : $("undoBtn")).click(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "y") { e.preventDefault(); $("redoBtn").click(); return; }
  if (e.ctrlKey || e.metaKey) return;
  if (k === " ") { e.preventDefault(); if (!S.spaceDown) { S.spaceDown = true; updateButtons(); } return; }
  if (k === "\\") { if (!S.holdBefore && headRev()) { S.holdBefore = true; redraw(); } return; }
  const map = { s: () => setTool("select"), b: () => setTool("brush"), e: () => setTool("eraser"), h: () => setTool("hand"), f: fit,
    1: () => $("zoom100Btn").click(), c: () => $("compareBtn").click(), m: () => $("mapBtn").click(), v: () => $("overlayBtn").click(), "?": () => $("helpBtn").click(),
    "[": () => { $("brushSize").value = Math.max(4, +$("brushSize").value - 6); $("brushSize").oninput(); },
    "]": () => { $("brushSize").value = Math.min(200, +$("brushSize").value + 6); $("brushSize").oninput(); },
    "+": () => { const { w, h } = stageSize(); zoomAt(1.25, w / 2, h / 2); }, "=": () => { const { w, h } = stageSize(); zoomAt(1.25, w / 2, h / 2); },
    "-": () => { const { w, h } = stageSize(); zoomAt(0.8, w / 2, h / 2); },
    Escape: () => {
      if (S.pending) { S.pending = null; renderPanels(); redraw(); }
      else if (S.proposal) { S.proposal = null; if (hasTargets()) setSelFor(null); renderAll(); redraw(); }
    },
    Enter: () => { if (S.pending && S.pending.candidates.length) keepPending(e.shiftKey ? "protected" : "target"); },
  };
  const fn = map[k] || map[k.toLowerCase?.()];
  if (fn) { e.preventDefault(); fn(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === " ") { S.spaceDown = false; updateButtons(); }
  if (e.key === "\\") { S.holdBefore = false; redraw(); }
});

window.addEventListener("hashchange", () => { const m = location.hash.match(/p=([A-Za-z0-9_.-]+)/); if (m && m[1] !== pid()) openProject(m[1]); });

// ---- adjust, rotate and crop: no model (lis/adjust.py) -------------------------------------------
// The sliders preview on the page and Apply sends one request; the server computes the same
// formula at full resolution and checks the saved file. Rotate, flip and crop make a new project.

var ADJ_KEYS = ["brightness", "contrast", "saturation", "warmth"];
var ADJ_PREVIEW_MAX = 1600;  // long side of the live preview; Apply always works at full resolution
var ADJ = { pid: null, values: { brightness: 0, contrast: 0, saturation: 0, warmth: 0 }, base: null, building: null, canvas: null, canvasKey: null, queued: false, busy: false };

function adjustActive() { return !!ADJ && ADJ_KEYS.some((k) => ADJ.values[k]); }
function adjustAreas() {
  const objs = P() ? P().objects : [];
  return { targets: objs.filter((o) => o.kind === "target"), kept: objs.filter((o) => o.kind === "protected") };
}
function adjustKey() {
  if (!P()) return "";
  return P().image_url + "|" + P().objects.map((o) => o.kind + ":" + o.url).join(",");
}
function adjustPreview() {
  return P() && adjustActive() && ADJ.canvas && ADJ.canvasKey === adjustKey() ? ADJ.canvas : null;
}

// The same steps as lis/adjust.py (see its docstring), on RGBA bytes: kept pixels and pixels outside
// the edit area are copied. The preview leaves out the few-pixel soft edge, which is under one
// preview pixel on most photos.
function adjustPixels(src, dst, n, v, target, kept) {
  const lut = new Float64Array(256);
  const gamma = Math.pow(2, -v.brightness / 100), kc = 1 + v.contrast / 100;
  for (let i = 0; i < 256; i++) {
    let x = i / 255;
    if (v.brightness) x = Math.pow(x, gamma);
    if (v.contrast) x = (x - 0.5) * kc + 0.5;
    lut[i] = x;
  }
  const ks = 1 + v.saturation / 100, wr = 1 + 0.2 * v.warmth / 100, wb = 1 - 0.2 * v.warmth / 100;
  const q = (x) => { const y = Math.floor(x * 255 + 0.5); return y < 0 ? 0 : y > 255 ? 255 : y; };
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    dst[j + 3] = 255;
    if ((kept && kept[i]) || (target && !target[i])) { dst[j] = src[j]; dst[j + 1] = src[j + 1]; dst[j + 2] = src[j + 2]; continue; }
    let r = lut[src[j]], g = lut[src[j + 1]], b = lut[src[j + 2]];
    if (v.saturation) { const L = 0.299 * r + 0.587 * g + 0.114 * b; r = L + (r - L) * ks; g = L + (g - L) * ks; b = L + (b - L) * ks; }
    if (v.warmth) { r *= wr; b *= wb; }
    dst[j] = q(r); dst[j + 1] = q(g); dst[j + 2] = q(b);
  }
}

async function adjustBase() {
  const key = adjustKey();
  if (ADJ.base && ADJ.base.key === key) return ADJ.base;
  if (ADJ.building && ADJ.building.key === key) return ADJ.building.promise;
  const p = P();
  const promise = (async () => {
    const im = await loadImage(p.image_url);
    const k = Math.min(1, ADJ_PREVIEW_MAX / Math.max(p.width, p.height));
    const w = Math.max(1, Math.round(p.width * k)), h = Math.max(1, Math.round(p.height * k));
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(im, 0, 0, w, h);
    const data = g.getImageData(0, 0, w, h).data;
    const union = async (objs) => {
      if (!objs.length) return null;
      const m = new Uint8Array(w * h);
      for (const o of objs) {
        const mi = await loadImage(o.url);
        g.clearRect(0, 0, w, h); g.drawImage(mi, 0, 0, w, h);
        const d = g.getImageData(0, 0, w, h).data;
        for (let i = 0; i < w * h; i++) if (d[i * 4] >= 128) m[i] = 1;
      }
      return m;
    };
    const { targets, kept } = adjustAreas();
    return { key, w, h, data, target: await union(targets), kept: await union(kept) };
  })();
  ADJ.building = { key, promise };
  try {
    const b = await promise;
    if (adjustKey() === key) ADJ.base = b;
    return b;
  } finally { if (ADJ.building && ADJ.building.key === key) ADJ.building = null; }
}

function queueAdjustPreview() {
  if (ADJ.queued) return;
  ADJ.queued = true;
  requestAnimationFrame(async () => {
    ADJ.queued = false;
    if (!P() || !adjustActive()) { ADJ.canvas = null; redraw(); return; }
    let b;
    try { b = await adjustBase(); } catch (e) { console.error(e); return; }
    if (b.key !== adjustKey()) { queueAdjustPreview(); return; }
    if (!ADJ.canvas || ADJ.canvas.width !== b.w || ADJ.canvas.height !== b.h) {
      ADJ.canvas = document.createElement("canvas"); ADJ.canvas.width = b.w; ADJ.canvas.height = b.h;
    }
    const g = ADJ.canvas.getContext("2d");
    const out = g.createImageData(b.w, b.h);
    adjustPixels(b.data, out.data, b.w * b.h, ADJ.values, b.target, b.kept);
    g.putImageData(out, 0, 0);
    ADJ.canvasKey = b.key;
    redraw();
  });
}

function readAdjustSliders() {
  for (const k of ADJ_KEYS) {
    const el = document.querySelector(`[data-adj="${k}"]`);
    ADJ.values[k] = el ? Math.round(+el.value) || 0 : 0;
    const out = el && el.parentElement.querySelector("output");
    if (out) out.textContent = ADJ.values[k] > 0 ? "+" + ADJ.values[k] : String(ADJ.values[k]);
  }
}
function resetAdjust(redrawNow = true) {
  for (const k of ADJ_KEYS) { const el = document.querySelector(`[data-adj="${k}"]`); if (el) el.value = 0; }
  readAdjustSliders();
  ADJ.canvas = null;
  if (redrawNow) { updateAdjustButtons(); redraw(); }
}

function renderAdjust() {
  if (!P() || !$("adjustPanel")) return;
  if (ADJ.pid !== pid()) { ADJ.pid = pid(); ADJ.base = null; resetAdjust(false); if (S.crop) cancelCrop(); }
  const { targets, kept } = adjustAreas();
  let t = targets.length ? `Applies to the edit area: ${targets.map((o) => o.name).join(", ")}.` : "Applies to the whole photo.";
  if (kept.length) t += ` Kept exactly: ${kept.map((o) => o.name).join(", ")}.`;
  $("adjustArea").textContent = t;
  if (adjustActive()) queueAdjustPreview();
  updateAdjustButtons();
}
function updateAdjustButtons() {
  if (!$("adjustPanel") || !ADJ) return;
  const off = !P() || !!S.job || ADJ.busy || S.status.state !== "ready";
  $("adjApplyBtn").disabled = off || !adjustActive();
  $("adjResetBtn").disabled = ADJ.busy || !adjustActive();
  const geoOff = off || adjustActive();  // unapplied sliders would be lost in the new project
  for (const id of ["rotLeftBtn", "rotRightBtn", "flipBtn", "cropBtn"]) $(id).disabled = geoOff;
  $("cropBtn").classList.toggle("on", !!S.crop);
  $("geoNote").textContent = P() && adjustActive() ? "Apply or reset the adjustment first: rotating and cropping use the image on screen." :
    "Rotating, flipping and cropping make a new project from the image on screen. This one stays as it is.";
  $("cropApplyBtn").disabled = off || !(S.crop && S.crop.rect);
}

// The result card for an adjustment: no model ran, so no edit check is claimed.
function adjustResultCopy(head) {
  const adj = head.kind === "adjust";
  $("moreBtn").hidden = adj; $("seedBtn").hidden = adj;
  if (!adj) return;
  const c = head.checks;
  const kept = (c.rows.find((r) => r.key === "protected") || {}).status === "pass";
  const title = c.overall === "fail" ? "Check failed" : "Adjusted: file checks passed";
  const sub = c.overall === "fail" ? "This version did not pass the file check." :
    "No model was used: a fixed formula, computed on this computer." + (kept ? " Kept areas are unchanged on the saved file." : "");
  $("verdict").className = "verdict " + c.overall;
  $("verdict").innerHTML = `${c.overall === "pass" ? ICON.shield : ICON[c.overall]}<div>${esc(title)}<small>${esc(sub)}</small></div>`;
  $("resultMeta").textContent = `no model · ${head.seconds} s`;
}

async function applyAdjust() {
  if (!P() || S.job || ADJ.busy || !adjustActive()) return;
  ADJ.busy = true; updateAdjustButtons(); setBusy(true, "Applying the adjustment");
  const r = await guard(() => api("POST", `/api/projects/${pid()}/adjust`, { ...ADJ.values, parent: P().head || "source" }));
  ADJ.busy = false; setBusy(false);
  if (!r) { updateAdjustButtons(); return; }
  S.project = r.project;
  resetAdjust(false);
  S.variants = null; S.expansionHint = false; S.pending = null; S.proposal = null;
  S.showMap = false; S.showOverlays = false; S.compare = true; S.compareX = 0.5;
  pic(r.revision.image_url); pic(r.revision.parent_url);
  const ov = r.revision.checks.overall;
  toast(ov === "fail" ? "The check failed." : "Adjusted. Checked on the saved file; no model was used.", ov === "fail" ? "error" : "ok");
  $("dockNote").textContent = "Drag the slider to compare. Adjust again, or describe an edit to keep going.";
  renderAll(); redraw();
}

async function geometry(op, box) {
  if (!P() || S.job || ADJ.busy) return;
  const what = { rotate_left: "the rotated copy", rotate_right: "the rotated copy", flip_horizontal: "the flipped copy", crop: "the cropped copy" }[op];
  ADJ.busy = true; updateAdjustButtons(); showOpening(what);
  const r = await guard(() => api("POST", `/api/projects/${pid()}/transform`, box ? { op, box } : { op }));
  ADJ.busy = false; hideOpening();
  if (!r) { updateAdjustButtons(); return; }
  if (S.crop) cancelCrop();
  setProject(r, true);
  toast(`${r.note}. This is a new project; the previous one is unchanged under Recent.`, "ok");
  loadRecent();
}

// ---- crop mode: drag a rectangle on the photo, Crop or Enter applies, Escape cancels ----

function startCrop() {
  if (S.move) endMove();
  if (!P() || S.job || adjustActive()) return;
  if (S.tool === "brush" || S.tool === "eraser") setTool("select");  // its bar is in the same corner
  S.crop = { pid: pid(), rect: null, drag: null };
  S.pending = null; S.compare = false; S.showMap = false; S.showOverlays = false;
  $("cropBar").hidden = false; $("cropSize").textContent = "Drag on the photo";
  bringStageIntoView();
  updateHint(); updateAdjustButtons(); updateButtons(); redraw();
}
function cancelCrop() {
  S.crop = null;
  $("cropBar").hidden = true;
  S.showOverlays = true;
  updateHint(); updateAdjustButtons(); updateButtons(); redraw();
}
function cropAspect() {
  const v = $("cropAspect").value;
  if (v === "free") return null;
  const [a, b] = v.split(":").map(Number);
  return a / b;
}
// A rectangle from the drag start a to the pointer b, in image pixels, kept inside the photo and,
// with a shape chosen, at that shape (turned upright when the drag is taller than wide).
function cropRect(a, b) {
  const W = P().width, H = P().height;
  const ax = Math.min(W, Math.max(0, a.x)), ay = Math.min(H, Math.max(0, a.y));
  const dx = b.x - ax, dy = b.y - ay;
  const sx = dx >= 0 ? 1 : -1, sy = dy >= 0 ? 1 : -1;
  let w = Math.abs(dx), h = Math.abs(dy);
  const roomX = sx > 0 ? W - ax : ax, roomY = sy > 0 ? H - ay : ay;
  let ar = cropAspect();
  if (ar) {
    if (h > w) ar = 1 / ar;
    if (w / Math.max(h, 1e-6) > ar) w = h * ar; else h = w / ar;
    const k = Math.min(1, roomX / Math.max(w, 1e-6), roomY / Math.max(h, 1e-6));
    w *= k; h *= k;
  } else { w = Math.min(w, roomX); h = Math.min(h, roomY); }
  const x0 = sx > 0 ? ax : ax - w, y0 = sy > 0 ? ay : ay - h;
  const r = [Math.max(0, Math.round(x0)), Math.max(0, Math.round(y0)), Math.round(w), Math.round(h)];
  r[2] = Math.min(r[2], W - r[0]); r[3] = Math.min(r[3], H - r[1]);
  return r;
}
function setCropRect(r) {
  S.crop.rect = r && r[2] >= 16 && r[3] >= 16 ? r : null;
  $("cropSize").textContent = S.crop.rect ? `${S.crop.rect[2]}x${S.crop.rect[3]}` : "Drag on the photo";
  updateAdjustButtons(); redraw();
}
function drawCrop(x, y, w, h) {
  const v = S.view, r = S.crop.rect;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,.55)";
  ctx.beginPath(); ctx.rect(x, y, w, h);
  if (r) ctx.rect(v.tx + (r[0] + r[2]) * v.s, v.ty + r[1] * v.s, -r[2] * v.s, r[3] * v.s);  // the opposite winding cuts the hole
  ctx.fill("nonzero");
  if (r) {
    const rx = v.tx + r[0] * v.s, ry = v.ty + r[1] * v.s, rw = r[2] * v.s, rh = r[3] * v.s;
    ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (const f of [1 / 3, 2 / 3]) { ctx.moveTo(rx + rw * f, ry); ctx.lineTo(rx + rw * f, ry + rh); ctx.moveTo(rx, ry + rh * f); ctx.lineTo(rx + rw, ry + rh * f); }
    ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, rw, rh);
  }
  ctx.restore();
}
function insideCrop(p) { const r = S.crop && S.crop.rect; return !!r && p.x >= r[0] && p.y >= r[1] && p.x <= r[0] + r[2] && p.y <= r[1] + r[3]; }
// Capture listeners run before the editor's own canvas handlers, so crop mode takes the drag first.
canvas.addEventListener("pointerdown", (e) => {
  if (!S.crop || !P() || pinch || e.button === 1 || e.button === 2 || S.spaceDown || S.tool === "hand") return;
  e.stopImmediatePropagation();
  canvas.setPointerCapture(e.pointerId);
  const p = toImage(e);
  S.crop.drag = insideCrop(p) ? { mode: "move", start: p, rect0: S.crop.rect.slice() } : { mode: "draw", start: p };
}, true);
canvas.addEventListener("pointermove", (e) => {
  if (!S.crop || !S.crop.drag) return;
  if (pinch) { S.crop.drag = null; return; }
  e.stopImmediatePropagation();
  const p = toImage(e), d = S.crop.drag;
  S.pointer = p;
  if (d.mode === "draw") setCropRect(cropRect(d.start, p));
  else {
    const W = P().width, H = P().height, r0 = d.rect0;
    const nx = Math.round(Math.min(W - r0[2], Math.max(0, r0[0] + p.x - d.start.x)));
    const ny = Math.round(Math.min(H - r0[3], Math.max(0, r0[1] + p.y - d.start.y)));
    setCropRect([nx, ny, r0[2], r0[3]]);
  }
}, true);
["pointerup", "pointercancel"].forEach((t) => canvas.addEventListener(t, (e) => {
  if (!S.crop || !S.crop.drag) return;
  e.stopImmediatePropagation();
  S.crop.drag = null;
  redraw();
}, true));
document.addEventListener("keydown", (e) => {
  if (!S.crop) return;
  const tag = document.activeElement?.tagName || "";
  if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); cancelCrop(); }
  else if (e.key === "Enter" && !/INPUT|TEXTAREA|BUTTON|SELECT/.test(tag)) { e.preventDefault(); e.stopImmediatePropagation(); if (S.crop.rect) geometry("crop", S.crop.rect); }
}, true);

if ($("adjustPanel")) {
  document.querySelectorAll("[data-adj]").forEach((el) => {
    el.addEventListener("input", () => {
      readAdjustSliders();
      if (adjustActive()) { S.compare = false; S.showMap = false; S.showOverlays = false; if (S.crop) cancelCrop(); }
      else { ADJ.canvas = null; S.showOverlays = true; }
      queueAdjustPreview(); updateAdjustButtons(); updateButtons(); redraw();
    });
    el.addEventListener("dblclick", () => { el.value = 0; el.dispatchEvent(new Event("input")); });  // a double click puts one slider back to 0
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); applyAdjust(); } });
  });
  $("adjResetBtn").onclick = () => { resetAdjust(); S.showOverlays = true; updateButtons(); redraw(); };
  $("adjApplyBtn").onclick = applyAdjust;
  $("rotLeftBtn").onclick = () => geometry("rotate_left");
  $("rotRightBtn").onclick = () => geometry("rotate_right");
  $("flipBtn").onclick = () => geometry("flip_horizontal");
  $("cropBtn").onclick = () => (S.crop ? cancelCrop() : startCrop());
  $("cropCancelBtn").onclick = cancelCrop;
  $("cropApplyBtn").onclick = () => { if (S.crop && S.crop.rect) geometry("crop", S.crop.rect); };
  $("cropAspect").onchange = () => {
    $("cropAspect").blur();  // give Enter and Escape back to crop mode
    const r = S.crop && S.crop.rect;
    if (r) setCropRect(cropRect({ x: r[0], y: r[1] }, { x: r[0] + r[2], y: r[1] + r[3] }));
  };
}

// ---- start ----------------------------------------------------------------------------------------

resize();
setTool("select");
renderCrumb();
requestAnimationFrame(loop);
pollStatus();
