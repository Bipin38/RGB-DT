"use strict";
// RGB-D-T Annotator — client. Drawing is 100% local (no server on mouse-move).
// The server is hit only on: scene load, ROI compute (on shape finish), and save.

const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};

const COLORS = ["#00FF00","#FF4444","#00FFFF","#FFD700","#FF00FF","#4488FF","#FFA500","#FFFFFF","#FF69B4","#7FFF00"];

const S = {
  config: null,
  sceneName: null,
  scene: null,          // server scene payload
  anns: [],             // annotations for current scene (authoritative local copy)
  meta: {},             // scene metadata
  qa: [],               // qa_pairs
  stagedRels: [],       // relations staged for the next added object
  mode: "rect",
  selected: -1,
  editing: -1,          // index of the saved object being edited (-1 = adding new)
  metaSaved: false,     // has scene metadata been saved for the current scene?
  // in-progress drawing
  bgImg: null,
  dragStart: null,
  dragRect: null,       // {x0,y0,x1,y1} in image coords
  polyPts: [],          // [{x,y}] image coords
  pending: null,        // finished shape awaiting "Add object": {shape,bbox,polygon,stats}
};

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ── helpers ──────────────────────────────────────────────────────────────────
function opt(sel, values, current) {
  sel.innerHTML = "";
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  }
}
function setSaveState(cls, txt) {
  const el = $("saveState"); el.className = "save-state " + cls; el.textContent = txt;
}
// Kitchen ID is a fixed dropdown: kitchen_001…kitchen_005 (plus any legacy value).
function optKitchen(current) {
  const sel = $("mKitchen"); sel.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = ""; blank.textContent = "— select kitchen —"; sel.appendChild(blank);
  for (let i = 1; i <= 5; i++) {
    const v = `kitchen_00${i}`;
    const o = document.createElement("option"); o.value = o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  }
  if (current && ![...sel.options].some(o => o.value === current)) {
    const o = document.createElement("option"); o.value = o.textContent = current; o.selected = true;
    sel.appendChild(o);   // keep any pre-existing out-of-range value instead of dropping it
  }
}
function setMetaSaved(saved) {
  S.metaSaved = saved;
  const el = $("metaStatus");
  if (!el) return;
  el.className = "meta-status " + (saved ? "ok" : "warn");
  el.textContent = saved ? "✓ Scene metadata saved" : "⚠ Unsaved — save scene metadata before completing";
  $("completeScene").classList.toggle("blocked", !saved);
}
function flashMeta() {
  const d = $("saveMeta").closest("details"); if (d) d.open = true;
  $("saveMeta").scrollIntoView({ behavior: "smooth", block: "center" });
  const el = $("metaStatus");
  el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1600);
}
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("rgbdt_theme", theme);
  const btn = $("themeToggle");
  if (btn) { btn.textContent = theme === "light" ? "☀️" : "🌙"; btn.title = `Theme: ${theme} — click for ${theme === "light" ? "dark" : "light"} (T)`; }
}
function toggleTheme() {
  applyTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
}
function evtToImg(e) {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width;
  const y = (e.clientY - r.top) * canvas.height / r.height;
  return {
    x: Math.max(0, Math.min(canvas.width, Math.round(x))),
    y: Math.max(0, Math.min(canvas.height, Math.round(y))),
  };
}
function annBbox(a) {
  if (a.bbox_original) return a.bbox_original;
  const xs = a.polygon_original.map(p => p.x), ys = a.polygon_original.map(p => p.y);
  return { x_min: Math.min(...xs), y_min: Math.min(...ys), x_max: Math.max(...xs), y_max: Math.max(...ys) };
}

// ── render loop ──────────────────────────────────────────────────────────────
function draw() {
  if (!S.bgImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(S.bgImg, 0, 0, canvas.width, canvas.height);

  // no-depth boundary shading
  const bx = S.scene && S.scene.no_depth_boundary_x;
  if (bx) {
    ctx.fillStyle = "rgba(255,100,0,0.22)";
    ctx.fillRect(0, 0, bx, canvas.height);
    ctx.strokeStyle = "rgba(255,100,0,0.9)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(bx, 0); ctx.lineTo(bx, canvas.height); ctx.stroke();
  }

  // saved annotations
  S.anns.forEach((a, i) => {
    const color = COLORS[i % COLORS.length];
    ctx.lineWidth = i === S.selected ? 4 : 2;
    ctx.strokeStyle = color;
    if (a.shape === "polygon" && a.polygon_original) {
      ctx.beginPath();
      a.polygon_original.forEach((p, k) => k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.closePath(); ctx.stroke();
    } else {
      const b = annBbox(a);
      ctx.strokeRect(b.x_min, b.y_min, b.x_max - b.x_min, b.y_max - b.y_min);
    }
  });

  // pending finished shape (awaiting Add object)
  ctx.setLineDash([6, 4]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  if (S.pending) {
    if (S.pending.shape === "polygon") {
      ctx.beginPath();
      S.pending.polygon.forEach((p, k) => k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.closePath(); ctx.stroke();
    } else {
      const b = S.pending.bbox;
      ctx.strokeRect(b.x_min, b.y_min, b.x_max - b.x_min, b.y_max - b.y_min);
    }
  }
  // active rectangle drag
  if (S.dragRect) {
    const r = S.dragRect;
    ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
  }
  ctx.setLineDash([]);
  // active polygon points
  if (S.polyPts.length) {
    ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath();
    S.polyPts.forEach((p, k) => k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.stroke();
    S.polyPts.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fill(); });
    // Highlight the first vertex once we have enough points to close — click it to finish.
    if (S.polyPts.length >= 3) {
      const f = S.polyPts[0];
      ctx.strokeStyle = "#00FF88"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(f.x, f.y, 9, 0, 7); ctx.stroke();
    }
  }
}

// ── canvas interaction ───────────────────────────────────────────────────────
canvas.addEventListener("mousedown", (e) => {
  if (S.mode !== "rect") return;
  const p = evtToImg(e);
  S.dragStart = p; S.dragRect = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
});
canvas.addEventListener("mousemove", (e) => {
  if (S.mode !== "rect" || !S.dragStart) return;
  const p = evtToImg(e);
  S.dragRect = { x0: S.dragStart.x, y0: S.dragStart.y, x1: p.x, y1: p.y };
  draw();
});
canvas.addEventListener("mouseup", (e) => {
  if (S.mode !== "rect" || !S.dragStart) return;
  const p = evtToImg(e);
  const x0 = Math.min(S.dragStart.x, p.x), x1 = Math.max(S.dragStart.x, p.x);
  const y0 = Math.min(S.dragStart.y, p.y), y1 = Math.max(S.dragStart.y, p.y);
  S.dragStart = null; S.dragRect = null;
  if (x1 - x0 < 8 || y1 - y0 < 8) { draw(); return; }   // MIN_BBOX
  finishShape({ shape: "rectangle", bbox: { x_min: x0, y_min: y0, x_max: x1, y_max: y1 } });
});
const POLY_SNAP = 12;   // px: click within this of the first vertex closes the polygon
canvas.addEventListener("click", (e) => {
  if (S.mode !== "polygon") return;
  const p = evtToImg(e);
  // Click on/near the first vertex to close the polygon.
  if (S.polyPts.length >= 3 && dist(p, S.polyPts[0]) <= POLY_SNAP) { finishPolygon(); return; }
  // Ignore a click that lands on the previous point — this also swallows the two
  // extra clicks a double-click fires, so double-click no longer adds stray vertices.
  const last = S.polyPts[S.polyPts.length - 1];
  if (last && dist(p, last) < 4) return;
  S.polyPts.push(p);
  draw(); syncToolbar();
});
canvas.addEventListener("dblclick", (e) => {
  if (S.mode === "polygon") { e.preventDefault(); finishPolygon(); }
});
// Right-click removes the last placed polygon vertex (undo).
canvas.addEventListener("contextmenu", (e) => {
  if (S.mode === "polygon" && S.polyPts.length) { e.preventDefault(); undoPolyPoint(); }
});

function undoPolyPoint() {
  if (S.mode === "polygon" && S.polyPts.length) {
    S.polyPts.pop();
    draw(); syncToolbar();
    setHint(S.polyPts.length ? "Removed last point" : "All points cleared — click to start again");
  }
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function finishPolygon() {
  // Drop consecutive near-duplicate points (belt-and-suspenders vs double-click).
  const pts = [];
  for (const p of S.polyPts) {
    const last = pts[pts.length - 1];
    if (!last || dist(p, last) >= 4) pts.push(p);
  }
  if (pts.length < 3) { setHint("Polygon needs ≥3 points — keep clicking, then click the first point (or double-click / Enter)."); return; }
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  S.polyPts = [];
  finishShape({
    shape: "polygon", polygon: pts,
    bbox: { x_min: Math.min(...xs), y_min: Math.min(...ys), x_max: Math.max(...xs), y_max: Math.max(...ys) },
  });
}

async function finishShape(shape) {
  if (S.editing >= 0) exitEdit();   // drawing a new shape means we're adding, not editing
  S.pending = shape;
  draw(); syncToolbar();
  $("pendingStats").textContent = "computing temp/depth…";
  try {
    const stats = await api(`/api/scene/${S.sceneName}/roi`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        shape: shape.shape, bbox: shape.bbox,
        polygon: shape.polygon || null,
        ambient_temp_c: null,
      }),
    });
    S.pending.stats = stats;
    const t = (v) => v == null ? "—" : v.toFixed(1);
    const hot = (stats.temp_max_c || 0) > 45;
    const el = $("pendingStats");
    el.className = "stats" + (hot ? " hot" : "");
    el.innerHTML = `<b>T</b> min ${t(stats.temp_min_c)} · mean ${t(stats.temp_mean_c)} · ` +
      `max ${t(stats.temp_max_c)} °C · <b>ΔT</b> ${t(stats.temp_delta_c)} · ` +
      `<b>depth</b> ${stats.median_depth_meters == null ? "—" : stats.median_depth_meters.toFixed(2)} m` +
      (stats.depth_warning ? `<br><span style="color:var(--warn)">${stats.depth_warning}</span>` : "");
  } catch (err) {
    $("pendingStats").textContent = "ROI error: " + err.message;
  }
}

// ── add / edit / delete object ───────────────────────────────────────────────
// The editable attribute fields (geometry + stats are NOT taken from the form).
function objFields() {
  const cls = $("objClass").value;
  return {
    class: cls, object_class: cls,
    object_state: $("objState").value,
    object_hazard_label: $("objHazLabel").value,
    object_hazard_tier: $("objHazTier").value,
    is_contextual: $("isCtx").checked,
    region_caption: $("regionCaption").value.trim(),
    spatial_relations: S.stagedRels.slice(),
  };
}
// Light reset after add/exit-edit: keep class/state/hazard "sticky" for the next
// object, but clear the per-object text + staged relations.
function resetObjForm() {
  $("regionCaption").value = "";
  S.stagedRels = []; renderStagedRels();
  if (S.editing < 0) {
    $("pendingStats").className = "stats muted";
    $("pendingStats").textContent = "Draw a shape to compute temp/depth…";
  }
}

// The primary button does double duty: "Add object" (new) or "Update object" (edit).
function addObject() {
  if (S.editing >= 0) return updateObject();
  if (!S.pending) { setHint("Draw a shape first"); return; }
  const ann = {
    ...objFields(),
    shape: S.pending.shape,
    bbox_original: S.pending.bbox,
    ...(S.pending.stats || {}),
  };
  delete ann.depth_warning;
  if (S.pending.shape === "polygon") ann.polygon_original = S.pending.polygon;
  S.anns.push(ann);
  S.pending = null;
  resetObjForm();
  renderObjects(); draw();
  saveScene();
}

function updateObject() {
  const a = S.anns[S.editing];
  if (!a) { exitEdit(); return; }
  Object.assign(a, objFields());   // fields only; geometry, id and stats preserved
  exitEdit();
  renderObjects(); draw();
  saveScene();
}

function loadFormFromAnn(a) {
  // The object's class may be a custom one not in the base list — add it so it selects.
  if (![...$("objClass").options].some(o => o.value === (a.object_class || a.class))) {
    const o = document.createElement("option");
    o.value = o.textContent = a.object_class || a.class; $("objClass").appendChild(o);
  }
  $("objClass").value = a.object_class || a.class || "";
  $("objState").value = a.object_state || "UNKNOWN";
  $("objHazLabel").value = a.object_hazard_label || "UNKNOWN";
  $("objHazTier").value = a.object_hazard_tier || "UNKNOWN";
  $("isCtx").checked = !!a.is_contextual;
  $("regionCaption").value = a.region_caption || "";
  S.stagedRels = (a.spatial_relations || []).slice();
  renderStagedRels();
}

function enterEdit(i) {
  const a = S.anns[i]; if (!a) return;
  if (S.polyPts.length || S.dragRect) { S.polyPts = []; S.dragRect = null; }
  S.editing = i; S.selected = i; S.pending = null;
  loadFormFromAnn(a);
  updateAddButton();
  const el = $("pendingStats"); el.className = "stats";
  el.innerHTML = `✏️ Editing object #${i + 1} (<b>${a.object_class}</b>) — change fields, then <b>Update object</b>. Shape/temperature unchanged.`;
  renderObjects(); draw(); syncToolbar();
}

function exitEdit() {
  S.editing = -1;
  updateAddButton();
  resetObjForm();
  $("pendingStats").className = "stats muted";
  $("pendingStats").textContent = "Draw a shape to compute temp/depth…";
  renderObjects(); draw(); syncToolbar();
}

function updateAddButton() {
  $("addObject").textContent = S.editing >= 0 ? "💾 Update object" : "➕ Add object";
  $("cancelEdit").hidden = S.editing < 0;
}

function deleteObject(i) {
  S.anns.splice(i, 1);
  if (S.editing === i) { S.editing = -1; updateAddButton(); resetObjForm(); }
  else if (S.editing > i) S.editing--;
  if (S.selected === i) S.selected = -1; else if (S.selected > i) S.selected--;
  renderObjects(); draw(); saveScene();
}

// ── save ─────────────────────────────────────────────────────────────────────
let saveTimer = null;
function collectMeta() {
  return {
    description: $("mDescription").value,
    ambient_temp_c: null,          // ambient removed — ΔT is now vs the thermal-scene min
    lighting: $("mLighting").value,
    scene_type: $("mSceneType").value,
    kitchen_id: $("mKitchen").value,
    scene_hazard_label: $("mHazLabel").value,
    scene_hazard_tier: $("mHazTier").value,
    scene_caption: $("mCaption").value,
    qa_pairs: S.qa,
  };
}
async function saveScene(completed) {
  setSaveState("saving", "saving…");
  try {
    const res = await api(`/api/scene/${S.sceneName}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotations: S.anns, scene_meta: collectMeta(),
        completed: completed === undefined ? null : completed }),
    });
    S.anns = res.annotations;          // server-recomputed stats are authoritative
    renderObjects(); draw();
    setSaveState("", "saved");
    return res;
  } catch (err) {
    setSaveState("error", "save failed");
    console.error(err);
  }
}

// ── rendering panels ─────────────────────────────────────────────────────────
function renderScenes() {
  const filter = $("sceneFilter").value.toLowerCase();
  const ul = $("scenes"); ul.innerHTML = "";
  let done = 0;
  for (const sc of S.config.scenes) {
    if (sc.completed) done++;
    if (filter && !sc.name.toLowerCase().includes(filter)) continue;
    const li = document.createElement("li");
    li.className = (sc.name === S.sceneName ? "active " : "") + (sc.completed ? "done" : "");
    li.innerHTML = `<span>${sc.name}</span><span class="badge">${sc.annotated || ""}</span>`;
    li.onclick = () => loadScene(sc.name);
    ul.appendChild(li);
  }
  $("progress").textContent = `${done}/${S.config.scenes.length}`;
}

function objIcon(a) {
  const haz = { HAZARD: "🔴", SAFE: "🟢" }[a.object_hazard_label] || "⚪";
  return (a.shape === "polygon" ? "⬡" : "▭") + " " + haz;
}
function renderObjects() {
  const ul = $("savedObjects"); ul.innerHTML = "";
  S.anns.forEach((a, i) => {
    const li = document.createElement("li");
    li.className = (i === S.selected ? "sel" : "") + (i === S.editing ? " editing" : "");
    li.style.borderLeftColor = COLORS[i % COLORS.length];
    const t = a.temp_max_c == null ? "—" : a.temp_max_c.toFixed(1);
    const d = a.median_depth_meters == null ? "—" : a.median_depth_meters.toFixed(2);
    const dt = a.temp_delta_c == null ? "" : ` ΔT ${a.temp_delta_c.toFixed(1)}`;
    li.innerHTML = `<div><div>${objIcon(a)} <b>${a.object_class}</b> [${a.object_state}]` +
      `${a.instance_id ? " · " + a.instance_id : ""}</div>` +
      `<div class="meta">${t}°C${dt} · ${d}m${a.spatial_relations && a.spatial_relations.length ? " · 🔗" + a.spatial_relations.length : ""}${a.region_caption ? " 📝" : ""}</div></div>` +
      `<button class="del" title="delete">🗑</button>`;
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) { deleteObject(i); return; }
      if (S.editing === i) { exitEdit(); return; }   // click the editing item again to stop
      enterEdit(i);
    };
    ul.appendChild(li);
  });
  $("objCount").textContent = S.anns.length ? `(${S.anns.length})` : "";
}

function renderStagedRels() {
  const ul = $("stagedRels"); ul.innerHTML = "";
  S.stagedRels.forEach((r, i) => {
    const li = document.createElement("li");
    const d = r.distance_meters ? ` @ ${r.distance_meters}m` : "";
    li.innerHTML = `<span>${r.relation} → ${r.target_instance_id}${d}</span><button class="del">✕</button>`;
    li.querySelector(".del").onclick = () => { S.stagedRels.splice(i, 1); renderStagedRels(); };
    ul.appendChild(li);
  });
  $("relCount").textContent = S.stagedRels.length ? `(${S.stagedRels.length})` : "";
}

function renderQa() {
  const ul = $("qaList"); ul.innerHTML = "";
  S.qa.forEach((q, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span><b>Q:</b> ${q.question}<br><b>A:</b> ${q.answer} <i>(${q.answer_type})</i></span><button class="del">🗑</button>`;
    li.querySelector(".del").onclick = () => { S.qa.splice(i, 1); renderQa(); saveScene(); };
    ul.appendChild(li);
  });
  $("qaCount").textContent = S.qa.length ? `(${S.qa.length})` : "";
}

// ── scene load ───────────────────────────────────────────────────────────────
function loadBg() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { S.bgImg = img; resolve(); };
    const params = new URLSearchParams({ _: Date.now() });
    if ($("enhance").checked) { params.set("enhance", "1"); }
    img.src = `/api/scene/${S.sceneName}/image/rgb?${params.toString()}`;
  });
}
async function loadScene(name) {
  if (S.anns && S.sceneName && S.sceneName !== name) await flushPending();
  S.sceneName = name;
  S.selected = -1; S.editing = -1; S.pending = null; S.polyPts = []; S.dragRect = null; S.stagedRels = [];
  updateAddButton();
  const sc = await api(`/api/scene/${name}`);
  S.scene = sc;
  S.anns = sc.annotations;
  S.meta = sc.scene_meta;
  S.qa = sc.scene_meta.qa_pairs || [];

  canvas.width = sc.width; canvas.height = sc.height;
  $("thermalImg").src = `/api/scene/${name}/image/thermal?_=${Date.now()}`;
  $("depthImg").src = `/api/scene/${name}/image/depth?_=${Date.now()}`;

  // banner
  const th = sc.thermal;
  let flag = "";
  if (th.max_c - th.min_c < 2) flag = "ℹ️ ambient (no heat source)";
  else if (th.max_c > 60) flag = "🔴 CRITICAL (>60°C)";
  else if (th.max_c > 45) flag = "🟠 HIGH (>45°C)";
  $("thermalBanner").innerHTML =
    `<b>${name}</b> — T_min ${th.min_c.toFixed(1)} · T_max <b>${th.max_c.toFixed(1)}</b> · ` +
    `mean ${th.mean_c.toFixed(1)} °C · ΔT ${(th.max_c - th.min_c).toFixed(1)} ${flag}`;
  $("boundaryNote").textContent = sc.no_depth_boundary_x
    ? `🟠 Orange zone (x < ${sc.no_depth_boundary_x}px): no depth data — depth = 0 m here.`
    : (sc.q_error ? `⚠️ Depth disabled: ${sc.q_error}` : "");

  // scene meta form
  const m = sc.scene_meta;
  $("mDescription").value = m.description || "";
  optKitchen(m.kitchen_id || "");
  opt($("mHazLabel"), S.config.enums.hazard_labels, m.scene_hazard_label || "UNKNOWN");
  opt($("mHazTier"), S.config.enums.hazard_tiers, m.scene_hazard_tier || "UNKNOWN");
  opt($("mLighting"), S.config.enums.lighting, m.lighting || "well_lit");
  opt($("mSceneType"), S.config.enums.scene_types, m.scene_type || "kitchen");
  $("mCaption").value = m.scene_caption || "";
  // Treat already-completed scenes, or scenes that already carry real metadata, as saved.
  setMetaSaved(!!(sc.completed || m.kitchen_id ||
                  (m.scene_hazard_label && m.scene_hazard_label !== "UNKNOWN")));

  renderScenes(); renderObjects(); renderStagedRels(); renderQa();
  await loadBg(); draw();
  $("pendingStats").className = "stats muted";
  $("pendingStats").textContent = "Draw a shape to compute temp/depth…";
}

async function flushPending() { /* placeholder: autosave already runs on each change */ }

// ── navigation ───────────────────────────────────────────────────────────────
function sceneIndex() { return S.config.scenes.findIndex(s => s.name === S.sceneName); }
function go(delta) {
  const n = S.config.scenes.length, i = (sceneIndex() + delta + n) % n;
  loadScene(S.config.scenes[i].name);
}
async function completeScene() {
  if (!S.metaSaved) {
    setHint("⚠ Save scene metadata before completing.");
    flashMeta();
    return;
  }
  const res = await saveScene(true);
  if (res) {
    const s = S.config.scenes.find(x => x.name === S.sceneName);
    if (s) s.completed = true;
  }
  go(1);
}

// ── toolbar / hints ──────────────────────────────────────────────────────────
function setHint(t) { $("hint").textContent = t; }
function syncToolbar() {
  document.querySelectorAll(".mode").forEach(b => b.classList.toggle("active", b.dataset.mode === S.mode));
  const drawingPoly = S.mode === "polygon" && S.polyPts.length;
  $("undoPoint").hidden = !drawingPoly;
  $("finishPoly").hidden = !drawingPoly;
  $("cancelShape").hidden = !(S.pending || S.polyPts.length || S.dragRect);
  setHint(S.mode === "rect"
    ? "Click-drag to draw a box"
    : (S.polyPts.length >= 3
        ? "Click the green first point / double-click / Enter to close · right-click or ⌫ to undo a point"
        : "Click each vertex (need ≥3) · right-click or ⌫ to undo · then close on the first point"));
}
function setMode(m) {
  S.mode = m; S.polyPts = []; S.dragRect = null;
  canvas.style.cursor = "crosshair";
  syncToolbar(); draw();
}
function cancelShape() {
  if (S.editing >= 0) { exitEdit(); return; }
  S.pending = null; S.polyPts = []; S.dragRect = null;
  $("pendingStats").className = "stats muted";
  $("pendingStats").textContent = "Draw a shape to compute temp/depth…";
  syncToolbar(); draw();
}

// ── wiring ───────────────────────────────────────────────────────────────────
async function init() {
  S.config = await api("/api/config");
  $("dataset").textContent = "· " + S.config.dataset;

  opt($("objClass"), S.config.classes);
  opt($("objState"), S.config.enums.object_states, "UNKNOWN");
  opt($("objHazLabel"), S.config.enums.hazard_labels, "UNKNOWN");
  opt($("objHazTier"), S.config.enums.hazard_tiers, "UNKNOWN");
  opt($("relType"), S.config.enums.relations);
  opt($("qaType"), S.config.enums.qa_answer_types);

  applyTheme(localStorage.getItem("rgbdt_theme") || "dark");   // sync button icon
  $("themeToggle").onclick = toggleTheme;

  document.querySelectorAll(".mode").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $("undoPoint").onclick = undoPolyPoint;
  $("finishPoly").onclick = finishPolygon;
  $("cancelShape").onclick = cancelShape;
  $("cancelEdit").onclick = exitEdit;
  $("enhance").onchange = () => loadBg().then(draw);
  $("addObject").onclick = addObject;
  $("saveMeta").onclick = async () => { const r = await saveScene(); if (r) setMetaSaved(true); };
  // Editing any metadata field marks it unsaved again (must re-save before completing).
  ["mDescription", "mKitchen", "mHazLabel", "mHazTier", "mLighting", "mSceneType", "mCaption"]
    .forEach(id => { $(id).addEventListener("input", () => setMetaSaved(false)); });
  $("prevScene").onclick = () => go(-1);
  $("nextScene").onclick = () => go(1);
  $("completeScene").onclick = completeScene;
  $("sceneFilter").oninput = renderScenes;

  $("addClassBtn").onclick = async () => {
    const name = prompt("New class name (e.g. toaster, heat_gun):");
    if (!name) return;
    const res = await api("/api/classes", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    S.config.classes = res.classes;
    opt($("objClass"), res.classes, res.name);
  };

  $("stageRel").onclick = () => {
    const target = $("relTarget").value.trim();
    if (!target) { setHint("Enter a target instance id"); return; }
    const d = parseFloat($("relDist").value) || 0;
    S.stagedRels.push({ relation: $("relType").value, target_instance_id: target,
      distance_meters: d > 0 ? d : null });
    $("relTarget").value = "";
    renderStagedRels();
  };

  $("addQa").onclick = () => {
    const q = $("qaQ").value.trim(), a = $("qaA").value.trim();
    if (!q || !a) { setHint("Question and answer required"); return; }
    const mods = [...document.querySelectorAll(".qamod:checked")].map(c => c.value);
    S.qa.push({ qa_id: `${S.sceneName}_qa_${S.qa.length + 1}`, question: q, answer: a,
      answer_type: $("qaType").value, requires_modality: mods });
    $("qaQ").value = ""; $("qaA").value = "";
    document.querySelectorAll(".qamod").forEach(c => c.checked = false);
    renderQa(); saveScene();
  };

  // keyboard shortcuts (ignore while typing in a field)
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.key === "n" || e.key === "N") go(1);
    else if (e.key === "p" || e.key === "P") go(-1);
    else if (e.key === "r" || e.key === "R") setMode("rect");
    else if (e.key === "g" || e.key === "G") setMode("polygon");
    else if (e.key === "t" || e.key === "T") toggleTheme();
    else if (e.key === "Enter" && S.mode === "polygon") finishPolygon();
    else if (e.key === "Escape") cancelShape();
    else if (e.key === "Delete" || e.key === "Backspace") {
      if (S.mode === "polygon" && S.polyPts.length) { e.preventDefault(); undoPolyPoint(); }
      else if (S.editing >= 0) { /* don't delete the object you're editing via keyboard */ }
      else if (S.selected >= 0) deleteObject(S.selected);
    }
    else if (/^[1-9]$/.test(e.key)) {
      const idx = +e.key - 1;
      if (idx < S.config.classes.length) $("objClass").value = S.config.classes[idx];
    }
  });

  setMode("rect");
  if (S.config.scenes.length) await loadScene(S.config.scenes[0].name);
}

init().catch(err => { alert("Init failed: " + err.message); console.error(err); });
