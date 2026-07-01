"""
RGB-D-T Thermal Hazard Dataset — Annotation Tool v2
====================================================
Upgraded schema for multi-model benchmarking.

New fields captured per annotation:
  - object_class     : unified taxonomy class (e.g. stove_burner, iron, outlet)
  - object_state     : ON / OFF / COOLING / UNKNOWN
  - hazard_tier      : CRITICAL / HIGH / MODERATE / CONTEXTUAL / SAFE / UNKNOWN
  - is_contextual    : bool — temperature-state-dependent hazard
  - shape            : rectangle / polygon
  - bbox_original    : {x_min, y_min, x_max, y_max} in original image pixels
  - polygon_original : [{x,y}...] for polygon shapes
  - temp_min_c       : min temperature in ROI (°C)
  - temp_mean_c      : mean temperature in ROI (°C)
  - temp_max_c       : max temperature in ROI (°C)
  - temp_delta_c     : max_temp - scene_ambient (°C)
  - max_temp         : kept for backward compat (= temp_max_c)
  - median_depth_meters : depth from stereo disparity + Q matrix
  - annotation_id    : stable UUID per annotation

New scene-level fields:
  - description, ambient_temp_c, lighting, scene_type
  - T_min/max/mean across full thermal map

Run from repo root:
    .venv/bin/streamlit run annotator/annotator_v2.py

From this folder:
    ../.venv/bin/streamlit run annotator_v2.py
"""

import base64
import io
import math
import os
import re
import uuid
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas, CanvasResult

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="RGB-D-T Annotation Tool v2")

ANNOTATOR_DIR           = os.path.dirname(os.path.abspath(__file__))
APP_VERSION             = "v2-benchmark-schema"
MIN_BBOX_DISPLAY_PIXELS = 8


def find_repo_root(start: str | None = None) -> str:
    """Locate dataset repo root (folder containing Dataset/ or new data/)."""
    cur = os.path.abspath(start or ANNOTATOR_DIR)
    for _ in range(6):
        for name in ("Annotating Dataset"):
            if os.path.isdir(os.path.join(cur, name)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(ANNOTATOR_DIR)


REPO_ROOT = find_repo_root()
BASE_DIR  = REPO_ROOT

# Taxonomy — edit this list to add new classes for your dataset
HAZARD_CLASSES = [
    "stove_burner",
    "iron",
    "oven_door",
    "kettle",
    "hot_surface",
    "electrical_outlet",
    "extension_cord",
    "space_heater",
    "radiator",
    "laptop_vent",
    "mug_cup",
    "other_hazard",
    "safe_object",
]

OBJECT_STATES   = ["ON", "OFF", "COOLING", "PLUGGED_IN", "ACTIVE", "UNKNOWN"]
LIGHTING_CONDS  = ["well_lit", "dim", "dark", "mixed", "artificial", "natural"]
SCENE_TYPES     = ["kitchen", "bathroom", "living_room", "bedroom", "office",
                   "laundry", "garage", "other"]
BOX_COLORS      = ["#00FF00","#FF4444","#00FFFF","#FFD700","#FF00FF",
                   "#4488FF","#FFA500","#FFFFFF","#FF69B4","#7FFF00"]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def file_slug(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "dataset"

def safe_float(val):
    if val is None: return None
    try:
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except: return None

# ─────────────────────────────────────────────────────────────────────────────
# Custom class persistence
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CLASSES_FILE = os.path.join(ANNOTATOR_DIR, "custom_classes.json")

def load_custom_classes():
    if os.path.exists(CUSTOM_CLASSES_FILE):
        try:
            with open(CUSTOM_CLASSES_FILE) as f:
                data = json.load(f)
            return [c for c in data if isinstance(c, str) and c.strip()]
        except Exception:
            pass
    return []

def save_custom_classes(classes):
    with open(CUSTOM_CLASSES_FILE, "w") as f:
        json.dump(classes, f, indent=2)

def get_all_classes():
    """Return built-in taxonomy + any user-added classes, deduplicated."""
    custom = load_custom_classes()
    seen = set(HAZARD_CLASSES)
    extras = [c for c in custom if c not in seen]
    return HAZARD_CLASSES + extras

# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery (accepts any scene folder name, not just "Scene_*")
# ─────────────────────────────────────────────────────────────────────────────

def _scene_valid(path):
    return (os.path.exists(os.path.join(path, "left_cam",  "left.png")) and
            os.path.exists(os.path.join(path, "thermal",   "rgb_matched_thermal.npy")) and
            os.path.exists(os.path.join(path, "depth",     "depth.npy")))

def get_available_dataset_names(base_dir):
    names = []
    for entry in sorted(os.listdir(base_dir), key=natural_sort_key):
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path) or entry.startswith(".") or entry == "__pycache__":
            continue
        try:
            if any(_scene_valid(os.path.join(path, c))
                   for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))):
                names.append(entry)
        except OSError:
            pass
    return names

def get_dataset_paths(name):
    slug = file_slug(name)
    data_dir = os.path.join(REPO_ROOT, name)
    ann      = os.path.join(ANNOTATOR_DIR, f"annotations_{slug}.json")
    prog     = os.path.join(ANNOTATOR_DIR, f"progress_{slug}.json")
    return data_dir, ann, prog

@st.cache_data
def get_dataset_files(data_dir):
    files = []
    if not os.path.exists(data_dir): return files
    for sd in sorted(os.listdir(data_dir), key=natural_sort_key):
        sp = os.path.join(data_dir, sd)
        if not os.path.isdir(sp): continue
        rgb      = os.path.join(sp, "left_cam",  "left.png")
        thermal  = os.path.join(sp, "thermal",   "rgb_matched__manual_calibrated_thermal.npy")
        depth    = os.path.join(sp, "depth",     "depth.npy")
        metadata = os.path.join(sp, "metadata.json")
        if os.path.exists(rgb) and os.path.exists(thermal) and os.path.exists(depth):
            files.append({"name": sd, "rgb": rgb, "thermal": thermal, "depth": depth,
                          "metadata": metadata if os.path.exists(metadata) else None})
    return files

# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f: return json.load(f)
        except: pass
    return []

def save_progress(lst, path):
    with open(path, "w") as f: json.dump(lst, f, indent=2); f.flush(); os.fsync(f.fileno())

def load_annotations(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f: return json.load(f)
        except: pass
    return {}

def save_annotations_atomic(anns, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(anns, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def get_scene_data(all_ann, scene):
    d = all_ann.get(scene, {})
    if isinstance(d, dict): return d
    return {"description": "", "annotations": d}

def save_scene(all_ann, scene, anns, scene_meta):
    existing = get_scene_data(all_ann, scene)
    all_ann[scene] = {**existing, **scene_meta, "annotations": anns}

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_annotation_bbox(ann):
    if "bbox_original" in ann: return ann["bbox_original"]
    poly = ann.get("polygon_original", [])
    if not poly: return None
    xs = [p["x"] for p in poly]; ys = [p["y"] for p in poly]
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}



# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm_colormap(arr, cmap):
    v = np.nan_to_num(arr, nan=0., posinf=0., neginf=0.)
    n = (np.zeros_like(v, dtype=np.uint8) if v.max()==v.min()
         else cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    return cv2.cvtColor(cv2.applyColorMap(n, cmap), cv2.COLOR_BGR2RGB)

def enhance_low_light(img: Image.Image, boost=1., gamma=1.):
    arr  = np.asarray(img.convert("RGB"))
    should = arr.mean() < 55 or np.percentile(arr, 99) < 150
    out  = arr.copy()
    if should:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        l,a,b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(3.0,(8,8)).apply(l)
        out = cv2.cvtColor(cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)
        lo,hi = np.percentile(out,1), np.percentile(out,99)
        if hi > lo: out = np.clip((out.astype(np.float32)-lo)*(255./(hi-lo)),0,255).astype(np.uint8)
        out = np.clip(out.astype(np.float32)*min(4.5,135./max(out.mean(),1.)),0,255).astype(np.uint8)
    if boost!=1.: out = np.clip(out.astype(np.float32)*boost,0,255).astype(np.uint8)
    if gamma!=1.: out = np.clip((out.astype(np.float32)/255.)**gamma*255,0,255).astype(np.uint8)
    return Image.fromarray(out,"RGB"), (should or boost!=1. or gamma!=1.)

def pil_to_data_url(img):
    buf=io.BytesIO(); img.convert("RGB").save(buf,"PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def fabric_image(img, w, h):
    return {"type":"image","version":"4.4.0","originX":"left","originY":"top",
            "left":0,"top":0,"width":w,"height":h,"fill":"rgb(0,0,0)","stroke":None,
            "strokeWidth":0,"scaleX":1,"scaleY":1,"angle":0,"opacity":1,
            "selectable":False,"evented":False,"hasControls":False,"hasBorders":False,
            "src":pil_to_data_url(img),"crossOrigin":None,"filters":[]}

# ─────────────────────────────────────────────────────────────────────────────
# Depth / Q-matrix
# ─────────────────────────────────────────────────────────────────────────────

def load_Q(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None, "metadata.json not found — depth disabled."
    try:
        with open(meta_path) as f: meta = json.load(f)
        Q = np.array(meta["calibration"]["Q"], dtype=np.float64)
        if Q.shape != (4,4): return None, f"Q shape {Q.shape} ≠ (4,4)"
        return Q, None
    except Exception as e: return None, str(e)

def disp_to_depth(roi, Q, cx, cy):
    valid = roi[roi > 0.1]
    if valid.size == 0: return 0.0, "No valid disparity in ROI — depth = 0 m."
    pt = Q @ np.array([cx, cy, float(np.median(valid)), 1.0])
    return float(pt[2]/pt[3]/1000.), None

# ─────────────────────────────────────────────────────────────────────────────
# Canvas wrapper (uses public st_canvas API)
# ─────────────────────────────────────────────────────────────────────────────

def stable_canvas(fill, sw, sc, initial, update, h, w, mode, key):
    return st_canvas(fill_color=fill, stroke_width=sw, stroke_color=sc,
                     background_color="", background_image=None,
                     update_streamlit=update, height=h, width=w,
                     drawing_mode=mode, initial_drawing=initial,
                     display_toolbar=True, point_display_radius=3, key=key)

# ─────────────────────────────────────────────────────────────────────────────
# Migrate old annotations to new schema
# ─────────────────────────────────────────────────────────────────────────────

CLASS_MIGRATION = {
    "burner 1": "stove_burner", "burner 2": "stove_burner",
    "burner 3": "stove_burner", "burner 4": "stove_burner",
    "burners":  "stove_burner", "burner":   "stove_burner",
    "iron": "iron", "kettle": "kettle", "oven": "oven_door",
    "outlet": "electrical_outlet", "heater": "space_heater",
    "radiator": "radiator", "laptop": "laptop_vent", "mug": "mug_cup",
}

def migrate_annotation(ann):
    """Upgrade a v1 annotation to v2 schema in-place. Idempotent."""
    if "annotation_id" not in ann:
        ann["annotation_id"] = str(uuid.uuid4())
    # Migrate class name to taxonomy
    old_cls = ann.get("class","").lower().strip()
    ann["object_class"] = CLASS_MIGRATION.get(old_cls, ann.get("object_class", old_cls or "other_hazard"))
    # Add missing fields with sensible defaults
    ann.setdefault("object_state", "UNKNOWN")
    ann.setdefault("hazard_tier",  "UNKNOWN")
    ann.setdefault("is_contextual", False)
    ann.setdefault("shape", "rectangle")
    ann.setdefault("temp_max_c",  ann.get("max_temp"))
    ann.setdefault("temp_min_c",  None)
    ann.setdefault("temp_mean_c", None)
    ann.setdefault("temp_delta_c", None)
    ann.setdefault("max_temp",    ann.get("temp_max_c"))   # backward compat
    ann.setdefault("median_depth_meters", None)
    return ann

def migrate_scene(scene_data):
    """Upgrade a v1 scene dict to v2 schema."""
    if isinstance(scene_data, list):
        scene_data = {"description": "", "annotations": scene_data}
    scene_data.setdefault("ambient_temp_c", None)
    scene_data.setdefault("lighting",       "UNKNOWN")
    scene_data.setdefault("scene_type",     "UNKNOWN")
    scene_data.setdefault("thermal_scene_min_c",  None)
    scene_data.setdefault("thermal_scene_max_c",  None)
    scene_data.setdefault("thermal_scene_mean_c", None)
    scene_data["annotations"] = [migrate_annotation(a) for a in scene_data.get("annotations",[])]
    return scene_data

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

st.title("RGB-D-T Thermal Hazard — Annotation Tool v2")
st.caption(
    f"Build: {APP_VERSION}  |  Data: `{REPO_ROOT}`  |  Annotations: `{ANNOTATOR_DIR}`"
)

# Dataset selector
available = get_available_dataset_names(BASE_DIR)
if not available:
    st.error(f"No dataset folders found under `{BASE_DIR}`. "
             "Each dataset needs scenes with left_cam/, thermal/, depth/ subfolders.")
    st.stop()

if ("dataset_selector" not in st.session_state
        or st.session_state.dataset_selector not in available):
    st.session_state.dataset_selector = available[0]

with st.sidebar:
    st.header("📂 Dataset")
    selected_ds = st.selectbox("Dataset Folder", available, key="dataset_selector")

DATA_DIR, OUTPUT_FILE, PROGRESS_FILE = get_dataset_paths(selected_ds)
dataset_files    = get_dataset_files(DATA_DIR)
completed_scenes = load_progress(PROGRESS_FILE)
all_annotations  = load_annotations(OUTPUT_FILE)

# Migrate any v1 data transparently on first load
migrated = False
for sn, sd in all_annotations.items():
    new_sd = migrate_scene(sd)
    if new_sd != sd:
        all_annotations[sn] = new_sd
        migrated = True
if migrated:
    save_annotations_atomic(all_annotations, OUTPUT_FILE)
    st.toast("Annotations migrated to v2 schema.", icon="✅")

if not dataset_files:
    st.warning(f"No valid scenes in `{DATA_DIR}`.")
    st.stop()

scene_names = [f["name"] for f in dataset_files]
if ("scene_selector" not in st.session_state
        or st.session_state.scene_selector not in scene_names):
    st.session_state.scene_selector = scene_names[0]

if "navigate_to_scene" in st.session_state:
    st.session_state.scene_selector = st.session_state.pop("navigate_to_scene")

def go_next():
    idx = scene_names.index(st.session_state.scene_selector)
    st.session_state.navigate_to_scene = scene_names[(idx+1) % len(scene_names)]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛 Controls")
    st.caption(f"Output: `{os.path.basename(OUTPUT_FILE)}`")

    def fmt(n): return f"✅ {n}" if n in completed_scenes else n
    sel_scene      = st.selectbox("Scene", scene_names, format_func=fmt, key="scene_selector")
    current_scene  = next(s for s in dataset_files if s["name"] == sel_scene)
    scene_blob     = migrate_scene(get_scene_data(all_annotations, sel_scene))
    scene_anns     = scene_blob.get("annotations", [])

    st.markdown("---")
    st.write("### ✏️ Drawing")
    stroke_width = st.slider("Stroke width", 1, 10, 2)
    shape_mode   = st.radio("Shape", ["Rectangle", "Polygon"], horizontal=True,
                             key=f"shape_{selected_ds}_{sel_scene}")
    if shape_mode == "Rectangle":
        draw_mode = "rect"
        st.caption("Click and drag on the image to draw a rectangle.")
    else:
        draw_mode = "point"
        st.caption("Single-click to add each vertex. Double-click to finish — the polygon closes back to the first point automatically.")

    st.markdown("---")
    st.write("### 🖼 RGB Display")
    show_orig  = st.checkbox("Original (no enhancement)", value=True)
    rgb_zoom   = st.slider("Zoom %", 50, 200, 100, 10)
    rgb_boost  = st.slider("Brightness boost", 1.0, 8.0, 1.0, 0.25, disabled=show_orig)
    rgb_gamma  = st.slider("Gamma", 0.3, 1.5, 1.0, 0.05, disabled=show_orig)

    stroke_color = BOX_COLORS[len(scene_anns) % len(BOX_COLORS)]
    st.markdown(f"**Pen:** <span style='color:{stroke_color}'>■■■</span>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📄 annotations.json", expanded=False):
        if all_annotations:
            st.json(json.dumps(all_annotations))
        else:
            st.caption("Empty.")

# ── Load scene data ──────────────────────────────────────────────────────────
orig_img         = Image.open(current_scene["rgb"]).convert("RGB")
orig_w, orig_h   = orig_img.size
disp_w           = max(1, int(orig_w * rgb_zoom/100))
disp_h           = max(1, int(orig_h * rgb_zoom/100))

if show_orig:
    disp_src, was_enh = orig_img, False
else:
    disp_src, was_enh = enhance_low_light(orig_img, rgb_boost, rgb_gamma)

disp_img = disp_src.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGBA")
sx, sy   = orig_w/disp_w, orig_h/disp_h

thermal_raw = np.load(current_scene["thermal"])
depth_raw   = np.load(current_scene["depth"])
Q, Q_err    = load_Q(current_scene["metadata"])

t_min  = float(np.nanmin(thermal_raw))
t_max  = float(np.nanmax(thermal_raw))
t_mean = float(np.nanmean(thermal_raw))

# Fixed 640px for display — st.image + use_column_width will scale to column width
VIS_SIZE = 640
thm_vis  = cv2.resize(norm_colormap(thermal_raw, cv2.COLORMAP_INFERNO), (VIS_SIZE, VIS_SIZE))
dep_vis  = cv2.resize(norm_colormap(depth_raw,   cv2.COLORMAP_VIRIDIS), (VIS_SIZE, VIS_SIZE))

# ── Build canvas ─────────────────────────────────────────────────────────────
canvas_objs = [fabric_image(disp_img, disp_w, disp_h)]
for idx, ann in enumerate(scene_anns):
    color = BOX_COLORS[idx % len(BOX_COLORS)]
    if ann.get("shape") == "polygon" and ann.get("polygon_original"):
        pts = ann["polygon_original"]
        path_cmds = [["M", pts[0]["x"]/sx, pts[0]["y"]/sy]]
        for p in pts[1:]:
            path_cmds.append(["L", p["x"]/sx, p["y"]/sy])
        path_cmds.append(["z"])
        canvas_objs.append({"type":"path","path":path_cmds,
                             "fill":"rgba(0,0,0,0)","stroke":color,
                             "strokeWidth":stroke_width,"selectable":False})
    else:
        bb = get_annotation_bbox(ann)
        if not bb: continue
        canvas_objs.append({"type":"rect","left":bb["x_min"]/sx,"top":bb["y_min"]/sy,
                             "width":(bb["x_max"]-bb["x_min"])/sx,"height":(bb["y_max"]-bb["y_min"])/sy,
                             "fill":"rgba(0,0,0,0)","stroke":color,"strokeWidth":stroke_width,
                             "scaleX":1,"scaleY":1,"selectable":False})

init_draw = {"version":"4.4.0","objects":canvas_objs}

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.write(f"**Dataset:** `{DATA_DIR}`  ·  **Annotations:** `{OUTPUT_FILE}`")

# Thermal summary banner
with st.expander("🌡 Thermal Scene Summary", expanded=True):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("T_min", f"{t_min:.2f} °C")
    c2.metric("T_max", f"{t_max:.2f} °C")
    c3.metric("T_mean",f"{t_mean:.2f} °C")
    c4.metric("ΔT",    f"{t_max-t_min:.2f} °C")
    if t_max - t_min < 2.0:
        st.info("ℹ️ Very narrow thermal range — scene appears ambient (no heat source visible in thermal).")
    elif t_max > 60:
        st.error("🔴 CRITICAL thermal signature detected (>60°C max).")
    elif t_max > 45:
        st.warning("🟠 HIGH thermal signature detected (>45°C max).")

if Q_err:
    st.warning(f"⚠️ Depth disabled: {Q_err}")

st.subheader(f"RGB — {current_scene['name']}")
rgb_col, ctrl_col = st.columns([2, 1], gap="large")

with rgb_col:
    if was_enh: st.caption("Display brightened — saved coordinates use original image.")
    if rgb_zoom != 100: st.caption(f"Zoom: {rgb_zoom}% ({disp_w}×{disp_h} px)")

    if shape_mode == "Rectangle":
        st.caption("🟩 Click and drag to draw a rectangle around the object.")
    else:
        st.caption(
            "🔷 **Polygon:** single-click to place each vertex. "
            "**Double-click** anywhere to finish — the polygon closes back to vertex 1 automatically. "
            "Then click 💾 Save."
        )

    # Key includes annotation count so the canvas resets cleanly after each save.
    canvas_key = f"canvas_{selected_ds}_{sel_scene}_{APP_VERSION}_{draw_mode}_{len(scene_anns)}"
    # point mode has no Fabric.js double-click builder, so update_streamlit=True is safe
    # for both modes — the canvas always pushes its current state to Streamlit on every
    # mouse event, ensuring canvas_result.json_data is populated when Save is clicked.
    canvas_result = stable_canvas(
        fill="rgba(0,0,0,0)", sw=stroke_width, sc=stroke_color,
        initial=init_draw, update=True,
        h=disp_h, w=disp_w, mode=draw_mode, key=canvas_key,
    )



# ── Control panel ────────────────────────────────────────────────────────────
with ctrl_col:
    st.subheader("Annotation Controls")

    # ── Scene-level metadata ─────────────────────────────────────────────────
    with st.expander("📋 Scene Metadata", expanded=True):
        desc_val = st.text_area("Description", value=scene_blob.get("description",""),
                                placeholder="What is in this scene?", height=80,
                                key=f"desc_{selected_ds}_{sel_scene}")
        col_a, col_b = st.columns(2)
        amb_temp  = col_a.number_input("Ambient °C", value=float(scene_blob.get("ambient_temp_c") or 22.0),
                                        step=0.5, key=f"amb_{selected_ds}_{sel_scene}")
        lighting  = col_b.selectbox("Lighting", LIGHTING_CONDS,
                                     index=LIGHTING_CONDS.index(scene_blob.get("lighting","well_lit"))
                                     if scene_blob.get("lighting") in LIGHTING_CONDS else 0,
                                     key=f"light_{selected_ds}_{sel_scene}")
        scene_type = st.selectbox("Scene type", SCENE_TYPES,
                                   index=SCENE_TYPES.index(scene_blob.get("scene_type","kitchen"))
                                   if scene_blob.get("scene_type") in SCENE_TYPES else 0,
                                   key=f"stype_{selected_ds}_{sel_scene}")
        if st.button("Save Scene Metadata", use_container_width=True):
            cur  = load_annotations(OUTPUT_FILE)
            blob = migrate_scene(get_scene_data(cur, sel_scene))
            blob.update({"description": desc_val, "ambient_temp_c": amb_temp,
                          "lighting": lighting, "scene_type": scene_type,
                          "thermal_scene_min_c": t_min, "thermal_scene_max_c": t_max,
                          "thermal_scene_mean_c": t_mean})
            cur[sel_scene] = blob
            save_annotations_atomic(cur, OUTPUT_FILE)
            st.success("Scene metadata saved.")

    st.markdown("---")

    # ── Per-object annotation fields ─────────────────────────────────────────
    st.write("**Object Annotation**")

    # ── Dynamic class list with add-new capability ────────────────────────────
    all_classes = get_all_classes()
    ADD_NEW = "＋ Add new class…"
    class_options = all_classes + [ADD_NEW]

    selected_option = st.selectbox(
        "Object Class", class_options,
        key=f"cls_{selected_ds}_{sel_scene}",
    )

    if selected_option == ADD_NEW:
        new_class_raw = st.text_input(
            "New class name",
            placeholder="e.g., toaster, heat_gun, soldering_iron",
            key=f"new_cls_{selected_ds}_{sel_scene}",
        )
        new_class = re.sub(r"[^a-zA-Z0-9]+", "_", new_class_raw.strip()).strip("_").lower()

        if st.button("✅ Add class", key=f"add_cls_{selected_ds}_{sel_scene}"):
            if not new_class:
                st.error("Please enter a class name.")
                st.stop()
            existing = get_all_classes()
            if new_class in existing:
                st.warning(f"'{new_class}' already exists — select it from the list.")
            else:
                custom = load_custom_classes()
                custom.append(new_class)
                save_custom_classes(custom)
                st.success(f"'{new_class}' added. Select it from the dropdown.")
                st.rerun()
        obj_class = new_class or ADD_NEW   # will be caught as invalid on Save
    else:
        obj_class = selected_option
    obj_state = st.selectbox("Object State", OBJECT_STATES,
                               key=f"state_{selected_ds}_{sel_scene}")
    is_ctx    = st.checkbox("Contextual hazard (state-dependent)",
                             value=False, key=f"ctx_{selected_ds}_{sel_scene}",
                             help="Check if this object is only dangerous above a certain temperature (e.g. a mug, laptop).")

    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        raw_objs = (canvas_result.json_data or {}).get("objects", [])

        if obj_class == ADD_NEW or not obj_class:
            st.error("Select or add a valid Object Class before saving.")
            st.stop()

        if shape_mode == "Rectangle":
            drawn = [o for o in raw_objs if o.get("type") == "rect"]
            existing_count = len([a for a in scene_anns if a.get("shape", "rectangle") == "rectangle"])
            if len(drawn) <= existing_count:
                st.error("Draw a NEW rectangle first."); st.stop()
            last = drawn[-1]
            rw = abs(last.get("width", 0) * last.get("scaleX", 1))
            rh = abs(last.get("height", 0) * last.get("scaleY", 1))
            if rw < MIN_BBOX_DISPLAY_PIXELS or rh < MIN_BBOX_DISPLAY_PIXELS:
                st.error("Box too small — drag a larger rectangle."); st.stop()
            x0 = max(0,      int(last["left"] * sx))
            y0 = max(0,      int(last["top"]  * sy))
            x1 = min(orig_w, int((last["left"] + rw) * sx))
            y1 = min(orig_h, int((last["top"]  + rh) * sy))
            if x1 <= x0 or y1 <= y0: st.error("Invalid area — redraw."); st.stop()

            t_roi    = thermal_raw[y0:y1, x0:x1]
            roi_min  = safe_float(np.nanmin(t_roi))
            roi_mean = safe_float(np.nanmean(t_roi))
            roi_max  = safe_float(np.nanmax(t_roi))
            amb      = scene_blob.get("ambient_temp_c") or t_min
            roi_delt = safe_float((roi_max or 0.) - amb) if roi_max else None

            depth_m, dw = 0., None
            if Q is not None:
                cx = x0 + (x1 - x0) / 2.; cy = y0 + (y1 - y0) / 2.
                depth_m, dw = disp_to_depth(depth_raw[y0:y1, x0:x1], Q, cx, cy)
                if dw: st.warning(dw)
            else:
                st.warning("Depth not computed — no Q matrix.")

            ann_entry = {
                "annotation_id":       str(uuid.uuid4()),
                "class":               obj_class,
                "object_class":        obj_class,
                "object_state":        obj_state,
                "is_contextual":       is_ctx,
                "shape":               "rectangle",
                "bbox_original":       {"x_min":x0,"y_min":y0,"x_max":x1,"y_max":y1},
                "temp_min_c":          roi_min,
                "temp_mean_c":         roi_mean,
                "temp_max_c":          roi_max,
                "temp_delta_c":        roi_delt,
                "max_temp":            roi_max,
                "median_depth_meters": safe_float(depth_m),
            }

        else:  # Polygon (point/click mode — no Fabric.js double-click issues)
            circles = [o for o in raw_objs if o.get("type") == "circle"]
            if not circles:
                st.error("Place polygon vertices first by clicking on the image."); st.stop()

            # Double-click detection: a double-click in point mode places two circles
            # at nearly the same position. Strip the trailing duplicate so the user
            # can double-click anywhere to signal "done" without affecting the shape.
            if len(circles) >= 2:
                last2, prev2 = circles[-1], circles[-2]
                dist = math.hypot(last2["left"] - prev2["left"], last2["top"] - prev2["top"])
                if dist < 10:
                    circles = circles[:-1]  # discard the duplicate

            # Fabric.js point mode: originX/Y='center', so left/top ARE the centre.
            poly_disp = [{"x": o["left"], "y": o["top"]} for o in circles]

            if len(poly_disp) < 3:
                st.error("Polygon needs at least 3 vertices — keep clicking, then double-click to finish."); st.stop()

            poly_orig = [{"x": max(0, min(orig_w, int(p["x"] * sx))),
                          "y": max(0, min(orig_h, int(p["y"] * sy)))} for p in poly_disp]
            xs = [p["x"] for p in poly_orig]; ys = [p["y"] for p in poly_orig]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            # Mask-based thermal stats (only pixels inside the polygon)
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            pts_arr = np.array([[p["x"], p["y"]] for p in poly_orig], dtype=np.int32)
            cv2.fillPoly(mask, [pts_arr], 1)
            t_roi    = thermal_raw[mask == 1]
            roi_min  = safe_float(np.nanmin(t_roi))  if t_roi.size else None
            roi_mean = safe_float(np.nanmean(t_roi)) if t_roi.size else None
            roi_max  = safe_float(np.nanmax(t_roi))  if t_roi.size else None
            amb      = scene_blob.get("ambient_temp_c") or t_min
            roi_delt = safe_float((roi_max or 0.) - amb) if roi_max else None

            depth_m, dw = 0., None
            if Q is not None:
                cx = float(np.mean(xs)); cy = float(np.mean(ys))
                depth_m, dw = disp_to_depth(depth_raw[y0:y1, x0:x1], Q, cx, cy)
                if dw: st.warning(dw)
            else:
                st.warning("Depth not computed — no Q matrix.")

            ann_entry = {
                "annotation_id":       str(uuid.uuid4()),
                "class":               obj_class,
                "object_class":        obj_class,
                "object_state":        obj_state,
                "is_contextual":       is_ctx,
                "shape":               "polygon",
                "bbox_original":       {"x_min":x0,"y_min":y0,"x_max":x1,"y_max":y1},
                "polygon_original":    poly_orig,
                "temp_min_c":          roi_min,
                "temp_mean_c":         roi_mean,
                "temp_max_c":          roi_max,
                "temp_delta_c":        roi_delt,
                "max_temp":            roi_max,
                "median_depth_meters": safe_float(depth_m),
            }

        cur  = load_annotations(OUTPUT_FILE)
        blob = migrate_scene(get_scene_data(cur, sel_scene))
        blob["annotations"].append(ann_entry)
        blob.update({"thermal_scene_min_c":t_min,"thermal_scene_max_c":t_max,
                      "thermal_scene_mean_c":t_mean})
        cur[sel_scene] = blob
        save_annotations_atomic(cur, OUTPUT_FILE)
        st.rerun()

    # ── Saved annotations list ───────────────────────────────────────────────
    st.markdown("---")
    st.write("**Saved Objects:**")
    fresh_anns = migrate_scene(get_scene_data(load_annotations(OUTPUT_FILE), sel_scene))["annotations"]
    if fresh_anns:
        for idx, ann in enumerate(fresh_anns):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            tc    = ann.get("temp_max_c") or ann.get("max_temp") or 0.
            dc    = ann.get("temp_delta_c")
            dep   = ann.get("median_depth_meters") or 0.
            state = ann.get("object_state","?")
            cls   = ann.get("object_class") or ann.get("class","?")
            dcstr = f" ΔT={dc:.1f}°C" if dc is not None else ""
            shape_icon = "⬡" if ann.get("shape") == "polygon" else "⬜"
            a_col, d_col = st.columns([4,1])
            with a_col:
                st.markdown(
                    f"<span style='color:{color}'>{shape_icon}</span> **{cls}** [{state}]<br>"
                    f"<small>{tc:.1f}°C{dcstr} · {dep:.2f}m</small>",
                    unsafe_allow_html=True)
            with d_col:
                if st.button("🗑", key=f"del_{selected_ds}_{sel_scene}_{idx}"):
                    cur  = load_annotations(OUTPUT_FILE)
                    blob = migrate_scene(get_scene_data(cur, sel_scene))
                    if idx < len(blob["annotations"]):
                        blob["annotations"].pop(idx)
                        cur[sel_scene] = blob
                        save_annotations_atomic(cur, OUTPUT_FILE)
                        st.rerun()
    else:
        st.caption("No saved annotations yet.")

    st.markdown("---")
    st.write("**Navigation**")
    pr_col, nx_col = st.columns(2)
    with pr_col:
        if st.button("✅ Complete", use_container_width=True):
            if sel_scene not in completed_scenes:
                completed_scenes.append(sel_scene)
                save_progress(completed_scenes, PROGRESS_FILE)
            go_next(); st.rerun()
    with nx_col:
        st.button("→ Next", use_container_width=True, on_click=go_next)

    # Progress bar
    total = len(dataset_files)
    done  = len(completed_scenes)
    st.progress(done/total if total else 0, text=f"{done}/{total} scenes completed")

# ── Modality views ───────────────────────────────────────────────────────────
st.markdown("---")
v1, v2 = st.columns(2)
with v1:
    st.subheader("🌡 Thermal (Inferno)")
    valid_d = depth_raw[depth_raw > 0.1]
    st.image(thm_vis, use_column_width=True,
             caption=f"min {t_min:.1f}°C  ·  max {t_max:.1f}°C  ·  ΔT {t_max-t_min:.1f}°C")
with v2:
    st.subheader("📏 Depth / Disparity (Viridis)")
    d_lo = float(valid_d.min()) if valid_d.size else 0.
    d_hi = float(valid_d.max()) if valid_d.size else 0.
    st.image(dep_vis, use_column_width=True,
             caption=f"Disparity  {d_lo:.1f} – {d_hi:.1f} px  ·  bright = closer")