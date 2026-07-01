"""
RGB-D-T Thermal Hazard Dataset — Annotation Tool v3
====================================================
Unified schema for multi-model benchmarking across all four tasks
(A: hazard classification, B: VQA, C: dense captioning, D: segmentation).
Annotate once at this level; every task reads the subset it needs.

Per-annotation fields:
  - object_class     : unified taxonomy class (e.g. stove_burner, iron, outlet)
  - instance_id      : human-readable stable id (e.g. stove_007_burner_frontleft) [B,C]
  - object_state     : ON / OFF / COOLING / UNKNOWN
  - object_hazard_label : HAZARD / SAFE / UNKNOWN — per-object ground truth [D]
  - object_hazard_tier  : CRITICAL / HIGH / MEDIUM / LOW / SAFE / UNKNOWN [D]
  - hazard_tier      : legacy graded hazard (kept for back-compat)
  - is_contextual    : bool — temperature-state-dependent hazard
  - shape            : rectangle / polygon
  - bbox_original    : {x_min, y_min, x_max, y_max} in original image pixels
  - polygon_original : [{x,y}...] for polygon shapes (the Task D mask)
  - temp_min/mean/max_c, temp_delta_c, max_temp : ROI thermal stats (°C)
  - median_depth_meters : depth from stereo disparity + Q matrix [C]
  - region_caption   : caption describing this object/region [C]
  - spatial_relations: [{relation, target_instance_id, distance_meters}] [B,C]
  - annotation_id    : stable UUID per annotation

Scene-level fields:
  - description, ambient_temp_c, lighting, scene_type
  - kitchen_id       : stable PHYSICAL-room id, shared across views — leak-free splits
  - capture_id       : optional finer session id
  - scene_hazard_label / scene_hazard_tier : Task A ground truth (judge by sight)
  - annotator_id     : who labeled — enables inter-annotator agreement
  - scene_caption    : scene-level caption [C]
  - qa_pairs         : [{qa_id, question, answer, answer_type, requires_modality}] [B]
  - T_min/max/mean across full thermal map

IMPORTANT: hazard labels must be judged by LOOKING at the scene, never derived
from temperature — otherwise the benchmark becomes circular.

Run from repo root:
    .venv/bin/streamlit run annotator/annotator_v3.py

From this folder:
    ../.venv/bin/streamlit run annotator_v3.py
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
# Compat shim: streamlit-drawable-canvas 0.9.3 calls
# streamlit.elements.image.image_to_url(image, width, ...), but Streamlit ≥1.31
# removed that symbol (it now lives in streamlit.elements.lib.image_utils with a
# LayoutConfig replacing the old width arg). Without this shim, the canvas's
# efficient background_image path raises, which is why the RGB frame used to be
# embedded as a base64 object and round-tripped ~0.5 MB on every mouse event.
# Restoring the old entry point lets us pass the image via background_image so it
# uploads ONCE and is never sent back — the single biggest win against drawing lag.
import streamlit.elements.image as _st_image_mod
if not hasattr(_st_image_mod, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            lc = _LayoutConfig(width=width if isinstance(width, int) else None)
            return _new_image_to_url(image, lc, clamp, channels, output_format, image_id)

        _st_image_mod.image_to_url = _image_to_url_compat
    except Exception:
        pass  # Fall back to the (slower) embedded-background path below.

_HAS_BG_URL = hasattr(_st_image_mod, "image_to_url")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="RGB-D-T Annotation Tool v2")

ANNOTATOR_DIR           = os.path.dirname(os.path.abspath(__file__))
APP_VERSION             = "v3-unified-schema"
MIN_BBOX_DISPLAY_PIXELS = 8


def find_repo_root(start: str | None = None) -> str:
    """Locate dataset repo root (folder containing Dataset/ or new data/)."""
    cur = os.path.abspath(start or ANNOTATOR_DIR)
    for _ in range(6):
        for name in ("Dataset", "data"):   # tuple of candidate names, not a string
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

# ── Hazard ground-truth taxonomy (judge by SIGHT, never by temperature) ───────
# Binary label used by Task A (scene) and Task D (object).
HAZARD_LABELS   = ["HAZARD", "SAFE", "UNKNOWN"]
# Graded tier; SAFE/LOW -> SAFE, MEDIUM/HIGH/CRITICAL -> HAZARD.
HAZARD_TIERS    = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE", "UNKNOWN"]
# Spatial relations between objects (Tasks B & C).
RELATION_TYPES  = ["near", "far", "on", "above", "below", "beside", "behind", "in_front_of"]
# Answer types for VQA pairs (Task B).
QA_ANSWER_TYPES = ["binary", "object", "count", "spatial", "other"]

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

@st.cache_data
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
        thermal  = os.path.join(sp, "thermal",   "rgb_matched_thermal.npy")
        depth    = os.path.join(sp, "depth",     "depth.npy")
        metadata = os.path.join(sp, "metadata.json")
        if os.path.exists(rgb) and os.path.exists(thermal) and os.path.exists(depth):
            files.append({"name": sd, "rgb": rgb, "thermal": thermal, "depth": depth,
                          "metadata": metadata if os.path.exists(metadata) else None})
    return files

@st.cache_data
def _load_rgb(path):
    return Image.open(path).convert("RGB")

@st.cache_data
def _load_npy(path):
    return np.load(path)

@st.cache_data
def _prepare_disp_image(rgb_path, disp_w, disp_h, show_orig, rgb_boost, rgb_gamma):
    img = Image.open(rgb_path).convert("RGB")
    if show_orig:
        return img.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGBA"), False
    src, was_enh = enhance_low_light(img, rgb_boost, rgb_gamma)
    return src.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGBA"), was_enh

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

@st.cache_data
def _modality_vis(path, cmap, size):
    """Colormapped, resized thermal/depth preview — cached by file path so the
    (now rare) full reruns don't recompute the colormap every time."""
    return cv2.resize(norm_colormap(np.load(path), cmap), (size, size))

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

def stable_canvas(fill, sw, sc, initial, update, h, w, mode, key, bg=None):
    # bg (a PIL image) goes through the component's background_image path: uploaded
    # once to Streamlit's media manager, referenced by URL, and — crucially — NOT
    # serialized back on mouse events. That keeps the per-event payload tiny.
    return st_canvas(fill_color=fill, stroke_width=sw, stroke_color=sc,
                     background_color="", background_image=bg,
                     update_streamlit=update, height=h, width=w,
                     drawing_mode=mode, initial_drawing=initial,
                     display_toolbar=True, point_display_radius=3, key=key)

# ── Fragment shim ─────────────────────────────────────────────────────────────
# st.fragment (stable since Streamlit 1.37, experimental since 1.33) scopes a
# rerun to a single function. Putting the canvas in a fragment means each drawing
# event re-runs ONLY the canvas — not the sidebar, controls, thermal/depth panels,
# or image decoding — which is what kills the flicker. On older Streamlit we fall
# back to a no-op decorator (behaves like before, still correct, just not isolated).
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if fragment is None:
    def fragment(func=None, **_kwargs):
        return func if func is not None else (lambda f: f)

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
    # ── New multi-task fields ────────────────────────────────────────────────
    ann.setdefault("instance_id", "")              # human-readable stable id (Tasks B/C)
    ann.setdefault("object_hazard_label", "UNKNOWN")  # per-object GT (Task D)
    ann.setdefault("object_hazard_tier",  ann.get("hazard_tier", "UNKNOWN"))  # graded (Task D)
    ann.setdefault("region_caption", "")           # per-object caption (Task C)
    ann.setdefault("spatial_relations", [])        # relations to other objects (Tasks B/C)
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
    # ── New multi-task scene fields ──────────────────────────────────────────
    scene_data.setdefault("kitchen_id", "")             # CRITICAL: stable room id for leak-free splits
    scene_data.setdefault("capture_id", "")             # optional finer session id
    scene_data.setdefault("scene_hazard_label", "UNKNOWN")  # Task A ground truth
    scene_data.setdefault("scene_hazard_tier",  "UNKNOWN")  # Task A graded
    scene_data.setdefault("annotator_id", "")           # reliability / agreement
    scene_data.setdefault("scene_caption", "")          # Task C scene-level caption
    scene_data.setdefault("qa_pairs", [])               # Task B VQA pairs
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

# Load annotations only when the file actually changes (skip disk reads on canvas re-runs)
_ann_mtime = os.path.getmtime(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
_ann_cache_key = f"anns__{OUTPUT_FILE}__{_ann_mtime}"
if st.session_state.get("_ann_cache_key") != _ann_cache_key:
    st.session_state["_ann_cache_key"] = _ann_cache_key
    st.session_state["_all_annotations"] = load_annotations(OUTPUT_FILE)
all_annotations = st.session_state["_all_annotations"]

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
orig_img         = _load_rgb(current_scene["rgb"])
orig_w, orig_h   = orig_img.size
disp_w           = max(1, int(orig_w * rgb_zoom/100))
disp_h           = max(1, int(orig_h * rgb_zoom/100))
sx, sy           = orig_w/disp_w, orig_h/disp_h

disp_img, was_enh = _prepare_disp_image(
    current_scene["rgb"], disp_w, disp_h, show_orig, rgb_boost, rgb_gamma
)
disp_img = disp_img.copy()  # detach from cache before in-place modifications

thermal_raw = _load_npy(current_scene["thermal"])
depth_raw   = _load_npy(current_scene["depth"])
Q, Q_err    = load_Q(current_scene["metadata"])

t_min  = float(np.nanmin(thermal_raw))
t_max  = float(np.nanmax(thermal_raw))
t_mean = float(np.nanmean(thermal_raw))

# Fixed 640px for display — st.image + use_column_width will scale to column width
VIS_SIZE = 640
thm_vis  = _modality_vis(current_scene["thermal"], cv2.COLORMAP_INFERNO, VIS_SIZE)
dep_vis  = _modality_vis(current_scene["depth"],   cv2.COLORMAP_VIRIDIS, VIS_SIZE)

# ── Depth coverage boundary: find leftmost column with valid disparity ────────
_depth_w = depth_raw.shape[1]
_valid_cols = np.where(np.any(depth_raw > 0.1, axis=0))[0]
if _valid_cols.size > 0 and _valid_cols[0] > 0:
    # Scale depth-image x → original RGB x → display x
    _no_depth_boundary_orig = int(_valid_cols[0] * orig_w / _depth_w)
    _no_depth_boundary_disp = int(_no_depth_boundary_orig / sx)
    # Shade the no-depth zone and draw a boundary line on the canvas background
    from PIL import ImageDraw
    _overlay = Image.new("RGBA", disp_img.size, (0, 0, 0, 0))
    ImageDraw.Draw(_overlay).rectangle(
        [0, 0, _no_depth_boundary_disp, disp_h - 1], fill=(255, 100, 0, 60)
    )
    disp_img = Image.alpha_composite(disp_img, _overlay)
    ImageDraw.Draw(disp_img).line(
        [(_no_depth_boundary_disp, 0), (_no_depth_boundary_disp, disp_h - 1)],
        fill=(255, 100, 0, 230), width=2,
    )
else:
    _no_depth_boundary_orig = None
    _no_depth_boundary_disp = None

# ── Build canvas ─────────────────────────────────────────────────────────────
# Fast path (_HAS_BG_URL): hand the RGB frame to the canvas via background_image
# (see stable_canvas) so it uploads once and is never serialized back on mouse
# events — this is what removes the drawing lag. Only annotation shapes go into
# initial_drawing. Slow fallback (old Streamlit lacking image_to_url): embed the
# frame as a base64 object, cached in session_state to avoid re-encoding it.
if _HAS_BG_URL:
    canvas_bg   = disp_img.convert("RGB")
    canvas_objs = []
else:
    canvas_bg = None
    _img_cache_key = f"bg_url_{selected_ds}_{sel_scene}_{disp_w}_{disp_h}_{show_orig}_{rgb_zoom}_{rgb_boost}_{rgb_gamma}"
    if _img_cache_key not in st.session_state:
        for k in [k for k in st.session_state if k.startswith("bg_url_")]:
            del st.session_state[k]
        st.session_state[_img_cache_key] = pil_to_data_url(disp_img)
    _bg_url = st.session_state[_img_cache_key]
    canvas_objs = [{"type":"image","version":"4.4.0","originX":"left","originY":"top",
                    "left":0,"top":0,"width":disp_w,"height":disp_h,"fill":"rgb(0,0,0)",
                    "stroke":None,"strokeWidth":0,"scaleX":1,"scaleY":1,"angle":0,
                    "opacity":1,"selectable":False,"evented":False,"hasControls":False,
                    "hasBorders":False,"src":_bg_url,"crossOrigin":None,"filters":[]}]
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

    # Key includes annotation count so the canvas resets cleanly after each save,
    # and canvas size so a zoom change remounts at the right dimensions.
    canvas_key = f"canvas_{selected_ds}_{sel_scene}_{APP_VERSION}_{draw_mode}_{disp_w}x{disp_h}_{len(scene_anns)}"

    # The canvas lives in a fragment so drawing events rerun ONLY this block, not the
    # whole page — that removes the flicker. update=True is still required (without it
    # the drawn shapes never reach Python); but because the rerun is now scoped to the
    # fragment, it's cheap. We stash the latest canvas JSON in session_state so the Save
    # button (which lives outside this fragment, in the control column) can read it.
    @fragment
    def _draw_canvas():
        result = stable_canvas(
            fill="rgba(0,0,0,0)", sw=stroke_width, sc=stroke_color,
            initial=init_draw, update=True,
            h=disp_h, w=disp_w, mode=draw_mode, key=canvas_key,
            bg=canvas_bg,
        )
        st.session_state["_canvas_json"] = result.json_data if result is not None else None

    _draw_canvas()

    if _no_depth_boundary_orig is not None:
        st.caption(
            f"🟠 Orange shaded region (left of x={_no_depth_boundary_orig} px): "
            "no depth data available here — annotations in this zone will have depth = 0 m."
        )


# ── Control panel ────────────────────────────────────────────────────────────
with ctrl_col:
    st.subheader("Annotation Controls")

    # ── Per-object annotation fields ─────────────────────────────────────────
    st.write("**Object Annotation**")
    all_classes = get_all_classes()

    # Add-new-class lives OUTSIDE the form: a form can't contain st.button, and the
    # class list must refresh immediately after adding. Rare action → collapsed.
    with st.expander("➕ Add a new object class", expanded=False):
        new_class_raw = st.text_input(
            "New class name", placeholder="e.g., toaster, heat_gun, soldering_iron",
            key=f"new_cls_{selected_ds}_{sel_scene}")
        if st.button("✅ Add class", key=f"add_cls_{selected_ds}_{sel_scene}"):
            new_class = re.sub(r"[^a-zA-Z0-9]+", "_", new_class_raw.strip()).strip("_").lower()
            if not new_class:
                st.error("Please enter a class name.")
            elif new_class in get_all_classes():
                st.warning(f"'{new_class}' already exists — select it below.")
            else:
                custom = load_custom_classes(); custom.append(new_class)
                save_custom_classes(custom)
                st.success(f"'{new_class}' added — select it below.")
                st.rerun()

    # Spatial relations editor (Tasks B & C) — also OUTSIDE the form (uses buttons).
    # Staged relations are stored in session_state and read when the annotation saves.
    with st.expander("🔗 Spatial relations", expanded=False):
        st.caption("Relations FROM this object to others (e.g. heat source → nearby cloth). "
                   "Only annotate relations that matter.")
        existing_targets = [a.get("instance_id","") for a in scene_anns if a.get("instance_id")]
        rel_type = st.selectbox("Relation", RELATION_TYPES,
                                key=f"reltype_{selected_ds}_{sel_scene}")
        rel_c1, rel_c2 = st.columns(2)
        if existing_targets:
            rel_target = rel_c1.selectbox(
                "Target instance", ["(type manually)"] + existing_targets,
                key=f"reltgtsel_{selected_ds}_{sel_scene}")
            if rel_target == "(type manually)":
                rel_target = rel_c1.text_input("Target instance id",
                                               key=f"reltgt_{selected_ds}_{sel_scene}")
        else:
            rel_target = rel_c1.text_input("Target instance id",
                                           placeholder="other object's instance_id",
                                           key=f"reltgt_{selected_ds}_{sel_scene}")
        rel_dist = rel_c2.number_input("Distance (m)", min_value=0.0, value=0.0, step=0.05,
                                       help="Metric distance between the objects, if known. 0 = unknown.",
                                       key=f"reldist_{selected_ds}_{sel_scene}")
        if "pending_relations" not in st.session_state:
            st.session_state.pending_relations = {}
        pr_key = f"{selected_ds}_{sel_scene}"
        st.session_state.pending_relations.setdefault(pr_key, [])
        if st.button("➕ Stage relation", key=f"addrel_{selected_ds}_{sel_scene}"):
            if rel_target and str(rel_target).strip():
                st.session_state.pending_relations[pr_key].append({
                    "relation": rel_type,
                    "target_instance_id": str(rel_target).strip(),
                    "distance_meters": (rel_dist if rel_dist > 0 else None),
                })
                st.rerun()
        staged = st.session_state.pending_relations.get(pr_key, [])
        if staged:
            st.caption("Staged (saved with the next 💾 Save Annotation):")
            for ri, r in enumerate(staged):
                d = r.get("distance_meters")
                dstr = f" @ {d}m" if d else ""
                rr1, rr2 = st.columns([5,1])
                rr1.markdown(f"<small>{r['relation']} → {r['target_instance_id']}{dstr}</small>",
                             unsafe_allow_html=True)
                if rr2.button("✕", key=f"rmrel_{selected_ds}_{sel_scene}_{ri}"):
                    st.session_state.pending_relations[pr_key].pop(ri)
                    st.rerun()

    # ── Save form ─────────────────────────────────────────────────────────────
    # Filling these fields causes NO rerun and NO canvas redraw — the page only
    # reruns when you press Save. This removes the per-widget flicker you saw while
    # choosing class / state / hazard on every keystroke or selection.
    with st.form(key=f"ann_form_{selected_ds}_{sel_scene}", clear_on_submit=False):
        obj_class = st.selectbox("Object Class", all_classes,
                                 key=f"cls_{selected_ds}_{sel_scene}")
        obj_state = st.selectbox("Object State", OBJECT_STATES,
                                 key=f"state_{selected_ds}_{sel_scene}")
        is_ctx    = st.checkbox("Contextual hazard (state-dependent)", value=False,
                                key=f"ctx_{selected_ds}_{sel_scene}",
                                help="Check if this object is only dangerous above a certain temperature (e.g. a mug, laptop).")
        instance_id = st.text_input(
            "Instance ID", placeholder="e.g. stove_007_burner_frontleft",
            help="Human-readable stable id for THIS physical object. Keep it consistent "
                 "across the left/right views. Define front/back/left/right RELATIVE TO "
                 "THE STOVE, not the camera. Used by Tasks B and C.",
            key=f"iid_{selected_ds}_{sel_scene}")
        oh_a, oh_b = st.columns(2)
        obj_haz_label = oh_a.selectbox(
            "Object hazard (Task D)", HAZARD_LABELS,
            help="Per-object ground truth for segmentation. A lit burner = HAZARD; the "
                 "cold burner beside it = SAFE. Judge by sight, not temperature.",
            key=f"ohl_{selected_ds}_{sel_scene}")
        obj_haz_tier = oh_b.selectbox(
            "Object tier", HAZARD_TIERS,
            help="Optional graded per-object hazard.",
            key=f"oht_{selected_ds}_{sel_scene}")
        region_caption = st.text_area(
            "Region caption (Task C)", placeholder="A lit gas burner glowing at ~115°C "
            "at the front-left of the stovetop with a pot on it.",
            height=70, key=f"rcap_{selected_ds}_{sel_scene}",
            help="A factual sentence describing THIS object/region for dense captioning.")
        submitted = st.form_submit_button("💾 Save Annotation", type="primary",
                                          width="stretch")

    if submitted:
        raw_objs = (st.session_state.get("_canvas_json") or {}).get("objects", [])
        # Collect staged spatial relations for this scene, then clear them.
        _pr_key = f"{selected_ds}_{sel_scene}"
        staged_relations = st.session_state.get("pending_relations", {}).get(_pr_key, [])

        if not obj_class:
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
                "instance_id":         instance_id.strip(),
                "object_state":        obj_state,
                "object_hazard_label": obj_haz_label,
                "object_hazard_tier":  obj_haz_tier,
                "is_contextual":       is_ctx,
                "shape":               "rectangle",
                "bbox_original":       {"x_min":x0,"y_min":y0,"x_max":x1,"y_max":y1},
                "temp_min_c":          roi_min,
                "temp_mean_c":         roi_mean,
                "temp_max_c":          roi_max,
                "temp_delta_c":        roi_delt,
                "max_temp":            roi_max,
                "median_depth_meters": safe_float(depth_m),
                "region_caption":      region_caption.strip(),
                "spatial_relations":   staged_relations,
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
                "instance_id":         instance_id.strip(),
                "object_state":        obj_state,
                "object_hazard_label": obj_haz_label,
                "object_hazard_tier":  obj_haz_tier,
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
                "region_caption":      region_caption.strip(),
                "spatial_relations":   staged_relations,
            }

        cur  = load_annotations(OUTPUT_FILE)
        blob = migrate_scene(get_scene_data(cur, sel_scene))
        blob["annotations"].append(ann_entry)
        blob.update({"thermal_scene_min_c":t_min,"thermal_scene_max_c":t_max,
                      "thermal_scene_mean_c":t_mean})
        cur[sel_scene] = blob
        save_annotations_atomic(cur, OUTPUT_FILE)
        # Staged relations have been persisted on this object — clear them.
        if "pending_relations" in st.session_state:
            st.session_state.pending_relations[_pr_key] = []
        st.rerun()

    # ── Saved annotations list ───────────────────────────────────────────────
    st.markdown("---")
    st.write("**Saved Objects:**")
    fresh_anns = scene_blob.get("annotations", [])
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
            haz   = ann.get("object_hazard_label","UNKNOWN")
            haz_icon = {"HAZARD":"🔴","SAFE":"🟢"}.get(haz, "⚪")
            iid   = ann.get("instance_id","")
            iid_str = f" · {iid}" if iid else ""
            cap_flag = "📝" if ann.get("region_caption") else ""
            nrel = len(ann.get("spatial_relations", []))
            rel_str = f" · 🔗{nrel}" if nrel else ""
            a_col, d_col = st.columns([4,1])
            with a_col:
                st.markdown(
                    f"<span style='color:{color}'>{shape_icon}</span> {haz_icon} **{cls}** [{state}]{iid_str}<br>"
                    f"<small>{tc:.1f}°C{dcstr} · {dep:.2f}m{rel_str} {cap_flag}</small>",
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

    # ── Scene-level metadata ─────────────────────────────────────────────────
    with st.expander("📋 Scene Metadata", expanded=True):
        desc_val = st.text_area("Description", value=scene_blob.get("description",""),
                                placeholder="What is in this scene?", height=80,
                                key=f"desc_{selected_ds}_{sel_scene}")

        # Identity / split-safety
        id_a, id_b = st.columns(2)
        kitchen_id = id_a.text_input(
            "Kitchen ID ⚠️", value=scene_blob.get("kitchen_id",""),
            placeholder="e.g. kitchen_007",
            help="CRITICAL. Stable id for the PHYSICAL room. Share the SAME id across "
                 "the left/right views and any other shots of this kitchen. Enables "
                 "leak-free train/test splits.",
            key=f"kid_{selected_ds}_{sel_scene}")
        capture_id = id_b.text_input(
            "Capture ID", value=scene_blob.get("capture_id",""),
            placeholder="e.g. kitchen_007_cap_03",
            help="Optional finer id for one capture session within a kitchen.",
            key=f"cid_{selected_ds}_{sel_scene}")

        # Scene hazard ground truth (Task A) — judge by sight, not temperature
        hz_a, hz_b = st.columns(2)
        scene_haz_label = hz_a.selectbox(
            "Scene hazard (Task A)", HAZARD_LABELS,
            index=HAZARD_LABELS.index(scene_blob.get("scene_hazard_label","UNKNOWN"))
                  if scene_blob.get("scene_hazard_label") in HAZARD_LABELS else len(HAZARD_LABELS)-1,
            help="Ground truth for Task A. Judge by LOOKING at the whole scene, "
                 "NOT by reading temperatures (that would make the benchmark circular).",
            key=f"shl_{selected_ds}_{sel_scene}")
        scene_haz_tier = hz_b.selectbox(
            "Scene tier", HAZARD_TIERS,
            index=HAZARD_TIERS.index(scene_blob.get("scene_hazard_tier","UNKNOWN"))
                  if scene_blob.get("scene_hazard_tier") in HAZARD_TIERS else len(HAZARD_TIERS)-1,
            help="Optional graded version of the scene hazard label.",
            key=f"sht_{selected_ds}_{sel_scene}")

        # Conditions
        col_a, col_b = st.columns(2)
        amb_temp  = col_a.number_input("Ambient °C", value=float(scene_blob.get("ambient_temp_c") or 22.0),
                                        step=0.5, key=f"amb_{selected_ds}_{sel_scene}")
        lighting  = col_b.selectbox("Lighting", LIGHTING_CONDS,
                                     index=LIGHTING_CONDS.index(scene_blob.get("lighting","well_lit"))
                                     if scene_blob.get("lighting") in LIGHTING_CONDS else 0,
                                     key=f"light_{selected_ds}_{sel_scene}")
        st_a, st_b = st.columns(2)
        scene_type = st_a.selectbox("Scene type", SCENE_TYPES,
                                   index=SCENE_TYPES.index(scene_blob.get("scene_type","kitchen"))
                                   if scene_blob.get("scene_type") in SCENE_TYPES else 0,
                                   key=f"stype_{selected_ds}_{sel_scene}")
        annotator_id = st_b.text_input(
            "Annotator ID", value=scene_blob.get("annotator_id",""),
            placeholder="e.g. annot_02",
            help="Who labeled this scene. Enables inter-annotator agreement.",
            key=f"aid_{selected_ds}_{sel_scene}")

        # Scene caption (Task C)
        scene_caption = st.text_area(
            "Scene caption (Task C)", value=scene_blob.get("scene_caption",""),
            placeholder="One or two factual sentences describing the whole scene.",
            height=70, key=f"scap_{selected_ds}_{sel_scene}")

        if st.button("Save Scene Metadata", width="stretch"):
            cur  = load_annotations(OUTPUT_FILE)
            blob = migrate_scene(get_scene_data(cur, sel_scene))
            blob.update({"description": desc_val, "ambient_temp_c": amb_temp,
                          "lighting": lighting, "scene_type": scene_type,
                          "kitchen_id": kitchen_id.strip(), "capture_id": capture_id.strip(),
                          "scene_hazard_label": scene_haz_label, "scene_hazard_tier": scene_haz_tier,
                          "annotator_id": annotator_id.strip(), "scene_caption": scene_caption,
                          "thermal_scene_min_c": t_min, "thermal_scene_max_c": t_max,
                          "thermal_scene_mean_c": t_mean})
            cur[sel_scene] = blob
            save_annotations_atomic(cur, OUTPUT_FILE)
            st.success("Scene metadata saved.")

    # ── Task B: VQA pair editor ──────────────────────────────────────────────
    with st.expander("❓ VQA Pairs (Task B)", expanded=False):
        existing_qa = scene_blob.get("qa_pairs", [])
        if existing_qa:
            for qi, qa in enumerate(existing_qa):
                q_txt = qa.get("question","")
                a_txt = qa.get("answer","")
                mods  = ",".join(qa.get("requires_modality", []))
                qrow, drow = st.columns([5,1])
                qrow.markdown(f"<small><b>Q{qi+1}:</b> {q_txt}<br><b>A:</b> {a_txt} "
                              f"<i>({qa.get('answer_type','?')}; needs: {mods or 'none'})</i></small>",
                              unsafe_allow_html=True)
                if drow.button("🗑", key=f"delqa_{selected_ds}_{sel_scene}_{qi}"):
                    cur  = load_annotations(OUTPUT_FILE)
                    blob = migrate_scene(get_scene_data(cur, sel_scene))
                    if qi < len(blob["qa_pairs"]):
                        blob["qa_pairs"].pop(qi)
                        cur[sel_scene] = blob
                        save_annotations_atomic(cur, OUTPUT_FILE)
                        st.rerun()
        else:
            st.caption("No QA pairs yet.")

        st.markdown("**Add a QA pair**")
        new_q = st.text_input("Question", key=f"newq_{selected_ds}_{sel_scene}",
                              placeholder="Is the cloth too close to the heat source?")
        qa_c1, qa_c2 = st.columns(2)
        new_a = qa_c1.text_input("Answer", key=f"newa_{selected_ds}_{sel_scene}",
                                placeholder="Yes")
        new_atype = qa_c2.selectbox("Answer type", QA_ANSWER_TYPES,
                                    key=f"newatype_{selected_ds}_{sel_scene}")
        new_mods = st.multiselect("Requires modality", ["rgb","thermal","depth"],
                                  default=[],
                                  help="Which modalities a human needs to answer. "
                                       "Tagging thermal/depth-only questions is what proves modality value.",
                                  key=f"newmods_{selected_ds}_{sel_scene}")
        if st.button("➕ Add QA pair", width="stretch",
                     key=f"addqa_{selected_ds}_{sel_scene}"):
            if not new_q.strip() or not new_a.strip():
                st.error("Both question and answer are required.")
            else:
                cur  = load_annotations(OUTPUT_FILE)
                blob = migrate_scene(get_scene_data(cur, sel_scene))
                blob.setdefault("qa_pairs", []).append({
                    "qa_id": f"{sel_scene}_qa_{len(blob.get('qa_pairs',[]))+1}",
                    "question": new_q.strip(),
                    "answer": new_a.strip(),
                    "answer_type": new_atype,
                    "requires_modality": new_mods,
                })
                cur[sel_scene] = blob
                save_annotations_atomic(cur, OUTPUT_FILE)
                st.rerun()

    st.markdown("---")
    st.write("**Navigation**")
    pr_col, nx_col = st.columns(2)
    with pr_col:
        if st.button("✅ Complete", width="stretch"):
            if sel_scene not in completed_scenes:
                completed_scenes.append(sel_scene)
                save_progress(completed_scenes, PROGRESS_FILE)
            go_next(); st.rerun()
    with nx_col:
        st.button("→ Next", width="stretch", on_click=go_next)

    # Progress bar. Count only completed scenes that exist in THIS dataset, otherwise a
    # progress file carried over from a larger dataset makes done/total exceed 1.0 and
    # st.progress raises (it requires a value in [0.0, 1.0]).
    total = len(dataset_files)
    done  = sum(1 for s in completed_scenes if s in scene_names)
    st.progress(min(1.0, done/total) if total else 0, text=f"{done}/{total} scenes completed")

# ── Modality views ───────────────────────────────────────────────────────────
st.markdown("---")
v1, v2 = st.columns(2)
with v1:
    st.subheader("🌡 Thermal (Inferno)")
    valid_d = depth_raw[depth_raw > 0.1]
    st.image(thm_vis, width="stretch",
             caption=f"min {t_min:.1f}°C  ·  max {t_max:.1f}°C  ·  ΔT {t_max-t_min:.1f}°C")
with v2:
    st.subheader("📏 Depth / Disparity (Viridis)")
    d_lo = float(valid_d.min()) if valid_d.size else 0.
    d_hi = float(valid_d.max()) if valid_d.size else 0.
    st.image(dep_vis, width="stretch",
             caption=f"Disparity  {d_lo:.1f} – {d_hi:.1f} px  ·  bright = closer")