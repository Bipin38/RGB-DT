"""
RGB-D-T annotation — core logic (no Streamlit).
====================================================
Pure functions extracted verbatim from annotator_v3.py so the fast FastAPI
backend (server.py) produces byte-identical thermal/depth stats and the exact
same JSON schema. Do NOT import streamlit here.

The one genuinely new function is `compute_roi_stats`, which lifts the inline
ROI math out of the old Streamlit save block (annotator_v3.py:843-856 for
rectangles, :905-921 for polygons) into a single reusable function.
"""

import functools
import json
import math
import os
import re
import uuid

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors legacy/annotator_v3.py)
# ─────────────────────────────────────────────────────────────────────────────
# core.py lives in <project>/app/ ; data (Dataset/, annotations/, *.json) lives
# at the project root one level up. PROJECT_ROOT can be overridden via env for
# non-standard layouts.
PACKAGE_DIR             = os.path.dirname(os.path.abspath(__file__))          # .../app
PROJECT_ROOT            = os.environ.get("RGBDT_ROOT") or os.path.dirname(PACKAGE_DIR)
APP_VERSION             = "v4-web"
MIN_BBOX_DISPLAY_PIXELS = 8

HAZARD_CLASSES = [
    "stove_burner", "iron", "oven_door", "kettle", "hot_surface",
    "electrical_outlet", "extension_cord", "space_heater", "radiator",
    "laptop_vent", "mug_cup", "other_hazard", "safe_object",
]
OBJECT_STATES   = ["ON", "OFF", "COOLING", "PLUGGED_IN", "ACTIVE", "UNKNOWN"]
LIGHTING_CONDS  = ["well_lit", "dim", "dark", "mixed", "artificial", "natural"]
SCENE_TYPES     = ["kitchen", "bathroom", "living_room", "bedroom", "office",
                   "laundry", "garage", "other"]
BOX_COLORS      = ["#00FF00", "#FF4444", "#00FFFF", "#FFD700", "#FF00FF",
                   "#4488FF", "#FFA500", "#FFFFFF", "#FF69B4", "#7FFF00"]
HAZARD_LABELS   = ["HAZARD", "SAFE", "UNKNOWN"]
HAZARD_TIERS    = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE", "UNKNOWN"]
RELATION_TYPES  = ["near", "far", "on", "above", "below", "beside", "behind", "in_front_of"]
QA_ANSWER_TYPES = ["binary", "object", "count", "spatial", "other"]


REPO_ROOT = PROJECT_ROOT   # dataset folders (e.g. Dataset/) live directly under here
BASE_DIR  = PROJECT_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def file_slug(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "dataset"


def safe_float(val):
    if val is None:
        return None
    try:
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Custom class persistence
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CLASSES_FILE = os.path.join(PROJECT_ROOT, "custom_classes.json")


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
    custom = load_custom_classes()
    seen = set(HAZARD_CLASSES)
    extras = [c for c in custom if c not in seen]
    return HAZARD_CLASSES + extras


def add_custom_class(raw):
    """Normalize + append a custom class. Returns (name, error)."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", (raw or "").strip()).strip("_").lower()
    if not name:
        return None, "empty class name"
    if name in get_all_classes():
        return name, "already exists"
    custom = load_custom_classes()
    custom.append(name)
    save_custom_classes(custom)
    return name, None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────
def _scene_valid(path):
    return (os.path.exists(os.path.join(path, "left_cam", "left.png")) and
            os.path.exists(os.path.join(path, "thermal", "rgb_matched_thermal.npy")) and
            os.path.exists(os.path.join(path, "depth", "depth.npy")))


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


def get_dataset_files(data_dir):
    files = []
    if not os.path.exists(data_dir):
        return files
    for sd in sorted(os.listdir(data_dir), key=natural_sort_key):
        sp = os.path.join(data_dir, sd)
        if not os.path.isdir(sp):
            continue
        rgb      = os.path.join(sp, "left_cam", "left.png")
        thermal  = os.path.join(sp, "thermal", "rgb_matched_thermal.npy")
        depth    = os.path.join(sp, "depth", "depth.npy")
        metadata = os.path.join(sp, "metadata.json")
        if os.path.exists(rgb) and os.path.exists(thermal) and os.path.exists(depth):
            files.append({"name": sd, "rgb": rgb, "thermal": thermal, "depth": depth,
                          "metadata": metadata if os.path.exists(metadata) else None})
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Loading / visualization (lru_cache replaces st.cache_data)
# ─────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def load_rgb(path):
    return Image.open(path).convert("RGB")


@functools.lru_cache(maxsize=32)
def load_npy(path):
    return np.load(path)


def norm_colormap(arr, cmap):
    v = np.nan_to_num(arr, nan=0., posinf=0., neginf=0.)
    n = (np.zeros_like(v, dtype=np.uint8) if v.max() == v.min()
         else cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    return cv2.cvtColor(cv2.applyColorMap(n, cmap), cv2.COLOR_BGR2RGB)


@functools.lru_cache(maxsize=64)
def modality_vis(path, cmap, size):
    """Colormapped, resized thermal/depth preview, cached by file path."""
    return cv2.resize(norm_colormap(np.load(path), cmap), (size, size))


def enhance_low_light(img, boost=1., gamma=1.):
    arr = np.asarray(img.convert("RGB"))
    should = arr.mean() < 55 or np.percentile(arr, 99) < 150
    out = arr.copy()
    if should:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
        out = cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        lo, hi = np.percentile(out, 1), np.percentile(out, 99)
        if hi > lo:
            out = np.clip((out.astype(np.float32) - lo) * (255. / (hi - lo)), 0, 255).astype(np.uint8)
        out = np.clip(out.astype(np.float32) * min(4.5, 135. / max(out.mean(), 1.)), 0, 255).astype(np.uint8)
    if boost != 1.:
        out = np.clip(out.astype(np.float32) * boost, 0, 255).astype(np.uint8)
    if gamma != 1.:
        out = np.clip((out.astype(np.float32) / 255.) ** gamma * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), (should or boost != 1. or gamma != 1.)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / depth
# ─────────────────────────────────────────────────────────────────────────────
def get_annotation_bbox(ann):
    if "bbox_original" in ann:
        return ann["bbox_original"]
    poly = ann.get("polygon_original", [])
    if not poly:
        return None
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}


def load_Q(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None, "metadata.json not found — depth disabled."
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        Q = np.array(meta["calibration"]["Q"], dtype=np.float64)
        if Q.shape != (4, 4):
            return None, f"Q shape {Q.shape} != (4,4)"
        return Q, None
    except Exception as e:
        return None, str(e)


def disp_to_depth(roi, Q, cx, cy):
    valid = roi[roi > 0.1]
    if valid.size == 0:
        return 0.0, "No valid disparity in ROI — depth = 0 m."
    pt = Q @ np.array([cx, cy, float(np.median(valid)), 1.0])
    return float(pt[2] / pt[3] / 1000.), None


def no_depth_boundary(depth_raw, orig_w):
    """Leftmost original-x column with valid disparity (or None)."""
    depth_w = depth_raw.shape[1]
    valid_cols = np.where(np.any(depth_raw > 0.1, axis=0))[0]
    if valid_cols.size > 0 and valid_cols[0] > 0:
        return int(valid_cols[0] * orig_w / depth_w)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ROI thermal/depth stats  (the one new function — exact port of the save block)
# ─────────────────────────────────────────────────────────────────────────────
def compute_roi_stats(thermal_raw, depth_raw, Q, shape, bbox=None, polygon=None,
                      scene_ambient_c=None):
    """
    Compute per-object thermal + depth stats, byte-identical to annotator_v3.py.

    shape: "rectangle" | "polygon"
    bbox:  {x_min,y_min,x_max,y_max} in ORIGINAL image pixels (required)
    polygon: [{x,y}, ...] in original pixels (required when shape == "polygon")
    scene_ambient_c: scene ambient temp (falls back to thermal-scene-min).

    Returns a dict of the temp_*/max_temp/median_depth_meters fields plus an
    optional "depth_warning" string.
    """
    orig_h, orig_w = thermal_raw.shape[:2]
    t_min = float(np.nanmin(thermal_raw))
    amb = safe_float(scene_ambient_c)
    if amb is None:
        amb = t_min

    x0 = int(bbox["x_min"]); y0 = int(bbox["y_min"])
    x1 = int(bbox["x_max"]); y1 = int(bbox["y_max"])
    x0 = max(0, min(orig_w, x0)); x1 = max(0, min(orig_w, x1))
    y0 = max(0, min(orig_h, y0)); y1 = max(0, min(orig_h, y1))

    if shape == "polygon" and polygon:
        # Mask-based thermal stats (only pixels inside the polygon).
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        pts_arr = np.array([[int(p["x"]), int(p["y"])] for p in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], 1)
        t_roi = thermal_raw[mask == 1]
        roi_min  = safe_float(np.nanmin(t_roi))  if t_roi.size else None
        roi_mean = safe_float(np.nanmean(t_roi)) if t_roi.size else None
        roi_max  = safe_float(np.nanmax(t_roi))  if t_roi.size else None
        xs = [int(p["x"]) for p in polygon]
        ys = [int(p["y"]) for p in polygon]
        cx = float(np.mean(xs)); cy = float(np.mean(ys))
    else:
        t_roi = thermal_raw[y0:y1, x0:x1]
        roi_min  = safe_float(np.nanmin(t_roi))  if t_roi.size else None
        roi_mean = safe_float(np.nanmean(t_roi)) if t_roi.size else None
        roi_max  = safe_float(np.nanmax(t_roi))  if t_roi.size else None
        cx = x0 + (x1 - x0) / 2.
        cy = y0 + (y1 - y0) / 2.

    roi_delt = safe_float((roi_max or 0.) - amb) if roi_max else None

    depth_m, dw = 0., None
    if Q is not None:
        depth_m, dw = disp_to_depth(depth_raw[y0:y1, x0:x1], Q, cx, cy)
    else:
        dw = "Depth not computed — no Q matrix."

    return {
        "temp_min_c":  roi_min,
        "temp_mean_c": roi_mean,
        "temp_max_c":  roi_max,
        "temp_delta_c": roi_delt,
        "max_temp":    roi_max,
        "median_depth_meters": safe_float(depth_m),
        "depth_warning": dw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schema migration (verbatim from annotator_v3.py)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_MIGRATION = {
    "burner 1": "stove_burner", "burner 2": "stove_burner",
    "burner 3": "stove_burner", "burner 4": "stove_burner",
    "burners": "stove_burner", "burner": "stove_burner",
    "iron": "iron", "kettle": "kettle", "oven": "oven_door",
    "outlet": "electrical_outlet", "heater": "space_heater",
    "radiator": "radiator", "laptop": "laptop_vent", "mug": "mug_cup",
}


def migrate_annotation(ann):
    if "annotation_id" not in ann:
        ann["annotation_id"] = str(uuid.uuid4())
    old_cls = ann.get("class", "").lower().strip()
    ann["object_class"] = CLASS_MIGRATION.get(old_cls, ann.get("object_class", old_cls or "other_hazard"))
    ann.setdefault("object_state", "UNKNOWN")
    ann.setdefault("hazard_tier", "UNKNOWN")
    ann.setdefault("is_contextual", False)
    ann.setdefault("shape", "rectangle")
    ann.setdefault("temp_max_c", ann.get("max_temp"))
    ann.setdefault("temp_min_c", None)
    ann.setdefault("temp_mean_c", None)
    ann.setdefault("temp_delta_c", None)
    ann.setdefault("max_temp", ann.get("temp_max_c"))
    ann.setdefault("median_depth_meters", None)
    ann.setdefault("instance_id", "")
    ann.setdefault("object_hazard_label", "UNKNOWN")
    ann.setdefault("object_hazard_tier", ann.get("hazard_tier", "UNKNOWN"))
    ann.setdefault("region_caption", "")
    ann.setdefault("spatial_relations", [])
    return ann


def migrate_scene(scene_data):
    if isinstance(scene_data, list):
        scene_data = {"description": "", "annotations": scene_data}
    scene_data.setdefault("description", "")
    scene_data.setdefault("ambient_temp_c", None)
    scene_data.setdefault("lighting", "UNKNOWN")
    scene_data.setdefault("scene_type", "UNKNOWN")
    scene_data.setdefault("thermal_scene_min_c", None)
    scene_data.setdefault("thermal_scene_max_c", None)
    scene_data.setdefault("thermal_scene_mean_c", None)
    scene_data.setdefault("kitchen_id", "")
    scene_data.setdefault("capture_id", "")
    scene_data.setdefault("scene_hazard_label", "UNKNOWN")
    scene_data.setdefault("scene_hazard_tier", "UNKNOWN")
    scene_data.setdefault("annotator_id", "")
    scene_data.setdefault("scene_caption", "")
    scene_data.setdefault("qa_pairs", [])
    scene_data["annotations"] = [migrate_annotation(a) for a in scene_data.get("annotations", [])]
    return scene_data


def get_scene_data(all_ann, scene):
    d = all_ann.get(scene, {})
    if isinstance(d, dict):
        return d
    return {"description": "", "annotations": d}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json_atomic(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
