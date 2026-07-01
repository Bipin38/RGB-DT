"""
RGB-D-T annotation — fast web backend (FastAPI).
====================================================
Replaces the laggy Streamlit tool. Drawing happens 100% client-side in the
browser (static/app.js); this server is contacted only to:
  • list scenes / classes                 (GET  /api/config)
  • load one scene + its images           (GET  /api/scene/{name} + /image/...)
  • compute live ROI temp/depth on a box  (POST /api/scene/{name}/roi)
  • persist a scene's annotations          (POST /api/scene/{name})

All thermal/depth math and the JSON schema are reused verbatim from
app/core.py, so outputs match annotator_v3.py exactly.

Storage: one file per scene under  annotations/<dataset_slug>/<scene>.json
(retires the monolithic annotations_dataset.json for conflict-free team merges).

Run:
    .venv/bin/uvicorn server:app --port 8000       # from this folder
Then open http://localhost:8000
"""

import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app import core

app = FastAPI(title="RGB-D-T Annotator")

STATIC_DIR = os.path.join(core.PACKAGE_DIR, "static")        # app/static (bundled)
ANN_ROOT   = os.path.join(core.PROJECT_ROOT, "annotations")  # per-scene output at root


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / scene helpers
# ─────────────────────────────────────────────────────────────────────────────
def _active_dataset():
    """Pick the dataset folder (env override or first discovered)."""
    override = os.environ.get("RGBDT_DATASET")
    available = core.get_available_dataset_names(core.BASE_DIR)
    if not available:
        raise HTTPException(500, f"No dataset folders under {core.BASE_DIR}")
    if override and override in available:
        return override
    return available[0]


def _scene_files():
    ds = _active_dataset()
    data_dir = os.path.join(core.REPO_ROOT, ds)
    return ds, {f["name"]: f for f in core.get_dataset_files(data_dir)}


def _ann_dir(ds):
    return os.path.join(ANN_ROOT, core.file_slug(ds))


def _ann_path(ds, scene):
    return os.path.join(_ann_dir(ds), f"{scene}.json")


def _load_scene_blob(ds, scene):
    """Per-scene file first; fall back to the legacy monolithic file."""
    path = _ann_path(ds, scene)
    if os.path.exists(path):
        return core.migrate_scene(core.load_json(path, {}))
    legacy = os.path.join(core.PROJECT_ROOT, f"annotations_{core.file_slug(ds)}.json")
    mono = core.load_json(legacy, {})
    return core.migrate_scene(core.get_scene_data(mono, scene))


def _scene_status(ds, scene):
    path = _ann_path(ds, scene)
    if not os.path.exists(path):
        return {"annotated": 0, "completed": False}
    blob = core.load_json(path, {})
    return {"annotated": len(blob.get("annotations", [])),
            "completed": bool(blob.get("completed", False))}


# ─────────────────────────────────────────────────────────────────────────────
# Config / listing
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    ds, files = _scene_files()
    scenes = [{"name": n, **_scene_status(ds, n)} for n in files]
    return {
        "dataset": ds,
        "scenes": scenes,
        "classes": core.get_all_classes(),
        "enums": {
            "object_states": core.OBJECT_STATES,
            "hazard_labels": core.HAZARD_LABELS,
            "hazard_tiers": core.HAZARD_TIERS,
            "lighting": core.LIGHTING_CONDS,
            "scene_types": core.SCENE_TYPES,
            "relations": core.RELATION_TYPES,
            "qa_answer_types": core.QA_ANSWER_TYPES,
        },
        "box_colors": core.BOX_COLORS,
    }


class NewClass(BaseModel):
    name: str


@app.post("/api/classes")
def add_class(body: NewClass):
    name, err = core.add_custom_class(body.name)
    if err == "empty class name":
        raise HTTPException(400, err)
    return {"name": name, "classes": core.get_all_classes(), "warning": err}


# ─────────────────────────────────────────────────────────────────────────────
# Scene load
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/scene/{scene}")
def get_scene(scene: str):
    ds, files = _scene_files()
    if scene not in files:
        raise HTTPException(404, f"Unknown scene {scene}")
    f = files[scene]
    rgb = core.load_rgb(f["rgb"])
    orig_w, orig_h = rgb.size
    thermal = core.load_npy(f["thermal"])
    depth = core.load_npy(f["depth"])
    Q, q_err = core.load_Q(f["metadata"])

    blob = _load_scene_blob(ds, scene)
    return {
        "name": scene,
        "width": orig_w,
        "height": orig_h,
        "q_available": Q is not None,
        "q_error": q_err,
        "no_depth_boundary_x": core.no_depth_boundary(depth, orig_w),
        "thermal": {
            "min_c": float(np.nanmin(thermal)),
            "max_c": float(np.nanmax(thermal)),
            "mean_c": float(np.nanmean(thermal)),
        },
        "annotations": blob.get("annotations", []),
        "scene_meta": {k: blob.get(k) for k in (
            "description", "ambient_temp_c", "lighting", "scene_type",
            "kitchen_id", "capture_id", "scene_hazard_label", "scene_hazard_tier",
            "annotator_id", "scene_caption", "qa_pairs",
        )},
        "completed": bool(blob.get("completed", False)),
    }


def _png(arr):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/scene/{scene}/image/{modality}")
def get_image(scene: str, modality: str,
              enhance: int = Query(0), boost: float = Query(1.0), gamma: float = Query(1.0)):
    ds, files = _scene_files()
    if scene not in files:
        raise HTTPException(404, f"Unknown scene {scene}")
    f = files[scene]
    if modality == "rgb":
        if enhance:
            img, _ = core.enhance_low_light(core.load_rgb(f["rgb"]), boost, gamma)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        return FileResponse(f["rgb"], media_type="image/png")
    if modality == "thermal":
        return _png(core.modality_vis(f["thermal"], cv2.COLORMAP_INFERNO, 640))
    if modality == "depth":
        return _png(core.modality_vis(f["depth"], cv2.COLORMAP_VIRIDIS, 640))
    raise HTTPException(404, f"Unknown modality {modality}")


# ─────────────────────────────────────────────────────────────────────────────
# Live ROI stats (called the instant a box is drawn)
# ─────────────────────────────────────────────────────────────────────────────
class BBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Point(BaseModel):
    x: int
    y: int


class ROIRequest(BaseModel):
    shape: str
    bbox: BBox
    polygon: list[Point] | None = None
    ambient_temp_c: float | None = None


def _compute(f, req: ROIRequest):
    thermal = core.load_npy(f["thermal"])
    depth = core.load_npy(f["depth"])
    Q, _ = core.load_Q(f["metadata"])
    return core.compute_roi_stats(
        thermal, depth, Q, req.shape,
        bbox=req.bbox.model_dump(),
        polygon=[p.model_dump() for p in req.polygon] if req.polygon else None,
        scene_ambient_c=req.ambient_temp_c,
    )


@app.post("/api/scene/{scene}/roi")
def roi(scene: str, req: ROIRequest):
    ds, files = _scene_files()
    if scene not in files:
        raise HTTPException(404, f"Unknown scene {scene}")
    return _compute(files[scene], req)


# ─────────────────────────────────────────────────────────────────────────────
# Save whole scene (annotations + scene metadata). Stats recomputed server-side.
# ─────────────────────────────────────────────────────────────────────────────
class SaveRequest(BaseModel):
    annotations: list[dict]
    scene_meta: dict
    completed: bool | None = None


@app.post("/api/scene/{scene}")
def save_scene(scene: str, req: SaveRequest):
    ds, files = _scene_files()
    if scene not in files:
        raise HTTPException(404, f"Unknown scene {scene}")
    f = files[scene]
    thermal = core.load_npy(f["thermal"])
    depth = core.load_npy(f["depth"])
    Q, _ = core.load_Q(f["metadata"])
    ambient = core.safe_float(req.scene_meta.get("ambient_temp_c"))

    # Recompute ROI stats for every annotation from its geometry — authoritative.
    clean = []
    for ann in req.annotations:
        ann = core.migrate_annotation(dict(ann))
        bbox = core.get_annotation_bbox(ann)
        if bbox is None:
            continue
        stats = core.compute_roi_stats(
            thermal, depth, Q, ann.get("shape", "rectangle"),
            bbox=bbox, polygon=ann.get("polygon_original"),
            scene_ambient_c=ambient,
        )
        stats.pop("depth_warning", None)
        ann.update(stats)
        ann["bbox_original"] = bbox
        clean.append(ann)

    blob = _load_scene_blob(ds, scene)
    blob.update(req.scene_meta)
    blob["annotations"] = clean
    blob["thermal_scene_min_c"] = float(np.nanmin(thermal))
    blob["thermal_scene_max_c"] = float(np.nanmax(thermal))
    blob["thermal_scene_mean_c"] = float(np.nanmean(thermal))
    if req.completed is not None:
        blob["completed"] = bool(req.completed)
    blob = core.migrate_scene(blob)
    if req.completed is not None:      # migrate_scene doesn't carry 'completed'
        blob["completed"] = bool(req.completed)

    core.save_json_atomic(blob, _ann_path(ds, scene))
    return {"ok": True, "annotations": blob["annotations"], "completed": blob.get("completed", False)}


# ─────────────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
