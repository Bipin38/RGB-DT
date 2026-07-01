"""
split_dataset.py — one-time migration.
====================================================
Explode the monolithic annotations_<slug>.json into per-scene files under
annotations/<slug>/<scene>.json, so teammates can each own different scenes
and merge conflict-free.

Idempotent: re-running only writes scenes that don't yet have a per-scene file
(pass --force to overwrite). Completion flags are seeded from progress_<slug>.json.

Usage:
    .venv/bin/python split_dataset.py [--dataset Dataset] [--force]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from app import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="dataset folder name (default: first found)")
    ap.add_argument("--force", action="store_true", help="overwrite existing per-scene files")
    args = ap.parse_args()

    available = core.get_available_dataset_names(core.BASE_DIR)
    if not available:
        raise SystemExit(f"No dataset folders under {core.BASE_DIR}")
    ds = args.dataset or available[0]
    slug = core.file_slug(ds)

    mono_path = os.path.join(core.PROJECT_ROOT, f"annotations_{slug}.json")
    prog_path = os.path.join(core.PROJECT_ROOT, f"progress_{slug}.json")
    out_dir = os.path.join(core.PROJECT_ROOT, "annotations", slug)
    os.makedirs(out_dir, exist_ok=True)

    mono = core.load_json(mono_path, {})
    completed = set(core.load_json(prog_path, []))
    if not mono:
        print(f"⚠️  {mono_path} is empty or missing — nothing to split.")
        return

    written = skipped = 0
    for scene, blob in mono.items():
        dst = os.path.join(out_dir, f"{scene}.json")
        if os.path.exists(dst) and not args.force:
            skipped += 1
            continue
        scene_blob = core.migrate_scene(blob)
        scene_blob["completed"] = scene in completed
        core.save_json_atomic(scene_blob, dst)
        written += 1

    print(f"✅ Split '{ds}': {written} scene files written, {skipped} skipped → {out_dir}")


if __name__ == "__main__":
    main()
