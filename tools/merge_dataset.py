"""
merge_dataset.py — recombine per-scene files into one dataset JSON.
====================================================
Maintainer step: after teammates push their annotations/<slug>/<scene>.json
files (via shared drive / git / cloud folder), merge them back into a single
annotations_<slug>.json in the EXACT original schema for downstream use.

Because each scene is a separate file, merging is conflict-free. If the same
scene was annotated by two people (duplicate detected across --extra folders),
the script WARNS and keeps the one with more annotations.

Usage:
    .venv/bin/python merge_dataset.py [--dataset Dataset] \
        [--extra /path/to/teammate/annotations] [--out annotations_dataset.json]
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from app import core


def _scene_from_path(p):
    return os.path.splitext(os.path.basename(p))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--extra", action="append", default=[],
                    help="additional annotations dirs from teammates (repeatable)")
    ap.add_argument("--out", default=None, help="output path (default annotations_<slug>.json)")
    args = ap.parse_args()

    available = core.get_available_dataset_names(core.BASE_DIR)
    ds = args.dataset or (available[0] if available else "Dataset")
    slug = core.file_slug(ds)

    dirs = [os.path.join(core.PROJECT_ROOT, "annotations", slug)]
    for e in args.extra:
        # accept either the .../annotations dir or the .../annotations/<slug> dir
        dirs.append(os.path.join(e, slug) if os.path.isdir(os.path.join(e, slug)) else e)

    merged = {}
    progress = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"⚠️  skipping missing dir: {d}")
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            scene = _scene_from_path(path)
            blob = core.migrate_scene(core.load_json(path, {}))
            if scene in merged:
                a_new = len(blob.get("annotations", []))
                a_old = len(merged[scene].get("annotations", []))
                print(f"⚠️  DUPLICATE {scene}: keeping the copy with more annotations "
                      f"({max(a_new, a_old)} vs {min(a_new, a_old)}).")
                if a_new <= a_old:
                    continue
            if blob.pop("completed", False):
                if scene not in progress:
                    progress.append(scene)
            else:
                blob.pop("completed", None)
            merged[scene] = blob

    out = args.out or os.path.join(core.PROJECT_ROOT, f"annotations_{slug}.json")
    core.save_json_atomic(merged, out)
    prog_out = os.path.join(core.PROJECT_ROOT, f"progress_{slug}.json")
    core.save_json_atomic(progress, prog_out)

    n_ann = sum(len(b.get("annotations", [])) for b in merged.values())
    print(f"✅ Merged {len(merged)} scenes, {n_ann} annotations → {out}")
    print(f"   {len(progress)} scenes marked complete → {prog_out}")


if __name__ == "__main__":
    main()
