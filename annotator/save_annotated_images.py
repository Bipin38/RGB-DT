#!/usr/bin/env python3
"""Draw annotation bounding boxes on dataset RGB images and save the results."""

import argparse
import json
import os
import re

import cv2
import numpy as np


BOX_COLORS_BGR = [
    (0, 255, 0),
    (68, 68, 255),
    (255, 255, 0),
    (0, 215, 255),
    (255, 0, 255),
    (255, 136, 68),
    (0, 165, 255),
    (255, 255, 255),
    (180, 105, 255),
    (0, 255, 127),
]


def natural_sort_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def get_bbox(annotation):
    bbox = annotation.get("bbox_original")
    if not bbox:
        return None
    required = ("x_min", "y_min", "x_max", "y_max")
    if not all(key in bbox for key in required):
        return None
    return bbox


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y - text_h - baseline - 4)
    cv2.rectangle(
        img,
        (x, y_top),
        (x + text_w + 6, y_top + text_h + baseline + 4),
        color,
        thickness=-1,
    )
    cv2.putText(
        img,
        text,
        (x + 3, y_top + text_h + 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(img, annotations, draw_polygons=True):
    annotated = img.copy()
    h, w = annotated.shape[:2]

    for idx, ann in enumerate(annotations):
        color = BOX_COLORS_BGR[idx % len(BOX_COLORS_BGR)]
        label = ann.get("object_class") or ann.get("class") or "object"
        label = label.replace("_", " ")

        if draw_polygons and ann.get("shape") == "polygon" and ann.get("polygon_original"):
            points = ann["polygon_original"]
            pts = np.array(
                [[int(max(0, min(w - 1, p["x"]))), int(max(0, min(h - 1, p["y"])))] for p in points],
                dtype=np.int32,
            )
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
            x_label = int(pts[:, 0].min())
            y_label = int(pts[:, 1].min())
            draw_label(annotated, label, x_label, y_label, color)
            continue

        bbox = get_bbox(ann)
        if not bbox:
            continue

        x1 = int(max(0, min(w - 1, bbox["x_min"])))
        y1 = int(max(0, min(h - 1, bbox["y_min"])))
        x2 = int(max(0, min(w - 1, bbox["x_max"])))
        y2 = int(max(0, min(h - 1, bbox["y_max"])))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_label(annotated, label, x1, y1, color)

    return annotated


def save_scene_annotations(scene_id, scene_data, dataset_dir, output_dir, draw_polygons):
    rgb_path = os.path.join(dataset_dir, scene_id, "left_cam", "left.png")
    if not os.path.exists(rgb_path):
        print(f"  Skipping {scene_id}: missing {rgb_path}")
        return False

    img = cv2.imread(rgb_path)
    if img is None:
        print(f"  Skipping {scene_id}: could not read {rgb_path}")
        return False

    annotations = scene_data.get("annotations", [])
    if not annotations:
        print(f"  Skipping {scene_id}: no annotations")
        return False

    annotated = annotate_image(img, annotations, draw_polygons=draw_polygons)

    scene_output_dir = os.path.join(output_dir, scene_id, "annotated")
    os.makedirs(scene_output_dir, exist_ok=True)
    output_path = os.path.join(scene_output_dir, "annotated.png")
    cv2.imwrite(output_path, annotated)
    print(f"  Saved {output_path} ({len(annotations)} annotation(s))")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from annotations_dataset.json onto RGB images."
    )
    parser.add_argument(
        "--annotations",
        default="annotator/annotations_dataset.json",
        help="Path to the annotations JSON file (default: annotator/annotations_dataset.json)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="Dataset",
        help="Root directory containing scene folders (default: Dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for annotated images (default: same as --dataset-dir)",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Process only the given scene id (e.g. Scene_35). Can be repeated.",
    )
    parser.add_argument(
        "--rect-only",
        action="store_true",
        help="Always draw rectangles from bbox_original, even for polygon annotations",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.dataset_dir

    with open(args.annotations, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    scene_ids = sorted(dataset.keys(), key=natural_sort_key)
    if args.scenes:
        requested = set(args.scenes)
        scene_ids = [scene_id for scene_id in scene_ids if scene_id in requested]
        missing = requested - set(scene_ids)
        for scene_id in sorted(missing, key=natural_sort_key):
            print(f"  Warning: {scene_id} not found in {args.annotations}")

    print(f"Loaded {len(scene_ids)} scene(s) from {args.annotations}")
    saved = 0
    for scene_id in scene_ids:
        if save_scene_annotations(
            scene_id,
            dataset[scene_id],
            args.dataset_dir,
            output_dir,
            draw_polygons=not args.rect_only,
        ):
            saved += 1

    print(f"\nDone. Saved {saved}/{len(scene_ids)} annotated image(s).")


if __name__ == "__main__":
    main()
