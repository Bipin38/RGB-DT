# RGB-D-T Annotation Tool

A fast web tool for annotating an **RGB + Depth + Thermal** dataset: draw
bounding boxes / polygons on the RGB frame and the app computes per-object
temperature (from the thermal `.npy`) and depth (from the depth `.npy`), plus
rich attributes (class, state, hazard tier, captions, VQA pairs).

Drawing runs **100% in the browser**, so there is no lag — the server is only
contacted to load a scene, compute a box's temp/depth, and save.

## Quick start
```bash
git clone <repo> && cd RGB-DT
./scripts/setup.sh        # create the virtualenv + install dependencies (once)
./scripts/run.sh          # start the app → http://localhost:8000
```
Copy the shared **`Dataset/`** folder to the project root first (it is not in
git). `run.sh` warns if no scenes are found.

- `./scripts/run.sh 8080` — use a different port.
- Stop the server with `Ctrl-C`.

## Project layout
```
RGB-DT/
├── scripts/
│   ├── setup.sh              # create venv + install requirements
│   └── run.sh                # launch the web app (auto-bootstraps + migrates)
├── app/                      # the web application (Python package)
│   ├── core.py               # thermal/depth math, schema, dataset discovery
│   ├── server.py             # FastAPI backend + serves the UI
│   └── static/               # browser UI (index.html, app.js, style.css)
├── tools/                    # dataset utilities
│   ├── split_dataset.py      # monolithic JSON → per-scene files (one-time)
│   └── merge_dataset.py      # per-scene files → one dataset JSON (maintainer)
├── legacy/                   # the old Streamlit tool (unmaintained fallback)
│   └── annotator_v3.py
├── requirements.txt
│
│   # ── data (lives at the project root) ──
├── Dataset/                  # raw RGB-D-T scenes — synced separately, gitignored
├── annotations/<slug>/*.json # per-scene annotations — COMMITTED & shared via git
├── annotations_dataset.json  # merged/legacy monolithic output
├── progress_dataset.json     # completed-scene list
└── custom_classes.json       # user-added object classes
```

## Using the app
- Pick a scene on the left. Draw a **Rectangle** (click-drag) or **Polygon**
  (click each vertex → click the green first point / double-click / `Enter` to
  close; right-click or `⌫` to undo a point).
- Fill the object fields, then **➕ Add object** (auto-saves). Click a saved
  object to **edit** its fields, or 🗑 to delete.
- Fill **Scene metadata** and click **💾 Save scene metadata** — required before
  **✅ Complete & next** will advance.
- Shortcuts: `N`/`P` next/prev · `R`/`G` rect/polygon · `1–9` pick class ·
  `T` light/dark theme · `Esc` cancel · `Del` delete selected.

## Team workflow (each person runs locally)
1. Everyone shares the same `Dataset/` and splits scenes so no two people
   annotate the same one.
2. Annotate locally; files save to `annotations/<slug>/<scene>.json`.
3. Commit & push your `annotations/` files (they are tracked in git).
4. Maintainer merges everyone's work into one dataset JSON:
   ```bash
   .venv/bin/python tools/merge_dataset.py \
       --extra /path/to/teammateA/annotations \
       --extra /path/to/teammateB/annotations
   ```
   Per-scene files merge conflict-free; duplicate scenes are flagged.

## Notes
- Dataset folder is auto-detected. Pin one with `RGBDT_DATASET=Dataset ./scripts/run.sh`.
- The legacy Streamlit tool in `legacy/` is kept only as a fallback; re-enable
  its deps in `requirements.txt` to run it.
