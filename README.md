# ComChop

Automated comic panel splitter. Upload a comic strip image and get each panel extracted individually — works for panels separated by whitespace gutters *or* touching panels separated only by a thin divider line.

## Stack

- **Backend**: FastAPI + OpenCV (`backend/`)
- **Frontend**: React + TypeScript + Vite + Tailwind (`frontend/`)

## Quick start

```bash
./run.sh
```

Starts the backend on `:8000`, the frontend on `:5173`, and opens the app in your browser.

## Manual setup

**Backend**

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## How it works

`backend/app/services/splitter.py` detects panel boundaries by scanning rows/columns for either:

1. Near-white gutters (whitespace between panels), or
2. Thin, uniform divider lines (e.g. a solid black rule with no surrounding gap)

Cut coordinates are derived from the median of each detected gutter/line band, then panels are cropped and returned as base64 JPEGs.

### Layer decomposition

Any panel can be split into layers — either automatically via `POST /api/decompose` (the "Split layers" button, or the **Layers only** upload mode for already-cut panels), or manually with the **Paint layers** tool, which runs entirely client-side.

**Automatic** (`backend/app/services/decomposer.py`):
- **Characters** — for line-art panels the foreground is found by flood-filling from the panel borders through non-ink pixels (median blur first so halftone dots don't block the flood; ink outlines do). Everything enclosed by outlines is content; pale colour washes flood away to the background. U2Net (`rembg`; ~176 MB auto-downloaded to `~/.u2net`) supplements the flood and takes over entirely on photo-style panels. Content is then split into individual characters via connected components refined by FastSAM instance segmentation (`ultralytics`, ~23 MB auto-downloaded), all running locally on CPU.
- **Speech bubbles** — classical CV: bright, convex, ink-enclosed blobs whose interiors hold several stroke-like dark components (text). The stroke-shape test keeps cartoon eyes (round pupils) out of the bubble layer.
- **Background** — the panel with both layers removed and the holes filled with OpenCV Telea inpainting.

**Manual** (`frontend/src/components/PaintPicker.tsx`) — for panels where auto-detection gets it wrong. Pick a category (add as many characters as needed, plus a fixed Bubbles category), paint directly over the pixels that belong to it with an adjustable brush, and hit Extract. What you paint *is* the layer — no model in the loop, so results are fully predictable. The background layer is the original panel with painted-over regions filled with a single solid color you pick (a color swatch in the picker), instead of inpainting. Runs instantly in the browser; nothing is sent to the backend.

Characters and bubbles are returned as transparent RGBA PNGs, the background as an opaque inpainted PNG.
