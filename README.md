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

Any panel can be split into layers via `POST /api/decompose` — either from the "Split layers" button on a panel card after a comic split, or by uploading already-cut panels directly with the **Layers only** mode toggle on the upload screen.

- **Characters** — foreground extracted with the U2Net salient-object model (`rembg`; ~176 MB auto-downloaded to `~/.u2net`), then separated into individual characters with FastSAM instance segmentation (`ultralytics`; ~23 MB auto-downloaded). Both run locally on CPU. Returns one RGBA PNG per character, ordered left to right; falls back to connected-component splitting if FastSAM finds no usable instances.
- **Speech bubbles** — classical CV in `backend/app/services/decomposer.py`: bright, convex contours containing text-like dark pixels
- **Background** — the panel with both layers removed and the holes filled with OpenCV Telea inpainting

Characters and bubbles are returned as transparent RGBA PNGs, the background as an opaque inpainted PNG.
