"""
ComChop – FastAPI Application Server
"""

from __future__ import annotations

import base64
import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.decomposer import DecomposeResult, decompose_panel
from app.services.splitter import SplitResult, split_panels

MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB
MAX_BATCH_FILES = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comchop")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ComChop API",
    description="Automated comic panel splitter powered by gutter-detection CV algorithms.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SplitResponse(BaseModel):
    panels: list[str]                  # base64 JPEG strings
    rows: int
    cols: int
    grid: list[tuple[int, int]]
    metadata: dict
    processing_time_ms: float


class ComicSplitResult(BaseModel):
    filename: str
    status: str                        # "ok" | "error"
    panels: list[str] = []
    rows: int = 0
    cols: int = 0
    grid: list[tuple[int, int]] = []
    metadata: dict = {}
    processing_time_ms: float = 0.0
    error: str | None = None


class BatchSplitResponse(BaseModel):
    results: list[ComicSplitResult]
    total_processing_time_ms: float


class DecomposeRequest(BaseModel):
    image_b64: str                     # base64-encoded panel image (JPEG/PNG)


class DecomposeResponse(BaseModel):
    characters: list[str]              # base64 PNGs (RGBA), one per character
    bubbles: str                       # base64 PNG (RGBA)
    background: str                    # base64 PNG (inpainted)
    metadata: dict
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="1.1.0")


@app.post("/api/split", response_model=SplitResponse, tags=["Splitter"])
async def split_comic(
    file: UploadFile = File(..., description="Comic image (PNG / JPG / WebP)"),
    white_median_threshold: int = Form(200, ge=100, le=255,
        description="Median brightness a row/col must exceed to count as a white gutter"),
    gutter_std_threshold: float = Form(30.0, ge=0.0, le=100.0,
        description="Max std-dev for a row/col to count as a flat, uniform white gutter"),
    min_panel_ratio: float = Form(0.15, ge=0.05, le=0.5,
        description="Min panel size as fraction of image dimension (0.05–0.5)"),
    detect_divider_lines: bool = Form(True,
        description="Also cut on thin solid border lines (no gap needed)"),
    dark_threshold: int = Form(70, ge=0, le=150,
        description="Mean brightness below which a row/col counts as a dark divider line"),
    line_std_threshold: float = Form(18.0, ge=0.0, le=60.0,
        description="Max std-dev for a row/col to count as a flat divider line"),
) -> SplitResponse:
    """
    Upload a comic image and receive individually extracted panel images.

    Returns each panel as a base64-encoded JPEG string alongside layout metadata.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be under 30 MB.")

    t0 = time.perf_counter()
    try:
        result: SplitResult = split_panels(
            image_bytes=image_bytes,
            white_median_threshold=white_median_threshold,
            gutter_std_threshold=gutter_std_threshold,
            min_panel_ratio=min_panel_ratio,
            detect_divider_lines=detect_divider_lines,
            dark_threshold=dark_threshold,
            line_std_threshold=line_std_threshold,
        )
    except Exception as exc:
        logger.exception("Splitter failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Processing error: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not result.panels:
        raise HTTPException(
            status_code=422,
            detail=(
                "No panels detected. Try lowering the white median threshold, "
                "raising the gutter std threshold, or reducing the min panel ratio."
            ),
        )

    logger.info(
        "Split %d panels (%dx%d grid) in %.1f ms",
        len(result.panels), result.rows, result.cols, elapsed_ms,
    )

    return SplitResponse(
        panels=result.panels,
        rows=result.rows,
        cols=result.cols,
        grid=result.grid,
        metadata=result.metadata,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/decompose", response_model=DecomposeResponse, tags=["Decomposer"])
def decompose(req: DecomposeRequest) -> DecomposeResponse:
    """
    Decompose a single panel image into character / speech-bubble /
    background layers. Characters are segmented with a local U2Net model,
    bubbles with classical CV, and the background is inpainted.

    Defined as a sync route so FastAPI runs it in the threadpool — U2Net
    inference takes a few seconds on CPU and must not block the event loop.
    """
    try:
        image_bytes = base64.b64decode(req.image_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be under 30 MB.")

    t0 = time.perf_counter()
    try:
        result: DecomposeResult = decompose_panel(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Decomposer failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Processing error: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Decomposed panel (%dx%d) in %.1f ms",
        result.metadata.get("width", 0), result.metadata.get("height", 0), elapsed_ms,
    )

    return DecomposeResponse(
        characters=result.characters,
        bubbles=result.bubbles,
        background=result.background,
        metadata=result.metadata,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/split-batch", response_model=BatchSplitResponse, tags=["Splitter"])
async def split_comics_batch(
    files: list[UploadFile] = File(..., description="Multiple comic images (PNG / JPG / WebP)"),
    white_median_threshold: int = Form(200, ge=100, le=255),
    gutter_std_threshold: float = Form(30.0, ge=0.0, le=100.0),
    min_panel_ratio: float = Form(0.15, ge=0.05, le=0.5),
    detect_divider_lines: bool = Form(True),
    dark_threshold: int = Form(70, ge=0, le=150),
    line_std_threshold: float = Form(18.0, ge=0.0, le=60.0),
) -> BatchSplitResponse:
    """
    Upload multiple comic images and receive panels for each, kept separate
    per source file (not merged into one flat list).
    """
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413, detail=f"Max {MAX_BATCH_FILES} files per batch."
        )

    t_batch0 = time.perf_counter()
    results: list[ComicSplitResult] = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(ComicSplitResult(
                filename=file.filename or "unknown",
                status="error",
                error="File must be an image.",
            ))
            continue

        image_bytes = await file.read()
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            results.append(ComicSplitResult(
                filename=file.filename or "unknown",
                status="error",
                error="Image must be under 30 MB.",
            ))
            continue

        t0 = time.perf_counter()
        try:
            result: SplitResult = split_panels(
                image_bytes=image_bytes,
                white_median_threshold=white_median_threshold,
                gutter_std_threshold=gutter_std_threshold,
                min_panel_ratio=min_panel_ratio,
                detect_divider_lines=detect_divider_lines,
                dark_threshold=dark_threshold,
                line_std_threshold=line_std_threshold,
            )
        except Exception as exc:
            logger.exception("Splitter failed on %s: %s", file.filename, exc)
            results.append(ComicSplitResult(
                filename=file.filename or "unknown",
                status="error",
                error=f"Processing error: {exc}",
            ))
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not result.panels:
            results.append(ComicSplitResult(
                filename=file.filename or "unknown",
                status="error",
                error="No panels detected.",
            ))
            continue

        results.append(ComicSplitResult(
            filename=file.filename or "unknown",
            status="ok",
            panels=result.panels,
            rows=result.rows,
            cols=result.cols,
            grid=result.grid,
            metadata=result.metadata,
            processing_time_ms=round(elapsed_ms, 2),
        ))

    total_elapsed_ms = (time.perf_counter() - t_batch0) * 1000
    logger.info(
        "Batch split %d files in %.1f ms (%d ok, %d failed)",
        len(files), total_elapsed_ms,
        sum(1 for r in results if r.status == "ok"),
        sum(1 for r in results if r.status == "error"),
    )

    return BatchSplitResponse(
        results=results,
        total_processing_time_ms=round(total_elapsed_ms, 2),
    )
