"""
ComChop – FastAPI Application Server
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.splitter import SplitResult, split_panels

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
    binary_threshold: int = Form(240, ge=200, le=255,
        description="Pixel brightness cut-off for binarisation (200–255)"),
    white_threshold: float = Form(0.99, ge=0.8, le=1.0,
        description="Min fraction of white pixels for a row/col to be a gutter (0.8–1.0)"),
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
    if len(image_bytes) > 30 * 1024 * 1024:  # 30 MB guard
        raise HTTPException(status_code=413, detail="Image must be under 30 MB.")

    t0 = time.perf_counter()
    try:
        result: SplitResult = split_panels(
            image_bytes=image_bytes,
            binary_threshold=binary_threshold,
            white_threshold=white_threshold,
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
                "No panels detected. Try lowering the binary threshold or "
                "the white threshold, or reduce the min panel ratio."
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
