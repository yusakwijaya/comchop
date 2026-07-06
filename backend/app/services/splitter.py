"""
ComChop – Comic Panel Splitter (Gutter-projection algorithm)
============================================================
Algorithm
---------
1. Decode image and convert to grayscale.
2. Compute per-row and per-column *median* brightness plus standard
   deviation. A row/col is a white-gutter candidate when its median exceeds
   `white_median_threshold` AND its standard deviation is below
   `gutter_std_threshold`.
   Median (rather than mean) is what makes this reliable for thin,
   anti-aliased or JPEG-compressed gutters: a couple of stray dark/blurred
   pixels can pull a strict mean-based test below threshold even though the
   row is overwhelmingly a clean gutter. The std check is what keeps this
   from false-positiving on a row that happens to pass through a bright
   speech bubble — that row has a high median too, but a high standard
   deviation (bubble + surrounding dark art), whereas a genuine gutter row
   is both bright *and* flat.
3. In addition to white gutters, rows/cols are also tagged as gutter
   candidates when they are a thin, uniform *divider line* (typically
   black) rather than whitespace — this covers comics whose panels touch
   directly and are separated only by a border line instead of a gap.
   A row/col qualifies when its mean brightness is below `dark_threshold`
   and its standard deviation is below `line_std_threshold` (i.e. it's a
   flat, solid line rather than dark artwork/textured content).
4. Consecutive gutter-candidate indices (white OR line) are grouped into
   bands. The *median* index of each band becomes a cut coordinate — this
   is more robust than taking the midpoint of the outermost pixels.
5. The image edges (0, height / 0, width) are inserted as cut coordinates
   only when the nearest detected gutter is more than `edge_margin` pixels
   away, preventing thin marginal slices.
6. Row bands are detected first (steps 2–5, scanning the *full width* of
   the image). Column cuts are then computed independently *within each row
   band* rather than across the whole image — this supports irregular
   grids where different rows have a different number of panels (e.g. a
   2-2-1 layout where the bottom row is one wide panel with no vertical
   divider).
7. Panels are the rectangular regions between adjacent cut coordinates.
   Regions narrower than `min_panel_ratio` of the total dimension (default
   15 %) are discarded as margins or artifact lines.
8. Surviving panels are JPEG-encoded and returned as base-64 strings.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("comchop.splitter")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    panels: List[str] = field(default_factory=list)   # base64-encoded JPEG strings
    rows: int = 0
    cols: int = 0
    grid: List[Tuple[int, int]] = field(default_factory=list)  # (row_idx, col_idx) per panel
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_base64_jpg(cv_img: np.ndarray, quality: int = 92) -> str:
    success, buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode panel to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _band_centers(
    indices: np.ndarray,
    max_val: int,
    max_gutter_ratio: float,
) -> List[int]:
    """
    Group consecutive indices into contiguous bands and return the median
    index of each band that survives — bands are always split by proximity
    (a run of dark-line candidates never merges with a run of white-gutter
    candidates as they are grouped separately by the caller), and any band
    thicker than `max_gutter_ratio` of `max_val` is discarded rather than
    turned into a cut. The thickness cap is what distinguishes a genuine,
    thin divider (a few px of whitespace or a border line) from a large
    panel that simply has a bright/uniform-colored background spanning most
    of its width or height — without it, a wide expanse of flat cream or
    pastel background would itself look like one giant "gutter".
    """
    if len(indices) == 0:
        return []

    diff = np.diff(indices)
    split_points = np.where(diff > 1)[0] + 1
    groups = np.split(indices, split_points)

    max_thickness = max_val * max_gutter_ratio
    groups = [g for g in groups if len(g) <= max_thickness]

    return [int(np.median(g)) for g in groups]


def _gutter_line_indices(
    mat: np.ndarray,
    white_median_threshold: int,
    gutter_std_threshold: float,
    detect_divider_lines: bool,
    dark_threshold: int,
    line_std_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a 2-D grayscale matrix, return two index arrays along axis 0:
    near-white gutter candidates, and thin uniform dark divider-line
    candidates. Each index is evaluated using stats computed across axis 1,
    so to detect column gutters pass in the transposed matrix. Kept separate
    (rather than unioned) so a thin dark line sitting inside a much larger
    bright background doesn't get merged into one thick band and discarded.
    """
    medians = np.median(mat, axis=1)
    stds = np.std(mat, axis=1)

    white_idx = np.where(
        (medians > white_median_threshold) & (stds < gutter_std_threshold)
    )[0]

    line_idx = np.array([], dtype=int)
    if detect_divider_lines:
        means = np.mean(mat, axis=1)
        line_idx = np.where(
            (means < dark_threshold) & (stds < line_std_threshold)
        )[0]

    return white_idx, line_idx


def _find_segments(
    white_idx: np.ndarray,
    line_idx: np.ndarray,
    max_val: int,
    edge_margin: int,
    white_max_gutter_ratio: float = 0.05,
    line_max_gutter_ratio: float = 0.02,
) -> List[int]:
    """
    Convert white-gutter and divider-line candidate indices into a sorted
    list of cut coordinates. The two candidate types are grouped and
    thickness-capped independently, then combined.

    Image edges (0 and max_val) are appended only when the nearest detected
    gutter is further than `edge_margin` pixels away.
    """
    centers = sorted(set(
        _band_centers(white_idx, max_val, white_max_gutter_ratio) +
        _band_centers(line_idx, max_val, line_max_gutter_ratio)
    ))

    if not centers:
        return [0, max_val]

    # Prepend 0 only if the first gutter is far enough from the top edge
    if centers[0] > edge_margin:
        centers.insert(0, 0)

    # Append max_val only if the last gutter is far enough from the bottom edge
    if (max_val - centers[-1]) > edge_margin:
        centers.append(max_val)

    return sorted(list(set(centers)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_panels(
    image_bytes: bytes,
    white_median_threshold: int = 200,
    gutter_std_threshold: float = 30.0,
    min_panel_ratio: float = 0.15,
    edge_margin: int = 15,
    jpeg_quality: int = 92,
    detect_divider_lines: bool = True,
    dark_threshold: int = 70,
    line_std_threshold: float = 18.0,
) -> SplitResult:
    """
    Detect and extract individual comic panels from an image.

    Parameters
    ----------
    image_bytes       : Raw image file bytes (any Pillow-supported format).
    white_median_threshold : Median brightness (0–255) a row/col must exceed
                        to be considered a white gutter candidate.
    gutter_std_threshold : Max standard deviation a row/col may have and
                        still count as a flat, uniform white gutter (as
                        opposed to a bright but busy region like a speech
                        bubble sitting inside otherwise dark artwork).
    min_panel_ratio   : Minimum panel size as a fraction of the total image
                        dimension.  Slices below this are discarded as borders
                        or watermark strips.
    edge_margin       : If the nearest gutter is within this many pixels of the
                        image edge, the edge itself is not inserted as an extra
                        cut (avoids hairline slices).
    jpeg_quality      : JPEG quality for output panels (1–100).
    detect_divider_lines : Also treat thin, uniform border lines (e.g. a solid
                        black rule with no surrounding whitespace) as cut
                        points, for comics whose panels touch directly.
    dark_threshold    : Mean brightness (0–255) below which a row/col is
                        considered dark enough to be a divider line.
    line_std_threshold: Max standard deviation a row/col may have and still be
                        considered a flat, solid line (as opposed to dark,
                        textured artwork).
    """
    # ── Load ──────────────────────────────────────────────────────────────
    pil_img = Image.open(io.BytesIO(image_bytes))
    cv_img = _pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def gutters(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _gutter_line_indices(
            mat, white_median_threshold, gutter_std_threshold,
            detect_divider_lines, dark_threshold, line_std_threshold,
        )

    # ── Row bands: detected once, scanning the full width ────────────────
    white_rows, line_rows = gutters(gray)
    y_cuts = _find_segments(white_rows, line_rows, h, edge_margin)

    logger.debug(
        "Gutter rows: %d white, %d line  |  Y cuts: %s",
        len(white_rows), len(line_rows), y_cuts,
    )

    # ── Crop and encode panels ─────────────────────────────────────────────
    result = SplitResult()
    x_cuts_by_row: List[List[int]] = []

    for ri in range(len(y_cuts) - 1):
        y1, y2 = y_cuts[ri], y_cuts[ri + 1]

        if (y2 - y1) <= h * min_panel_ratio:
            x_cuts_by_row.append([])
            continue

        # ── Column cuts computed independently within this row band ──────
        # (supports irregular grids, e.g. a wide bottom row with no
        # vertical divider even though rows above it do have one)
        band_gray = gray[y1:y2, :]
        white_cols, line_cols = gutters(band_gray.T)
        x_cuts = _find_segments(white_cols, line_cols, w, edge_margin)
        x_cuts_by_row.append(x_cuts)

        for ci in range(len(x_cuts) - 1):
            x1, x2 = x_cuts[ci], x_cuts[ci + 1]

            # Discard narrow margin strips
            if (x2 - x1) <= w * min_panel_ratio:
                continue

            panel = cv_img[y1:y2, x1:x2]
            if panel.size == 0:
                continue

            result.panels.append(_cv_to_base64_jpg(panel, jpeg_quality))
            result.grid.append((ri, ci))

    # Re-derive actual grid dimensions from surviving panels
    if result.grid:
        result.rows = max(r for r, _ in result.grid) + 1
        result.cols = max(c for _, c in result.grid) + 1

    result.metadata = {
        "original_size": [w, h],
        "y_cuts": y_cuts,
        "x_cuts_by_row": x_cuts_by_row,
        "gutter_row_count": int(len(white_rows) + len(line_rows)),
    }

    return result
