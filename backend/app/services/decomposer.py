"""
ComChop – Panel Layer Decomposer
================================
Splits a single comic panel into three layers:

1. **Speech bubbles** – detected with classical CV: bright, mostly-convex
   contours that contain dark "text-like" pixels inside. Returned as an
   RGBA PNG (transparent outside the bubbles).
2. **Characters** – foreground extracted with the U2Net salient object
   model (via `rembg`), then split into individual character instances
   with FastSAM (instance segmentation, runs locally on CPU). Bubble
   regions are subtracted so a bubble overlapping a character ends up in
   the bubble layer only. Returned as a list of RGBA PNGs, one per
   character, ordered left to right. Falls back to connected-component
   splitting when FastSAM yields no usable instances.
3. **Background** – the panel with characters + bubbles removed and the
   holes filled with OpenCV Telea inpainting. Returned as an opaque PNG.

Models are loaded lazily on first use and cached for the process
lifetime (U2Net ~176 MB to ~/.u2net; FastSAM-s ~23 MB to the backend
working directory).
"""

from __future__ import annotations

import base64
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("comchop.decomposer")

# ---------------------------------------------------------------------------
# Lazy rembg session (module-level singleton, thread-safe init)
# ---------------------------------------------------------------------------
_session = None
_session_lock = threading.Lock()

_sam_model = None
_sam_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                from rembg import new_session
                logger.info("Loading U2Net segmentation model...")
                _session = new_session("u2net")
                logger.info("U2Net model ready")
    return _session


def _get_sam():
    global _sam_model
    if _sam_model is None:
        with _sam_lock:
            if _sam_model is None:
                from ultralytics import FastSAM
                logger.info("Loading FastSAM instance segmentation model...")
                _sam_model = FastSAM("FastSAM-s.pt")
                logger.info("FastSAM model ready")
    return _sam_model


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class DecomposeResult:
    characters: list[str]    # base64 PNGs (RGBA), one per character, left→right
    bubbles: str             # base64 PNG (RGBA)
    background: str          # base64 PNG (opaque, inpainted)
    metadata: dict


# ---------------------------------------------------------------------------
# Bubble detection (classical CV)
# ---------------------------------------------------------------------------
def _detect_bubble_mask(
    bgr: np.ndarray,
    white_threshold: int = 230,
    min_area_ratio: float = 0.004,
    max_area_ratio: float = 0.45,
    min_solidity: float = 0.72,
    text_dark_threshold: int = 110,
    min_text_ratio: float = 0.004,
    max_text_ratio: float = 0.40,
) -> np.ndarray:
    """
    Return a uint8 mask (255 = bubble) of speech bubbles.

    A bubble is a bright, fairly convex blob whose interior contains a
    small-but-nonzero fraction of dark pixels (the lettering). Pure white
    gutters/backgrounds fail the text test; dark artwork fails the
    brightness test; busy bright areas fail the solidity test.
    """
    h, w = bgr.shape[:2]
    panel_area = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    _, bright = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    # Close small gaps so lettering doesn't fragment the bubble interior.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    # Drop white components touching the image border (gutter margins, page
    # background). Left in place, a border frame becomes the outermost
    # contour and RETR_EXTERNAL would hide any bubble nested inside it.
    n_labels, labels = cv2.connectedComponents(bright)
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)
    for lbl in border_labels:
        bright[labels == lbl] = 0

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)
    kept = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * panel_area or area > max_area_ratio * panel_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < min_solidity:
            continue

        # Fill the candidate and look for text-like dark pixels inside.
        cand = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cand, [cnt], -1, 255, thickness=cv2.FILLED)
        # Erode so the bubble's own dark outline doesn't count as text.
        interior = cv2.erode(cand, kernel, iterations=2)
        interior_px = int(np.count_nonzero(interior))
        if interior_px == 0:
            continue
        dark_px = int(np.count_nonzero((gray < text_dark_threshold) & (interior > 0)))
        text_ratio = dark_px / interior_px
        if not (min_text_ratio <= text_ratio <= max_text_ratio):
            continue

        # Bubble confirmed – include its outline by slightly dilating.
        cand = cv2.dilate(cand, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, cand)
        kept += 1

    logger.info("Bubble detection: %d candidates, %d kept", len(contours), kept)
    return mask


# ---------------------------------------------------------------------------
# Character segmentation (U2Net via rembg)
# ---------------------------------------------------------------------------
def _detect_character_mask(bgr: np.ndarray, alpha_threshold: int = 120) -> np.ndarray:
    """Return a uint8 mask (255 = character/foreground) from U2Net."""
    from rembg import remove

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(
        rgb,
        session=_get_session(),
        only_mask=False,
        post_process_mask=True,
    )
    alpha = np.asarray(rgba)[:, :, 3]
    mask = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)

    # Clean up speckles and small holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# Character instance splitting (FastSAM)
# ---------------------------------------------------------------------------
def _split_character_instances(
    bgr: np.ndarray,
    char_mask: np.ndarray,
    min_instance_ratio: float = 0.05,
    dominant_ratio: float = 0.90,
    dedup_iou: float = 0.70,
) -> list[np.ndarray]:
    """
    Split the combined character mask into per-character masks using
    FastSAM instance segmentation. Returns a list of uint8 masks ordered
    left to right. Falls back to connected components when FastSAM
    produces no usable instances.

    Ratios are relative to the total character-region area, not the panel.
    """
    char_area = int(np.count_nonzero(char_mask))
    if char_area == 0:
        return []

    candidates: list[np.ndarray] = []
    try:
        results = _get_sam()(
            bgr, device="cpu", retina_masks=True, imgsz=640,
            conf=0.4, iou=0.9, verbose=False,
        )
        r = results[0]
        if r.masks is not None:
            h, w = char_mask.shape
            for m in r.masks.data.cpu().numpy():
                inst = (m > 0.5).astype(np.uint8) * 255
                if inst.shape != (h, w):
                    inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_NEAREST)
                clipped = cv2.bitwise_and(inst, char_mask)
                area = int(np.count_nonzero(clipped))
                if area >= min_instance_ratio * char_area:
                    candidates.append(clipped)
    except Exception as exc:
        logger.warning("FastSAM inference failed, using fallback: %s", exc)

    # Drop near-duplicates of the whole foreground when real sub-instances
    # exist (SAM often emits one mask covering every character at once).
    if len(candidates) > 1:
        subs = [c for c in candidates
                if np.count_nonzero(c) < dominant_ratio * char_area]
        if len(subs) >= 2:
            candidates = subs

    # Deduplicate by IoU, keeping larger masks.
    candidates.sort(key=lambda m: -np.count_nonzero(m))
    kept: list[np.ndarray] = []
    for cand in candidates:
        cand_area = np.count_nonzero(cand)
        dup = False
        for k in kept:
            inter = np.count_nonzero(cv2.bitwise_and(cand, k))
            union = cand_area + np.count_nonzero(k) - inter
            if union > 0 and inter / union > dedup_iou:
                dup = True
                break
        if not dup:
            kept.append(cand)

    # Greedy paint, largest first — each pixel belongs to one instance.
    instances: list[np.ndarray] = []
    claimed = np.zeros_like(char_mask)
    for cand in kept:
        remaining = cv2.bitwise_and(cand, cv2.bitwise_not(claimed))
        if np.count_nonzero(remaining) >= min_instance_ratio * char_area:
            instances.append(remaining)
            claimed = cv2.bitwise_or(claimed, remaining)

    # Unclaimed character pixels: attach each leftover blob to the
    # touching instance with the most overlap, or promote it to its own
    # instance when big enough.
    leftover = cv2.bitwise_and(char_mask, cv2.bitwise_not(claimed))
    n, labels = cv2.connectedComponents(leftover)
    touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for lbl in range(1, n):
        blob = np.where(labels == lbl, 255, 0).astype(np.uint8)
        blob_area = np.count_nonzero(blob)
        grown = cv2.dilate(blob, touch_kernel)
        overlaps = [int(np.count_nonzero(cv2.bitwise_and(grown, inst)))
                    for inst in instances]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] > 0:
            instances[best] = cv2.bitwise_or(instances[best], blob)
        elif blob_area >= min_instance_ratio * char_area:
            instances.append(blob)

    # Fallback: no SAM instances at all → connected components.
    if not instances:
        n, labels = cv2.connectedComponents(char_mask)
        for lbl in range(1, n):
            blob = np.where(labels == lbl, 255, 0).astype(np.uint8)
            if np.count_nonzero(blob) >= min_instance_ratio * char_area:
                instances.append(blob)
        if not instances:
            instances = [char_mask]

    # Order left → right by centroid for stable numbering.
    def centroid_x(m: np.ndarray) -> float:
        xs = np.nonzero(m)[1]
        return float(xs.mean()) if xs.size else 0.0

    instances.sort(key=centroid_x)
    logger.info("Character instances: %d (from %d SAM candidates)",
                len(instances), len(kept))
    return instances


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _encode_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _layer_rgba(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """BGR image + mask -> BGRA layer, transparent outside mask."""
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def decompose_panel(image_bytes: bytes) -> DecomposeResult:
    """Decompose a comic panel image into character / bubble / background layers."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")

    h, w = bgr.shape[:2]

    bubble_mask = _detect_bubble_mask(bgr)
    char_mask = _detect_character_mask(bgr)

    # A bubble overlapping a character belongs to the bubble layer only.
    # Subtract a dilated bubble mask so the bubble's anti-aliased outline
    # doesn't survive as a ring in the character layer.
    sub_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bubble_dilated = cv2.dilate(bubble_mask, sub_kernel, iterations=2)
    char_mask = cv2.bitwise_and(char_mask, cv2.bitwise_not(bubble_dilated))

    # Background: remove both layers, dilate to also eat outline halos,
    # then inpaint the holes.
    removed = cv2.bitwise_or(char_mask, bubble_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_mask = cv2.dilate(removed, kernel, iterations=1)
    background = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    instance_masks = _split_character_instances(bgr, char_mask)

    char_px = int(np.count_nonzero(char_mask))
    bubble_px = int(np.count_nonzero(bubble_mask))
    total_px = h * w

    return DecomposeResult(
        characters=[_encode_png_b64(_layer_rgba(bgr, m)) for m in instance_masks],
        bubbles=_encode_png_b64(_layer_rgba(bgr, bubble_mask)),
        background=_encode_png_b64(background),
        metadata={
            "width": w,
            "height": h,
            "character_count": len(instance_masks),
            "character_coverage": round(char_px / total_px, 4),
            "bubble_coverage": round(bubble_px / total_px, 4),
        },
    )
