"""
ComChop – Panel Layer Decomposer
================================
Splits a single comic panel into three layers:

1. **Speech bubbles** – classical CV: bright, mostly-convex blobs that are
   enclosed by ink (not part of the page background) and contain several
   separate dark components inside (text letters). The letter-count
   requirement is what keeps cartoon eyes — white blob + one dark pupil —
   out of the bubble layer.
2. **Characters** – the panel's "content" region (see below) split into
   individual character instances, either automatically (connected
   components refined by FastSAM) or from user-supplied click points
   (FastSAM point prompts). Returned as RGBA PNGs ordered left to right.
3. **Background** – the panel with characters + bubbles removed and the
   holes filled with OpenCV Telea inpainting.

Content detection
-----------------
For line-art comics the foreground is found with a flood fill from the
panel borders through non-ink pixels (median blur first, so halftone
dots don't block the flood; ink outlines do). Everything the flood
cannot reach is enclosed by outlines and counts as content — this keeps
characters' white interiors as foreground and sends pale colour washes
to the background. U2Net (rembg) adds any foreground it finds outside
the flooded page, and becomes the sole source when the flood covers too
little of the panel to look like line art (photo-style backgrounds).

Models load lazily and are cached for the process lifetime (U2Net
~176 MB to ~/.u2net; FastSAM-s ~23 MB to the backend working directory).
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
# Lazy model singletons (thread-safe init)
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
# Page background (flood fill from borders through non-ink)
# ---------------------------------------------------------------------------
def _page_background_mask(
    bgr: np.ndarray,
    page_threshold: int = 200,
    blur_ksize: int = 7,
) -> tuple[np.ndarray, float]:
    """
    Return (mask, coverage) where mask marks the page background: bright
    regions connected to the panel border. Median blur first so halftone
    dot patterns read as solid page; a slight erosion of the bright mask
    closes small gaps in ink outlines so the flood doesn't leak into
    enclosed shapes.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, blur_ksize)
    bright = (blur > page_threshold).astype(np.uint8) * 255
    eroded = cv2.erode(bright, np.ones((3, 3), np.uint8), iterations=1)

    n, labels = cv2.connectedComponents(eroded)
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)
    page = np.isin(labels, list(border_labels)).astype(np.uint8) * 255
    page = cv2.dilate(page, np.ones((3, 3), np.uint8), iterations=1)
    # Never claim ink pixels for the page.
    page[blur <= 120] = 0

    coverage = float(np.count_nonzero(page)) / page.size
    return page, coverage


def _flat_wash_mask(
    bgr: np.ndarray,
    page_mask: np.ndarray,
    min_area_ratio: float = 0.08,
    max_edge_density: float = 0.02,
) -> np.ndarray:
    """
    Detect flat decorative colour washes (pale blobs behind characters)
    that are too dark for the page flood: large, border- or page-touching
    mid-tone regions with almost no internal edges.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    mid = ((blur > 130) & (blur <= 220)).astype(np.uint8) * 255
    mid[page_mask > 0] = 0

    edges = cv2.Canny(gray, 60, 160)
    page_grown = cv2.dilate(page_mask, np.ones((5, 5), np.uint8))

    wash = np.zeros((h, w), dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mid)
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area_ratio * h * w:
            continue
        blob = (labels == lbl)
        touches_border = blob[0, :].any() or blob[-1, :].any() \
            or blob[:, 0].any() or blob[:, -1].any()
        touches_page = bool(np.count_nonzero(page_grown[blob]))
        if not (touches_border or touches_page):
            continue
        edge_density = float(np.count_nonzero(edges[blob])) / area
        if edge_density <= max_edge_density:
            wash[blob] = 255
    return wash


# ---------------------------------------------------------------------------
# Bubble detection
# ---------------------------------------------------------------------------
def _detect_bubble_mask(
    bgr: np.ndarray,
    page_mask: np.ndarray,
    white_threshold: int = 225,
    min_area_ratio: float = 0.006,
    max_area_ratio: float = 0.45,
    min_solidity: float = 0.70,
    text_dark_threshold: int = 110,
    min_text_ratio: float = 0.004,
    max_text_ratio: float = 0.40,
    min_text_components: int = 3,
) -> np.ndarray:
    """
    Return a uint8 mask (255 = bubble) of speech bubbles.

    A bubble is a bright, fairly convex blob that is NOT part of the page
    background and whose interior contains several separate dark
    components (lettering). Requiring >= min_text_components is what
    rejects cartoon eyes: an eye is a bright blob with a single dark
    pupil, text has one component per letter stroke.
    """
    h, w = bgr.shape[:2]
    panel_area = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    _, bright = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    bright[page_mask > 0] = 0
    # Close small gaps so lettering doesn't fragment the bubble interior.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

        cand = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cand, [cnt], -1, 255, thickness=cv2.FILLED)
        # Erode so the bubble's own dark outline doesn't count as text.
        interior = cv2.erode(cand, kernel, iterations=2)
        interior_px = int(np.count_nonzero(interior))
        if interior_px == 0:
            continue

        dark = ((gray < text_dark_threshold) & (interior > 0)).astype(np.uint8)
        dark_px = int(np.count_nonzero(dark))
        text_ratio = dark_px / interior_px
        if not (min_text_ratio <= text_ratio <= max_text_ratio):
            continue

        # Text check: several separate dark components AND stroke-like
        # shapes. Letters are thin strokes (low circularity); the dark
        # bits inside an eye or a face — pupils — are round blobs (high
        # circularity). This is what separates bubbles from characters
        # that also have bright interiors with dark details.
        dark_contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        circ: list[float] = []
        for dc in dark_contours:
            a = cv2.contourArea(dc)
            p = cv2.arcLength(dc, True)
            if a >= 6 and p > 0:
                circ.append(4 * np.pi * a / (p * p))
        if len(circ) < min_text_components:
            continue
        if float(np.median(circ)) > 0.55:
            continue

        # Bubble confirmed – include its outline by slightly dilating.
        cand = cv2.dilate(cand, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, cand)
        kept += 1

    logger.info("Bubble detection: %d contours, %d bubbles", len(contours), kept)
    return mask


# ---------------------------------------------------------------------------
# Content region (character candidates before instance split)
# ---------------------------------------------------------------------------
def _u2net_foreground(bgr: np.ndarray, alpha_threshold: int = 120) -> np.ndarray:
    from rembg import remove

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=_get_session(), post_process_mask=True)
    alpha = np.asarray(rgba)[:, :, 3]
    mask = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _content_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return (content, page_mask, mode). Content = candidate character
    pixels; page_mask marks the page/wash background.

    mode "lineart": flood-fill page detection worked (page covers enough
    of the panel) — content is everything enclosed by ink, plus any
    U2Net foreground outside the page. mode "saliency": flood found no
    meaningful page (photo-style panel) — content is the U2Net mask.
    """
    page, coverage = _page_background_mask(bgr)

    if coverage >= 0.20:
        wash = _flat_wash_mask(bgr, page)
        page = cv2.bitwise_or(page, wash)
        content = cv2.bitwise_not(page)
        # Rescue foreground U2Net sees that the flood misclassified is
        # intentionally NOT done for page pixels (the flood is more
        # reliable on line art); U2Net only adds regions the flood never
        # reached, e.g. dark shapes floating on washes.
        u2 = _u2net_foreground(bgr)
        u2[page > 0] = 0
        content = cv2.bitwise_or(content, u2)
        mode = "lineart"
    else:
        content = _u2net_foreground(bgr)
        mode = "saliency"

    # Tidy: drop speckles (stray marks, floating sfx letters stay in the
    # background layer), fill pinholes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)
    h, w = content.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(content)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < 0.003 * h * w:
            content[labels == lbl] = 0

    logger.info("Content mode=%s page_coverage=%.2f", mode, coverage)
    return content, page, mode


# ---------------------------------------------------------------------------
# Character instances — automatic
# ---------------------------------------------------------------------------
def _sam_everything_masks(bgr: np.ndarray) -> list[np.ndarray]:
    h, w = bgr.shape[:2]
    out: list[np.ndarray] = []
    try:
        results = _get_sam()(
            bgr, device="cpu", retina_masks=True, imgsz=640,
            conf=0.4, iou=0.9, verbose=False,
        )
        r = results[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                inst = (m > 0.5).astype(np.uint8) * 255
                if inst.shape != (h, w):
                    inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_NEAREST)
                out.append(inst)
    except Exception as exc:
        logger.warning("FastSAM everything-mode failed: %s", exc)
    return out


def _auto_instances(
    bgr: np.ndarray,
    char_mask: np.ndarray,
    min_component_ratio: float = 0.02,
    split_min_share: float = 0.20,
    split_min_union: float = 0.60,
) -> list[np.ndarray]:
    """
    Automatic character split. Base unit = connected component of the
    content mask (an enclosed outline = one object). A component is only
    subdivided when FastSAM proposes >= 2 sub-masks that each cover a
    substantial share of it — this avoids shattering one character into
    head/body parts, at the cost of keeping heavily-overlapping
    characters merged (use click points for those).
    """
    char_area = int(np.count_nonzero(char_mask))
    if char_area == 0:
        return []

    h, w = char_mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(char_mask)
    big: list[np.ndarray] = []
    small: list[np.ndarray] = []
    for lbl in range(1, n):
        blob = np.where(labels == lbl, 255, 0).astype(np.uint8)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_ratio * char_area:
            big.append(blob)
        else:
            small.append(blob)

    sam_masks = _sam_everything_masks(bgr) if big else []

    instances: list[np.ndarray] = []
    for comp in big:
        comp_area = int(np.count_nonzero(comp))
        cands = []
        for sm in sam_masks:
            clipped = cv2.bitwise_and(sm, comp)
            area = int(np.count_nonzero(clipped))
            if split_min_share * comp_area <= area <= 0.85 * comp_area:
                cands.append((area, clipped))
        # Dedup heavy overlaps, keep larger.
        cands.sort(key=lambda t: -t[0])
        picked: list[np.ndarray] = []
        for area, cm in cands:
            overlap = any(
                np.count_nonzero(cv2.bitwise_and(cm, p)) > 0.5 * area
                for p in picked
            )
            if not overlap:
                picked.append(cm)

        union_px = int(np.count_nonzero(
            np.bitwise_or.reduce(picked) if picked else np.zeros_like(comp)))
        if len(picked) >= 2 and union_px >= split_min_union * comp_area:
            claimed = np.zeros_like(comp)
            for cm in picked:
                rest = cv2.bitwise_and(cm, cv2.bitwise_not(claimed))
                if np.count_nonzero(rest) > 0:
                    instances.append(rest)
                    claimed = cv2.bitwise_or(claimed, rest)
            leftover = cv2.bitwise_and(comp, cv2.bitwise_not(claimed))
            _attach_blobs(leftover, instances)
        else:
            instances.append(comp)

    # Attach small components (clouds, accessories) to the nearest
    # instance; unattached ones stay in the background.
    for blob in small:
        _attach_blobs(blob, instances)

    instances.sort(key=_centroid_x)
    logger.info("Auto instances: %d", len(instances))
    return instances


def _attach_blobs(mask: np.ndarray, instances: list[np.ndarray],
                  reach: int = 15) -> None:
    """Merge each blob of `mask` into the touching/nearest instance (in place)."""
    if not instances or not np.count_nonzero(mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    n, labels = cv2.connectedComponents(mask)
    for lbl in range(1, n):
        blob = np.where(labels == lbl, 255, 0).astype(np.uint8)
        grown = cv2.dilate(blob, kernel)
        overlaps = [int(np.count_nonzero(cv2.bitwise_and(grown, inst)))
                    for inst in instances]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0:
            instances[best] = cv2.bitwise_or(instances[best], blob)


def _centroid_x(m: np.ndarray) -> float:
    xs = np.nonzero(m)[1]
    return float(xs.mean()) if xs.size else 0.0


# ---------------------------------------------------------------------------
# Character instances — user click points
# ---------------------------------------------------------------------------
def _point_instances(
    bgr: np.ndarray,
    points: list[tuple[int, int]],
    group_cap_ratio: float = 0.70,
) -> list[np.ndarray]:
    """
    One FastSAM point-prompt per click. The prompt returns nested
    candidates (part / object / group). Selection rule: prefer the
    largest candidate that does NOT contain any *other* click point —
    each click is one character, so a mask swallowing two clicks is a
    group. With a single click that rule can't discriminate, so
    group-level masks covering most of the panel's ink are dropped
    instead. Pixels claimed by an earlier click are excluded from later
    ones.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink_area = max(int(np.count_nonzero(gray < 200)), 1)

    pts = [(int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1)))
           for (px, py) in points]

    instances: list[np.ndarray] = []
    claimed = np.zeros((h, w), dtype=np.uint8)
    for idx, (px, py) in enumerate(pts):
        other_pts = [p for i, p in enumerate(pts) if i != idx]
        best: np.ndarray | None = None
        try:
            results = _get_sam()(
                bgr, device="cpu", retina_masks=True, imgsz=640,
                points=[[px, py]], labels=[1], verbose=False,
            )
            r = results[0]
            if r.masks is not None:
                cands = []
                for m in r.masks.data.cpu().numpy():
                    inst = (m > 0.5).astype(np.uint8) * 255
                    if inst.shape != (h, w):
                        inst = cv2.resize(inst, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                    if inst[py, px] == 0:
                        continue
                    cands.append((int(np.count_nonzero(inst)), inst))
                exclusive = [(a, m) for a, m in cands
                             if not any(m[oy, ox] > 0 for (ox, oy) in other_pts)]
                non_group = [(a, m) for a, m in cands
                             if a <= group_cap_ratio * ink_area]
                pool = exclusive or non_group or cands
                if pool:
                    _, best = max(pool, key=lambda t: t[0])
        except Exception as exc:
            logger.warning("FastSAM point prompt failed at (%d,%d): %s",
                           px, py, exc)

        if best is None:
            continue
        rest = cv2.bitwise_and(best, cv2.bitwise_not(claimed))
        if np.count_nonzero(rest) > 0:
            instances.append(rest)
            claimed = cv2.bitwise_or(claimed, rest)

    logger.info("Point instances: %d from %d clicks", len(instances), len(points))
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
def decompose_panel(
    image_bytes: bytes,
    points: list[tuple[int, int]] | None = None,
) -> DecomposeResult:
    """
    Decompose a comic panel into character / bubble / background layers.

    `points`: optional click coordinates (pixel space of the input
    image). When given, character extraction is driven by FastSAM point
    prompts — one character per click — instead of the automatic split.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")

    h, w = bgr.shape[:2]

    content, page_mask, mode = _content_mask(bgr)
    bubble_mask = _detect_bubble_mask(bgr, page_mask)

    # A bubble overlapping a character belongs to the bubble layer only;
    # dilate before subtracting so the bubble's anti-aliased outline
    # doesn't survive as a ring in a character layer.
    sub_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bubble_dilated = cv2.dilate(bubble_mask, sub_kernel, iterations=2)
    char_region = cv2.bitwise_and(content, cv2.bitwise_not(bubble_dilated))

    if points:
        instance_masks = _point_instances(bgr, points)
        for i, m in enumerate(instance_masks):
            instance_masks[i] = cv2.bitwise_and(
                m, cv2.bitwise_not(bubble_dilated))
        instance_masks = [m for m in instance_masks if np.count_nonzero(m)]
        instance_masks.sort(key=_centroid_x)
        split_mode = "points"
    else:
        instance_masks = _auto_instances(bgr, char_region)
        split_mode = "auto"

    # Background: remove extracted characters + bubbles, inpaint holes.
    removed = bubble_mask.copy()
    for m in instance_masks:
        removed = cv2.bitwise_or(removed, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_mask = cv2.dilate(removed, kernel, iterations=1)
    background = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5,
                             flags=cv2.INPAINT_TELEA)

    char_px = int(sum(np.count_nonzero(m) for m in instance_masks))
    bubble_px = int(np.count_nonzero(bubble_mask))
    total_px = h * w

    return DecomposeResult(
        characters=[_encode_png_b64(_layer_rgba(bgr, m)) for m in instance_masks],
        bubbles=_encode_png_b64(_layer_rgba(bgr, bubble_mask)),
        background=_encode_png_b64(background),
        metadata={
            "width": w,
            "height": h,
            "content_mode": mode,
            "split_mode": split_mode,
            "character_count": len(instance_masks),
            "character_coverage": round(char_px / total_px, 4),
            "bubble_coverage": round(bubble_px / total_px, 4),
        },
    )
