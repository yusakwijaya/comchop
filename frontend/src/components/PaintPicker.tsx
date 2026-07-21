import { useCallback, useEffect, useRef, useState } from 'react'
import type { LayerSet } from './LayerResults'

// ── Category palette ─────────────────────────────────────────────────────────
const CHAR_COLORS = ['#f97316', '#a855f7', '#22c55e', '#eab308', '#ec4899', '#06b6d4', '#f43f5e', '#84cc16']
const BUBBLE_ID = 90
const BUBBLE_COLOR = '#38bdf8'
const MAX_HISTORY = 40

type Tool = 'brush' | 'bucket'

/**
 * 3x3 median filter over RGB. Comic line art is full of halftone dot
 * screens; a raw flood fill stops at every dot. Median removes isolated
 * specks while leaving ink outlines (which are thick and contiguous)
 * intact, so the bucket fills a shaded region but still stops at edges.
 */
function medianSmooth(src: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h * 3)
  const win = new Uint8Array(9)
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let n = 0
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy
          if (yy < 0 || yy >= h) continue
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx
            if (xx < 0 || xx >= w) continue
            win[n++] = src[(yy * w + xx) * 4 + c]
          }
        }
        // Insertion sort — n is at most 9.
        for (let i = 1; i < n; i++) {
          const v = win[i]
          let j = i - 1
          while (j >= 0 && win[j] > v) { win[j + 1] = win[j]; j-- }
          win[j + 1] = v
        }
        out[(y * w + x) * 3 + c] = win[n >> 1]
      }
    }
  }
  return out
}

interface Category {
  id: number
  label: string
  color: string
}

interface Props {
  imageB64: string
  /**
   * Existing layer result (e.g. from the automatic ML decompose) to
   * pre-fill the paint as a starting point — the user then corrects it
   * with the brush instead of painting everything from scratch. Each
   * layer's alpha channel is read back as that category's painted mask.
   */
  initialLayers?: { characters: string[]; bubbles: string }
  onApply: (layers: LayerSet) => void
  onCancel: () => void
}

/** Decode a base64 RGBA PNG and claim its opaque pixels into ownerMap. */
async function claimLayerAlpha(
  b64: string, categoryId: number, ownerMap: Uint8Array, w: number, h: number,
): Promise<void> {
  const img = new Image()
  await new Promise<void>(resolve => {
    img.onload = () => resolve()
    img.onerror = () => resolve()
    img.src = `data:image/png;base64,${b64}`
  })
  if (!img.naturalWidth) return
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(img, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h).data
  for (let i = 0; i < w * h; i++) {
    if (data[i * 4 + 3] > 127) ownerMap[i] = categoryId
  }
}

export default function PaintPicker({ imageB64, initialLayers, onApply, onCancel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ownerMapRef = useRef<Uint8Array | null>(null)
  const origDataRef = useRef<ImageData | null>(null)
  const overlayDataRef = useRef<ImageData | null>(null)
  const sizeRef = useRef({ w: 0, h: 0 })
  const paintingRef = useRef(false)
  const dirtyRef = useRef(false)
  const rafRef = useRef<number | null>(null)
  const undoStackRef = useRef<Uint8Array[]>([])
  const redoStackRef = useRef<Uint8Array[]>([])
  const strokeSnapshotTakenRef = useRef(false)
  // Last point painted (natural image coords) — Photoshop-style shift+click
  // paints a straight line from here to the new click point.
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  // Median-smoothed copy of the panel, built lazily on first bucket use
  // and reused for every later fill.
  const smoothedRef = useRef<Uint8Array | null>(null)

  const [ready, setReady] = useState(false)
  const [categories, setCategories] = useState<Category[]>([
    { id: 1, label: 'Character 1', color: CHAR_COLORS[0] },
  ])
  const [activeId, setActiveId] = useState(1)
  const [tool, setTool] = useState<Tool>('brush')
  const [tolerance, setTolerance] = useState(40)
  const [brushRadius, setBrushRadius] = useState(24)
  const [bgColor, setBgColor] = useState('#ffffff')
  const [paintedIds, setPaintedIds] = useState<Set<number>>(new Set())
  const [cursor, setCursor] = useState<{ x: number; y: number; scale: number } | null>(null)
  const [undoCount, setUndoCount] = useState(0)
  const [redoCount, setRedoCount] = useState(0)

  // ── Load image into canvas ────────────────────────────────────────────
  useEffect(() => {
    const img = new Image()
    img.onload = async () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const w = img.naturalWidth
      const h = img.naturalHeight
      canvas.width = w
      canvas.height = h
      sizeRef.current = { w, h }
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      const orig = ctx.getImageData(0, 0, w, h)
      origDataRef.current = orig
      overlayDataRef.current = new ImageData(
        new Uint8ClampedArray(orig.data), w, h,
      )
      const ownerMap = new Uint8Array(w * h)
      ownerMapRef.current = ownerMap
      setBrushRadius(Math.max(8, Math.round(Math.min(w, h) / 20)))

      // Pre-fill from an existing result so the user corrects it with the
      // brush rather than repainting everything from scratch.
      if (initialLayers) {
        for (let i = 0; i < initialLayers.characters.length; i++) {
          await claimLayerAlpha(initialLayers.characters[i], i + 1, ownerMap, w, h)
        }
        await claimLayerAlpha(initialLayers.bubbles, BUBBLE_ID, ownerMap, w, h)

        const n = Math.max(1, initialLayers.characters.length)
        setCategories(Array.from({ length: n }, (_, i) => ({
          id: i + 1,
          label: `Character ${i + 1}`,
          color: CHAR_COLORS[i % CHAR_COLORS.length],
        })))

        const ids = new Set<number>()
        const overlay = overlayDataRef.current!.data
        for (let i = 0; i < w * h; i++) {
          const id = ownerMap[i]
          if (id === 0) continue
          ids.add(id)
          const hex = id === BUBBLE_ID
            ? BUBBLE_COLOR
            : CHAR_COLORS[(id - 1) % CHAR_COLORS.length]
          blendPixel(i, orig.data, overlay, hexToRgb(hex), 0.5)
        }
        setPaintedIds(ids)
        ctx.putImageData(overlayDataRef.current!, 0, 0)
      }

      setReady(true)
    }
    img.src = `data:image/png;base64,${imageB64}`
    // The picker unmounts whenever it closes, so this runs once per open;
    // initialLayers is captured at mount and intentionally not a dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageB64])

  const categoryById = useCallback((id: number): Category | undefined => {
    if (id === BUBBLE_ID) return { id: BUBBLE_ID, label: 'Bubbles', color: BUBBLE_COLOR }
    return categories.find(c => c.id === id)
  }, [categories])

  const hexToRgb = (hex: string): [number, number, number] => {
    const v = parseInt(hex.slice(1), 16)
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255]
  }

  const scheduleRedraw = useCallback(() => {
    dirtyRef.current = true
    if (rafRef.current != null) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      if (!dirtyRef.current) return
      dirtyRef.current = false
      const canvas = canvasRef.current
      const overlay = overlayDataRef.current
      if (!canvas || !overlay) return
      canvas.getContext('2d')!.putImageData(overlay, 0, 0)
    })
  }, [])

  // Blend `color` at `alpha` onto the overlay buffer for one pixel index.
  const blendPixel = (idx: number, orig: Uint8ClampedArray, overlay: Uint8ClampedArray,
                      color: [number, number, number] | null, alpha: number) => {
    const o = idx * 4
    if (color === null) {
      overlay[o] = orig[o]; overlay[o + 1] = orig[o + 1]; overlay[o + 2] = orig[o + 2]
      return
    }
    overlay[o] = orig[o] * (1 - alpha) + color[0] * alpha
    overlay[o + 1] = orig[o + 1] * (1 - alpha) + color[1] * alpha
    overlay[o + 2] = orig[o + 2] * (1 - alpha) + color[2] * alpha
  }

  const paintAt = useCallback((cx: number, cy: number, categoryId: number) => {
    const { w, h } = sizeRef.current
    const ownerMap = ownerMapRef.current
    const orig = origDataRef.current
    const overlay = overlayDataRef.current
    if (!ownerMap || !orig || !overlay) return

    // Snapshot once per stroke — right before its first mutation — so
    // undo reverts a whole stroke, not one brush dab.
    if (!strokeSnapshotTakenRef.current) {
      strokeSnapshotTakenRef.current = true
      const stack = undoStackRef.current
      stack.push(ownerMap.slice())
      if (stack.length > MAX_HISTORY) stack.shift()
      redoStackRef.current = []
      setUndoCount(stack.length)
      setRedoCount(0)
    }

    const cat = categoryById(categoryId)
    const color = cat ? hexToRgb(cat.color) : null
    const r = brushRadius
    const x0 = Math.max(0, Math.floor(cx - r))
    const x1 = Math.min(w - 1, Math.ceil(cx + r))
    const y0 = Math.max(0, Math.floor(cy - r))
    const y1 = Math.min(h - 1, Math.ceil(cy + r))

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        // +0.5 treats (x, y) as a pixel's center rather than its corner,
        // so the painted circle is geometrically centered on (cx, cy).
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy
        if (dx * dx + dy * dy > r * r) continue
        const idx = y * w + x
        ownerMap[idx] = categoryId
        blendPixel(idx, orig.data, overlay.data, color, 0.5)
      }
    }
    setPaintedIds(prev => (prev.has(categoryId) ? prev : new Set(prev).add(categoryId)))
    lastPointRef.current = { x: cx, y: cy }
    scheduleRedraw()
  }, [brushRadius, categoryById, scheduleRedraw])

  // Photoshop-style shift+click: stamp overlapping dabs along the segment
  // from the last painted point to (cx, cy) so the stroke looks continuous
  // rather than a series of separate circles.
  const paintLine = useCallback((x0: number, y0: number, x1: number, y1: number, categoryId: number) => {
    const dist = Math.hypot(x1 - x0, y1 - y0)
    const step = Math.max(1, brushRadius / 3)
    const steps = Math.max(1, Math.ceil(dist / step))
    for (let i = 1; i <= steps; i++) {
      const t = i / steps
      paintAt(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, categoryId)
    }
  }, [brushRadius, paintAt])

  const toImageCoords = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      // Display px per natural px — used to size the brush cursor to match
      // the actual painted radius regardless of how the canvas is scaled.
      displayX: e.clientX - rect.left,
      displayY: e.clientY - rect.top,
      scale: rect.width / canvas.width,
    }
  }, [])

  const recomputePaintedIds = useCallback(() => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    const ids = new Set<number>()
    for (let i = 0; i < ownerMap.length; i++) {
      if (ownerMap[i] !== 0) ids.add(ownerMap[i])
    }
    setPaintedIds(ids)
  }, [])

  const pushHistory = useCallback(() => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    const stack = undoStackRef.current
    stack.push(ownerMap.slice())
    if (stack.length > MAX_HISTORY) stack.shift()
    redoStackRef.current = []
    setUndoCount(stack.length)
    setRedoCount(0)
  }, [])

  /**
   * Bucket fill: claim every pixel reachable from (sx, sy) whose colour
   * stays within `tolerance` of the clicked colour. Runs on the
   * median-smoothed copy so halftone dots don't stop the flood, while
   * ink outlines still bound it.
   */
  const bucketFill = useCallback((sx: number, sy: number, categoryId: number) => {
    const { w, h } = sizeRef.current
    const ownerMap = ownerMapRef.current
    const orig = origDataRef.current
    const overlay = overlayDataRef.current
    if (!ownerMap || !orig || !overlay) return

    const x0 = Math.floor(sx), y0 = Math.floor(sy)
    if (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) return

    if (!smoothedRef.current) {
      smoothedRef.current = medianSmooth(orig.data, w, h)
    }
    const sm = smoothedRef.current

    const start = (y0 * w + x0) * 3
    const tr = sm[start], tg = sm[start + 1], tb = sm[start + 2]
    const tol = tolerance

    const cat = categoryById(categoryId)
    const color = cat ? hexToRgb(cat.color) : null

    // Scanline flood fill — far fewer queue operations than per-pixel
    // 4-way flooding over the large flat regions typical of comic art.
    const seen = new Uint8Array(w * h)
    const match = (x: number, y: number): boolean => {
      const i = y * w + x
      if (seen[i]) return false
      const o = i * 3
      return Math.abs(sm[o] - tr) <= tol
        && Math.abs(sm[o + 1] - tg) <= tol
        && Math.abs(sm[o + 2] - tb) <= tol
    }

    // Nothing to do if the click can't even seed a fill.
    if (!match(x0, y0)) return
    pushHistory()

    const stack: number[] = [x0, y0]
    while (stack.length) {
      const y = stack.pop()!
      let x = stack.pop()!
      if (!match(x, y)) continue

      let left = x
      while (left > 0 && match(left - 1, y)) left--
      let right = x
      while (right < w - 1 && match(right + 1, y)) right++

      for (x = left; x <= right; x++) {
        const i = y * w + x
        seen[i] = 1
        ownerMap[i] = categoryId
        blendPixel(i, orig.data, overlay.data, color, 0.5)
        if (y > 0 && match(x, y - 1)) stack.push(x, y - 1)
        if (y < h - 1 && match(x, y + 1)) stack.push(x, y + 1)
      }
    }

    recomputePaintedIds()
    lastPointRef.current = { x: sx, y: sy }
    scheduleRedraw()
  }, [tolerance, categoryById, recomputePaintedIds, scheduleRedraw, pushHistory])

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = toImageCoords(e)

    if (tool === 'bucket') {
      // A fill is one discrete action — never the start of a drag.
      bucketFill(x, y, activeId)
      return
    }

    paintingRef.current = true
    strokeSnapshotTakenRef.current = false
    if (e.shiftKey && lastPointRef.current) {
      const { x: lx, y: ly } = lastPointRef.current
      paintLine(lx, ly, x, y, activeId)
      // A shift+click is a discrete action, not the start of a drag —
      // ignore any further movement until the button is released again.
      paintingRef.current = false
    } else {
      paintAt(x, y, activeId)
    }
  }, [toImageCoords, paintAt, paintLine, bucketFill, tool, activeId])

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y, displayX, displayY, scale } = toImageCoords(e)
    setCursor({ x: displayX, y: displayY, scale })
    if (!paintingRef.current) return
    // Connect to the previous sample instead of stamping a single dab —
    // on a fast drag the browser can report mousemove far less often than
    // the pointer actually travels, leaving visible gaps and making the
    // cursor ring look ahead of / uncentered on the painted trail.
    const prev = lastPointRef.current
    if (prev) paintLine(prev.x, prev.y, x, y, activeId)
    else paintAt(x, y, activeId)
  }, [toImageCoords, paintAt, paintLine, activeId])

  const stopPainting = useCallback(() => { paintingRef.current = false }, [])
  const hideCursor = useCallback(() => { paintingRef.current = false; setCursor(null) }, [])

  const rebuildOverlay = useCallback(() => {
    const { w, h } = sizeRef.current
    const ownerMap = ownerMapRef.current
    const orig = origDataRef.current
    const overlay = overlayDataRef.current
    if (!ownerMap || !orig || !overlay) return
    for (let i = 0; i < w * h; i++) {
      const id = ownerMap[i]
      const cat = id === 0 ? undefined : categoryById(id)
      blendPixel(i, orig.data, overlay.data, cat ? hexToRgb(cat.color) : null, 0.5)
    }
    scheduleRedraw()
  }, [categoryById, scheduleRedraw])

  const restoreSnapshot = useCallback((snapshot: Uint8Array) => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    ownerMap.set(snapshot)
    recomputePaintedIds()
    rebuildOverlay()
  }, [recomputePaintedIds, rebuildOverlay])

  const undo = useCallback(() => {
    const ownerMap = ownerMapRef.current
    const snapshot = undoStackRef.current.pop()
    if (!ownerMap || !snapshot) return
    redoStackRef.current.push(ownerMap.slice())
    restoreSnapshot(snapshot)
    setUndoCount(undoStackRef.current.length)
    setRedoCount(redoStackRef.current.length)
  }, [restoreSnapshot])

  const redo = useCallback(() => {
    const ownerMap = ownerMapRef.current
    const snapshot = redoStackRef.current.pop()
    if (!ownerMap || !snapshot) return
    undoStackRef.current.push(ownerMap.slice())
    restoreSnapshot(snapshot)
    setUndoCount(undoStackRef.current.length)
    setRedoCount(redoStackRef.current.length)
  }, [restoreSnapshot])

  const clearCategory = useCallback((id: number) => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    pushHistory()
    for (let i = 0; i < ownerMap.length; i++) {
      if (ownerMap[i] === id) ownerMap[i] = 0
    }
    setPaintedIds(prev => { const n = new Set(prev); n.delete(id); return n })
    rebuildOverlay()
  }, [rebuildOverlay, pushHistory])

  const clearAll = useCallback(() => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    pushHistory()
    ownerMap.fill(0)
    setPaintedIds(new Set())
    lastPointRef.current = null
    rebuildOverlay()
  }, [rebuildOverlay, pushHistory])

  const addCharacter = useCallback(() => {
    setCategories(prev => {
      const nextId = prev.length ? Math.max(...prev.map(c => c.id)) + 1 : 1
      const color = CHAR_COLORS[(nextId - 1) % CHAR_COLORS.length]
      const next = [...prev, { id: nextId, label: `Character ${nextId}`, color }]
      setActiveId(nextId)
      return next
    })
  }, [])

  const removeCharacter = useCallback((id: number) => {
    setCategories(prev => prev.filter(c => c.id !== id))
    clearCategory(id)
    setActiveId(prev => (prev === id ? (categories.find(c => c.id !== id)?.id ?? 0) : prev))
  }, [clearCategory, categories])

  // ── Build final layers from ownerMap (direct-paint = layer, no ML) ────
  const buildLayerCanvas = useCallback((predicate: (id: number) => boolean): string => {
    const { w, h } = sizeRef.current
    const orig = origDataRef.current!
    const ownerMap = ownerMapRef.current!
    const out = new ImageData(w, h)
    for (let i = 0; i < w * h; i++) {
      const o = i * 4
      const claimed = predicate(ownerMap[i])
      out.data[o] = orig.data[o]
      out.data[o + 1] = orig.data[o + 1]
      out.data[o + 2] = orig.data[o + 2]
      out.data[o + 3] = claimed ? 255 : 0
    }
    const c = document.createElement('canvas')
    c.width = w; c.height = h
    c.getContext('2d')!.putImageData(out, 0, 0)
    return c.toDataURL('image/png').split(',')[1]
  }, [])

  const buildBackgroundCanvas = useCallback((bgHex: string): string => {
    const { w, h } = sizeRef.current
    const orig = origDataRef.current!
    const ownerMap = ownerMapRef.current!
    const [br, bgc, bb] = hexToRgb(bgHex)
    const out = new ImageData(w, h)
    for (let i = 0; i < w * h; i++) {
      const o = i * 4
      const claimed = ownerMap[i] !== 0
      out.data[o] = claimed ? br : orig.data[o]
      out.data[o + 1] = claimed ? bgc : orig.data[o + 1]
      out.data[o + 2] = claimed ? bb : orig.data[o + 2]
      out.data[o + 3] = 255
    }
    const c = document.createElement('canvas')
    c.width = w; c.height = h
    c.getContext('2d')!.putImageData(out, 0, 0)
    return c.toDataURL('image/png').split(',')[1]
  }, [])

  const apply = useCallback(() => {
    const t0 = performance.now()
    const charIds = categories.map(c => c.id).filter(id => paintedIds.has(id))
    const characters = charIds.map(id => buildLayerCanvas(owner => owner === id))
    const bubbles = paintedIds.has(BUBBLE_ID)
      ? buildLayerCanvas(owner => owner === BUBBLE_ID)
      : buildLayerCanvas(() => false)
    const background = buildBackgroundCanvas(bgColor)
    onApply({
      characters,
      bubbles,
      background,
      processingMs: performance.now() - t0,
      splitMode: 'paint',
    })
  }, [categories, paintedIds, buildLayerCanvas, buildBackgroundCanvas, bgColor, onApply])

  // Keyboard shortcuts: Cmd/Ctrl+Z to undo, Shift+Cmd/Ctrl+Z (or Ctrl+Y) to redo.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [undo, redo])

  const activeCat = activeId === BUBBLE_ID
    ? { id: BUBBLE_ID, label: 'Bubbles', color: BUBBLE_COLOR }
    : categories.find(c => c.id === activeId)

  return (
    <div className="px-3 pb-3 flex flex-col gap-2" id="paint-picker">
      <div className="text-[10px] text-surface-200/50 uppercase tracking-widest">
        Paint each layer, then extract
      </div>
      <div className="text-[10px] text-surface-200/30 -mt-1">
        {tool === 'brush'
          ? 'Tip: hold Shift and click to draw a straight line from your last stroke'
          : 'Tip: click inside an outlined area to fill it — raise Tolerance if the fill stops too early'}
      </div>

      {/* Tool picker */}
      <div className="flex items-center gap-1.5">
        <button
          id="paint-tool-brush"
          onClick={() => setTool('brush')}
          className={`text-[10px] rounded-lg px-2.5 py-1 border transition-colors ${
            tool === 'brush'
              ? 'bg-accent-600 border-accent-500 text-white'
              : 'glass border-surface-500/40 text-surface-200/60 hover:text-surface-200'
          }`}
        >
          Brush
        </button>
        <button
          id="paint-tool-bucket"
          onClick={() => setTool('bucket')}
          className={`text-[10px] rounded-lg px-2.5 py-1 border transition-colors ${
            tool === 'bucket'
              ? 'bg-accent-600 border-accent-500 text-white'
              : 'glass border-surface-500/40 text-surface-200/60 hover:text-surface-200'
          }`}
        >
          Bucket
        </button>
      </div>

      {/* Category chips */}
      <div className="flex flex-wrap items-center gap-1.5">
        {categories.map(cat => (
          <CategoryChip
            key={cat.id}
            cat={cat}
            active={activeId === cat.id}
            onClick={() => setActiveId(cat.id)}
            onRemove={categories.length > 1 ? () => removeCharacter(cat.id) : undefined}
          />
        ))}
        <button
          id="paint-add-character"
          onClick={addCharacter}
          className="text-[10px] rounded-full px-2.5 py-1 border border-dashed border-surface-500/50 text-surface-200/50 hover:text-surface-200 hover:border-surface-400 transition-colors"
        >
          + Character
        </button>
        <CategoryChip
          cat={{ id: BUBBLE_ID, label: 'Bubbles', color: BUBBLE_COLOR }}
          active={activeId === BUBBLE_ID}
          onClick={() => setActiveId(BUBBLE_ID)}
        />
        <button
          id="paint-eraser"
          onClick={() => setActiveId(0)}
          className={`text-[10px] rounded-full px-2.5 py-1 border transition-colors ${
            activeId === 0
              ? 'bg-surface-200/20 border-surface-200/50 text-surface-200'
              : 'border-surface-500/40 text-surface-200/50 hover:text-surface-200'
          }`}
        >
          Eraser
        </button>
      </div>

      {/* Canvas */}
      <div className="relative rounded-lg overflow-hidden border border-accent-500/50">
        <canvas
          ref={canvasRef}
          className={`w-full h-auto block select-none ${
            tool === 'bucket' ? 'cursor-crosshair' : 'cursor-none'
          }`}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={stopPainting}
          onMouseLeave={hideCursor}
        />
        {cursor && tool === 'brush' && (
          <div
            className="absolute rounded-full border-2 pointer-events-none"
            style={{
              left: cursor.x,
              top: cursor.y,
              width: brushRadius * 2 * cursor.scale,
              height: brushRadius * 2 * cursor.scale,
              transform: 'translate(-50%, -50%)',
              borderColor: activeId === 0 ? '#e2e4f0' : (activeCat?.color ?? '#e2e4f0'),
              backgroundColor: activeId === 0 ? 'transparent' : `${activeCat?.color ?? '#e2e4f0'}33`,
            }}
          />
        )}
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-900/60 text-xs text-surface-200/50">
            Loading…
          </div>
        )}
      </div>

      {/* Brush + bg color controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-1.5">
          <button
            id="paint-undo"
            onClick={undo}
            disabled={undoCount === 0}
            title="Undo (Ctrl/Cmd+Z)"
            className="rounded-lg p-1.5 glass border border-surface-500/40 text-surface-200/60 transition-colors hover:text-surface-200 disabled:opacity-30"
          >
            <UndoIcon size={13} />
          </button>
          <button
            id="paint-redo"
            onClick={redo}
            disabled={redoCount === 0}
            title="Redo (Shift+Ctrl/Cmd+Z)"
            className="rounded-lg p-1.5 glass border border-surface-500/40 text-surface-200/60 transition-colors hover:text-surface-200 disabled:opacity-30"
          >
            <UndoIcon size={13} flip />
          </button>
        </div>
        {tool === 'brush' ? (
          <label className="flex items-center gap-2 text-[10px] text-surface-200/60">
            Brush
            <input
              id="paint-brush-size"
              type="range" min={4} max={100} value={brushRadius}
              onChange={e => setBrushRadius(Number(e.target.value))}
              className="w-20 accent-accent-500"
            />
          </label>
        ) : (
          <label className="flex items-center gap-2 text-[10px] text-surface-200/60">
            Tolerance
            <input
              id="paint-tolerance"
              type="range" min={0} max={120} value={tolerance}
              onChange={e => setTolerance(Number(e.target.value))}
              className="w-20 accent-accent-500"
            />
            <span className="font-mono text-surface-200/40 w-6">{tolerance}</span>
          </label>
        )}
        <label className="flex items-center gap-2 text-[10px] text-surface-200/60">
          Background fill
          <input
            type="color" value={bgColor}
            onChange={e => setBgColor(e.target.value)}
            className="w-6 h-6 rounded border border-surface-500/40 bg-transparent cursor-pointer"
          />
        </label>
        {activeCat && (
          <span className="text-[10px] text-surface-200/40">
            Painting: <span style={{ color: activeCat.color }}>{activeCat.label}</span>
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          id="paint-apply"
          onClick={apply}
          disabled={paintedIds.size === 0}
          className="flex-1 rounded-lg px-3 py-1.5 bg-accent-600 hover:bg-accent-500 text-white text-xs font-semibold transition-colors disabled:opacity-40"
        >
          Extract layers
        </button>
        <button
          id="paint-clear-all"
          onClick={clearAll}
          disabled={paintedIds.size === 0}
          className="rounded-lg px-3 py-1.5 glass border border-surface-500/40 text-surface-200/60 text-xs transition-colors hover:text-surface-200 disabled:opacity-40"
        >
          Clear all
        </button>
        <button
          id="paint-cancel"
          onClick={onCancel}
          className="rounded-lg px-3 py-1.5 glass border border-surface-500/40 text-surface-200/60 text-xs transition-colors hover:text-surface-200"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

function UndoIcon({ size = 16, flip = false }: { size?: number; flip?: boolean }) {
  return (
    <svg
      width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2}
      viewBox="0 0 24 24" style={flip ? { transform: 'scaleX(-1)' } : undefined}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 14L4 9l5-5" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 9h10.5a5.5 5.5 0 015.5 5.5v0a5.5 5.5 0 01-5.5 5.5H11" />
    </svg>
  )
}

function CategoryChip({
  cat, active, onClick, onRemove,
}: {
  cat: Category
  active: boolean
  onClick: () => void
  onRemove?: () => void
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] rounded-full pl-1 pr-2.5 py-1 border cursor-pointer transition-colors ${
        active ? 'text-white' : 'text-surface-200/70 hover:text-surface-200'
      }`}
      style={{
        borderColor: cat.color,
        backgroundColor: active ? cat.color : 'transparent',
      }}
      onClick={onClick}
    >
      <span
        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
        style={{ backgroundColor: cat.color }}
      />
      {cat.label}
      {onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove() }}
          className="ml-0.5 opacity-70 hover:opacity-100"
        >
          ×
        </button>
      )}
    </span>
  )
}
