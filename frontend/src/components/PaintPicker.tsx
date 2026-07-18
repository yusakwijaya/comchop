import { useCallback, useEffect, useRef, useState } from 'react'
import type { LayerSet } from './LayerResults'

// ── Category palette ─────────────────────────────────────────────────────────
const CHAR_COLORS = ['#f97316', '#a855f7', '#22c55e', '#eab308', '#ec4899', '#06b6d4', '#f43f5e', '#84cc16']
const BUBBLE_ID = 90
const BUBBLE_COLOR = '#38bdf8'

interface Category {
  id: number
  label: string
  color: string
}

interface Props {
  imageB64: string
  onApply: (layers: LayerSet) => void
  onCancel: () => void
}

export default function PaintPicker({ imageB64, onApply, onCancel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ownerMapRef = useRef<Uint8Array | null>(null)
  const origDataRef = useRef<ImageData | null>(null)
  const overlayDataRef = useRef<ImageData | null>(null)
  const sizeRef = useRef({ w: 0, h: 0 })
  const paintingRef = useRef(false)
  const dirtyRef = useRef(false)
  const rafRef = useRef<number | null>(null)

  const [ready, setReady] = useState(false)
  const [categories, setCategories] = useState<Category[]>([
    { id: 1, label: 'Character 1', color: CHAR_COLORS[0] },
  ])
  const [activeId, setActiveId] = useState(1)
  const [brushRadius, setBrushRadius] = useState(24)
  const [bgColor, setBgColor] = useState('#ffffff')
  const [paintedIds, setPaintedIds] = useState<Set<number>>(new Set())
  const [cursor, setCursor] = useState<{ x: number; y: number; scale: number } | null>(null)

  // ── Load image into canvas ────────────────────────────────────────────
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
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
      ownerMapRef.current = new Uint8Array(w * h)
      setBrushRadius(Math.max(8, Math.round(Math.min(w, h) / 20)))
      setReady(true)
    }
    img.src = `data:image/png;base64,${imageB64}`
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

    const cat = categoryById(categoryId)
    const color = cat ? hexToRgb(cat.color) : null
    const r = brushRadius
    const x0 = Math.max(0, Math.floor(cx - r))
    const x1 = Math.min(w - 1, Math.ceil(cx + r))
    const y0 = Math.max(0, Math.floor(cy - r))
    const y1 = Math.min(h - 1, Math.ceil(cy + r))

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy
        if (dx * dx + dy * dy > r * r) continue
        const idx = y * w + x
        ownerMap[idx] = categoryId
        blendPixel(idx, orig.data, overlay.data, color, 0.5)
      }
    }
    setPaintedIds(prev => (prev.has(categoryId) ? prev : new Set(prev).add(categoryId)))
    scheduleRedraw()
  }, [brushRadius, categoryById, scheduleRedraw])

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

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    paintingRef.current = true
    const { x, y } = toImageCoords(e)
    paintAt(x, y, activeId)
  }, [toImageCoords, paintAt, activeId])

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y, displayX, displayY, scale } = toImageCoords(e)
    setCursor({ x: displayX, y: displayY, scale })
    if (!paintingRef.current) return
    paintAt(x, y, activeId)
  }, [toImageCoords, paintAt, activeId])

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

  const clearCategory = useCallback((id: number) => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    for (let i = 0; i < ownerMap.length; i++) {
      if (ownerMap[i] === id) ownerMap[i] = 0
    }
    setPaintedIds(prev => { const n = new Set(prev); n.delete(id); return n })
    rebuildOverlay()
  }, [rebuildOverlay])

  const clearAll = useCallback(() => {
    const ownerMap = ownerMapRef.current
    if (!ownerMap) return
    ownerMap.fill(0)
    setPaintedIds(new Set())
    rebuildOverlay()
  }, [rebuildOverlay])

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

  const activeCat = activeId === BUBBLE_ID
    ? { id: BUBBLE_ID, label: 'Bubbles', color: BUBBLE_COLOR }
    : categories.find(c => c.id === activeId)

  return (
    <div className="px-3 pb-3 flex flex-col gap-2" id="paint-picker">
      <div className="text-[10px] text-surface-200/50 uppercase tracking-widest">
        Paint each layer, then extract
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
          className="w-full h-auto block cursor-none select-none"
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={stopPainting}
          onMouseLeave={hideCursor}
        />
        {cursor && (
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
        <label className="flex items-center gap-2 text-[10px] text-surface-200/60">
          Brush
          <input
            type="range" min={4} max={100} value={brushRadius}
            onChange={e => setBrushRadius(Number(e.target.value))}
            className="w-20 accent-accent-500"
          />
        </label>
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
