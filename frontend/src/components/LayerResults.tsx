import { useCallback, useRef, useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────
export interface DecomposeResponse {
  characters: string[]
  bubbles: string
  background: string
  metadata: Record<string, unknown>
  processing_time_ms: number
}

export interface LayerSet {
  characters: string[]
  bubbles: string
  background: string
  processingMs: number
  splitMode: string
}

export type LayersState =
  | { status: 'loading' }
  | { status: 'done'; layers: LayerSet }
  | { status: 'error'; error: string }

export type Point = [number, number]

// ── API helper ───────────────────────────────────────────────────────────────
export async function decomposeImage(
  imageB64: string,
  points?: Point[],
): Promise<LayerSet> {
  const res = await fetch('/api/decompose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: imageB64, points: points ?? null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Server error')
  }
  const data: DecomposeResponse = await res.json()
  return {
    characters: data.characters,
    bubbles: data.bubbles,
    background: data.background,
    processingMs: data.processing_time_ms,
    splitMode: String(data.metadata?.split_mode ?? 'auto'),
  }
}

// ── Layer results display ────────────────────────────────────────────────────
interface Props {
  state: LayersState | undefined
  baseName: string
  idSuffix: string | number
  /** Original image (base64) — needed for the manual character picker. */
  imageB64: string
  /** Re-run decomposition; `points` are pixel coords in the original image. */
  onDecompose: (points?: Point[]) => void
}

export default function LayerResults({
  state, baseName, idSuffix, imageB64, onDecompose,
}: Props) {
  const [picking, setPicking] = useState(false)
  const [points, setPoints] = useState<Point[]>([])
  const imgRef = useRef<HTMLImageElement>(null)

  const downloadLayer = useCallback((b64: string, name: string) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${b64}`
    link.download = `${name}.png`
    link.click()
  }, [])

  const addPoint = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    const img = imgRef.current
    if (!img) return
    const rect = img.getBoundingClientRect()
    // Map click from displayed size to natural pixel coordinates.
    const x = Math.round((e.clientX - rect.left) / rect.width * img.naturalWidth)
    const y = Math.round((e.clientY - rect.top) / rect.height * img.naturalHeight)
    setPoints(p => [...p, [x, y]])
  }, [])

  const applyPoints = useCallback(() => {
    if (points.length === 0) return
    setPicking(false)
    onDecompose(points)
  }, [points, onDecompose])

  if (!state) return null

  if (state.status === 'loading') {
    return (
      <div className="px-3 pb-3 flex items-center gap-2 text-xs text-surface-200/50">
        <span className="w-3 h-3 rounded-full border-2 border-accent-400 border-t-transparent animate-spin" />
        Segmenting layers…
      </div>
    )
  }

  // ── Manual character picker ──────────────────────────────────────────
  if (picking) {
    return (
      <div className="px-3 pb-3 flex flex-col gap-2" id={`picker-${idSuffix}`}>
        <div className="text-[10px] text-surface-200/50 uppercase tracking-widest">
          Click each character once ({points.length} selected)
        </div>
        <div className="relative rounded-lg overflow-hidden border border-accent-500/50 cursor-crosshair">
          <img
            ref={imgRef}
            src={`data:image/png;base64,${imageB64}`}
            alt="Pick characters"
            className="w-full h-auto object-contain select-none"
            onClick={addPoint}
            draggable={false}
          />
          {points.map(([x, y], i) => {
            const img = imgRef.current
            const left = img ? (x / img.naturalWidth) * 100 : 0
            const top = img ? (y / img.naturalHeight) * 100 : 0
            return (
              <span
                key={i}
                className="absolute w-5 h-5 -ml-2.5 -mt-2.5 rounded-full bg-accent-500 border-2 border-white text-white text-[10px] font-bold flex items-center justify-center pointer-events-none shadow"
                style={{ left: `${left}%`, top: `${top}%` }}
              >
                {i + 1}
              </span>
            )
          })}
        </div>
        <div className="flex gap-2">
          <button
            id={`picker-apply-${idSuffix}`}
            onClick={applyPoints}
            disabled={points.length === 0}
            className="flex-1 rounded-lg px-3 py-1.5 bg-accent-600 hover:bg-accent-500 text-white text-xs font-semibold transition-colors disabled:opacity-40"
          >
            Extract {points.length || ''} character{points.length === 1 ? '' : 's'}
          </button>
          <button
            id={`picker-clear-${idSuffix}`}
            onClick={() => setPoints([])}
            disabled={points.length === 0}
            className="rounded-lg px-3 py-1.5 glass border border-surface-500/40 text-surface-200/60 text-xs transition-colors hover:text-surface-200 disabled:opacity-40"
          >
            Clear
          </button>
          <button
            id={`picker-cancel-${idSuffix}`}
            onClick={() => { setPicking(false); setPoints([]) }}
            className="rounded-lg px-3 py-1.5 glass border border-surface-500/40 text-surface-200/60 text-xs transition-colors hover:text-surface-200"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div className="px-3 pb-3 flex flex-col gap-2">
        <div className="text-xs text-red-400">
          Layer split failed: {state.error}
        </div>
        <PickButton idSuffix={idSuffix} onClick={() => { setPoints([]); setPicking(true) }} />
      </div>
    )
  }

  const layers: { key: string; label: string; b64: string; transparent: boolean }[] = [
    ...state.layers.characters.map((b64, i) => ({
      key: `character_${i + 1}`,
      label: state.layers.characters.length > 1 ? `Character ${i + 1}` : 'Character',
      b64,
      transparent: true,
    })),
    { key: 'bubbles', label: 'Bubbles', b64: state.layers.bubbles, transparent: true },
    { key: 'background', label: 'Background', b64: state.layers.background, transparent: false },
  ]

  return (
    <div className="px-3 pb-3 flex flex-col gap-2" id={`layers-${idSuffix}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-[10px] text-surface-200/30 uppercase tracking-widest">
          Layers · {state.layers.characters.length} character
          {state.layers.characters.length === 1 ? '' : 's'}
          {state.layers.splitMode === 'points' ? ' · manual' : ''}
          {' · '}{state.layers.processingMs.toFixed(0)} ms
        </div>
        <PickButton idSuffix={idSuffix} onClick={() => { setPoints([]); setPicking(true) }} />
      </div>
      <div className="grid grid-cols-3 gap-2">
        {layers.map(layer => (
          <div key={layer.key} className="flex flex-col gap-1">
            <div
              className={`rounded-lg overflow-hidden border border-surface-500/40 ${
                layer.transparent ? 'bg-checker' : 'bg-surface-900/40'
              }`}
            >
              <img
                src={`data:image/png;base64,${layer.b64}`}
                alt={`${layer.label} layer`}
                className="w-full h-auto object-contain"
                loading="lazy"
              />
            </div>
            <button
              id={`download-layer-${layer.key}-${idSuffix}`}
              onClick={() => downloadLayer(layer.b64, `${baseName}_${layer.key}`)}
              className="text-[10px] text-surface-200/50 hover:text-brand-300 transition-colors flex items-center justify-center gap-1"
            >
              <DownloadIcon size={10} />
              {layer.label}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function PickButton({ idSuffix, onClick }: { idSuffix: string | number; onClick: () => void }) {
  return (
    <button
      id={`pick-characters-${idSuffix}`}
      onClick={onClick}
      className="text-[10px] text-accent-400 hover:text-accent-500 transition-colors flex items-center gap-1 flex-shrink-0"
    >
      <TargetIcon size={11} />
      Pick characters
    </button>
  )
}

function TargetIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="3" />
      <line x1="12" y1="1" x2="12" y2="5" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="1" y1="12" x2="5" y2="12" />
      <line x1="19" y1="12" x2="23" y2="12" />
    </svg>
  )
}

function DownloadIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
  )
}
