import { useCallback, useState } from 'react'
import JSZip from 'jszip'
import PaintPicker from './PaintPicker'

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

// ── API helper (auto mode only — manual mode runs entirely client-side) ──────
export async function decomposeImage(imageB64: string): Promise<LayerSet> {
  const res = await fetch('/api/decompose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: imageB64 }),
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
  /** Original image (base64) — needed for the manual paint picker. */
  imageB64: string
  /** Re-run automatic (server) decomposition. */
  onDecompose: () => void
  /** Apply a manually-painted layer set (computed entirely in the browser). */
  onManualResult: (layers: LayerSet) => void
}

export default function LayerResults({
  state, baseName, idSuffix, imageB64, onDecompose, onManualResult,
}: Props) {
  const [painting, setPainting] = useState(false)

  const downloadLayer = useCallback((b64: string, name: string) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${b64}`
    link.download = `${name}.png`
    link.click()
  }, [])

  const downloadAllLayers = useCallback(async (
    layerList: { key: string; b64: string }[],
  ) => {
    const zip = new JSZip()
    layerList.forEach(layer => {
      zip.file(`${baseName}_${layer.key}.png`, layer.b64, { base64: true })
    })
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${baseName}_layers.zip`
    link.click()
    URL.revokeObjectURL(url)
  }, [baseName])

  const applyPaint = useCallback((layers: LayerSet) => {
    setPainting(false)
    onManualResult(layers)
  }, [onManualResult])

  if (!state) return null

  if (state.status === 'loading') {
    return (
      <div className="px-3 pb-3 flex items-center gap-2 text-xs text-surface-200/50">
        <span className="w-3 h-3 rounded-full border-2 border-accent-400 border-t-transparent animate-spin" />
        Segmenting layers…
      </div>
    )
  }

  // ── Manual paint picker ──────────────────────────────────────────────
  if (painting) {
    return (
      <PaintPicker
        imageB64={imageB64}
        onApply={applyPaint}
        onCancel={() => setPainting(false)}
      />
    )
  }

  if (state.status === 'error') {
    return (
      <div className="px-3 pb-3 flex flex-col gap-2">
        <div className="text-xs text-red-400">
          Layer split failed: {state.error}
        </div>
        <div className="flex items-center gap-3">
          <button
            id={`retry-auto-${idSuffix}`}
            onClick={onDecompose}
            className="text-[10px] text-brand-400 hover:text-brand-300 transition-colors"
          >
            Retry auto
          </button>
          <PaintButton idSuffix={idSuffix} onClick={() => setPainting(true)} />
        </div>
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
          {state.layers.splitMode === 'paint' ? ' · manual' : ''}
          {' · '}{state.layers.processingMs.toFixed(0)} ms
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            id={`download-all-layers-${idSuffix}`}
            onClick={() => downloadAllLayers(layers)}
            className="text-[10px] text-surface-200/50 hover:text-brand-300 transition-colors flex items-center gap-1"
          >
            <DownloadIcon size={11} />
            Download all
          </button>
          <PaintButton idSuffix={idSuffix} onClick={() => setPainting(true)} />
        </div>
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

function PaintButton({ idSuffix, onClick }: { idSuffix: string | number; onClick: () => void }) {
  return (
    <button
      id={`paint-layers-${idSuffix}`}
      onClick={onClick}
      className="text-[10px] text-accent-400 hover:text-accent-500 transition-colors flex items-center gap-1 flex-shrink-0"
    >
      <BrushIcon size={11} />
      Paint layers
    </button>
  )
}

function BrushIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.53 16.122a3 3 0 00-5.78 1.128 2.25 2.25 0 01-2.4 2.245 4.5 4.5 0 008.4-2.245c0-.399-.078-.78-.22-1.128zm0 0a15.998 15.998 0 003.388-1.62m-5.043-.025a15.994 15.994 0 011.622-3.395m3.42 3.42a15.995 15.995 0 004.764-4.648l3.876-5.814a1.151 1.151 0 00-1.597-1.597L14.146 6.32a15.996 15.996 0 00-4.649 4.763m3.42 3.42a6.776 6.776 0 00-3.42-3.42" />
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
