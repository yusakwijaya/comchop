import { useCallback } from 'react'

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
}

export type LayersState =
  | { status: 'loading' }
  | { status: 'done'; layers: LayerSet }
  | { status: 'error'; error: string }

// ── API helper ───────────────────────────────────────────────────────────────
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
  }
}

// ── Layer results display ────────────────────────────────────────────────────
interface Props {
  state: LayersState | undefined
  baseName: string
  idSuffix: string | number
}

export default function LayerResults({ state, baseName, idSuffix }: Props) {
  const downloadLayer = useCallback((b64: string, name: string) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${b64}`
    link.download = `${name}.png`
    link.click()
  }, [])

  if (!state) return null

  if (state.status === 'loading') {
    return (
      <div className="px-3 pb-3 flex items-center gap-2 text-xs text-surface-200/50">
        <span className="w-3 h-3 rounded-full border-2 border-accent-400 border-t-transparent animate-spin" />
        Segmenting layers…
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <div className="px-3 pb-3 text-xs text-red-400">
        Layer split failed: {state.error}
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
      <div className="text-[10px] text-surface-200/30 uppercase tracking-widest">
        Layers · {state.layers.characters.length} character
        {state.layers.characters.length === 1 ? '' : 's'} · {state.layers.processingMs.toFixed(0)} ms
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

function DownloadIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
  )
}
