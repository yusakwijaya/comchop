import { useCallback, useState } from 'react'
import JSZip from 'jszip'
import LayerResults, { LayersState, Point, decomposeImage } from './LayerResults'

export interface Panel {
  b64: string      // base64 PNG
  row: number
  col: number
  index: number
}

interface Props {
  panels: Panel[]
  rows: number
  cols: number
  processingMs: number
  originalName: string
}

export default function PanelGrid({ panels, rows, cols, processingMs, originalName }: Props) {
  // Strip extension from original filename to use as base name
  const baseName = originalName.replace(/\.[^.]+$/, '')

  // Layer decomposition state, keyed by panel index
  const [layersByPanel, setLayersByPanel] = useState<Record<number, LayersState>>({})

  const decomposePanel = useCallback(async (panel: Panel, points?: Point[]) => {
    setLayersByPanel(s => ({ ...s, [panel.index]: { status: 'loading' } }))
    try {
      const layers = await decomposeImage(panel.b64, points)
      setLayersByPanel(s => ({ ...s, [panel.index]: { status: 'done', layers } }))
    } catch (err) {
      setLayersByPanel(s => ({
        ...s,
        [panel.index]: {
          status: 'error',
          error: err instanceof Error ? err.message : 'Unknown error',
        },
      }))
    }
  }, [])

  const downloadSingle = useCallback((panel: Panel) => {
    const link = document.createElement('a')
    link.href = `data:image/jpeg;base64,${panel.b64}`
    link.download = `${baseName}_${panel.index + 1}.jpg`
    link.click()
  }, [baseName])

  const downloadAll = useCallback(async () => {
    const zip = new JSZip()
    panels.forEach((p) => {
      const byteString = atob(p.b64)
      const ab = new Uint8Array(byteString.length)
      for (let i = 0; i < byteString.length; i++) ab[i] = byteString.charCodeAt(i)
      zip.file(`${baseName}_${p.index + 1}.jpg`, ab)
    })
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${baseName}_panels.zip`
    link.click()
    URL.revokeObjectURL(url)
  }, [panels, baseName])

  return (
    <section className="w-full animate-fade-in" id="panel-results">
      {/* ── Stats bar ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <span className="glass rounded-xl px-4 py-2 text-sm font-semibold text-brand-300 border border-brand-700/30">
            {panels.length} panels
          </span>
          <span className="glass rounded-xl px-4 py-2 text-sm text-surface-200/60 border border-surface-500/30">
            {rows}×{cols} grid
          </span>
          <span className="glass rounded-xl px-4 py-2 text-sm text-surface-200/40 border border-surface-500/20">
            ⚡ {processingMs.toFixed(0)} ms
          </span>
        </div>

        <button
          id="download-all-btn"
          onClick={downloadAll}
          className="
            btn-glow flex items-center gap-2 rounded-xl px-5 py-2.5
            bg-brand-600 hover:bg-brand-500
            text-white text-sm font-semibold
            transition-colors duration-200
          "
        >
          <DownloadIcon />
          Download all (.zip)
        </button>
      </div>

      {/* ── Panel grid ─────────────────────────────────────────────── */}
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${Math.min(cols, 4)}, minmax(0, 1fr))`,
        }}
      >
        {panels.map((panel) => (
          <div
            key={panel.index}
            id={`panel-card-${panel.index}`}
            className="panel-card glass rounded-xl overflow-hidden border border-surface-500/40 group"
          >
            <div className="relative bg-surface-900/40">
              <img
                src={`data:image/jpeg;base64,${panel.b64}`}
                alt={`Panel ${panel.index + 1} (row ${panel.row + 1}, col ${panel.col + 1})`}
                className="w-full h-auto object-contain"
                loading="lazy"
              />
              {/* hover overlay */}
              <div className="
                absolute inset-0 flex items-center justify-center
                bg-surface-900/70 opacity-0 group-hover:opacity-100
                transition-opacity duration-200 rounded-t-xl
              ">
                <div className="flex flex-col items-center gap-2">
                  <button
                    id={`download-panel-${panel.index}`}
                    onClick={() => downloadSingle(panel)}
                    className="
                      glass border border-brand-500/50 rounded-xl
                      px-4 py-2 text-sm font-semibold text-brand-300
                      hover:bg-brand-600/30 transition-colors duration-150
                      flex items-center gap-2
                    "
                  >
                    <DownloadIcon size={14} />
                    Save PNG
                  </button>
                  <button
                    id={`decompose-panel-${panel.index}`}
                    onClick={() => decomposePanel(panel)}
                    disabled={layersByPanel[panel.index]?.status === 'loading'}
                    className="
                      glass border border-accent-500/50 rounded-xl
                      px-4 py-2 text-sm font-semibold text-accent-400
                      hover:bg-accent-600/30 transition-colors duration-150
                      flex items-center gap-2 disabled:opacity-50
                    "
                  >
                    <LayersIcon size={14} />
                    Split layers
                  </button>
                </div>
              </div>
            </div>

            <div className="px-3 py-2 flex items-center justify-between">
              <span className="text-xs text-surface-200/40 font-mono">
                #{panel.index + 1}
              </span>
              <span className="text-xs text-surface-200/30">
                R{panel.row + 1} · C{panel.col + 1}
              </span>
            </div>

            <LayerResults
              state={layersByPanel[panel.index]}
              baseName={`${baseName}_${panel.index + 1}`}
              idSuffix={panel.index}
              imageB64={panel.b64}
              onDecompose={(points) => decomposePanel(panel, points)}
            />
          </div>
        ))}
      </div>
    </section>
  )
}

function LayersIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3l9 5-9 5-9-5 9-5z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13l9 5 9-5" />
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
