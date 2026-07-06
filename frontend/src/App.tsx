import { useState, useCallback, useRef } from 'react'
import ComicUploader from './components/ComicUploader'
import PanelGrid, { Panel } from './components/PanelGrid'

// ── Types ────────────────────────────────────────────────────────────────────
interface ComicSplitResult {
  filename: string
  status: 'ok' | 'error'
  panels: string[]
  rows: number
  cols: number
  grid: [number, number][]
  metadata: Record<string, unknown>
  processing_time_ms: number
  error: string | null
}

interface BatchSplitResponse {
  results: ComicSplitResult[]
  total_processing_time_ms: number
}

interface ComicGroup {
  filename: string
  status: 'ok' | 'error'
  panels: Panel[]
  rows: number
  cols: number
  processingMs: number
  error: string | null
}

interface AppState {
  status: 'idle' | 'processing' | 'done' | 'error'
  comics: ComicGroup[]
  error: string | null
}

// ── Settings defaults ─────────────────────────────────────────────────────────
const DEFAULT_WHITE_MEDIAN_THRESHOLD = 200
const DEFAULT_GUTTER_STD_THRESHOLD = 30
const DEFAULT_MIN_PANEL = 0.15

export default function App() {
  const [state, setState] = useState<AppState>({
    status: 'idle',
    comics: [],
    error: null,
  })

  // Advanced settings
  const [whiteMedianThreshold, setWhiteMedianThreshold] = useState(DEFAULT_WHITE_MEDIAN_THRESHOLD)
  const [gutterStdThreshold, setGutterStdThreshold] = useState(DEFAULT_GUTTER_STD_THRESHOLD)
  const [minPanel, setMinPanel] = useState(DEFAULT_MIN_PANEL)
  const [showSettings, setShowSettings] = useState(false)

  // Keep the last uploaded files around so "re-run" can resend them with
  // new parameter values without requiring the user to re-upload.
  const lastFilesRef = useRef<File[]>([])

  const handleUpload = useCallback(async (files: File[]) => {
    lastFilesRef.current = files
    setState(s => ({ ...s, status: 'processing', error: null }))

    const form = new FormData()
    files.forEach(f => form.append('files', f))
    form.append('white_median_threshold', String(whiteMedianThreshold))
    form.append('gutter_std_threshold', String(gutterStdThreshold))
    form.append('min_panel_ratio', String(minPanel))

    try {
      const res = await fetch('/api/split-batch', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? 'Server error')
      }
      const data: BatchSplitResponse = await res.json()

      const comics: ComicGroup[] = data.results.map(r => ({
        filename: r.filename,
        status: r.status,
        error: r.error,
        rows: r.rows,
        cols: r.cols,
        processingMs: r.processing_time_ms,
        panels: r.panels.map((b64, i) => ({
          b64,
          row: r.grid[i][0],
          col: r.grid[i][1],
          index: i,
        })),
      }))

      setState({ status: 'done', comics, error: null })
    } catch (err) {
      setState(s => ({
        ...s,
        status: 'error',
        error: err instanceof Error ? err.message : 'Unknown error',
      }))
    }
  }, [whiteMedianThreshold, gutterStdThreshold, minPanel])

  const rerun = useCallback(() => {
    if (lastFilesRef.current.length > 0) handleUpload(lastFilesRef.current)
  }, [handleUpload])

  const reset = () => setState({ status: 'idle', comics: [], error: null })

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ───────────────────────────────────────────────────── */}
      <header className="glass sticky top-0 z-50 border-b border-surface-600/50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo mark */}
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-lg shadow-brand-900/40">
              <ScissorsIcon />
            </div>
            <div>
              <h1 className="font-display font-bold text-xl leading-none gradient-text">
                ComChop
              </h1>
              <p className="text-[10px] text-surface-200/40 leading-none mt-0.5 tracking-widest uppercase">
                Panel Splitter
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {state.status === 'done' && (
              <button
                id="reset-btn"
                onClick={reset}
                className="text-sm text-surface-200/50 hover:text-surface-200 transition-colors duration-150 flex items-center gap-1.5"
              >
                <RefreshIcon />
                New image
              </button>
            )}
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="glass rounded-lg px-3 py-1.5 text-xs text-surface-200/50 hover:text-surface-200 transition-colors border border-surface-500/30"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </header>

      {/* ── Main ─────────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-12 flex flex-col gap-10">

        {/* Hero copy */}
        {state.status === 'idle' && (
          <div className="text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 text-xs text-brand-300 border border-brand-700/30 mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-pulse-slow" />
              Zero-config · Local CV · No ML weights
            </div>
            <h2 className="font-display font-bold text-4xl sm:text-5xl leading-tight mb-4">
              Drop a comic.{' '}
              <span className="gradient-text">Get the panels.</span>
            </h2>
            <p className="text-surface-200/50 max-w-lg mx-auto text-base leading-relaxed">
              ComChop uses gutter-density projection to detect panel boundaries in any
              comic layout — 2×2, 1×4, webtoon strips — without any ML models.
            </p>
          </div>
        )}

        {/* Upload zone */}
        {state.status !== 'done' && (
          <div className="animate-slide-up">
            <ComicUploader
              onUpload={handleUpload}
              isProcessing={state.status === 'processing'}
            />
            <SettingsPanel
              show={showSettings}
              onToggle={() => setShowSettings(s => !s)}
              whiteMedianThreshold={whiteMedianThreshold}
              setWhiteMedianThreshold={setWhiteMedianThreshold}
              gutterStdThreshold={gutterStdThreshold}
              setGutterStdThreshold={setGutterStdThreshold}
              minPanel={minPanel}
              setMinPanel={setMinPanel}
            />
          </div>
        )}

        {/* Error */}
        {state.status === 'error' && (
          <div className="glass rounded-2xl border border-red-800/50 bg-red-900/10 p-5 flex items-start gap-3 animate-fade-in">
            <AlertIcon />
            <div>
              <p className="font-semibold text-red-400 text-sm">Processing failed</p>
              <p className="text-surface-200/60 text-sm mt-0.5">{state.error}</p>
              <button onClick={reset} className="mt-3 text-xs text-brand-400 underline">
                Try again
              </button>
            </div>
          </div>
        )}

        {/* Results — one section per uploaded comic, kept separate */}
        {state.status === 'done' && (
          <div className="flex flex-col gap-12">
            {/* Not happy with a result? Tweak the settings and re-run without re-uploading */}
            <div className="glass rounded-2xl border border-surface-500/30 p-5 animate-fade-in">
              <div className="flex items-center justify-between gap-4">
                <button
                  id="toggle-settings-btn"
                  onClick={() => setShowSettings(s => !s)}
                  className="text-xs text-surface-200/40 hover:text-surface-200/70 transition-colors duration-150 inline-flex items-center gap-1"
                >
                  <ChevronIcon open={showSettings} />
                  Not quite right? Adjust settings and re-run
                </button>
                <button
                  id="rerun-btn"
                  onClick={rerun}
                  disabled={state.status !== 'done'}
                  className="
                    flex items-center gap-2 rounded-lg px-4 py-1.5
                    bg-brand-600 hover:bg-brand-500
                    text-white text-xs font-semibold
                    transition-colors duration-200 flex-shrink-0
                  "
                >
                  <RefreshIcon />
                  Re-run
                </button>
              </div>

              {showSettings && (
                <div className="mt-4 grid sm:grid-cols-3 gap-5 animate-fade-in">
                  <SliderField
                    id="white-median-threshold-slider-rerun"
                    label="White Median Threshold"
                    hint="Median brightness a row/col must exceed to count as a white gutter"
                    min={100} max={255} step={1}
                    value={whiteMedianThreshold}
                    onChange={setWhiteMedianThreshold}
                    format={(v) => String(v)}
                  />
                  <SliderField
                    id="gutter-std-threshold-slider-rerun"
                    label="Gutter Flatness"
                    hint="Max variation allowed for a row/col to still count as a flat, uniform gutter"
                    min={0} max={100} step={1}
                    value={gutterStdThreshold}
                    onChange={setGutterStdThreshold}
                    format={(v) => String(v)}
                  />
                  <SliderField
                    id="min-panel-slider-rerun"
                    label="Min Panel Size"
                    hint="Strips smaller than this fraction of the image are discarded"
                    min={0.05} max={0.45} step={0.01}
                    value={minPanel}
                    onChange={setMinPanel}
                    format={(v) => (v * 100).toFixed(0) + '%'}
                  />
                </div>
              )}
            </div>

            {state.comics.map((comic, i) => (
              <div key={`${comic.filename}-${i}`} id={`comic-group-${i}`}>
                <h3 className="text-sm font-semibold text-surface-200/70 mb-3 truncate">
                  {comic.filename}
                </h3>
                {comic.status === 'ok' ? (
                  <PanelGrid
                    panels={comic.panels}
                    rows={comic.rows}
                    cols={comic.cols}
                    processingMs={comic.processingMs}
                    originalName={comic.filename}
                  />
                ) : (
                  <div className="glass rounded-2xl border border-red-800/50 bg-red-900/10 p-5 flex items-start gap-3">
                    <AlertIcon />
                    <div>
                      <p className="font-semibold text-red-400 text-sm">Processing failed</p>
                      <p className="text-surface-200/60 text-sm mt-0.5">{comic.error}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="border-t border-surface-600/30 py-5 text-center">
        <p className="text-xs text-surface-200/25">
          ComChop · Gutter-detection CV · MIT License
        </p>
      </footer>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────
interface SettingsPanelProps {
  show: boolean
  onToggle: () => void
  whiteMedianThreshold: number
  setWhiteMedianThreshold: (v: number) => void
  gutterStdThreshold: number
  setGutterStdThreshold: (v: number) => void
  minPanel: number
  setMinPanel: (v: number) => void
}

function SettingsPanel({
  show, onToggle,
  whiteMedianThreshold, setWhiteMedianThreshold,
  gutterStdThreshold, setGutterStdThreshold,
  minPanel, setMinPanel,
}: SettingsPanelProps) {
  return (
    <>
      <div className="mt-4 text-center">
        <button
          id="toggle-settings-btn"
          onClick={onToggle}
          className="text-xs text-surface-200/40 hover:text-surface-200/70 transition-colors duration-150 inline-flex items-center gap-1"
        >
          <ChevronIcon open={show} />
          Advanced settings
        </button>
      </div>

      {show && (
        <div className="mt-4 glass rounded-2xl border border-surface-500/30 p-5 grid sm:grid-cols-3 gap-5 animate-fade-in">
          <SliderField
            id="white-median-threshold-slider"
            label="White Median Threshold"
            hint="Median brightness a row/col must exceed to count as a white gutter"
            min={100} max={255} step={1}
            value={whiteMedianThreshold}
            onChange={setWhiteMedianThreshold}
            format={(v) => String(v)}
          />
          <SliderField
            id="gutter-std-threshold-slider"
            label="Gutter Flatness"
            hint="Max variation allowed for a row/col to still count as a flat, uniform gutter"
            min={0} max={100} step={1}
            value={gutterStdThreshold}
            onChange={setGutterStdThreshold}
            format={(v) => String(v)}
          />
          <SliderField
            id="min-panel-slider"
            label="Min Panel Size"
            hint="Strips smaller than this fraction of the image are discarded"
            min={0.05} max={0.45} step={0.01}
            value={minPanel}
            onChange={setMinPanel}
            format={(v) => (v * 100).toFixed(0) + '%'}
          />
        </div>
      )}
    </>
  )
}

interface SliderFieldProps {
  id: string
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
  format: (v: number) => string
}

function SliderField({ id, label, hint, min, max, step, value, onChange, format }: SliderFieldProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-baseline">
        <label htmlFor={id} className="text-xs font-semibold text-surface-200/70">{label}</label>
        <span className="text-xs font-mono text-brand-400">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full bg-surface-500 appearance-none cursor-pointer accent-brand-500"
      />
      <p className="text-[10px] text-surface-200/30 leading-tight">{hint}</p>
    </div>
  )
}

function ScissorsIcon() {
  return (
    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
      <circle cx="6" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <line x1="20" y1="4" x2="8.12" y2="15.88" />
      <line x1="14.47" y1="14.48" x2="20" y2="20" />
      <line x1="8.12" y1="8.12" x2="12" y2="12" />
    </svg>
  )
}

function RefreshIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
    </svg>
  )
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg className={`w-3 h-3 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function AlertIcon() {
  return (
    <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  )
}
