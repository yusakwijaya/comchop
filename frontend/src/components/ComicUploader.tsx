import React, { useCallback, useState, useRef } from 'react'

interface Props {
  onUpload: (file: File) => void
  isProcessing: boolean
}

export default function ComicUploader({ onUpload, isProcessing }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return
    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)
    onUpload(file)
  }, [onUpload])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = () => setIsDragging(false)
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleSampleLoad = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const response = await fetch('/comic_grid.png')
      const blob = await response.blob()
      const file = new File([blob], 'comic_grid.png', { type: 'image/png' })
      handleFile(file)
    } catch (err) {
      console.error("Failed to load sample image", err)
    }
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        id="comic-drop-zone"
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !isProcessing && inputRef.current?.click()}
        className={`
          glass rounded-2xl border-2 border-dashed
          transition-all duration-300 cursor-pointer select-none
          ${isDragging ? 'drop-active' : 'border-surface-500 hover:border-brand-600'}
          ${isProcessing ? 'cursor-not-allowed opacity-70' : ''}
        `}
      >
        <input
          ref={inputRef}
          id="comic-file-input"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onInputChange}
          disabled={isProcessing}
        />

        {preview ? (
          <div className="relative p-4">
            <img
              src={preview}
              alt="Comic preview"
              className="w-full max-h-72 object-contain rounded-xl"
            />
            {/* overlay on processing */}
            {isProcessing && (
              <div className="absolute inset-4 rounded-xl flex flex-col items-center justify-center bg-surface-900/80 backdrop-blur-sm">
                <SpinnerIcon />
                <p className="mt-3 text-brand-400 font-medium text-sm tracking-wide">
                  Detecting gutters…
                </p>
              </div>
            )}
            {!isProcessing && (
              <div className="absolute top-6 right-6 glass rounded-lg px-3 py-1 text-xs text-surface-200/70">
                Click to change
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-20 px-6 text-center gap-4">
            <UploadIcon dragging={isDragging} />
            <div>
              <p className="text-lg font-semibold text-surface-200">
                {isDragging ? 'Release to analyse' : 'Drop your comic here'}
              </p>
              <p className="text-sm text-surface-200/50 mt-1">
                PNG · JPG · WebP · up to 30 MB
              </p>
            </div>
            <div className="flex gap-3">
              <span className="glass rounded-full px-5 py-2 text-sm font-medium text-brand-300 border border-brand-700/40">
                Browse files
              </span>
              <button
                id="load-sample-btn"
                type="button"
                onClick={handleSampleLoad}
                className="glass rounded-full px-5 py-2 text-sm font-medium text-accent-300 border border-accent-700/40 hover:bg-accent-700/20 transition-all duration-150"
              >
                Use Sample Grid
              </button>
            </div>
          </div>
        )}
      </div>

      {fileName && !isProcessing && (
        <p className="mt-2 text-xs text-center text-surface-200/40">
          {fileName}
        </p>
      )}
    </div>
  )
}

function UploadIcon({ dragging }: { dragging: boolean }) {
  return (
    <div className={`
      w-16 h-16 rounded-2xl flex items-center justify-center
      transition-all duration-300
      ${dragging
        ? 'bg-brand-600/30 scale-110'
        : 'bg-surface-600/60'
      }
    `}>
      <svg className={`w-8 h-8 transition-colors duration-300 ${dragging ? 'text-brand-400' : 'text-surface-200/50'}`} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    </div>
  )
}

function SpinnerIcon() {
  return (
    <svg className="w-10 h-10 animate-spin text-brand-500" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8z" />
    </svg>
  )
}
