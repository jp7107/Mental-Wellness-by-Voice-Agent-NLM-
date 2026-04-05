'use client'
import { useRef, useState, useCallback } from 'react'
import { createWorkletBlobURL } from '@/lib/audioWorklet'
import { SAMPLE_RATE } from '@/lib/constants'

interface UseAudioCaptureOptions {
  onAudioChunk: (pcmBytes: ArrayBuffer) => void
}

export function useAudioCapture({ onAudioChunk }: UseAudioCaptureOptions) {
  const [isCapturing, setIsCapturing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ctxRef   = useRef<AudioContext | null>(null)
  const nodeRef  = useRef<AudioWorkletNode | null>(null)
  const srcRef   = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const blobRef  = useRef<string | null>(null)

  const start = useCallback(async () => {
    if (isCapturing) return
    setError(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: SAMPLE_RATE, echoCancellation: true, noiseSuppression: true }
      })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
      ctxRef.current = ctx

      const blobURL = createWorkletBlobURL()
      blobRef.current = blobURL
      await ctx.audioWorklet.addModule(blobURL)

      const workletNode = new AudioWorkletNode(ctx, 'mindease-processor')
      workletNode.port.onmessage = (e) => {
        if (e.data?.pcm16) onAudioChunk(e.data.pcm16)
      }
      nodeRef.current = workletNode

      const src = ctx.createMediaStreamSource(stream)
      srcRef.current = src
      src.connect(workletNode)
      workletNode.connect(ctx.destination)

      setIsCapturing(true)
    } catch (err: any) {
      setError(err?.message ?? 'Microphone access denied')
    }
  }, [isCapturing, onAudioChunk])

  const stop = useCallback(() => {
    nodeRef.current?.disconnect()
    srcRef.current?.disconnect()
    streamRef.current?.getTracks().forEach(t => t.stop())
    ctxRef.current?.close()
    if (blobRef.current) URL.revokeObjectURL(blobRef.current)
    nodeRef.current = null
    srcRef.current = null
    streamRef.current = null
    ctxRef.current = null
    setIsCapturing(false)
  }, [])

  return { isCapturing, error, start, stop }
}
