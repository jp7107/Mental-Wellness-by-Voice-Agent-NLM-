'use client'
import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import VoiceOrb from '@/components/VoiceOrb'
import MoodRing from '@/components/MoodRing'
import ConversationLog, { Message } from '@/components/ConversationLog'
import SafetyBanner from '@/components/SafetyBanner'
import LatencyBar from '@/components/LatencyBar'
import { useAudioCapture } from '@/hooks/useAudioCapture'
import { useWebSocket } from '@/hooks/useWebSocket'

interface SafetyState {
  active: boolean
  message: string
  resources: Array<{ name: string; contact: string; region?: string }>
}

interface LatencyLayer { name: string; ms: number; target: number }

export default function HomePage() {
  const [emotion, setEmotion] = useState('calm')
  const [confidence, setConfidence] = useState(0)
  const [scores, setScores] = useState<Record<string, number>>({})
  const [moodScore, setMoodScore] = useState(1)
  const [messages, setMessages] = useState<Message[]>([])
  const [safety, setSafety] = useState<SafetyState>({ active: false, message: '', resources: [] })
  const [isProcessing, setIsProcessing] = useState(false)
  const [latencyLayers, setLatencyLayers] = useState<LatencyLayer[]>([])

  const audioQueueRef = useRef<AudioBuffer[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)
  const nextPlayTimeRef = useRef(0)
  const emotionRef = useRef(emotion)
  useEffect(() => { emotionRef.current = emotion }, [emotion])

  // ── WebSocket ──
  const { status, connect, disconnect, sendAudio } = useWebSocket({
    onMessage: useCallback((msg: Record<string, unknown>) => {
      const t = msg.type as string
      if (t === 'transcript') {
        setIsProcessing(true)
        setMessages(prev => [...prev, {
          id: `u-${Date.now()}`,
          role: 'user',
          text: msg.text as string,
          timestamp: new Date(),
        }])
      } else if (t === 'emotion') {
        setEmotion(msg.label as string)
        setConfidence(msg.confidence as number)
        setScores((msg.scores as Record<string, number>) ?? {})
        setLatencyLayers(prev => {
          const rest = prev.filter(l => l.name !== 'Emotion')
          return [...rest, { name: 'Emotion', ms: (msg.duration_ms as number) ?? 0, target: 80 }]
        })
      } else if (t === 'mood_update') {
        setMoodScore(msg.score as number)
      } else if (t === 'safety_alert') {
        setSafety({ active: true, message: msg.message as string, resources: (msg.resources as any[]) ?? [] })
        setIsProcessing(false)
      } else if (t === 'response') {
        setIsProcessing(false)
        const responseText = msg.text as string
        if (responseText) {
          setMessages(prev => [...prev, {
            id: `a-${Date.now()}`,
            role: 'assistant',
            text: responseText,
            emotion: emotionRef.current,
            timestamp: new Date(),
          }])
        }
        setLatencyLayers(prev => {
          const rest = prev.filter(l => l.name !== 'LLM')
          return [...rest, { name: 'LLM', ms: (msg.duration_ms as number) ?? 0, target: 250 }]
        })
      } else if (t === 'tts_start') {
        nextPlayTimeRef.current = 0
      } else if (t === 'tts_chunk') {
        // Decode and queue audio
        const b64 = msg.audio as string
        const binary = atob(b64)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        playPCM16(bytes.buffer)
      }
    }, []),
  })

  // ── Audio playback ──
  const playPCM16 = useCallback((buffer: ArrayBuffer) => {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext({ sampleRate: 24000 })
    const ctx = audioCtxRef.current

    if (ctx.state === 'suspended') {
      ctx.resume().catch(e => console.error('AudioContext resume failed', e))
    }

    if (nextPlayTimeRef.current < ctx.currentTime) {
      nextPlayTimeRef.current = ctx.currentTime
    }

    const int16 = new Int16Array(buffer)
    const float32 = new Float32Array(int16.length)
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768
    const ab = ctx.createBuffer(1, float32.length, 24000)
    ab.copyToChannel(float32, 0)
    
    const src = ctx.createBufferSource()
    src.buffer = ab
    src.connect(ctx.destination)
    src.start(nextPlayTimeRef.current)
    
    // Increment the next play time by the duration of this chunk
    nextPlayTimeRef.current += ab.duration
  }, [])

  // ── Audio capture ──
  const { isCapturing, error: micError, start: startMic, stop: stopMic } = useAudioCapture({
    onAudioChunk: sendAudio,
  })

  // Auto-start mic when WebSocket becomes ready
  const prevStatusRef = useRef(status)
  useEffect(() => {
    if (prevStatusRef.current !== 'ready' && status === 'ready' && !isCapturing) {
      startMic()
    }
    prevStatusRef.current = status
  }, [status, isCapturing, startMic])

  const isListening = isCapturing && status === 'ready'

  const handleToggle = useCallback(async () => {
    if (status === 'disconnected' || status === 'error') {
      connect()
      return
    }
    if (status !== 'ready') return

    if (isCapturing) {
      stopMic()
    } else {
      await startMic()
    }
  }, [status, isCapturing, connect, startMic, stopMic])

  // Add assistant message when response arrives
  // Assistant messages are added in the onMessage 'response' handler above

  const statusLabel = useMemo(() => {
    if (status === 'disconnected') return 'Tap to Connect'
    if (status === 'connecting')   return 'Connecting…'
    if (status === 'error')        return 'Connection Error — Tap to Retry'
    if (!isCapturing)              return 'Tap to Speak'
    if (isProcessing)              return 'Processing…'
    return 'Listening…'
  }, [status, isCapturing, isProcessing])

  return (
    <main className="main-layout" style={{
      position: 'relative', zIndex: 1,
      minHeight: '100vh',
      display: 'grid',
      gridTemplateColumns: '1fr 380px',
      gridTemplateRows: 'auto 1fr auto',
      gap: 20, padding: 24,
      maxWidth: 1200, margin: '0 auto',
    }}>

      {/* ── Header ── */}
      <header style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontFamily: 'Outfit, sans-serif', fontSize: 26, fontWeight: 800,
            background: 'linear-gradient(135deg, #7c6af7, #5eb8ff)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            letterSpacing: '-0.5px', margin: 0 }}>
            MIND EASE
          </h1>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            Private · On-Device · Real-Time
          </p>
        </div>

        {/* Connection indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: status === 'ready' ? '#6af79a'
              : status === 'connecting' ? '#f7c26a'
              : '#f76a8a',
            boxShadow: status === 'ready' ? '0 0 8px #6af79a' : undefined,
            transition: 'all 0.3s ease',
          }} />
          <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'capitalize' }}>
            {status}
          </span>
        </div>
      </header>

      {/* ── Conversation (left) ── */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 16, overflow: 'hidden' }}>
        {safety.active && (
          <SafetyBanner
            message={safety.message}
            resources={safety.resources}
            onDismiss={() => setSafety(s => ({ ...s, active: false }))}
          />
        )}

        <div className="glass" style={{ flex: 1, overflowY: 'auto', padding: 20, minHeight: 300 }}>
          <ConversationLog messages={messages} />
        </div>

        {micError && (
          <p style={{ fontSize: 12, color: '#f76a8a', textAlign: 'center' }}>
            ⚠ {micError}
          </p>
        )}
      </section>

      {/* ── Right panel ── */}
      <aside style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>

        {/* Orb + status */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, padding: '32px 0' }}>
          <VoiceOrb
            isListening={isListening}
            isProcessing={isProcessing}
            onToggle={handleToggle}
            emotion={emotion}
          />
          <p style={{ fontSize: 13, color: 'var(--text-muted)', letterSpacing: 0.5 }}>
            {statusLabel}
          </p>
        </div>

        {/* Mood ring */}
        <MoodRing
          emotion={emotion}
          confidence={confidence}
          score={moodScore}
          scores={scores}
        />

        {/* Latency overlay */}
        <LatencyBar layers={latencyLayers} />

        {/* Offline badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 14px', borderRadius: 20,
          background: 'rgba(106,247,154,0.1)', border: '1px solid rgba(106,247,154,0.2)',
        }}>
          <span style={{ fontSize: 10 }}>🔒</span>
          <span style={{ fontSize: 11, color: '#6af79a' }}>All processing on-device</span>
        </div>
      </aside>

      {/* ── Footer ── */}
      <footer style={{ gridColumn: '1 / -1', textAlign: 'center',
        fontSize: 11, color: 'var(--text-muted)', paddingTop: 8 }}>
        MIND EASE is not a substitute for professional mental health care.
        If you are in crisis, please contact emergency services.
      </footer>

      <style>{`
        @media (max-width: 800px) {
          .main-layout {
            grid-template-columns: 1fr !important;
            padding: 16px !important;
          }
        }
      `}</style>
    </main>
  )
}
