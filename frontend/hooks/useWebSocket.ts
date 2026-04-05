'use client'
import { useRef, useState, useCallback, useEffect } from 'react'
import { WS_URL } from '@/lib/constants'

export type WSStatus = 'disconnected' | 'connecting' | 'ready' | 'error'

interface UseWebSocketOptions {
  onMessage: (msg: Record<string, unknown>) => void
}

export function useWebSocket({ onMessage }: UseWebSocketOptions) {
  const [status, setStatus] = useState<WSStatus>('disconnected')
  const wsRef = useRef<WebSocket | null>(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')
    const ws = new WebSocket(WS_URL)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => setStatus('connecting') // wait for engine ready msg

    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'status' && msg.status === 'ready') setStatus('ready')
          onMessageRef.current(msg)
        } catch {}
      }
    }

    ws.onclose = () => { setStatus('disconnected'); wsRef.current = null }
    ws.onerror = () => setStatus('error')

    wsRef.current = ws
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
  }, [])

  const sendAudio = useCallback((pcmBuffer: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(pcmBuffer)
    }
  }, [])

  const sendJSON = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  useEffect(() => () => { wsRef.current?.close() }, [])

  return { status, connect, disconnect, sendAudio, sendJSON }
}
