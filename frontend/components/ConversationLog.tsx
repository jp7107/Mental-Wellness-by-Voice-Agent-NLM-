'use client'
import React, { useEffect, useRef } from 'react'
import { EMOTION_COLORS } from '@/lib/constants'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  emotion?: string
  timestamp: Date
}

interface ConversationLogProps {
  messages: Message[]
}

export default function ConversationLog({ messages }: ConversationLogProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', flexDirection: 'column', gap: 12 }}>
        <div style={{ fontSize: 40, opacity: 0.3 }}>💬</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 14, textAlign: 'center' }}>
          Tap the orb and start speaking.<br/>Your conversation will appear here.
        </p>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '4px 0' }}>
      {messages.map(msg => {
        const isUser = msg.role === 'user'
        const emotionColor = msg.emotion ? (EMOTION_COLORS[msg.emotion] ?? '#7c6af7') : undefined

        return (
          <div key={msg.id} style={{
            display: 'flex',
            justifyContent: isUser ? 'flex-end' : 'flex-start',
            animation: 'msg-in 0.3s ease',
          }}>
            <div style={{
              maxWidth: '80%',
              padding: '12px 16px',
              borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              background: isUser
                ? 'linear-gradient(135deg, rgba(124,106,247,0.35), rgba(94,184,255,0.25))'
                : 'rgba(255,255,255,0.05)',
              border: `1px solid ${isUser
                ? 'rgba(124,106,247,0.3)'
                : emotionColor ? `${emotionColor}33` : 'rgba(255,255,255,0.08)'}`,
              boxShadow: emotionColor && !isUser ? `0 2px 16px ${emotionColor}15` : undefined,
            }}>
              {!isUser && msg.emotion && (
                <div style={{
                  fontSize: 10, letterSpacing: 1.5, textTransform: 'uppercase',
                  color: emotionColor ?? 'var(--text-muted)',
                  marginBottom: 4, opacity: 0.8,
                }}>
                  Mind Ease · {msg.emotion}
                </div>
              )}
              <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--text-primary)', margin: 0 }}>
                {msg.text}
              </p>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, textAlign: isUser ? 'right' : 'left' }}>
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        )
      })}
      <div ref={endRef} />
      <style>{`
        @keyframes msg-in {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
