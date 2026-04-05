export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws/session'

export const SAMPLE_RATE = 16000
export const CHUNK_DURATION_MS = 2000
export const CHUNK_SIZE = SAMPLE_RATE * (CHUNK_DURATION_MS / 1000)

export const EMOTION_COLORS: Record<string, string> = {
  calm:       '#6af79a',
  sad:        '#5eb8ff',
  anxious:    '#f7c26a',
  angry:      '#f77b6a',
  fearful:    '#d06af7',
  distressed: '#f76a8a',
}

export const EMOTION_LABELS: Record<string, string> = {
  calm:       'Calm',
  sad:        'Sad',
  anxious:    'Anxious',
  angry:      'Angry',
  fearful:    'Fearful',
  distressed: 'Distressed',
}
