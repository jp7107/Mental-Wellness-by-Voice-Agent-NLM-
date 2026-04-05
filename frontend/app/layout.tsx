import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MIND EASE — Mental Wellness by Voice',
  description: 'On-device, real-time, private voice AI for mental wellness support.',
  keywords: 'mental health, AI, voice, wellness, offline, private',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
