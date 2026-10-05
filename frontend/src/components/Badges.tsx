import type { Risk } from '../api'

const EMOTION_EMOJI: Record<string, string> = {
  sadness: '😢', fear: '😨', anger: '😠', anxiety: '😰', loneliness: '🫥', joy: '😊', neutral: '😶',
}

export function RiskBadge({ level }: { level: Risk }) {
  const label = { low: 'Low risk', moderate: 'Moderate risk', high: 'High risk' }[level]
  return <span className={`badge risk-${level}`} title="Automated indication, not a diagnosis">{label}</span>
}

export function EmotionBadge({ emotion }: { emotion: string }) {
  return <span className="badge emotion">{EMOTION_EMOJI[emotion] ?? ''} {emotion}</span>
}

export function SentimentBadge({ sentiment }: { sentiment: string }) {
  return <span className={`badge sent-${sentiment}`}>{sentiment}</span>
}

export function ScoreBars({ scores }: { scores: Record<string, number> }) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1])
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div key={k} className="bar-row">
          <span className="bar-label">{k}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.round(v * 100)}%` }} /></span>
          <span className="bar-val">{Math.round(v * 100)}%</span>
        </div>
      ))}
    </div>
  )
}
