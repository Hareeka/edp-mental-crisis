import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar, BarChart, Legend } from 'recharts'
import { api, MOODS, type Dashboard as D } from '../api'
import { EmotionBadge, RiskBadge } from '../components/Badges'

const EMOTION_COLORS: Record<string, string> = {
  sadness: '#6b8fd6', fear: '#a77bd1', anger: '#e07a5f', anxiety: '#f2b84b', loneliness: '#7aa6a1', joy: '#81c784', neutral: '#b0b7c3',
}

export default function Dashboard() {
  const [d, setD] = useState<D | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.dashboard().then(setD).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="error">{error}</div>
  if (!d) return <div className="muted">Loading…</div>

  const moodLabel = (s: number | null) => (s == null ? '—' : MOODS[Math.max(0, Math.min(4, Math.round(s) - 1))].emoji + ' ' + s.toFixed(1))
  const trend = d.trend.map((p) => ({ ...p, label: p.date.slice(5) }))
  const emotionKeys = Object.keys(EMOTION_COLORS).filter((k) => trend.some((p) => p.emotions[k]))

  return (
    <div>
      <header className="page-head">
        <h1>Dashboard</h1>
        <p className="muted">Your last 7–14 days at a glance.</p>
      </header>
      <div className="stats">
        <div className="card stat"><div className="muted small">7-day mood average</div><div className="stat-val">{moodLabel(d.mood_average_7d)}</div></div>
        <div className="card stat">
          <div className="muted small">Latest mood</div>
          <div className="stat-val">{d.latest_mood ? `${MOODS.find((m) => m.key === d.latest_mood!.mood)?.emoji} ${d.latest_mood.mood.replace('_', ' ')}` : '—'}</div>
        </div>
        <div className="card stat"><div className="muted small">Conversations (7d)</div><div className="stat-val">{d.interactions_7d}</div></div>
        <div className="card stat">
          <div className="muted small">Most frequent emotion (7d)</div>
          <div className="stat-val">{d.dominant_emotion_7d ? <EmotionBadge emotion={d.dominant_emotion_7d} /> : '—'}</div>
        </div>
      </div>

      <div className="grid2">
        <div className="card">
          <h3>Mood trend (14 days)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e8ef" />
              <XAxis dataKey="label" fontSize={11} />
              <YAxis domain={[1, 5]} ticks={[1, 2, 3, 4, 5]} fontSize={11} />
              <Tooltip />
              <Line type="monotone" dataKey="avg_mood" name="Avg mood" stroke="#5b6ee1" strokeWidth={2} connectNulls dot />
            </LineChart>
          </ResponsiveContainer>
          {trend.every((p) => p.avg_mood == null) && <p className="muted small">No moods yet — <Link to="/mood">log one</Link>.</p>}
        </div>
        <div className="card">
          <h3>Emotions detected in conversations</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={trend.map((p) => ({ label: p.label, ...p.emotions }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e8ef" />
              <XAxis dataKey="label" fontSize={11} />
              <YAxis allowDecimals={false} fontSize={11} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {emotionKeys.map((k) => <Bar key={k} dataKey={k} stackId="e" fill={EMOTION_COLORS[k]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3>Recent interactions</h3>
        {d.recent_interactions.length === 0 ? (
          <p className="muted">No conversations yet. <Link to="/chat">Talk to your companion</Link>.</p>
        ) : (
          <table className="table">
            <thead><tr><th>When</th><th>Message</th><th>Emotion</th><th>Risk</th></tr></thead>
            <tbody>
              {d.recent_interactions.map((r) => (
                <tr key={r.id}>
                  <td className="nowrap small">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="truncate">{r.message ?? <span className="muted">(not stored)</span>}</td>
                  <td><EmotionBadge emotion={r.emotion} /></td>
                  <td><RiskBadge level={r.risk_level} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
