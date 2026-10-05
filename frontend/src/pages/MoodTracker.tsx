import { useEffect, useState, type FormEvent } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, MOODS, type Mood } from '../api'

export default function MoodTracker() {
  const [mood, setMood] = useState<string | null>(null)
  const [intensity, setIntensity] = useState(5)
  const [note, setNote] = useState('')
  const [history, setHistory] = useState<Mood[]>([])
  const [days, setDays] = useState(30)
  const [msg, setMsg] = useState<string | null>(null)

  const load = () => api.moods(days).then(setHistory).catch(() => undefined)
  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [days])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!mood) return
    await api.addMood(mood, intensity, note)
    setMsg('Mood saved')
    setMood(null)
    setNote('')
    setIntensity(5)
    load()
    setTimeout(() => setMsg(null), 2500)
  }

  const chart = history.map((m) => ({ t: new Date(m.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }), score: m.score, intensity: m.intensity }))

  return (
    <div>
      <header className="page-head">
        <h1>Mood Tracker</h1>
        <p className="muted">Log how you feel. Patterns over time help the companion respond better.</p>
      </header>
      <form className="card" onSubmit={submit}>
        <h3>How are you feeling right now?</h3>
        <div className="mood-picker">
          {MOODS.map((m) => (
            <button type="button" key={m.key} className={`mood-btn ${mood === m.key ? 'selected' : ''}`} onClick={() => setMood(m.key)}>
              <span className="emoji">{m.emoji}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </div>
        <label>
          Intensity: <strong>{intensity}</strong>/10
          <input type="range" min={1} max={10} value={intensity} onChange={(e) => setIntensity(Number(e.target.value))} />
        </label>
        <label>Note (optional, encrypted)<input value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} placeholder="What's contributing to this?" /></label>
        <div className="row">
          <button className="primary" disabled={!mood}>Save mood</button>
          {msg && <span className="ok">{msg}</span>}
        </div>
      </form>

      <div className="card">
        <div className="row spread">
          <h3>History</h3>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>
        </div>
        {history.length === 0 ? (
          <p className="muted">No moods logged in this period.</p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={chart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e8ef" />
                <XAxis dataKey="t" fontSize={11} />
                <YAxis yAxisId="m" domain={[1, 5]} ticks={[1, 2, 3, 4, 5]} fontSize={11} />
                <YAxis yAxisId="i" orientation="right" domain={[1, 10]} fontSize={11} />
                <Tooltip />
                <Line yAxisId="m" dataKey="score" name="Mood (1-5)" stroke="#5b6ee1" strokeWidth={2} />
                <Line yAxisId="i" dataKey="intensity" name="Intensity (1-10)" stroke="#f2b84b" strokeDasharray="4 3" />
              </LineChart>
            </ResponsiveContainer>
            <table className="table">
              <thead><tr><th>When</th><th>Mood</th><th>Intensity</th><th>Note</th></tr></thead>
              <tbody>
                {[...history].reverse().slice(0, 20).map((m) => (
                  <tr key={m.id}>
                    <td className="nowrap small">{new Date(m.created_at).toLocaleString()}</td>
                    <td>{MOODS.find((x) => x.key === m.mood)?.emoji} {m.mood.replace('_', ' ')}</td>
                    <td>{m.intensity}</td>
                    <td className="truncate">{m.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  )
}
