import { useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

export default function Settings() {
  const { user, refresh, logout } = useAuth()
  const [msg, setMsg] = useState<string | null>(null)
  const [rating, setRating] = useState(0)
  const [comment, setComment] = useState('')

  if (!user) return null

  const toggleConsent = async () => {
    await api.setConsent(!user.consent_to_research)
    await refresh()
  }
  const clear = async () => {
    if (!confirm('Delete all your conversation history? This cannot be undone.')) return
    await api.clearInteractions()
    setMsg('Conversation history deleted')
  }
  const del = async () => {
    if (!confirm('Permanently delete your account and all data?')) return
    await api.deleteAccount()
    logout()
  }
  const sendFeedback = async () => {
    await api.feedback({ experience: rating || null, comment: comment || null })
    setRating(0)
    setComment('')
    setMsg('Thanks for your feedback!')
  }

  return (
    <div>
      <header className="page-head"><h1>Privacy &amp; Settings</h1></header>
      {msg && <div className="ok card">{msg}</div>}
      <div className="card">
        <h3>Your data</h3>
        <p className="small muted">Messages, mood notes and feedback comments are encrypted at rest. Logs contain no message content.</p>
        <label className="row">
          <input type="checkbox" checked={user.consent_to_research} onChange={toggleConsent} />
          Allow my anonymised conversations to be used to improve the models (off by default)
        </label>
        <div className="row">
          <button onClick={clear}>Delete conversation history</button>
          <button className="danger" onClick={del}>Delete account</button>
        </div>
      </div>
      <div className="card">
        <h3>Overall experience</h3>
        <div className="row">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} className={`star ${n <= rating ? 'on' : ''}`} onClick={() => setRating(n)} aria-label={`${n} stars`}>★</button>
          ))}
        </div>
        <textarea rows={3} value={comment} maxLength={2000} onChange={(e) => setComment(e.target.value)} placeholder="Anything we could do better?" />
        <button className="primary" disabled={!rating && !comment} onClick={sendFeedback}>Send feedback</button>
      </div>
    </div>
  )
}
