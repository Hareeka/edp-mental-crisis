import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { api, type ChatResponse } from '../api'
import { EmotionBadge, RiskBadge, ScoreBars, SentimentBadge } from '../components/Badges'

interface Msg {
  role: 'user' | 'assistant'
  text: string
  analysis?: ChatResponse
}

function Feedback({ interactionId }: { interactionId: number }) {
  const [sent, setSent] = useState<string | null>(null)
  const send = async (usefulness: number) => {
    await api.feedback({ interaction_id: interactionId, usefulness, relevance: usefulness, appropriateness: usefulness })
    setSent(usefulness >= 4 ? 'Thanks for the feedback' : "Thanks — we'll use this to improve")
  }
  if (sent) return <span className="small muted">{sent}</span>
  return (
    <span className="feedback">
      <span className="small muted">Was this helpful?</span>
      <button className="icon" aria-label="Helpful" onClick={() => send(5)}>👍</button>
      <button className="icon" aria-label="Not helpful" onClick={() => send(2)}>👎</button>
    </span>
  )
}

function AssistantMessage({ m }: { m: Msg }) {
  const [open, setOpen] = useState(false)
  const a = m.analysis
  const crisis = a?.risk.level === 'high'
  return (
    <div className={`msg assistant ${crisis ? 'crisis' : ''}`}>
      {crisis && <div className="crisis-title">Your safety matters</div>}
      <div className="msg-text">{m.text}</div>
      {a && a.resources.length > 0 && (
        <ul className="resources">
          {a.resources.map((r) => (
            <li key={r.name}><strong>{r.name}:</strong> {r.detail}</li>
          ))}
        </ul>
      )}
      {a && a.recommendations.length > 0 && (
        <div className="recs">
          <div className="small muted">{a.risk.level === 'moderate' ? 'Things that might help' : 'Wellness ideas'}</div>
          <ul>{a.recommendations.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      )}
      {a && (
        <div className="analysis">
          <div className="badges">
            <SentimentBadge sentiment={a.sentiment} />
            <EmotionBadge emotion={a.emotion} />
            <RiskBadge level={a.risk.level} />
            <button className="link small" onClick={() => setOpen(!open)}>{open ? 'Hide analysis' : 'Show analysis'}</button>
          </div>
          {open && (
            <div className="analysis-detail">
              <div><div className="small muted">Emotions</div><ScoreBars scores={a.emotion_scores} /></div>
              <div><div className="small muted">Sentiment</div><ScoreBars scores={a.sentiment_scores} /></div>
              <div>
                <div className="small muted">Risk probabilities</div>
                <ScoreBars scores={a.risk.probabilities} />
                {a.risk.indicators.length > 0 && <div className="small">Indicators: {a.risk.indicators.join(', ')}</div>}
                <div className="small muted">Decision: {a.risk.reasons.join(' · ')} · model {a.risk.model_version} · {Math.round(a.latency_ms)} ms</div>
              </div>
              <p className="small muted">{a.disclaimer}</p>
            </div>
          )}
          <Feedback interactionId={a.interaction_id} />
        </div>
      )}
    </div>
  )
}

export default function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.interactions().then((rows) =>
      setMsgs(
        rows.flatMap((r) => [
          ...(r.message ? [{ role: 'user' as const, text: r.message }] : []),
          ...(r.response ? [{ role: 'assistant' as const, text: r.response }] : []),
        ]),
      ),
    ).catch(() => undefined)
  }, [])

  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [msgs, busy])

  const send = async (e: FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setError(null)
    setMsgs((m) => [...m, { role: 'user', text }])
    setBusy(true)
    try {
      const res = await api.chat(text)
      setMsgs((m) => [...m, { role: 'assistant', text: res.reply, analysis: res }])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="chat-page">
      <header className="page-head">
        <h1>AI Companion</h1>
        <p className="muted">Share what's on your mind. In an emergency, please use <Link to="/support">crisis support</Link>.</p>
      </header>
      <div className="chat-log">
        {msgs.length === 0 && (
          <div className="empty muted">Hi, I'm here to listen. How are you feeling today?</div>
        )}
        {msgs.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="msg user"><div className="msg-text">{m.text}</div></div>
          ) : (
            <AssistantMessage key={i} m={m} />
          ),
        )}
        {busy && <div className="msg assistant typing">…</div>}
        <div ref={endRef} />
      </div>
      {error && <div className="error">{error}</div>}
      <form className="chat-input" onSubmit={send}>
        <textarea
          value={input}
          maxLength={5000}
          placeholder="Type a message…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) send(e)
          }}
          rows={2}
        />
        <button className="primary" disabled={busy || !input.trim()}>Send</button>
      </form>
    </div>
  )
}
