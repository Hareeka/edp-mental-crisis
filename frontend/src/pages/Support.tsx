import { useEffect, useState } from 'react'
import { api, type Resource } from '../api'

const WELLNESS = [
  { title: 'Box breathing', body: 'Breathe in for 4 seconds, hold 4, out 4, hold 4. Repeat for two minutes.' },
  { title: '5-4-3-2-1 grounding', body: 'Notice 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.' },
  { title: 'Move a little', body: 'A 10-minute walk, stretch or dance can shift your mood and energy.' },
  { title: 'Sleep routine', body: 'Keep a consistent bedtime and put screens away 30 minutes before sleep.' },
  { title: 'Connect', body: 'Message or call one person today, even briefly.' },
  { title: 'Journal', body: 'Write down what you are feeling and one thing you are grateful for.' },
]

export default function Support() {
  const [crisis, setCrisis] = useState<Resource[]>([])
  useEffect(() => {
    api.resources().then((r) => setCrisis(r.crisis)).catch(() => undefined)
  }, [])

  return (
    <div>
      <header className="page-head">
        <h1>Support</h1>
        <p className="muted">Resources for everyday wellbeing and for moments of crisis.</p>
      </header>
      <div className="card crisis-card">
        <h3>If you are in crisis or thinking about harming yourself</h3>
        <p>
          Please contact a person who can help right now. MindCompanion is an AI and <strong>cannot</strong> respond to emergencies.
        </p>
        <ul className="resources">
          {crisis.map((r) => <li key={r.name}><strong>{r.name}:</strong> {r.detail}</li>)}
        </ul>
        <p className="small">Reaching out to a trusted friend, family member, doctor or counsellor is also a strong step.</p>
      </div>
      <div className="card">
        <h3>Professional support</h3>
        <p>
          A GP/primary-care doctor, licensed counsellor, psychologist or psychiatrist can assess and treat mental health conditions.
          Many universities and workplaces offer free, confidential counselling services.
        </p>
      </div>
      <h3>Wellness practices</h3>
      <div className="tiles">
        {WELLNESS.map((w) => (
          <div key={w.title} className="card tile"><strong>{w.title}</strong><p className="small">{w.body}</p></div>
        ))}
      </div>
    </div>
  )
}
