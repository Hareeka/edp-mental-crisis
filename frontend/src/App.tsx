import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import Login from './pages/Login'
import MoodTracker from './pages/MoodTracker'
import Settings from './pages/Settings'
import Support from './pages/Support'

function Shell() {
  const { user, loading, logout } = useAuth()
  if (loading) return <div className="center muted">Loading…</div>
  if (!user) return <Login />

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">◐</span> MindCompanion
        </div>
        <nav>
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/chat">AI Companion</NavLink>
          <NavLink to="/mood">Mood Tracker</NavLink>
          <NavLink to="/support">Support</NavLink>
          <NavLink to="/settings">Privacy &amp; Settings</NavLink>
        </nav>
        <div className="sidebar-foot">
          <div className="muted small">Signed in as {user.alias}</div>
          <button className="link" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/mood" element={<MoodTracker />} />
          <Route path="/support" element={<Support />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <footer className="disclaimer">
          MindCompanion is an AI support tool, not a therapist. Risk levels are automated indications, not a diagnosis.
          If you are in danger, contact local emergency services — see <NavLink to="/support">Support</NavLink>.
        </footer>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  )
}
