export type Risk = 'low' | 'moderate' | 'high'

export interface User {
  id: number
  alias: string
  email: string
  is_admin: boolean
  consent_to_research: boolean
  created_at: string
}

export interface Resource {
  name: string
  detail: string
}

export interface ChatResponse {
  interaction_id: number
  reply: string
  reply_mode: 'template' | 'llm' | 'safety'
  sentiment: string
  sentiment_scores: Record<string, number>
  emotion: string
  emotion_scores: Record<string, number>
  risk: { level: Risk; probabilities: Record<Risk, number>; indicators: string[]; reasons: string[]; model_version: string }
  recommendations: string[]
  resources: Resource[]
  disclaimer: string
  latency_ms: number
  created_at: string
}

export interface Interaction {
  id: number
  message: string | null
  response: string | null
  sentiment: string
  emotion: string
  risk_level: Risk
  created_at: string
}

export interface Mood {
  id: number
  mood: string
  score: number
  intensity: number
  note: string | null
  created_at: string
}

export interface TrendPoint {
  date: string
  avg_mood: number | null
  mood_count: number
  interactions: number
  negative_ratio: number | null
  emotions: Record<string, number>
  risk: Record<string, number>
}

export interface Dashboard {
  mood_average_7d: number | null
  latest_mood: Mood | null
  interactions_7d: number
  dominant_emotion_7d: string | null
  recent_interactions: Interaction[]
  trend: TrendPoint[]
}

export const MOODS: { key: string; label: string; emoji: string; score: number }[] = [
  { key: 'very_low', label: 'Very low', emoji: '😞', score: 1 },
  { key: 'low', label: 'Low', emoji: '🙁', score: 2 },
  { key: 'okay', label: 'Okay', emoji: '😐', score: 3 },
  { key: 'good', label: 'Good', emoji: '🙂', score: 4 },
  { key: 'very_good', label: 'Very good', emoji: '😄', score: 5 },
]

const TOKEN_KEY = 'mc_token'
export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t: string | null) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY))

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof URLSearchParams)) headers.set('Content-Type', 'application/json')
  const res = await fetch(path, { ...init, headers })
  if (res.status === 401 && token) {
    setToken(null)
    window.dispatchEvent(new Event('mc-logout'))
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      msg = typeof body.detail === 'string' ? body.detail : body.detail?.[0]?.msg ?? msg
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  register: (alias: string, email: string, password: string) =>
    request<{ access_token: string }>('/api/auth/register', { method: 'POST', body: JSON.stringify({ alias, email, password }) }),
  login: (email: string, password: string) =>
    request<{ access_token: string }>('/api/auth/token', {
      method: 'POST',
      body: new URLSearchParams({ username: email, password }),
    }),
  me: () => request<User>('/api/auth/me'),
  setConsent: (consent: boolean) =>
    request<User>('/api/auth/me/consent', { method: 'PATCH', body: JSON.stringify({ consent_to_research: consent }) }),
  deleteAccount: () => request<void>('/api/auth/me', { method: 'DELETE' }),
  chat: (message: string) => request<ChatResponse>('/api/chat', { method: 'POST', body: JSON.stringify({ message }) }),
  interactions: () => request<Interaction[]>('/api/interactions?limit=50'),
  clearInteractions: () => request<void>('/api/interactions', { method: 'DELETE' }),
  addMood: (mood: string, intensity: number, note?: string) =>
    request<Mood>('/api/moods', { method: 'POST', body: JSON.stringify({ mood, intensity, note: note || null }) }),
  moods: (days = 30) => request<Mood[]>(`/api/moods?days=${days}`),
  trends: (days = 14) => request<TrendPoint[]>(`/api/trends?days=${days}`),
  dashboard: () => request<Dashboard>('/api/dashboard'),
  resources: () => request<{ crisis: Resource[]; disclaimer: string }>('/api/resources'),
  feedback: (body: Record<string, unknown>) => request<{ id: number }>('/api/feedback', { method: 'POST', body: JSON.stringify(body) }),
}
