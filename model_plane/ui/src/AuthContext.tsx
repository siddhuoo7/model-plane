/**
 * AuthContext — provides session state (token, user) across the app.
 * Persists token in localStorage. Exposes login/logout helpers.
 */
import React, { createContext, useContext, useEffect, useState } from 'react'
import { api, getSessionToken, setSessionToken, clearSessionToken } from './api/client'

interface User { email: string; name: string; id: string }

interface AuthState {
  token: string | null
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, name: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState>({
  token: null, user: null, loading: true,
  login: async () => {}, signup: async () => {}, logout: () => {},
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(getSessionToken)
  const [user, setUser]   = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // Verify token on mount
  useEffect(() => {
    if (!token) { setLoading(false); return }
    api.auth.me()
      .then(u => { setUser(u); setLoading(false) })
      .catch(() => { clearSessionToken(); setToken(null); setLoading(false) })
  }, [token])

  const login = async (email: string, password: string) => {
    const res = await api.auth.login(email, password)
    setSessionToken(res.token)
    setToken(res.token)
    setUser({ email: res.email, name: res.name, id: '' })
  }

  const signup = async (email: string, password: string, name: string) => {
    const res = await api.auth.signup(email, password, name)
    setSessionToken(res.token)
    setToken(res.token)
    setUser({ email: res.email, name: res.name, id: '' })
  }

  const logout = () => {
    clearSessionToken()
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() { return useContext(AuthContext) }
