/**
 * Login / Signup page.
 * Shows signup form when no users exist; switches to login otherwise.
 * Also auto-detects via GET /admin/api/auth/has-users.
 */
import React, { useEffect, useState } from 'react'
import {
  TextInput,
  Button,
  InlineNotification,
  Loading,
} from '@carbon/react'
import { useAuth } from '../AuthContext'
import { api } from '../api/client'

export default function LoginPage() {
  const { login, signup } = useAuth()

  const [mode, setMode]       = useState<'login' | 'signup' | 'loading'>('loading')
  const [email, setEmail]     = useState('')
  const [password, setPassword] = useState('')
  const [name, setName]       = useState('')
  const [error, setError]     = useState<string | null>(null)
  const [busy, setBusy]       = useState(false)

  useEffect(() => {
    api.auth.hasUsers()
      .then(r => setMode(r.has_users ? 'login' : 'signup'))
      .catch(() => setMode('login'))
  }, [])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (mode === 'signup') {
        await signup(email, password, name)
      } else {
        await login(email, password)
      }
    } catch (err: any) {
      const msg = err?.message ?? String(err)
      // Extract the "detail" field from FastAPI error body if present
      const detailMatch = msg.match(/"detail":"([^"]+)"/)
      setError(detailMatch ? detailMatch[1] : msg)
    } finally {
      setBusy(false)
    }
  }

  if (mode === 'loading') {
    return (
      <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center' }}>
        <Loading withOverlay={false} />
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'linear-gradient(135deg, #0f0f0f 0%, #161616 60%, #1a1a2e 100%)',
    }}>
      <div style={{
        background: '#fff', borderRadius: 12, padding: '2.5rem',
        width: '100%', maxWidth: 400,
        boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
      }}>
        {/* Logo / title */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg, #0f62fe 0%, #8a3ffc 100%)',
            color: '#fff', borderRadius: 10,
            width: 48, height: 48, fontSize: '1.25rem', fontWeight: 800,
            marginBottom: '0.75rem',
          }}>MP</div>
          <h1 style={{ fontSize: '1.375rem', fontWeight: 700, color: '#161616', margin: 0 }}>
            Model Plane
          </h1>
          <p style={{ fontSize: '0.8125rem', color: '#6f6f6f', margin: '0.25rem 0 0' }}>
            {mode === 'signup' ? 'Create your admin account' : 'Sign in to continue'}
          </p>
        </div>

        {error && (
          <InlineNotification
            kind="error"
            title={error}
            hideCloseButton={false}
            onClose={() => setError(null)}
            style={{ marginBottom: '1rem' }}
          />
        )}

        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {mode === 'signup' && (
            <TextInput
              id="login-name"
              labelText="Your name"
              placeholder="Ada Lovelace"
              value={name}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
            />
          )}
          <TextInput
            id="login-email"
            labelText="Email address"
            placeholder="you@example.com"
            type="email"
            value={email}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
            required
          />
          <TextInput
            id="login-password"
            labelText="Password"
            placeholder="••••••••"
            type="password"
            value={password}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
            required
          />

          <Button
            type="submit"
            style={{ width: '100%', maxWidth: '100%', justifyContent: 'center' }}
            disabled={busy}
          >
            {busy ? 'Please wait…' : mode === 'signup' ? 'Create account' : 'Sign in'}
          </Button>
        </form>

        <p style={{ textAlign: 'center', fontSize: '0.8125rem', color: '#6f6f6f', marginTop: '1.25rem' }}>
          {mode === 'signup'
            ? 'Already have an account? '
            : 'No account yet? '}
          <button
            onClick={() => { setError(null); setMode(mode === 'signup' ? 'login' : 'signup') }}
            style={{ background: 'none', border: 'none', color: '#0f62fe', cursor: 'pointer', textDecoration: 'underline', fontSize: 'inherit' }}
          >
            {mode === 'signup' ? 'Sign in' : 'Create one'}
          </button>
        </p>
      </div>
    </div>
  )
}
