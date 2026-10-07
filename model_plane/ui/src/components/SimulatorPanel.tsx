/**
 * Simulator panel — sends a test prompt to POST /admin/api/simulate and
 * renders a detailed strata-style 7-step pipeline trace:
 *   step tabs, feature dimension bars, candidate list with scores, JSON viewer.
 */
import React, { useState, useCallback } from 'react'
import { Button, Tag, InlineLoading } from '@carbon/react'
import { Play, Copy, CheckmarkFilled } from '@carbon/icons-react'
import { api, SimulateResponse } from '../api/client'
import { TierBadge } from './TierBadge'

// ── Tier colour helper ─────────────────────────────────────────────────────
const TIER_COLORS: Record<string, string> = {
  simple:    '#6fdc8c',
  medium:    '#82cfff',
  complex:   '#ffafd2',
  reasoning: '#be95ff',
}
function tierColor(t: string | null) { return TIER_COLORS[t ?? ''] ?? '#c6c6c6' }

// ── Score bar ─────────────────────────────────────────────────────────────
function ScoreBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(1, Math.max(0, value / max)) * 100
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem', fontSize: '0.75rem', fontFamily: 'monospace' }}>
      <span style={{ minWidth: 160, color: '#c6c6c6' }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: '#393939', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#8a3ffc', borderRadius: 3 }} />
      </div>
      <span style={{ minWidth: 42, textAlign: 'right', color: '#f4f4f4' }}>{value.toFixed(4)}</span>
    </div>
  )
}

// ── Sensitivity colour helper ──────────────────────────────────────────────
const SENS_COLOR: Record<string, string> = {
  public: '#24a148', internal: '#0f62fe', confidential: '#f1c21b', restricted: '#da1e28',
}
function SensBadge({ v }: { v: string }) {
  const c = SENS_COLOR[v] ?? '#6f6f6f'
  return (
    <span style={{ fontSize: '0.7rem', fontWeight: 700, padding: '1px 8px', borderRadius: 10, background: c + '20', color: c, border: `1px solid ${c}50` }}>
      {v.toUpperCase()}
    </span>
  )
}

// ── Step tabs ─────────────────────────────────────────────────────────────
const STEP_KEYS = ['features', 'classifier', 'scorer', 'blend', 'ml', 'policy_applied', 'winner'] as const
const STEP_LABELS: Record<string, string> = {
  features: 'Feat…', classifier: 'Clas…', scorer: 'Score',
  blend: 'Blend', ml: 'Tier ML', policy_applied: 'Poli…', winner: 'Pick',
}

type StepKey = typeof STEP_KEYS[number]

function StepTabBar({ active, result, onSelect }: {
  active: StepKey; result: SimulateResponse; onSelect: (k: StepKey) => void
}) {
  const ps = result.pipeline_steps
  const summaries: Record<string, string> = {
    features:       `~${result.features.estimated_tokens}t`,
    classifier:     ps.classifier?.classifier_tier ?? '—',
    scorer:         ps.scorer?.raw_score?.toFixed(2) ?? '—',
    blend:          ps.blend?.blended_tier ?? '—',
    ml:             ps.ml?.enabled ? (ps.ml.predicted_tier ?? 'off') : 'off',
    policy_applied: ps.policy_applied?.resolved_tier ?? '—',
    winner:         result.selected_model?.split('/').pop()?.slice(0, 4) ?? '—',
  }
  return (
    <div style={{ display: 'flex', gap: 2, marginBottom: '1rem', overflowX: 'auto' }}>
      {STEP_KEYS.map((k, i) => (
        <button key={k} onClick={() => onSelect(k)} style={{
          flex: 1, minWidth: 44, padding: '0.375rem 0.25rem', border: 'none', borderRadius: 4, cursor: 'pointer',
          background: active === k ? '#6929c4' : '#393939',
          color: active === k ? '#fff' : '#c6c6c6',
          fontSize: '0.6rem', fontFamily: 'monospace', lineHeight: 1.4, textAlign: 'center',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{i + 1}</div>
          <div>{STEP_LABELS[k]}</div>
          <div style={{
            marginTop: 2,
            color: active === k ? '#e5d5ff' : '#8a8a8a',
            maxWidth: 44, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>{summaries[k]}</div>
        </button>
      ))}
    </div>
  )
}

// ── Step detail panes ─────────────────────────────────────────────────────
function StepDetail({ stepKey, result }: { stepKey: StepKey; result: SimulateResponse }) {
  const ps = result.pipeline_steps
  const label = (s: string) => s.replace(/_/g, ' ')

  if (stepKey === 'features') {
    const f = ps.features
    return (
      <div>
        <p style={{ fontSize: '0.75rem', color: '#c6c6c6', marginBottom: '0.75rem' }}>
          <strong style={{ color: '#f4f4f4' }}>STEP 1 — FEATURES</strong>
          <br />~{f?.total_tokens ?? 0} tokens &nbsp;·&nbsp; {f?.user_messages ?? 0} msg
          {f?.has_tools ? ' · tools' : ''}
          {f?.has_images ? ' · images' : ''}
          {f?.language_hint ? ` · ${f.language_hint}` : ''}
        </p>
        {Object.entries(f?.dimensions ?? {}).map(([k, v]) => (
          <ScoreBar key={k} label={label(k)} value={v} />
        ))}
      </div>
    )
  }

  if (stepKey === 'classifier') {
    const c = ps.classifier
    const sourceColors: Record<string, string> = { laya: '#be95ff', mbert: '#82cfff', regex: '#6f6f6f' }
    const src = c?.source ?? 'regex'
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STEP 2 — CLASSIFIER</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#f4f4f4' }}>{c?.task_type ?? '—'}</span>
          <span style={{ background: tierColor(c?.classifier_tier ?? null), borderRadius: 20, padding: '0 8px', fontSize: '0.7rem', fontWeight: 600, color: '#161616' }}>{c?.classifier_tier ?? '—'}</span>
          <span style={{ fontSize: '0.75rem', color: '#9f9f9f' }}>confidence {c?.confidence?.toFixed(2) ?? '—'}</span>
          {/* source badge — prominent when non-regex */}
          <span style={{
            background: sourceColors[src] ?? '#6f6f6f',
            borderRadius: 20, padding: '0 8px', fontSize: '0.65rem',
            fontWeight: 700, color: src === 'regex' ? '#c6c6c6' : '#161616',
            fontFamily: 'monospace',
          }}>{src}</span>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {Object.entries(c?.fired_signals ?? {}).map(([k, v]) => (
            <span key={k} style={{ background: '#393939', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', color: '#c6c6c6', fontFamily: 'monospace' }}>
              {label(k)} {v > 0 && v !== 1 ? `(${v.toFixed(2)})` : ''}
            </span>
          ))}
        </div>
      </div>
    )
  }

  if (stepKey === 'scorer') {
    const s = ps.scorer
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STEP 3 — SCORER</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#f4f4f4' }}>raw score {s?.raw_score?.toFixed(4) ?? '—'}</span>
          <span style={{ background: tierColor(s?.scorer_tier ?? null), borderRadius: 20, padding: '0 8px', fontSize: '0.7rem', fontWeight: 600, color: '#161616' }}>{s?.scorer_tier ?? '—'}</span>
        </div>
        {Object.entries(s?.dimension_scores ?? {}).map(([k, v]) => (
          <ScoreBar key={k} label={label(k)} value={v} />
        ))}
      </div>
    )
  }

  if (stepKey === 'blend') {
    const b = ps.blend
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STEP 4 — BLEND</p>
        <p style={{ fontSize: '0.8rem', color: '#c6c6c6', fontFamily: 'monospace' }}>
          blend: max({b?.scorer_tier ?? '—'}, {b?.classifier_tier ?? '—'}) = <strong style={{ color: '#f4f4f4' }}>{b?.blended_tier ?? '—'}</strong>
        </p>
      </div>
    )
  }

  if (stepKey === 'ml') {
    const m = ps.ml
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STEP 5 — ML TIER GUARD</p>
        <p style={{ fontSize: '0.75rem', color: '#8a8a8a', marginBottom: '0.5rem', fontStyle: 'italic' }}>
          XGBoost model that can override the blended tier. Separate from the task classifier in Step 2.
        </p>
        <p style={{ fontSize: '0.8rem', color: '#c6c6c6' }}>
          {m?.enabled
            ? <><span style={{ color: '#f4f4f4', fontFamily: 'monospace' }}>{m.predicted_tier ?? '—'}</span>&nbsp; conf {m.confidence.toFixed(2)} threshold {m.threshold} {m.active ? '· active' : ''}</>
            : <span style={{ color: '#8a8a8a' }}>Tier ML guard off — enable via ML &amp; Training page</span>
          }
        </p>
      </div>
    )
  }

  if (stepKey === 'policy_applied') {
    const p = ps.policy_applied
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STEP 6 — POLICY</p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {p?.task_override_used && (
            <span style={{ background: '#393939', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', color: '#c6c6c6', fontFamily: 'monospace' }}>task override applied</span>
          )}
          <span style={{ background: '#393939', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', color: '#c6c6c6', fontFamily: 'monospace' }}>resolved → {p?.resolved_tier ?? '—'}</span>
        </div>
      </div>
    )
  }

  if (stepKey === 'winner') {
    return (
      <div>
        <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.75rem' }}>STEP 7 — CANDIDATES</p>
        {result.candidates.map((c, i) => {
          const isWinner = c.model === result.selected_model
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.375rem 0.625rem',
              borderRadius: 6, marginBottom: 4,
              border: isWinner ? '1px solid #6929c4' : '1px solid #393939',
              background: isWinner ? '#1c0f33' : '#262626',
            }}>
              <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: '#f4f4f4', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {c.model}
              </span>
              <span style={{ background: tierColor(c.tier), borderRadius: 20, padding: '0 6px', fontSize: '0.65rem', fontWeight: 600, color: '#161616', flexShrink: 0 }}>{c.tier}</span>
              <span style={{ fontSize: '0.65rem', color: '#9f9f9f', flexShrink: 0 }}>${c.input_per_mtok.toFixed(2)}/M</span>
              {isWinner && <span style={{ fontSize: '0.65rem', color: '#6fdc8c', fontWeight: 700, flexShrink: 0 }}>✓ winner</span>}
            </div>
          )
        })}
      </div>
    )
  }

  return null
}

// ── Sample prompts ─────────────────────────────────────────────────────────
const SAMPLES = [
  'Hi, how are you?',
  'Summarize the key differences between REST and GraphQL',
  'Design a microservices architecture for an e-commerce platform',
  'Solve: if 2x + 5 = 17, prove the solution step by step',
  'Write a Python async web scraper with retry logic',
]

// ── Main panel ────────────────────────────────────────────────────────────
export function SimulatorPanel() {
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SimulateResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeStep, setActiveStep] = useState<StepKey>('features')
  const [showJson, setShowJson] = useState(false)
  const [copied, setCopied] = useState(false)
  const [activeTab, setActiveTab] = useState<'trace' | 'security' | 'cache' | 'compression' | 'rule'>('trace')

  async function run() {
    if (!prompt.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.simulate([{ role: 'user', content: prompt }])
      setResult(res)
      setActiveStep('features')
      setShowJson(false)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run() }
  }, [prompt])

  function copyJson() {
    if (!result) return
    navigator.clipboard.writeText(JSON.stringify(result, null, 2)).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const jsonStr = result ? JSON.stringify(result, null, 2) : ''

  return (
    <div style={{
      background: '#161616', color: '#f4f4f4', borderRadius: 8,
      padding: '1.25rem', minHeight: 400,
      fontFamily: '-apple-system, "Segoe UI", system-ui, sans-serif',
    }}>
      <h4 style={{ margin: '0 0 0.25rem', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', color: '#9f9f9f' }}>SIMULATOR</h4>
      <p style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: '#8a8a8a', lineHeight: 1.4 }}>
        Type a message to trace exactly how the smart router picks a model — no call is made.
      </p>

      {/* Sample prompts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', marginBottom: '0.75rem' }}>
        {SAMPLES.map(s => (
          <button key={s} onClick={() => setPrompt(s)} style={{
            background: '#262626', border: '1px solid #393939', borderRadius: 20,
            padding: '3px 10px', fontSize: '0.7rem', color: '#c6c6c6', cursor: 'pointer',
            maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {s}
          </button>
        ))}
      </div>

      {/* Prompt textarea */}
      <textarea
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={3}
        placeholder="Enter a prompt…"
        style={{
          width: '100%', boxSizing: 'border-box', resize: 'vertical',
          background: '#262626', border: '1px solid #393939', borderRadius: 4,
          color: '#f4f4f4', fontSize: '0.875rem', padding: '0.5rem 0.625rem',
          fontFamily: 'inherit', outline: 'none',
        }}
      />

      <button
        onClick={run}
        disabled={loading || !prompt.trim()}
        style={{
          marginTop: '0.625rem', width: '100%', padding: '0.625rem',
          background: loading || !prompt.trim() ? '#6929c480' : '#6929c4',
          border: 'none', borderRadius: 6, color: '#fff', fontSize: '0.875rem',
          fontWeight: 600, cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
        }}
      >
        {loading ? <InlineLoading description="Simulating…" /> : <>Simulate ⌘↵</>}
      </button>

      {error && (
        <p style={{ color: '#ff8389', fontSize: '0.75rem', marginTop: '0.5rem' }}>{error}</p>
      )}

      {result?.classifier_warning && (
        <div style={{
          marginTop: '0.75rem', background: '#3b2700', border: '1px solid #f1c21b',
          borderRadius: 6, padding: '0.625rem 0.875rem',
          display: 'flex', alignItems: 'flex-start', gap: '0.5rem',
        }}>
          <span style={{ color: '#f1c21b', fontSize: '0.9rem', flexShrink: 0 }}>⚠</span>
          <p style={{ margin: 0, fontSize: '0.75rem', color: '#ffd966', lineHeight: 1.5 }}>
            {result.classifier_warning}
          </p>
        </div>
      )}

      {result && (
        <div style={{ marginTop: '1rem' }}>
          {/* Tab bar */}
          <div style={{ display: 'flex', background: '#262626', borderRadius: 6, padding: 3, marginBottom: '1rem', gap: 2 }}>
            {([
              ['trace', 'Pipeline'],
              ['security', '🔒 Security'],
              ['cache', '⚡ Cache'],
              ['compression', '📦 Compress'],
              ['rule', 'Rule'],
            ] as const).map(([t, label]) => (
              <button key={t} onClick={() => setActiveTab(t as any)} style={{
                flex: 1, padding: '0.375rem 0.25rem', border: 'none', borderRadius: 4, cursor: 'pointer',
                background: activeTab === t ? '#393939' : 'transparent',
                color: activeTab === t ? '#f4f4f4' : '#8a8a8a', fontSize: '0.7rem', fontWeight: 600,
              }}>
                {label}
              </button>
            ))}
          </div>

          {activeTab === 'security' && result.security && (
            <div style={{ fontSize: '0.8rem', color: '#c6c6c6', lineHeight: 1.7 }}>
              <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STAGE 0.5 — SECURITY GUARD</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.625rem', flexWrap: 'wrap' }}>
                <SensBadge v={result.security.sensitivity} />
                {!result.security.enabled && <span style={{ fontSize: '0.7rem', color: '#f1c21b' }}>⚠ guard disabled</span>}
              </div>
              {result.security.pii_types.length > 0 && (
                <div style={{ marginBottom: '0.375rem' }}>
                  <span style={{ color: '#9f9f9f', fontSize: '0.7rem' }}>PII detected: </span>
                  {result.security.pii_types.map(p => (
                    <span key={p} style={{ background: '#3b1414', border: '1px solid #da1e2850', borderRadius: 20, padding: '0 6px', marginRight: 4, fontSize: '0.65rem', color: '#ff8389', fontFamily: 'monospace' }}>{p}</span>
                  ))}
                </div>
              )}
              {result.security.secret_types.length > 0 && (
                <div style={{ marginBottom: '0.375rem' }}>
                  <span style={{ color: '#9f9f9f', fontSize: '0.7rem' }}>Secrets detected: </span>
                  {result.security.secret_types.map(p => (
                    <span key={p} style={{ background: '#3b1414', border: '1px solid #da1e2850', borderRadius: 20, padding: '0 6px', marginRight: 4, fontSize: '0.65rem', color: '#ff8389', fontFamily: 'monospace' }}>{p}</span>
                  ))}
                </div>
              )}
              {result.security.blocked_providers.length > 0 ? (
                <div style={{ marginBottom: '0.375rem' }}>
                  <span style={{ color: '#9f9f9f', fontSize: '0.7rem' }}>Blocked deployments: </span>
                  {result.security.blocked_providers.map(p => (
                    <span key={p} style={{ background: '#262626', border: '1px solid #da1e2830', borderRadius: 20, padding: '0 6px', marginRight: 4, fontSize: '0.65rem', color: '#ff8389', fontFamily: 'monospace' }}>{p}</span>
                  ))}
                  <span style={{ fontSize: '0.7rem', color: '#9f9f9f' }}>({result.security.candidates_removed} removed from candidacy)</span>
                </div>
              ) : (
                <p style={{ color: '#6fdc8c', fontSize: '0.75rem' }}>✓ No deployments blocked by policy</p>
              )}
              {result.security.audit_summary && (
                <p style={{ fontSize: '0.7rem', color: '#8a8a8a', fontFamily: 'monospace', marginTop: '0.375rem' }}>{result.security.audit_summary}</p>
              )}
            </div>
          )}

          {activeTab === 'cache' && result.cache_routing && (
            <div style={{ fontSize: '0.8rem', color: '#c6c6c6', lineHeight: 1.7 }}>
              <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>STAGE 6e — CONTEXT REUSE SCORE</p>
              <div style={{ marginBottom: '0.75rem' }}>
                <ScoreBar label="context_reuse_score" value={result.cache_routing.context_reuse_score} />
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', marginBottom: '0.5rem' }}>
                <span style={{ background: '#262626', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', fontFamily: 'monospace', color: '#c6c6c6' }}>
                  turn #{result.cache_routing.turn_index}
                </span>
                <span style={{ background: result.cache_routing.cache_warm ? '#0a2a1a' : '#262626', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', fontFamily: 'monospace', color: result.cache_routing.cache_warm ? '#6fdc8c' : '#8a8a8a' }}>
                  {result.cache_routing.cache_warm ? '● warm' : '○ cold'}
                </span>
                {!result.cache_routing.enabled && (
                  <span style={{ background: '#262626', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', color: '#f1c21b', fontFamily: 'monospace' }}>⚠ stage disabled</span>
                )}
              </div>
              <p style={{ fontSize: '0.75rem', color: '#c6c6c6', fontFamily: 'monospace', marginBottom: '0.25rem' }}>
                <strong style={{ color: '#9f9f9f' }}>reason:</strong> {result.cache_routing.routing_reason}
              </p>
              {result.cache_routing.session_id && (
                <p style={{ fontSize: '0.7rem', color: '#8a8a8a', fontFamily: 'monospace' }}>
                  session: {result.cache_routing.session_id}
                </p>
              )}
              {result.cache_routing.prefix_hash && (
                <p style={{ fontSize: '0.7rem', color: '#8a8a8a', fontFamily: 'monospace' }}>
                  prefix_hash: {result.cache_routing.prefix_hash}
                </p>
              )}
            </div>
          )}

          {activeTab === 'compression' && result.compression && (
            <div style={{ fontSize: '0.8rem', color: '#c6c6c6', lineHeight: 1.7 }}>
              <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>COMPRESSION GATE</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.625rem', flexWrap: 'wrap' }}>
                <span style={{ background: '#262626', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', fontFamily: 'monospace', color: '#c6c6c6' }}>
                  profile: {result.compression.profile}
                </span>
                {result.compression.cache_aware_skip && (
                  <span style={{ background: '#0a2a1a', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', fontFamily: 'monospace', color: '#6fdc8c' }}>
                    ⚡ cache-aware skip (reuse ≥ {result.compression.cache_aware_threshold})
                  </span>
                )}
              </div>
              <div style={{ marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.75rem' }}>
                  <span><span style={{ color: '#9f9f9f' }}>before:</span> {result.compression.tokens_before}t</span>
                  <span><span style={{ color: '#9f9f9f' }}>after:</span> {result.compression.tokens_after}t</span>
                  <span style={{ color: result.compression.savings_pct > 0 ? '#6fdc8c' : '#8a8a8a' }}>
                    {result.compression.savings_pct > 0 ? `−${result.compression.savings_pct}%` : 'no savings'}
                  </span>
                </div>
                {result.compression.tokens_before > 0 && (
                  <div style={{ marginTop: '0.375rem' }}>
                    <ScoreBar label="compression ratio" value={result.compression.savings_pct / 100} />
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'trace' ? (
            <>
              {/* Winner hero card */}
              <div style={{
                background: '#1c1c3a', border: '1px solid #6929c4', borderRadius: 8,
                padding: '0.75rem 1rem', marginBottom: '1rem',
              }}>
                <div style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '1rem', color: '#f4f4f4', marginBottom: 4 }}>
                  {result.selected_model ?? '—'}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', fontSize: '0.75rem', color: '#9f9f9f' }}>
                  <span style={{ background: tierColor(result.tier), borderRadius: 20, padding: '0 8px', fontWeight: 700, color: '#161616' }}>{result.tier ?? '—'}</span>
                  <span style={{ fontFamily: 'monospace' }}>{result.task_type ?? '—'}</span>
                  <span>{result.routing_stage ?? ''}</span>
                </div>
              </div>

              {/* Step tabs */}
              <StepTabBar active={activeStep} result={result} onSelect={setActiveStep} />

              {/* Step detail */}
              <div style={{ minHeight: 120 }}>
                <StepDetail stepKey={activeStep} result={result} />
              </div>

              {/* Policy tags */}
              {result.pipeline_steps.policy_applied && (
                <div style={{ marginTop: '1rem' }}>
                  <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>POLICY APPLIED</p>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {result.pipeline_steps.policy_applied.resolved_tier && (
                      <span style={{ background: '#393939', borderRadius: 20, padding: '2px 10px', fontSize: '0.65rem', color: '#c6c6c6', fontFamily: 'monospace' }}>
                        resolved {result.pipeline_steps.policy_applied.resolved_tier}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Input signals */}
              <div style={{ marginTop: '1rem' }}>
                <p style={{ fontSize: '0.7rem', fontWeight: 700, color: '#9f9f9f', letterSpacing: '0.08em', marginBottom: '0.25rem' }}>INPUT SIGNALS</p>
                <p style={{ fontSize: '0.75rem', color: '#c6c6c6', fontFamily: 'monospace' }}>
                  ~{result.features.estimated_tokens} tokens&nbsp; {result.features.message_count} msg
                  {result.features.has_tools ? ' · tools' : ''}
                </p>
              </div>

              {/* JSON viewer */}
              <div style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <button onClick={() => setShowJson(v => !v)} style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: '0.75rem', color: '#9f9f9f', padding: 0,
                  }}>
                    {showJson ? '▲ Hide full JSON' : '▼ Show full JSON'}
                  </button>
                  {showJson && (
                    <button onClick={copyJson} style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontSize: '0.75rem', color: copied ? '#6fdc8c' : '#78a9ff',
                      display: 'flex', alignItems: 'center', gap: 4, padding: 0,
                    }}>
                      {copied ? <><CheckmarkFilled size={12} /> copied</> : <><Copy size={12} /> copy</>}
                    </button>
                  )}
                </div>
                {showJson && (
                  <pre style={{
                    marginTop: '0.5rem', background: '#262626', borderRadius: 6,
                    padding: '0.75rem', fontSize: '0.65rem', color: '#c6c6c6',
                    overflowX: 'auto', maxHeight: 400, lineHeight: 1.6,
                    fontFamily: '"IBM Plex Mono", "Courier New", monospace',
                  }}>
                    {jsonStr}
                  </pre>
                )}
              </div>
            </>
          ) : (
            /* Rule match tab — shows routing stage / source info */
            <div style={{ fontSize: '0.8rem', color: '#c6c6c6' }}>
              <p style={{ fontFamily: 'monospace', marginBottom: '0.5rem' }}>
                <strong style={{ color: '#f4f4f4' }}>routing_stage:</strong> {result.routing_stage ?? '—'}
              </p>
              <p style={{ fontFamily: 'monospace', marginBottom: '0.5rem' }}>
                <strong style={{ color: '#f4f4f4' }}>task_type:</strong> {result.task_type ?? '—'}
              </p>
              <p style={{ fontFamily: 'monospace', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <strong style={{ color: '#f4f4f4' }}>classifier_source:</strong>{' '}
                <span style={{
                  background: ({ laya: '#be95ff', mbert: '#82cfff', regex: '#6f6f6f' } as Record<string,string>)[result.pipeline_steps.classifier?.source ?? 'regex'] ?? '#6f6f6f',
                  borderRadius: 20, padding: '0 8px', fontSize: '0.7rem', fontWeight: 700,
                  color: result.pipeline_steps.classifier?.source === 'regex' ? '#c6c6c6' : '#161616',
                }}>
                  {result.pipeline_steps.classifier?.source ?? '—'}
                </span>
              </p>
              <p style={{ fontFamily: 'monospace', marginBottom: '0.5rem' }}>
                <strong style={{ color: '#f4f4f4' }}>ml_active:</strong> {result.pipeline_steps.ml?.active ? 'yes' : 'no'}
              </p>
              {result.pipeline_steps.winner?.reasoning && (
                <p style={{ fontFamily: 'monospace' }}>
                  <strong style={{ color: '#f4f4f4' }}>reasoning:</strong> {result.pipeline_steps.winner.reasoning}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
