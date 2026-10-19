/**
 * Request Explorer — filterable, paginated table of routing decisions.
 * Expanding a row shows a full 7-step pipeline breakdown matching the
 * simulator panel: classifier, scorer, blend, cache, security, compression,
 * plus a raw-JSON view of the entire record.
 */
import React, { useState, useCallback } from 'react'
import {
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  TableExpandHeader,
  TableExpandRow,
  TableExpandedRow,
  TextInput,
  Select,
  SelectItem,
  Button,
  Pagination,
  InlineNotification,
} from '@carbon/react'
import { Search, Reset, Time, ThumbsUp, ThumbsDown } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import { api, RequestRecord } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

const PAGE_SIZES = [25, 50, 100]

// ── colour maps ──────────────────────────────────────────────────────────────

const TIER_COLOR: Record<string, string> = {
  simple: '#24a148', medium: '#0f62fe', complex: '#ff832b', reasoning: '#8a3ffc',
}
const SENS_COLOR: Record<string, string> = {
  public: '#24a148', internal: '#0f62fe', confidential: '#f1c21b', restricted: '#da1e28',
}
const CLF_COLOR: Record<string, { bg: string; fg: string }> = {
  laya:  { bg: '#6929c430', fg: '#6929c4' },
  mbert: { bg: '#0043ce30', fg: '#0043ce' },
  regex: { bg: '#52525220', fg: '#525252' },
}

// ── small shared primitives ───────────────────────────────────────────────────

function Pill({ label, color = '#525252', bg = '#52525220' }: { label: string; color?: string; bg?: string }) {
  return (
    <span style={{
      display: 'inline-block', fontSize: '0.65rem', fontWeight: 700,
      padding: '1px 7px', borderRadius: 20, background: bg, color,
      fontFamily: 'monospace', whiteSpace: 'nowrap',
    }}>{label}</span>
  )
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: 4 }}>
      <span style={{ fontSize: '0.65rem', color: '#525252', width: 160, flexShrink: 0, textAlign: 'right', fontFamily: 'monospace' }}>
        {label.replace(/_/g, ' ')}
      </span>
      <div style={{ flex: 1, height: 6, background: '#e0e0e0', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#0f62fe', borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: '0.65rem', color: '#161616', width: 36, fontFamily: 'monospace' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

function SensChip({ value }: { value: string | undefined }) {
  if (!value || value === 'disabled') return <span style={{ color: '#a8a8a8', fontSize: '0.75rem' }}>—</span>
  const color = SENS_COLOR[value] ?? '#6f6f6f'
  return (
    <span style={{
      display: 'inline-block', fontSize: '0.6875rem', fontWeight: 700,
      padding: '1px 7px', borderRadius: 10,
      background: color + '15', color, border: `1px solid ${color}40`,
    }}>{value}</span>
  )
}

// ── pipeline drawer ───────────────────────────────────────────────────────────

type DrawerTab = 'pipeline' | 'security' | 'cache' | 'compression' | 'json'

const DRAWER_TABS: { id: DrawerTab; label: string }[] = [
  { id: 'pipeline',    label: '⚡ Pipeline' },
  { id: 'security',   label: '🔒 Security' },
  { id: 'cache',      label: '⚡ Cache' },
  { id: 'compression',label: '🗜 Compress' },
  { id: 'json',       label: '</> JSON' },
]

function DrawerTabBar({ active, onChange }: { active: DrawerTab; onChange: (t: DrawerTab) => void }) {
  return (
    <div style={{ display: 'flex', gap: 2, marginBottom: '0.75rem', borderBottom: '1px solid #e0e0e0', paddingBottom: 0 }}>
      {DRAWER_TABS.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            padding: '0.375rem 0.75rem',
            fontSize: '0.75rem', fontWeight: active === t.id ? 700 : 400,
            color: active === t.id ? '#0f62fe' : '#525252',
            borderBottom: active === t.id ? '2px solid #0f62fe' : '2px solid transparent',
            marginBottom: -1,
          }}
        >{t.label}</button>
      ))}
    </div>
  )
}

// Pipeline tab — all 7 steps matching simulator depth
function PipelineTab({ rec }: { rec: RequestRecord }) {
  const sb = rec.score_breakdown ?? {}
  const allDims = Object.entries(sb)  // all 14, sorted by value desc
    .sort(([, a], [, b]) => b - a)
  const clfStyle = CLF_COLOR[rec.classifier_source ?? 'regex'] ?? CLF_COLOR.regex

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      {/* Step nav bar */}
      <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
        {[
          { n: '1', label: 'Features' },
          { n: '2', label: 'Classify' },
          { n: '3', label: 'Score' },
          { n: '4', label: 'Blend' },
          { n: '5', label: 'ML' },
          { n: '6', label: 'Policy' },
          { n: '7', label: 'Winner' },
        ].map(s => (
          <div key={s.n} style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            padding: '0.25rem 0.625rem', borderRadius: 6,
            background: s.n === '7' ? '#1c0f33' : '#f4f4f4',
            border: s.n === '7' ? '1px solid #6929c4' : '1px solid #e0e0e0',
            minWidth: 60,
          }}>
            <span style={{ fontSize: '0.6rem', color: s.n === '7' ? '#be95ff' : '#8d8d8d' }}>{s.n}</span>
            <span style={{ fontSize: '0.65rem', fontWeight: 600, color: s.n === '7' ? '#f4f4f4' : '#161616' }}>{s.label}</span>
          </div>
        ))}
      </div>

      {/* Step 1 — Features */}
      <div>
        <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>STEP 1 — FEATURES</div>
        <p style={{ margin: '0 0 0.5rem', fontSize: '0.75rem', color: '#525252' }}>
          ~{rec.feature_tokens ?? rec.input_tokens ?? 0} tokens
          {' · '}{1} msg
          {rec.feature_has_tools ? ' · tools' : ''}
          {rec.feature_language ? ` · ${rec.feature_language}` : ''}
        </p>
        {allDims.length > 0 && allDims.map(([k, v]) => (
          <ScoreBar key={k} label={k} value={v} />
        ))}
      </div>

      {/* Step 2 — Classifier */}
      <div>
        <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>STEP 2 — CLASSIFIER</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.375rem' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '0.8rem' }}>{rec.task_type ?? '—'}</span>
          <span style={{
            background: TIER_COLOR[rec.classifier_tier ?? rec.tier ?? ''] ?? '#8d8d8d',
            borderRadius: 20, padding: '0 8px', fontSize: '0.65rem', fontWeight: 600, color: '#fff',
          }}>{rec.classifier_tier ?? rec.tier ?? '—'}</span>
          {rec.classifier_source && (
            <span style={{
              background: clfStyle.bg, color: clfStyle.fg,
              borderRadius: 20, padding: '0 8px', fontSize: '0.65rem', fontWeight: 700, fontFamily: 'monospace',
            }}>{rec.classifier_source}</span>
          )}
          {rec.classifier_confidence != null && (
            <span style={{ fontSize: '0.7rem', color: '#525252' }}>confidence {rec.classifier_confidence.toFixed(2)}</span>
          )}
        </div>
        {rec.classifier_signals && Object.keys(rec.classifier_signals).length > 0 && (
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            {Object.entries(rec.classifier_signals).map(([k, v]) => (
              <span key={k} style={{
                background: '#e8e8e8', borderRadius: 20, padding: '2px 9px',
                fontSize: '0.65rem', color: '#393939', fontFamily: 'monospace',
              }}>
                {k.replace(/_/g, ' ')}{v > 0 && v !== 1 ? ` (${v.toFixed(2)})` : ''}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Step 3 — Scorer */}
      <div>
        <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>STEP 3 — SCORER</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '0.75rem' }}>
            raw score {rec.raw_score != null ? rec.raw_score.toFixed(4) : '—'}
          </span>
          {rec.scorer_tier && (
            <span style={{
              background: TIER_COLOR[rec.scorer_tier] ?? '#8d8d8d',
              borderRadius: 20, padding: '0 8px', fontSize: '0.65rem', fontWeight: 600, color: '#fff',
            }}>{rec.scorer_tier}</span>
          )}
          {rec.raw_score != null && (
            <span style={{ fontSize: '0.7rem', color: '#525252' }}>
              (dimensions shown in Step 1)
            </span>
          )}
        </div>
      </div>

      {/* Step 4 — Blend */}
      <div>
        <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>STEP 4 — BLEND</div>
        <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#161616' }}>
          {rec.blend_clf_used === false ? (
            <>
              scorer({rec.scorer_tier ?? '—'}) <span style={{ color: '#8d8d8d' }}>clf suppressed (low conf)</span> → <strong>{rec.tier ?? '—'}</strong>
            </>
          ) : (
            <>max({rec.scorer_tier ?? '—'}, {rec.clf_tier_pre_blend ?? rec.classifier_tier ?? '—'}) → <strong>{rec.tier ?? '—'}</strong></>
          )}
        </span>
        {rec.blend_clf_used === false && (
          <div style={{ fontSize: '0.68rem', color: '#6f6f6f', marginTop: '0.2rem' }}>
            Classifier ({rec.clf_tier_pre_blend ?? rec.classifier_tier ?? '—'}) ignored — confidence {rec.classifier_confidence?.toFixed(2) ?? '—'} below threshold
          </div>
        )}
      </div>

      {/* Step 7 — Winner */}
      <div style={{
        padding: '0.625rem 0.75rem', borderRadius: 6,
        background: '#f0e6ff', border: '1px solid #6929c430',
      }}>
        <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#6929c4', letterSpacing: '0.08em', marginBottom: '0.375rem' }}>STEP 7 — WINNER</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '0.8rem', color: '#161616' }}>{rec.deployment ?? '—'}</span>
          <span style={{ fontSize: '0.7rem', color: '#525252' }}>{rec.provider}</span>
          <TierBadge tier={rec.tier} />
        </div>
        {rec.routing_reason && (
          <div style={{ marginTop: '0.375rem', fontSize: '0.7rem', color: '#525252', fontFamily: 'monospace' }}>
            {rec.routing_reason}
          </div>
        )}
        <div style={{ marginTop: '0.375rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <Pill label={`stage: ${rec.routing_source ?? '—'}`} color="#525252" />
          <Pill label={`conf ${(rec.routing_confidence ?? 0).toFixed(2)}`} color="#525252" />
          <Pill label={`${rec.latency_ms} ms`} color="#525252" />
          <Pill label={`$${Number(rec.cost_usd).toFixed(6)}`} color="#525252" />
          <Pill label={`in ${rec.input_tokens ?? 0} / out ${rec.output_tokens ?? 0} tok`} color="#525252" />
        </div>
      </div>

    </div>
  )
}

// Security tab
function SecurityTab({ rec }: { rec: RequestRecord }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.25rem' }}>STAGE 0.5 — SECURITY GUARD</div>
      <div style={{ fontSize: '0.75rem' }}>
        <span style={{ color: '#525252' }}>Sensitivity: </span>
        <SensChip value={rec.security_sensitivity} />
      </div>
      {rec.context_reuse_score != null && (
        <div style={{ fontSize: '0.75rem' }}>
          <span style={{ color: '#525252' }}>Context reuse score: </span>
          <span style={{ fontWeight: 600 }}>{rec.context_reuse_score.toFixed(3)}</span>
          <span style={{ color: '#525252', marginLeft: 6, fontSize: '0.65rem' }}>
            {rec.context_reuse_score >= 0.8 ? '· strong affinity'
              : rec.context_reuse_score >= 0.5 ? '· moderate affinity'
              : rec.context_reuse_score > 0 ? '· weak signal'
              : '· cold start'}
          </span>
        </div>
      )}
      {rec.security_blocked && rec.security_blocked.length > 0 ? (
        <div style={{ fontSize: '0.75rem' }}>
          <span style={{ color: '#525252' }}>Blocked providers: </span>
          <span style={{ color: '#da1e28', fontWeight: 600 }}>{rec.security_blocked.join(', ')}</span>
        </div>
      ) : (
        <div style={{ fontSize: '0.75rem', color: '#24a148' }}>✓ No providers blocked by policy</div>
      )}
    </div>
  )
}

// Cache tab
function CacheTab({ rec }: { rec: RequestRecord }) {
  const score = rec.context_reuse_score ?? 0
  const pct = Math.round(score * 100)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.25rem' }}>STAGE 6E — CONTEXT REUSE</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{ flex: 1, height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: '#6929c4', borderRadius: 4 }} />
        </div>
        <span style={{ fontFamily: 'monospace', fontSize: '0.8rem', fontWeight: 700, color: '#6929c4', minWidth: 36 }}>
          {score.toFixed(2)}
        </span>
      </div>
      <div style={{ fontSize: '0.7rem', color: '#525252' }}>
        {score >= 0.8 ? 'Strong cache affinity — session routed to warm model'
          : score >= 0.5 ? 'Moderate cache affinity — context-reuse bonus applied'
          : score > 0 ? 'Weak cache signal — base scoring dominated'
          : 'No session or cold start'}
      </div>
    </div>
  )
}

// Compression tab
function CompressionTab({ rec }: { rec: RequestRecord }) {
  const before = rec.tokens_before_compression ?? 0
  const after = rec.tokens_after_compression ?? before
  const saved = before > 0 ? Math.round((1 - after / before) * 100) : 0
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ fontSize: '0.6rem', fontWeight: 700, color: '#8d8d8d', letterSpacing: '0.08em', marginBottom: '0.25rem' }}>COMPRESSION</div>
      <div style={{ fontSize: '0.75rem' }}>
        <span style={{ color: '#525252' }}>Profile: </span>
        <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{rec.compression_profile ?? 'passthrough'}</span>
      </div>
      {before > 0 && (
        <>
          <div style={{ fontSize: '0.75rem' }}>
            <span style={{ color: '#525252' }}>Tokens: </span>
            <span>{before.toLocaleString()}</span>
            <span style={{ color: '#525252' }}> → </span>
            <span style={{ color: '#24a148', fontWeight: 600 }}>{after.toLocaleString()}</span>
            <span style={{ color: '#525252' }}> ({saved}% saved)</span>
          </div>
          <div style={{ height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden', maxWidth: 300 }}>
            <div style={{ width: `${100 - saved}%`, height: '100%', background: '#24a148', borderRadius: 4 }} />
          </div>
        </>
      )}
      {(rec.volatile_tokens_found ?? 0) > 0 && (
        <div style={{ fontSize: '0.75rem', color: '#da6a00' }}>
          ⚠ <strong>{rec.volatile_tokens_found}</strong> volatile token(s) in system prompt detected by CacheAligner
        </div>
      )}
      {before === 0 && (
        <div style={{ fontSize: '0.75rem', color: '#8d8d8d' }}>No compression applied to this request.</div>
      )}
    </div>
  )
}

// JSON tab — full height with expand toggle
function JsonTab({ rec }: { rec: RequestRecord }) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const json = JSON.stringify(rec, null, 2)
  function copy() {
    navigator.clipboard?.writeText(json)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginBottom: '0.375rem' }}>
        <button onClick={() => setExpanded(v => !v)} style={{
          background: 'none', border: '1px solid #e0e0e0', borderRadius: 4,
          padding: '2px 10px', fontSize: '0.7rem', cursor: 'pointer', color: '#525252',
        }}>
          {expanded ? '↑ Collapse' : '↓ Expand'}
        </button>
        <button onClick={copy} style={{
          background: 'none', border: '1px solid #e0e0e0', borderRadius: 4,
          padding: '2px 10px', fontSize: '0.7rem', cursor: 'pointer', color: '#525252',
        }}>
          {copied ? '✓ Copied' : 'Copy JSON'}
        </button>
      </div>
      <pre style={{
        margin: 0, padding: '0.75rem', background: '#161616', color: '#f4f4f4',
        borderRadius: 6, fontSize: '0.65rem', lineHeight: 1.5, overflow: 'auto',
        height: expanded ? 'auto' : 280, fontFamily: 'monospace',
        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>{json}</pre>
    </div>
  )
}

// ── Feedback bar ─────────────────────────────────────────────────────────────

const TIERS = ['simple', 'medium', 'complex', 'reasoning'] as const

function FeedbackBar({ rec }: { rec: RequestRecord }) {
  const [vote, setVote]               = useState<'up' | 'down' | null>(null)
  const [correctedTier, setCorrected] = useState('')
  const [status, setStatus]           = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [errorMsg, setErrorMsg]       = useState('')

  const submit = useCallback(async () => {
    if (vote === null) return
    setStatus('saving')
    try {
      await api.feedback.submit(
        rec.id,
        vote === 'up',
        correctedTier || undefined,
      )
      setStatus('saved')
    } catch (e: any) {
      setErrorMsg(e?.message ?? 'Failed to save feedback')
      setStatus('error')
    }
  }, [vote, correctedTier, rec.id])

  if (status === 'saved') {
    return (
      <div style={{ marginBottom: '0.75rem' }}>
        <InlineNotification
          kind="success"
          title="Feedback saved"
          subtitle="This routing decision has been added to the training dataset."
          hideCloseButton
          lowContrast
        />
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap',
      marginBottom: '0.875rem', padding: '0.625rem 0.75rem',
      background: '#f4f4f4', borderRadius: 6, border: '1px solid #e0e0e0',
    }}>
      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#525252', whiteSpace: 'nowrap' }}>
        Was this routing correct?
      </span>

      {/* Thumbs up */}
      <button
        onClick={() => setVote('up')}
        title="Routing was correct"
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          padding: '4px 12px', borderRadius: 20, cursor: 'pointer',
          fontSize: '0.75rem', fontWeight: 600,
          border: vote === 'up' ? '2px solid #24a148' : '1px solid #c6c6c6',
          background: vote === 'up' ? '#defbe6' : '#fff',
          color: vote === 'up' ? '#198038' : '#525252',
        }}
      >
        <ThumbsUp size={14} /> Yes
      </button>

      {/* Thumbs down */}
      <button
        onClick={() => setVote('down')}
        title="Routing was wrong"
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          padding: '4px 12px', borderRadius: 20, cursor: 'pointer',
          fontSize: '0.75rem', fontWeight: 600,
          border: vote === 'down' ? '2px solid #da1e28' : '1px solid #c6c6c6',
          background: vote === 'down' ? '#fff1f1' : '#fff',
          color: vote === 'down' ? '#da1e28' : '#525252',
        }}
      >
        <ThumbsDown size={14} /> No
      </button>

      {/* Tier correction — only shown after thumbs-down */}
      {vote === 'down' && (
        <select
          value={correctedTier}
          onChange={e => setCorrected(e.target.value)}
          style={{
            padding: '4px 8px', borderRadius: 4, border: '1px solid #c6c6c6',
            fontSize: '0.75rem', background: '#fff', cursor: 'pointer',
          }}
        >
          <option value="">Should have been… (optional)</option>
          {TIERS.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      )}

      {/* Submit */}
      {vote !== null && (
        <button
          onClick={submit}
          disabled={status === 'saving'}
          style={{
            padding: '4px 14px', borderRadius: 4, cursor: status === 'saving' ? 'not-allowed' : 'pointer',
            fontSize: '0.75rem', fontWeight: 600,
            background: '#0f62fe', color: '#fff', border: 'none',
            opacity: status === 'saving' ? 0.6 : 1,
          }}
        >
          {status === 'saving' ? 'Saving…' : 'Submit feedback'}
        </button>
      )}

      {status === 'error' && (
        <span style={{ fontSize: '0.7rem', color: '#da1e28' }}>⚠ {errorMsg}</span>
      )}
    </div>
  )
}

// Full expanded drawer
function RequestDrawer({ rec }: { rec: RequestRecord }) {
  const [tab, setTab] = useState<DrawerTab>('pipeline')
  return (
    <div style={{ padding: '0.75rem 1rem 1rem', background: '#fafafa', borderTop: '1px solid #e0e0e0' }}>
      <FeedbackBar rec={rec} />
      <DrawerTabBar active={tab} onChange={setTab} />
      <div style={{ maxWidth: 900 }}>
        {tab === 'pipeline'    && <PipelineTab    rec={rec} />}
        {tab === 'security'   && <SecurityTab    rec={rec} />}
        {tab === 'cache'      && <CacheTab       rec={rec} />}
        {tab === 'compression'&& <CompressionTab rec={rec} />}
        {tab === 'json'       && <JsonTab        rec={rec} />}
      </div>
    </div>
  )
}

// ── table headers ─────────────────────────────────────────────────────────────

const headers = [
  { key: 'timestamp',            header: 'Time' },
  { key: 'deployment',           header: 'Deployment' },
  { key: 'tier',                 header: 'Tier' },
  { key: 'task_type',            header: 'Task Type' },
  { key: 'provider',             header: 'Provider' },
  { key: 'security_sensitivity', header: 'Sensitivity' },
  { key: 'context_reuse_score',  header: 'Reuse' },
  { key: 'latency_ms',           header: 'Latency' },
  { key: 'cost_usd',             header: 'Cost' },
]

// ── time range presets ────────────────────────────────────────────────────────

const TIME_RANGES = [
  { label: 'Last 1 hour',   value: '1h',  seconds: 60 * 60 },
  { label: 'Last 6 hours',  value: '6h',  seconds: 6 * 60 * 60 },
  { label: 'Last 24 hours', value: '24h', seconds: 24 * 60 * 60 },
  { label: 'Last 7 days',   value: '7d',  seconds: 7 * 24 * 60 * 60 },
  { label: 'All',           value: 'all', seconds: 0 },
] as const

type TimeRangeValue = typeof TIME_RANGES[number]['value']

function formatTimestamp(ts: number | undefined | null) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  if (isNaN(d.getTime())) return '—'
  return (
    d.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
    ' ' +
    d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

export default function RequestExplorerPage() {
  const [provider, setProvider] = useState('')
  const [tier, setTier] = useState('')
  const [taskType, setTaskType] = useState('')
  const [sensitivity, setSensitivity] = useState('')
  const [timeRange, setTimeRange] = useState<TimeRangeValue>('24h')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['requests', provider, tier, taskType],
    queryFn: () =>
      api.requests({
        limit: 1000,
        provider: provider || undefined,
        tier: tier || undefined,
        task_type: taskType || undefined,
      }),
    refetchInterval: 15_000,
  })

  const rangeSeconds = TIME_RANGES.find(r => r.value === timeRange)?.seconds ?? 0
  const cutoff = rangeSeconds > 0 ? (Date.now() / 1000) - rangeSeconds : 0

  const allRecords: RequestRecord[] = (data?.records ?? []).filter(
    (r: any) =>
      (r.deployment || r.provider || r.tier) &&
      (!sensitivity || r.security_sensitivity === sensitivity) &&
      (cutoff === 0 || (r.timestamp ?? 0) >= cutoff)
  )
  const totalItems = allRecords.length
  const start = (page - 1) * pageSize
  const pageRecords = allRecords.slice(start, start + pageSize)

  const oldestTs = allRecords.length > 0
    ? Math.min(...allRecords.map((r: any) => r.timestamp ?? Infinity))
    : null
  const newestTs = allRecords.length > 0
    ? Math.max(...allRecords.map((r: any) => r.timestamp ?? 0))
    : null

  function reset() {
    setProvider(''); setTier(''); setTaskType(''); setSensitivity('')
    setTimeRange('24h'); setPage(1)
  }

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 1rem', color: '#161616' }}>Request Explorer</h2>

      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <Select id="filter-time-range" labelText="Time range" value={timeRange}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => { setTimeRange(e.target.value as TimeRangeValue); setPage(1) }}
          size="sm" style={{ maxWidth: 180 }}>
          {TIME_RANGES.map(r => <SelectItem key={r.value} value={r.value} text={r.label} />)}
        </Select>
        <TextInput id="filter-provider" labelText="Provider" placeholder="e.g. openai"
          value={provider}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setProvider(e.target.value); setPage(1) }}
          size="sm" style={{ maxWidth: 160 }} />
        <Select id="filter-tier" labelText="Tier" value={tier}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => { setTier(e.target.value); setPage(1) }}
          size="sm" style={{ maxWidth: 160 }}>
          <SelectItem value="" text="All tiers" />
          {['simple', 'medium', 'complex', 'reasoning'].map(t => (
            <SelectItem key={t} value={t} text={t} />
          ))}
        </Select>
        <TextInput id="filter-task" labelText="Task type" placeholder="e.g. code_generation"
          value={taskType}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setTaskType(e.target.value); setPage(1) }}
          size="sm" style={{ maxWidth: 200 }} />
        <Select id="filter-sensitivity" labelText="Sensitivity" value={sensitivity}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => { setSensitivity(e.target.value); setPage(1) }}
          size="sm" style={{ maxWidth: 160 }}>
          <SelectItem value="" text="All" />
          {['public', 'internal', 'confidential', 'restricted'].map(s => (
            <SelectItem key={s} value={s} text={s} />
          ))}
        </Select>
        <Button kind="ghost" size="sm" renderIcon={Reset} onClick={reset}>Reset</Button>
        <Button kind="tertiary" size="sm" renderIcon={Search} onClick={() => refetch()}>Refresh</Button>
      </div>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            <p style={{ margin: 0, fontSize: '0.75rem', color: '#525252' }}>
              Showing {pageRecords.length} of {totalItems} (buffer: {data.total_buffered})
            </p>
            {oldestTs != null && newestTs != null && totalItems > 0 && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem', color: '#525252' }}>
                <Time size={12} />
                {formatTimestamp(oldestTs)} — {formatTimestamp(newestTs)}
              </span>
            )}
          </div>

          <DataTable
            rows={pageRecords.map(r => ({ ...r, id: r.id ?? String(r.timestamp) }))}
            headers={headers}
          >
            {({ rows, headers: hs, getTableProps, getHeaderProps, getRowProps }: any) => (
              <TableContainer>
                <Table {...getTableProps()} size="sm">
                  <TableHead>
                    <TableRow>
                      <TableExpandHeader />
                      {hs.map((h: any) => (
                        <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {rows.map((row: any) => {
                      const rec = pageRecords.find(r => (r.id ?? String(r.timestamp)) === row.id)
                      return (
                        <React.Fragment key={row.id}>
                          <TableExpandRow {...getRowProps({ row })}>
                            {row.cells.map((cell: any) => (
                              <TableCell key={cell.id}>
                                {cell.info.header === 'tier' ? (
                                  <TierBadge tier={cell.value} />
                                ) : cell.info.header === 'security_sensitivity' ? (
                                  <SensChip value={cell.value} />
                                ) : cell.info.header === 'context_reuse_score' ? (
                                  cell.value != null ? (
                                    <span
                                      title={`Context reuse score: ${Number(cell.value).toFixed(4)}`}
                                      style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.75rem', fontFamily: 'monospace', cursor: 'help' }}
                                    >
                                      <span style={{
                                        display: 'inline-block', width: 28, height: 6, borderRadius: 3,
                                        background: `linear-gradient(90deg, #6929c4 ${Math.round(Number(cell.value) * 100)}%, #e0e0e0 ${Math.round(Number(cell.value) * 100)}%)`,
                                      }} />
                                      {Number(cell.value).toFixed(2)}
                                    </span>
                                  ) : <span style={{ color: '#8d8d8d', fontSize: '0.75rem' }}>—</span>
                                ) : cell.info.header === 'timestamp' ? (
                                  formatTimestamp(cell.value)
                                ) : cell.info.header === 'cost_usd' ? (
                                  `$${Number(cell.value ?? 0).toFixed(5)}`
                                ) : cell.info.header === 'latency_ms' ? (
                                  `${cell.value ?? '—'} ms`
                                ) : (
                                  cell.value ?? '—'
                                )}
                              </TableCell>
                            ))}
                          </TableExpandRow>
                          <TableExpandedRow colSpan={hs.length + 1}>
                            {rec ? <RequestDrawer rec={rec} /> : null}
                          </TableExpandedRow>
                        </React.Fragment>
                      )
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </DataTable>

          <Pagination
            totalItems={totalItems}
            pageSize={pageSize}
            pageSizes={PAGE_SIZES}
            page={page}
            onChange={({ page: p, pageSize: ps }: { page: number; pageSize: number }) => {
              setPage(p); setPageSize(ps)
            }}
          />
        </>
      )}
    </div>
  )
}
