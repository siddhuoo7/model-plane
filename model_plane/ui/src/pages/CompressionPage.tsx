/**
 * Compression page — live control of the context-compression gate.
 *
 * Profile picker: a Carbon Select dropdown on the left; a description card
 * on the right; and below that a simple before/after example block for the
 * selected profile.
 */
import React, { useEffect, useState } from 'react'
import {
  Button,
  Toggle,
  Slider,
  Select,
  SelectItem,
  InlineNotification,
  Tag,
} from '@carbon/react'
import { Save, Renew } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, CompressionConfig } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

// ── Profile metadata ─────────────────────────────────────────────────────────

const PROFILE_META: Record<string, {
  label: string; badge: string; color: string
  what: string; input: string; output: string
}> = {
  auto: {
    label: 'Auto',        badge: '🤖', color: '#24a148',
    what: 'Picks the best profile per request — no fixed rule.',
    input: 'Any messages array',
    output: 'Routed to the best matching profile below',
  },
  passthrough: {
    label: 'Passthrough', badge: '⏭', color: '#6f6f6f',
    what: 'No changes — messages forwarded exactly as received.',
    input: '[ sys, user, assistant, … ] unchanged',
    output: '[ sys, user, assistant, … ] unchanged',
  },
  tool_output_compaction: {
    label: 'Tool Compaction', badge: '🔧', color: '#0f62fe',
    what: 'Truncates oversized tool-call results (role: tool) to ≤800 tokens each.',
    input: '[ …, { role:"tool", content: "…15 000 tokens…" } ]',
    output: '[ …, { role:"tool", content: "…800 tokens…[truncated]" } ]',
  },
  conversation_summary: {
    label: 'Conv. Summary', badge: '📋', color: '#6929c4',
    what: 'Drops middle turns to fit a token budget. Keeps system prompt + last 6 turns.',
    input: '[ sys, u1, a1, u2, a2, …, u10, a10 ] — 12 000 tokens',
    output: '[ sys, [summary placeholder], u9, a9, u10, a10 ] — ~3 000 tokens',
  },
  code_context_reduction: {
    label: 'Code Reduction', badge: '💻', color: '#00539a',
    what: 'Truncates fenced code blocks (``` … ```) to first 40 lines each.',
    input: '[ user: "…```python\\n<800 lines of code>\\n```…" ]',
    output: '[ user: "…```python\\n<first 40 lines>\\n# [truncated]\\n```…" ]',
  },
}

const PROFILE_ORDER = ['passthrough', 'tool_output_compaction', 'conversation_summary', 'code_context_reduction']

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }: {
  label: string; value: string | number; sub?: string; accent?: string
}) {
  return (
    <div style={{
      background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
      borderLeft: `4px solid ${accent ?? '#0f62fe'}`,
      padding: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    }}>
      <p style={{ margin: '0 0 0.25rem', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: accent ?? '#0f62fe' }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: '#161616', lineHeight: 1 }}>{value}</p>
      {sub && <p style={{ margin: '0.25rem 0 0', fontSize: '0.75rem', color: '#525252' }}>{sub}</p>}
    </div>
  )
}

function MiniBar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = total > 0 ? Math.round(value / total * 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '0.375rem' }}>
      <span style={{ minWidth: 200, fontSize: '0.8125rem', fontFamily: 'monospace', color: '#161616' }}>{label}</span>
      <div style={{ flex: 1, height: 8, background: '#f0f0f0', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <span style={{ minWidth: 48, textAlign: 'right', fontSize: '0.8125rem', fontFamily: 'monospace', color: '#525252' }}>
        {value} <span style={{ color: '#a8a8a8' }}>({pct}%)</span>
      </span>
    </div>
  )
}

// ── Profile picker section ────────────────────────────────────────────────────
// Left: Carbon Select dropdown.  Right: description card + before/after block.

function ProfilePicker({
  enabled,
  profile,
  onSelectProfile,
}: {
  enabled: boolean
  profile: string
  onSelectProfile: (p: string) => void
}) {
  const ALL_PROFILES = ['auto', ...PROFILE_ORDER]
  const meta = PROFILE_META[profile] ?? PROFILE_META['auto']

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: '1.25rem', alignItems: 'start' }}>
      {/* ── Left: dropdown ── */}
      <div>
        <Select
          id="compression-profile"
          labelText="Algorithm profile"
          value={profile}
          disabled={!enabled}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => onSelectProfile(e.target.value)}
        >
          {ALL_PROFILES.map((p) => (
            <SelectItem key={p} value={p} text={`${PROFILE_META[p].badge}  ${PROFILE_META[p].label}`} />
          ))}
        </Select>
        {!enabled && (
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#6f6f6f' }}>
            Enable compression above to change the profile.
          </p>
        )}
      </div>

      {/* ── Right: description card + before/after example ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {/* Description card */}
        <div style={{
          background: enabled ? (meta.color + '0d') : '#f4f4f4',
          border: `1px solid ${enabled ? meta.color + '40' : '#e0e0e0'}`,
          borderLeft: `4px solid ${enabled ? meta.color : '#c0c0c0'}`,
          borderRadius: 6, padding: '0.875rem',
        }}>
          <p style={{ margin: '0 0 0.375rem', fontSize: '0.8125rem', fontWeight: 700, color: enabled ? meta.color : '#6f6f6f' }}>
            {meta.badge} {meta.label}
          </p>
          <p style={{ margin: 0, fontSize: '0.8125rem', color: '#525252', lineHeight: 1.5 }}>{meta.what}</p>
        </div>

        {/* Before / After example */}
        {profile !== 'auto' && (
          <div style={{
            background: '#fff', border: '1px solid #e0e0e0', borderRadius: 6,
            padding: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem',
          }}>
            <div>
              <p style={{ margin: '0 0 0.375rem', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#6f6f6f' }}>Before</p>
              <code style={{ display: 'block', fontSize: '0.75rem', color: '#161616', background: '#f4f4f4', padding: '0.375rem 0.5rem', borderRadius: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-all', lineHeight: 1.5 }}>
                {meta.input}
              </code>
            </div>
            <div>
              <p style={{ margin: '0 0 0.375rem', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: meta.color }}>After</p>
              <code style={{ display: 'block', fontSize: '0.75rem', color: meta.color, background: meta.color + '0d', padding: '0.375rem 0.5rem', borderRadius: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-all', lineHeight: 1.5, border: `1px solid ${meta.color}30` }}>
                {meta.output}
              </code>
            </div>
          </div>
        )}
        {profile === 'auto' && (
          <div style={{ background: '#f4f4f4', border: '1px solid #e0e0e0', borderRadius: 6, padding: '0.75rem' }}>
            <p style={{ margin: '0 0 0.25rem', fontSize: '0.75rem', fontWeight: 700, color: '#525252' }}>Auto-select rules</p>
            <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.75rem', color: '#525252', lineHeight: 1.7 }}>
              {PROFILE_ORDER.map((p) => (
                <li key={p}><span style={{ color: PROFILE_META[p].color, fontWeight: 600 }}>{PROFILE_META[p].label}</span> — {PROFILE_META[p].what.split('.')[0]}.</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function CompressionPage() {
  const qc = useQueryClient()
  const [saveMsg, setSaveMsg] = useState<string | null>(null)

  const [form, setForm] = useState<CompressionConfig>({
    enabled: true,
    profile: 'auto',
    token_threshold: 6000,
    cache_aware_gate_enabled: true,
    cache_aware_threshold: 0.60,
  })
  const [dirty, setDirty] = useState(false)

  const { data: cfg, isLoading: cfgLoading, error: cfgError } = useQuery({
    queryKey: ['compression-config'],
    queryFn: api.compression.getConfig,
  })

  const { data: metrics, refetch: refetchMetrics, isFetching: metricsFetching } = useQuery({
    queryKey: ['compression-metrics'],
    queryFn: api.compression.metrics,
    refetchInterval: 30_000,
  })

  useEffect(() => {
    if (cfg && !dirty) {
      setForm({
        enabled: cfg.enabled,
        profile: cfg.profile,
        token_threshold: cfg.token_threshold,
        cache_aware_gate_enabled: cfg.cache_aware_gate_enabled,
        cache_aware_threshold: cfg.cache_aware_threshold,
      })
    }
  }, [cfg])

  const saveMut = useMutation({
    mutationFn: (body: Partial<CompressionConfig>) => api.compression.setConfig(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['compression-config'] })
      qc.invalidateQueries({ queryKey: ['security-flags'] })
      setDirty(false)
      setSaveMsg('Saved — takes effect on the next request.')
      setTimeout(() => setSaveMsg(null), 3500)
    },
  })

  function update(patch: Partial<CompressionConfig>) {
    setForm(f => ({ ...f, ...patch }))
    setDirty(true)
  }

  const mTotal = metrics?.total_requests_sampled ?? 0
  const profileDist = metrics?.profile_distribution ?? {}
  const PROFILE_COLORS: Record<string, string> = {
    passthrough: '#6f6f6f', tool_output_compaction: '#0f62fe',
    conversation_summary: '#6929c4', code_context_reduction: '#00539a', auto: '#24a148',
  }

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: 960 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 700, color: '#161616' }}>Compression</h2>
          <p style={{ margin: 0, fontSize: '0.875rem', color: '#525252' }}>
            Reduces prompt tokens before dispatch. Changes take effect on the next request — no restart required.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Button kind="ghost" size="sm" renderIcon={Renew} onClick={() => refetchMetrics()} disabled={metricsFetching}>
            {metricsFetching ? 'Refreshing…' : 'Refresh metrics'}
          </Button>
          <Button kind="primary" size="sm" renderIcon={Save} disabled={!dirty || saveMut.isPending} onClick={() => saveMut.mutate(form)}>
            {saveMut.isPending ? 'Saving…' : 'Save changes'}
          </Button>
        </div>
      </div>

      {saveMsg && (
        <InlineNotification kind="success" title={saveMsg} hideCloseButton={false} onClose={() => setSaveMsg(null)} style={{ marginBottom: '1rem' }} />
      )}

      {cfgLoading && <LoadingState />}
      {cfgError && <ErrorState message={String(cfgError)} />}

      {!cfgLoading && (
        <>
          {/* ── Enable / disable ── */}
          <section style={{
            background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
            padding: '1.25rem', marginBottom: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3 style={{ margin: '0 0 0.25rem', fontSize: '0.875rem', fontWeight: 700 }}>Compression enabled</h3>
                <p style={{ margin: 0, fontSize: '0.75rem', color: '#525252' }}>
                  When off, all requests are forwarded unchanged — the flow diagram below collapses to a passthrough arrow.
                </p>
              </div>
              <Toggle
                id="compression-enabled" size="sm" labelText="" hideLabel
                labelA="Off" labelB="On"
                toggled={form.enabled}
                onToggle={(v: boolean) => update({ enabled: v })}
              />
            </div>
          </section>

          {/* ── Profile picker ── */}
          <section style={{
            background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
            padding: '1.25rem', marginBottom: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          }}>
            <h3 style={{ margin: '0 0 1rem', fontSize: '0.875rem', fontWeight: 700 }}>Algorithm profile</h3>
            <ProfilePicker
              enabled={form.enabled}
              profile={form.profile}
              onSelectProfile={(p) => update({ profile: p })}
            />
          </section>

          {/* ── Token threshold (auto mode only) ── */}
          {form.enabled && (
            <section style={{
              background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
              padding: '1.25rem', marginBottom: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
              opacity: form.profile === 'auto' ? 1 : 0.5,
              pointerEvents: form.profile === 'auto' ? 'auto' : 'none',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                <h3 style={{ margin: 0, fontSize: '0.875rem', fontWeight: 700 }}>Token threshold</h3>
                {form.profile !== 'auto' && <Tag type="gray" size="sm">auto mode only</Tag>}
              </div>
              <p style={{ margin: '0 0 1rem', fontSize: '0.75rem', color: '#525252' }}>
                In <strong>auto</strong> mode, compression only triggers above this token count. Below it, passthrough is used.
              </p>
              <Slider
                id="compression-threshold"
                labelText={`Threshold: ${form.token_threshold.toLocaleString()} tokens`}
                min={512} max={32000} step={512}
                value={form.token_threshold}
                onChange={({ value }: { value: number }) => update({ token_threshold: value })}
              />
            </section>
          )}

          {/* ── Cache-aware gate ── */}
          {form.enabled && (
            <section style={{
              background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
              padding: '1.25rem', marginBottom: '1.5rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                <div style={{ flex: 1 }}>
                  <h3 style={{ margin: '0 0 0.25rem', fontSize: '0.875rem', fontWeight: 700 }}>Cache-aware gate</h3>
                  <p style={{ margin: '0 0 0.875rem', fontSize: '0.75rem', color: '#525252', lineHeight: 1.5 }}>
                    When <code>context_reuse_score</code> ≥ threshold, compression is skipped to preserve the KV-cache hit.
                    Compressing a warm prefix changes its token sequence, causing a cache miss on the provider side.
                  </p>
                  <Slider
                    id="cache-threshold"
                    labelText={`Skip threshold: ${form.cache_aware_threshold.toFixed(2)}`}
                    min={0} max={1} step={0.05}
                    value={form.cache_aware_threshold}
                    onChange={({ value }: { value: number }) => update({ cache_aware_threshold: value })}
                    style={{ width: 320 }}
                  />
                </div>
                <Toggle
                  id="cache-gate-enabled" size="sm" labelText="" hideLabel labelA="Off" labelB="On"
                  toggled={form.cache_aware_gate_enabled}
                  onToggle={(v: boolean) => update({ cache_aware_gate_enabled: v })}
                />
              </div>
            </section>
          )}

          {/* ── Savings metrics ── */}
          <section style={{
            background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
            padding: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          }}>
            <h3 style={{ margin: '0 0 1rem', fontSize: '0.875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
              Savings metrics — last 500 requests
            </h3>
            {metrics ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(155px, 1fr))', gap: '0.875rem', marginBottom: '1.5rem' }}>
                  <StatCard label="Overall savings" value={`${metrics.overall_savings_pct}%`} sub={`${metrics.total_savings_tokens.toLocaleString()} tokens saved`} accent="#24a148" />
                  <StatCard label="Compressed" value={metrics.compressed_count} sub={`of ${mTotal} sampled`} accent="#0f62fe" />
                  <StatCard label="Cache skips" value={metrics.cache_skip_count} sub="prefix preserved" accent="#6929c4" />
                  <StatCard label="Tokens before" value={metrics.total_tokens_before.toLocaleString()} accent="#8d8d8d" />
                  <StatCard label="Tokens after"  value={metrics.total_tokens_after.toLocaleString()}  accent="#8d8d8d" />
                </div>
                {Object.keys(profileDist).length > 0 && (
                  <div>
                    <p style={{ margin: '0 0 0.75rem', fontSize: '0.8125rem', fontWeight: 700, color: '#525252' }}>Profile distribution</p>
                    {Object.entries(profileDist).sort((a, b) => b[1] - a[1]).map(([p, c]) => (
                      <MiniBar key={p} label={`${PROFILE_META[p]?.badge ?? ''} ${PROFILE_META[p]?.label ?? p}`} value={c} total={mTotal} color={PROFILE_COLORS[p] ?? '#6f6f6f'} />
                    ))}
                  </div>
                )}
                {mTotal === 0 && (
                  <p style={{ fontSize: '0.875rem', color: '#525252', textAlign: 'center', padding: '1.5rem' }}>
                    No requests in buffer yet. Metrics appear once traffic flows through the gateway.
                  </p>
                )}
              </>
            ) : (
              <p style={{ fontSize: '0.875rem', color: '#525252' }}>Loading metrics…</p>
            )}
          </section>
        </>
      )}
    </div>
  )
}
