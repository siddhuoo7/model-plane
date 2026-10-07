/**
 * Security & Governance page.
 *
 * Shows:
 *   1. Feature-flag status (security guard enabled / context-reuse enabled)
 *   2. Policy matrix — which DataSensitivity × ProviderTrust combinations are allowed
 *   3. Provider trust map — current trust level per provider (default + overrides)
 *   4. Sensitivity distribution — donut of recent requests by data class
 *   5. Blocked deployment audit — which deployments were blocked and how often
 *   6. Recent high-sensitivity events table
 */
import React, { useState } from 'react'
import {
  Tag,
  Toggle,
  InlineNotification,
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  Button,
  Select,
  SelectItem,
} from '@carbon/react'
import { Renew, Security, CheckmarkFilled, Save } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

// ── Palette ────────────────────────────────────────────────────────────────────

const SENSITIVITY_COLOR: Record<string, string> = {
  public:       '#24a148',
  internal:     '#0f62fe',
  confidential: '#f1c21b',
  restricted:   '#da1e28',
}

const TRUST_COLOR: Record<string, string> = {
  external_public:   '#da1e28',
  approved_external: '#f1c21b',
  private_cloud:     '#0f62fe',
  on_prem:           '#24a148',
}

const TRUST_LABEL: Record<string, string> = {
  external_public:   'External Public',
  approved_external: 'Approved External',
  private_cloud:     'Private Cloud',
  on_prem:           'On-Prem',
}

const TRUST_OPTIONS = [
  { value: 'external_public',   label: 'External Public' },
  { value: 'approved_external', label: 'Approved External' },
  { value: 'private_cloud',     label: 'Private Cloud' },
  { value: 'on_prem',           label: 'On-Prem' },
]

// PII rule explanations shown in the Configure tab
const PII_RULES = [
  {
    id: 'EMAIL',
    label: 'Email Address',
    description: 'Matches patterns like user@domain.com',
    example: 'john.doe@example.com',
    default: true,
  },
  {
    id: 'PHONE',
    label: 'Phone Number',
    description: 'Detects international and domestic phone formats',
    example: '+1 (555) 867-5309',
    default: true,
  },
  {
    id: 'SSN',
    label: 'Social Security Number',
    description: 'Matches NNN-NN-NNNN patterns',
    example: '123-45-6789',
    default: true,
  },
  {
    id: 'CREDIT_CARD',
    label: 'Credit Card Number',
    description: 'Detects 13–16 digit card numbers with optional separators',
    example: '4111 1111 1111 1111',
    default: true,
  },
  {
    id: 'IP_ADDRESS',
    label: 'IP Address',
    description: 'Detects IPv4 addresses (e.g. in logs or request data)',
    example: '192.168.1.1',
    default: false,
  },
  {
    id: 'AWS_KEY',
    label: 'AWS Access Key',
    description: 'Detects AWS access key IDs starting with AKIA…',
    example: 'AKIAIOSFODNN7EXAMPLE',
    default: true,
  },
  {
    id: 'API_KEY',
    label: 'API Key / Secret',
    description: 'Detects api_key=, secret_key=, bearer token patterns',
    example: 'api_key=abcdef1234567890',
    default: true,
  },
  {
    id: 'PRIVATE_KEY',
    label: 'Private Key Block',
    description: 'Detects PEM-encoded RSA/EC private key headers',
    example: '-----BEGIN RSA PRIVATE KEY-----',
    default: true,
  },
]

// ── Small helpers ──────────────────────────────────────────────────────────────

function SensTag({ value }: { value: string }) {
  const color = SENSITIVITY_COLOR[value] ?? '#6f6f6f'
  return (
    <span style={{
      display: 'inline-block', fontSize: '0.6875rem', fontWeight: 700,
      padding: '1px 8px', borderRadius: 10, letterSpacing: '0.04em',
      background: color + '18', color, border: `1px solid ${color}50`,
    }}>
      {value.toUpperCase()}
    </span>
  )
}

function TrustTag({ value }: { value: string }) {
  const color = TRUST_COLOR[value] ?? '#6f6f6f'
  return (
    <span style={{
      display: 'inline-block', fontSize: '0.6875rem', fontWeight: 600,
      padding: '1px 8px', borderRadius: 10,
      background: color + '12', color, border: `1px solid ${color}40`,
    }}>
      {TRUST_LABEL[value] ?? value}
    </span>
  )
}

function FlagRowInfo({
  label, enabled, onToggle, what, whenDisabled, envVar, envValue,
}: {
  label: string
  enabled: boolean
  onToggle?: (val: boolean) => void
  what: string
  whenDisabled: string
  envVar: string
  envValue: string
}) {
  const [open, setOpen] = React.useState(false)
  return (
    <div style={{ borderBottom: '1px solid #f0f0f0' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0.875rem 0',
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <p style={{ margin: 0, fontWeight: 700, fontSize: '0.875rem' }}>{label}</p>
            <button
              onClick={() => setOpen(v => !v)}
              title="What does this do?"
              style={{
                border: 'none', cursor: 'pointer',
                width: 18, height: 18, borderRadius: '50%',
                background: open ? '#0f62fe' : '#e8e8e8',
                color: open ? '#fff' : '#525252',
                fontSize: '0.6875rem', fontWeight: 700,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}
            >ⓘ</button>
          </div>
        </div>
        {onToggle ? (
          <Toggle
            id={`flag-toggle-${label.replace(/\s+/g, '-').toLowerCase()}`}
            labelA="Off"
            labelB="On"
            toggled={enabled}
            onToggle={onToggle}
            size="sm"
          />
        ) : (
          <span style={{
            fontSize: '0.75rem', fontWeight: 700, padding: '2px 10px', borderRadius: 10,
            background: enabled ? '#defbe6' : '#fff1f1',
            color: enabled ? '#198038' : '#da1e28',
            border: `1px solid ${enabled ? '#a7f0ba' : '#ffb3b8'}`,
            flexShrink: 0,
          }}>
            {enabled ? '● Enabled' : '○ Disabled'}
          </span>
        )}
      </div>

      {open && (
        <div style={{
          margin: '0 0 0.875rem',
          background: '#f4f4f4', borderRadius: 8,
          padding: '1rem 1.125rem',
          borderLeft: '3px solid #0f62fe',
          fontSize: '0.8125rem', lineHeight: 1.6,
        }}>
          <p style={{ margin: '0 0 0.5rem', fontWeight: 700, color: '#161616' }}>What it does</p>
          <p style={{ margin: '0 0 0.875rem', color: '#525252' }}>{what}</p>

          <p style={{ margin: '0 0 0.375rem', fontWeight: 700, color: '#161616' }}>When disabled</p>
          <p style={{ margin: '0 0 0.875rem', color: '#da1e28' }}>{whenDisabled}</p>

          <p style={{ margin: '0 0 0.375rem', fontWeight: 700, color: '#161616' }}>
            Enable / disable via environment variable
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <code style={{
              background: '#e8e8e8', padding: '2px 8px', borderRadius: 4,
              fontSize: '0.8125rem', fontFamily: 'monospace', color: '#161616',
            }}>
              {envVar}={envValue}
            </code>
            <span style={{ fontSize: '0.75rem', color: '#525252' }}>
              Set in <code>.env</code> or as an OS environment variable — takes effect on next restart.
              Current session: already applied via settings object.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function formatTime(ts: number | null | undefined) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return isNaN(d.getTime()) ? '—' : d.toLocaleTimeString()
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function SecurityPage() {
  const qc = useQueryClient()
  const [activeTab, setActiveTab] = useState<'matrix' | 'configure' | 'events'>('matrix')

  // Editable trust overrides (local copy of routing config provider_trust_overrides)
  const [trustEdits, setTrustEdits] = useState<Record<string, string>>({})
  const [trustDirty, setTrustDirty] = useState(false)
  const [trustSaveMsg, setTrustSaveMsg] = useState<string | null>(null)

  // PII rule toggles — load saved disabled rules from the server
  const { data: piiRulesData } = useQuery({
    queryKey: ['pii-rules'],
    queryFn: () => api.security.piiRules ? api.security.piiRules() : fetch('/admin/api/security/pii-rules', {
      headers: {
        'Content-Type': 'application/json',
        ...(api.auth ? { 'Authorization': `Bearer ${localStorage.getItem('mp_token') || ''}` } : {})
      },
    }).then(r => r.json()) as Promise<{ disabled: string[] }>,
  })

  // Optimistic local disabled set — updated immediately on toggle, synced from server once loaded.
  const [localDisabled, setLocalDisabled] = React.useState<Set<string> | null>(null)
  React.useEffect(() => {
    if (piiRulesData && localDisabled === null) {
      setLocalDisabled(new Set(piiRulesData.disabled))
    }
  }, [piiRulesData])

  const disabledSet: Set<string> = localDisabled ?? new Set(piiRulesData?.disabled ?? [])
  const piiEnabled: Record<string, boolean> = Object.fromEntries(
    PII_RULES.map(r => [r.id, !disabledSet.has(r.id)])
  )

  function togglePiiRule(ruleId: string, enabled: boolean) {
    // Update local state immediately (no snap-back)
    const nextDisabled = new Set(disabledSet)
    if (enabled) nextDisabled.delete(ruleId)
    else nextDisabled.add(ruleId)
    setLocalDisabled(nextDisabled)

    fetch('/admin/api/security/pii-rules', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('mp_token') || ''}`
      },
      body: JSON.stringify({ disabled: Array.from(nextDisabled) }),
    }).then(() => qc.invalidateQueries({ queryKey: ['pii-rules'] }))
      .catch(() => {
        // Roll back on failure
        setLocalDisabled(new Set(piiRulesData?.disabled ?? []))
      })
  }

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['security-summary'],
    queryFn: api.security.summary,
    refetchInterval: 10_000,
  })

  const { data: flags } = useQuery({
    queryKey: ['security-flags'],
    queryFn: api.security.flags,
  })

  // Optimistic local copy — updated immediately on toggle so the UI never snaps back.
  const [securityEnabled, setSecurityEnabled] = React.useState<boolean | null>(null)
  // Sync from server once loaded (but don't override a local edit in flight)
  React.useEffect(() => {
    if (flags && securityEnabled === null) {
      setSecurityEnabled(flags.security_routing_enabled)
    }
  }, [flags])

  const saveFlagMut = useMutation({
    mutationFn: async (patch: Record<string, unknown>) => {
      const res = await fetch('/admin/api/security/flags', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('mp_token') || ''}`
        },
        body: JSON.stringify(patch),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },
    onSuccess: (_data, variables) => {
      // Merge the saved value back into the query cache so a refetch doesn't
      // clobber the displayed value with a stale server response.
      qc.setQueryData(['security-flags'], (old: any) =>
        old ? { ...old, ...variables } : old
      )
      qc.invalidateQueries({ queryKey: ['security-flags'] })
    },
    onError: () => {
      // Roll back to server value on failure
      if (flags) setSecurityEnabled(flags.security_routing_enabled)
    },
  })

  function toggleFlag(key: string, val: boolean) {
    if (key === 'security_routing_enabled') setSecurityEnabled(val)
    saveFlagMut.mutate({ [key]: val })
  }

  // Load current routing config to populate trust edits
  const { data: routingCfg } = useQuery({
    queryKey: ['routing-config'],
    queryFn: api.routing.getConfig,
  })

  // Populate trustEdits when routing config loads
  React.useEffect(() => {
    if (routingCfg && !trustDirty) {
      const overrides = (routingCfg as any).provider_trust_overrides ?? {}
      setTrustEdits(overrides)
    }
  }, [routingCfg])

  const saveTrustMut = useMutation({
    mutationFn: async (overrides: Record<string, string>) => {
      const current = routingCfg ?? {}
      return api.routing.setConfig({ ...current, provider_trust_overrides: overrides } as any)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['routing-config'] })
      qc.invalidateQueries({ queryKey: ['security-summary'] })
      setTrustDirty(false)
      setTrustSaveMsg('Trust overrides saved.')
      setTimeout(() => setTrustSaveMsg(null), 3000)
    },
  })

  function setTrustOverride(provider: string, trust: string) {
    setTrustEdits(prev => ({ ...prev, [provider]: trust }))
    setTrustDirty(true)
  }

  const TRUST_COLS = ['external_public', 'approved_external', 'private_cloud', 'on_prem']

  // Sensitivity distribution total
  const sensTotal = data
    ? Object.values(data.sensitivity_counts as Record<string, number>).reduce((a, b) => a + b, 0)
    : 0

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: 960 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '0.25rem' }}>
            <Security size={20} style={{ color: '#0f62fe' }} />
            <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: '#161616' }}>Security &amp; Governance</h2>
          </div>
          <p style={{ margin: 0, fontSize: '0.875rem', color: '#525252' }}>
            Provider trust policy matrix, data-sensitivity routing rules, and real-time security events.
          </p>
        </div>
        <Button kind="ghost" size="sm" renderIcon={Renew} onClick={() => refetch()} disabled={isFetching}>
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </Button>
      </div>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && flags && (
        <>
          {/* Security Guard */}
          <div style={{ borderBottom: '1px solid #e0e0e0', paddingBottom: '1.5rem', marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '0.8125rem', fontWeight: 700, margin: '0 0 0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#525252' }}>
              Security Guard
            </h3>
            <p style={{ margin: '0 0 0.75rem', fontSize: '0.875rem', color: '#161616', lineHeight: 1.6 }}>
              Classifies each request's data sensitivity (PII, secrets, keywords) and removes providers
              that are not permitted to handle that sensitivity class. This is a hard eligibility filter
              — blocked providers are never evaluated for cost or cache scoring.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <p style={{ margin: 0, fontSize: '0.875rem', color: '#525252' }}>
                Enable or disable the security hard-filter across all requests.
              </p>
              <Toggle
                id="security-guard-toggle"
                labelA="Off"
                labelB="On"
                toggled={securityEnabled ?? flags.security_routing_enabled}
                onToggle={(v: boolean) => toggleFlag('security_routing_enabled', v)}
                size="sm"
              />
            </div>
          </div>

          {/* Stats row */}
          {sensTotal > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.875rem', marginBottom: '1.5rem' }}>
              {(['public', 'internal', 'confidential', 'restricted'] as const).map(s => {
                const count = (data.sensitivity_counts as Record<string, number>)[s] ?? 0
                const pct = sensTotal > 0 ? Math.round(count / sensTotal * 100) : 0
                return (
                  <div key={s} style={{
                    background: '#fff', border: `1px solid ${SENSITIVITY_COLOR[s]}30`,
                    borderLeft: `4px solid ${SENSITIVITY_COLOR[s]}`,
                    borderRadius: 8, padding: '1rem',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                  }}>
                    <p style={{ margin: '0 0 0.25rem', fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: SENSITIVITY_COLOR[s] }}>
                      {s}
                    </p>
                    <p style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: '#161616' }}>{count}</p>
                    <p style={{ margin: 0, fontSize: '0.75rem', color: '#525252' }}>{pct}% of sampled</p>
                  </div>
                )
              })}
            </div>
          )}

          {/* Tabs: Matrix | Configure | Events */}
          <div style={{ display: 'flex', gap: 0, marginBottom: '1.25rem', borderBottom: '1px solid #e0e0e0' }}>
            {([
              ['matrix', 'Policy Matrix & Trust Map'],
              ['configure', '⚙ Configure'],
              ['events', `Security Events${data.recent_events.length > 0 ? ` (${data.recent_events.length})` : ''}`],
            ] as const).map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  padding: '0.625rem 1.25rem',
                  fontSize: '0.875rem', fontWeight: activeTab === tab ? 700 : 400,
                  color: activeTab === tab ? '#0f62fe' : '#525252',
                  borderBottom: activeTab === tab ? '2px solid #0f62fe' : '2px solid transparent',
                  marginBottom: -1,
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {activeTab === 'matrix' && (
            <>
              {/* Policy Matrix */}
              <section style={{
                background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                padding: '1.25rem', marginBottom: '1.5rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              }}>
                <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: '0 0 0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                  Data Sensitivity → Provider Trust Policy Matrix
                </h3>
                <p style={{ fontSize: '0.8125rem', color: '#525252', marginBottom: '1rem' }}>
                  A request can only be routed to a provider if the intersection of its data class and provider trust level is <strong style={{ color: '#24a148' }}>Allowed</strong>.
                  This is a hard filter — it cannot be overridden by cache score or cost.
                </p>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '0.5rem 0.75rem', background: '#f4f4f4', borderBottom: '2px solid #e0e0e0', fontSize: '0.75rem', fontWeight: 700, color: '#525252' }}>
                          Data class
                        </th>
                        {TRUST_COLS.map(tc => (
                          <th key={tc} style={{ textAlign: 'center', padding: '0.5rem 0.75rem', background: '#f4f4f4', borderBottom: '2px solid #e0e0e0' }}>
                            <TrustTag value={tc} />
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(data.policy_matrix as any[]).map((row: any, i: number) => (
                        <tr key={row.sensitivity} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                          <td style={{ padding: '0.625rem 0.75rem', borderBottom: '1px solid #f0f0f0' }}>
                            <SensTag value={row.sensitivity} />
                          </td>
                          {TRUST_COLS.map(tc => (
                            <td key={tc} style={{ padding: '0.625rem 0.75rem', textAlign: 'center', borderBottom: '1px solid #f0f0f0' }}>
                              {row[tc]
                                ? <span style={{ color: '#24a148', fontWeight: 700, fontSize: '1rem' }}>✓</span>
                                : <span style={{ color: '#da1e28', fontWeight: 700, fontSize: '1rem' }}>✕</span>
                              }
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Provider Trust Map */}
              <section style={{
                background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                padding: '1.25rem', marginBottom: '1.5rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                    Provider Trust Levels
                  </h3>
                  {Object.keys(data.trust_overrides as Record<string, string>).length > 0 && (
                    <Tag type="blue" size="sm">
                      {Object.keys(data.trust_overrides as Record<string, string>).length} active override(s)
                    </Tag>
                  )}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.625rem' }}>
                  {Object.entries(data.provider_trust_map as Record<string, string>).sort().map(([prov, trust]) => {
                    const isOverride = (data.trust_overrides as Record<string, string>)[prov] !== undefined
                    return (
                      <div key={prov} style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '0.5rem 0.75rem',
                        background: isOverride ? '#edf5ff' : '#f4f4f4',
                        border: `1px solid ${isOverride ? '#0f62fe30' : '#e0e0e0'}`,
                        borderRadius: 6,
                      }}>
                        <span style={{ fontSize: '0.8125rem', fontWeight: 600, fontFamily: 'monospace' }}>{prov}</span>
                        <TrustTag value={trust} />
                      </div>
                    )
                  })}
                </div>
              </section>

              {/* Blocked deployment counts */}
              {Object.keys(data.blocked_counts as Record<string, number>).length > 0 && (
                <section style={{
                  background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                  padding: '1.25rem',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: '0 0 0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                    Blocked Deployment Audit (last 500 requests)
                  </h3>
                  <div style={{ display: 'flex', gap: '0.625rem', flexWrap: 'wrap' }}>
                    {Object.entries(data.blocked_counts as Record<string, number>).sort((a, b) => b[1] - a[1]).map(([dep, count]) => (
                      <div key={dep} style={{
                        background: '#fff1f1', border: '1px solid #ffb3b8',
                        borderRadius: 6, padding: '0.5rem 0.875rem',
                      }}>
                        <span style={{ fontSize: '0.8125rem', fontWeight: 600 }}>{dep}</span>
                        <span style={{ marginLeft: '0.5rem', fontSize: '0.8125rem', color: '#da1e28', fontWeight: 700 }}>{count}×</span>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}

          {activeTab === 'configure' && (
            <div>
              {trustSaveMsg && (
                <InlineNotification kind="success" title={trustSaveMsg} hideCloseButton={false} onClose={() => setTrustSaveMsg(null)} style={{ marginBottom: '1rem' }} />
              )}

              {/* Trust level overrides */}
              <section style={{
                background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                padding: '1.25rem', marginBottom: '1.5rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div>
                    <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                      Provider Trust Level Overrides
                    </h3>
                    <p style={{ fontSize: '0.75rem', color: '#525252', margin: '0.25rem 0 0' }}>
                      Override default provider trust levels. Changes take effect immediately.
                    </p>
                  </div>
                  <Button
                    kind="primary" size="sm" renderIcon={Save}
                    disabled={!trustDirty || saveTrustMut.isPending}
                    onClick={() => saveTrustMut.mutate(trustEdits)}
                  >
                    {saveTrustMut.isPending ? 'Saving…' : 'Save changes'}
                  </Button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.75rem' }}>
                  {Object.entries(data.provider_trust_map as Record<string, string>).sort().map(([prov, defaultTrust]) => {
                    const currentTrust = trustEdits[prov] ?? defaultTrust
                    const isOverridden = trustEdits[prov] !== undefined && trustEdits[prov] !== defaultTrust
                    return (
                      <div key={prov} style={{
                        background: isOverridden ? '#edf5ff' : '#f4f4f4',
                        border: `1px solid ${isOverridden ? '#0f62fe40' : '#e0e0e0'}`,
                        borderRadius: 8, padding: '0.75rem',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                          <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '0.875rem' }}>{prov}</span>
                          {isOverridden && <Tag type="blue" size="sm">overridden</Tag>}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#525252', marginBottom: '0.375rem' }}>
                          Default: <TrustTag value={defaultTrust} />
                        </div>
                        <Select
                          id={`trust-${prov}`}
                          labelText=""
                          hideLabel
                          size="sm"
                          value={currentTrust}
                          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setTrustOverride(prov, e.target.value)}
                        >
                          {TRUST_OPTIONS.map(o => (
                            <SelectItem key={o.value} value={o.value} text={o.label} />
                          ))}
                        </Select>
                      </div>
                    )
                  })}
                </div>
              </section>

              {/* PII rule toggles */}
              <section style={{
                background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                padding: '1.25rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              }}>
                <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: '0 0 0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                  PII Detection Rules
                </h3>
                <p style={{ fontSize: '0.75rem', color: '#525252', marginBottom: '1rem' }}>
                  Disable individual detection patterns. Requests matching only disabled patterns will not be elevated to RESTRICTED sensitivity.
                  Changes are persisted per user and take effect immediately.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.125rem' }}>
                  {PII_RULES.map(rule => (
                    <div key={rule.id} style={{
                      display: 'flex', alignItems: 'flex-start', gap: '1rem',
                      padding: '0.75rem', borderBottom: '1px solid #f0f0f0',
                    }}>
                      <Toggle
                        id={`pii-${rule.id}`}
                        size="sm"
                        labelText=""
                        hideLabel
                        toggled={piiEnabled[rule.id] ?? rule.default}
                        onToggle={(v: boolean) => togglePiiRule(rule.id, v)}
                      />
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontWeight: 700, fontSize: '0.875rem' }}>{rule.label}</span>
                          <span style={{
                            fontSize: '0.65rem', fontFamily: 'monospace', padding: '0 6px', borderRadius: 20,
                            background: '#f4f4f4', border: '1px solid #e0e0e0', color: '#525252',
                          }}>{rule.id}</span>
                        </div>
                        <p style={{ margin: '0.125rem 0 0', fontSize: '0.75rem', color: '#525252' }}>{rule.description}</p>
                        <p style={{ margin: '0.125rem 0 0', fontSize: '0.7rem', fontFamily: 'monospace', color: '#8d8d8d' }}>e.g. {rule.example}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          {activeTab === 'events' && (
            <section style={{
              background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
              padding: '1.25rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            }}>
              <h3 style={{ fontSize: '0.875rem', fontWeight: 700, margin: '0 0 0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#525252' }}>
                Recent High-Sensitivity Events
              </h3>
              <p style={{ fontSize: '0.75rem', color: '#525252', marginBottom: '1rem' }}>
                Requests at INTERNAL sensitivity or above, or any request where one or more providers were blocked.
                Sampled from the last {data.total_requests_sampled} buffered requests. Auto-refreshes every 10 s.
              </p>
              {data.recent_events.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2.5rem', color: '#525252' }}>
                  <CheckmarkFilled size={32} style={{ marginBottom: '0.5rem', color: '#24a148' }} />
                  <p style={{ margin: 0, fontWeight: 600 }}>No high-sensitivity events in recent buffer</p>
                  <p style={{ margin: '0.25rem 0 0', fontSize: '0.8125rem' }}>All sampled requests were classified as PUBLIC or INTERNAL with no provider blocks.</p>
                </div>
              ) : (
                <DataTable
                  rows={(data.recent_events as any[]).map((e: any, i: number) => ({ id: String(i), ...e }))}
                  headers={[
                    { key: 'timestamp', header: 'Time' },
                    { key: 'sensitivity', header: 'Sensitivity' },
                    { key: 'reason', header: 'Detected Trigger / Reason' },
                    { key: 'deployment', header: 'Routed to' },
                    { key: 'provider', header: 'Provider' },
                    { key: 'blocked_count', header: 'Blocked Models' },
                  ]}
                >
                  {({ rows, headers, getTableProps, getHeaderProps, getRowProps }: any) => (
                    <TableContainer>
                      <Table {...getTableProps()} size="sm">
                        <TableHead>
                          <TableRow>
                            {headers.map((h: any) => (
                              <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {rows.map((row: any) => {
                            const origItem = (data.recent_events as any[])[parseInt(row.id, 10)] || {}
                            return (
                              <TableRow {...getRowProps({ row })} key={row.id}>
                                {row.cells.map((cell: any) => (
                                  <TableCell key={cell.id}>
                                    {cell.info.header === 'sensitivity' ? (
                                      <SensTag value={cell.value ?? '—'} />
                                    ) : cell.info.header === 'reason' ? (
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                        <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#161616' }}>
                                          {origItem.reason || cell.value || 'Policy check'}
                                        </span>
                                        {origItem.pii_types?.length > 0 && (
                                          <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                                            {origItem.pii_types.map((p: string) => (
                                              <Tag key={p} type="magenta" size="sm">{p}</Tag>
                                            ))}
                                          </div>
                                        )}
                                        {origItem.secret_types?.length > 0 && (
                                          <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                                            {origItem.secret_types.map((s: string) => (
                                              <Tag key={s} type="red" size="sm">{s}</Tag>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    ) : cell.info.header === 'blocked_count' ? (
                                      origItem.blocked?.length ? (
                                        <Tag type="red" size="sm" title={origItem.blocked.join(', ')}>
                                          {origItem.blocked.length} model(s) blocked
                                        </Tag>
                                      ) : (
                                        <Tag type="green" size="sm">0 blocked</Tag>
                                      )
                                    ) : cell.info.header === 'timestamp' ? (
                                      formatTime(cell.value)
                                    ) : (
                                      cell.value ?? '—'
                                    )}
                                  </TableCell>
                                ))}
                              </TableRow>
                            )
                          })}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )}
                </DataTable>
              )}
            </section>
          )}
        </>
      )}
    </div>
  )
}
