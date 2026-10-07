/**
 * Dashboard page — GET /admin/api/traffic + /health + /catalog
 * Auto-refreshes every 10 s.
 */
import React from 'react'
import {
  Grid,
  Column,
  Tile,
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  Tag,
  Button,
  SkeletonPlaceholder,
} from '@carbon/react'
import {
  Activity,
  Money,
  ServerProxy,
  MachineLearning,
  CircleFilled,
  WatsonHealthStatusPartialFail,
  Renew,
  ArrowUpRight,
  ArrowDownRight,
} from '@carbon/icons-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, TrafficResponse, HealthResponse } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { ErrorState } from '../components/States'

// ── Colour palette ─────────────────────────────────────────────────────────────

const C = {
  blue:    '#0f62fe',
  green:   '#24a148',
  purple:  '#8a3ffc',
  teal:    '#00b0a0',
  yellow:  '#f1c21b',
  red:     '#da1e28',
  gray:    '#6f6f6f',
  border:  '#e0e0e0',
  surface: '#f4f4f4',
  white:   '#ffffff',
  text:    '#161616',
  muted:   '#525252',
}

const TIER_COLORS: Record<string, string> = {
  simple:    C.green,
  medium:    C.blue,
  complex:   C.teal,
  reasoning: C.purple,
}

const PROVIDER_ACCENT: Record<string, string> = {
  openai:    '#10a37f',
  anthropic: '#d4a853',
  watsonx:   '#0f62fe',
  bedrock:   '#ff9900',
  azure:     '#0078d4',
  local:     '#6f6f6f',
  vllm:      '#8a3ffc',
}

// ── Skeleton card ──────────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div style={{
      background: C.white, border: `1px solid ${C.border}`, borderRadius: 8,
      padding: '1.25rem', height: 110,
    }}>
      <SkeletonPlaceholder style={{ height: 12, width: '50%', marginBottom: 12 }} />
      <SkeletonPlaceholder style={{ height: 32, width: '65%', marginBottom: 8 }} />
      <SkeletonPlaceholder style={{ height: 10, width: '40%' }} />
    </div>
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, accent, icon: Icon, delta,
}: {
  label: string; value: string | number; sub?: string; accent?: string
  icon?: React.ElementType; delta?: { val: number; label: string }
}) {
  const color = accent ?? C.blue
  return (
    <div style={{
      background: C.white,
      border: `1px solid ${C.border}`,
      borderTop: `3px solid ${color}`,
      borderRadius: '0 0 8px 8px',
      padding: '1.125rem 1.25rem',
      height: '100%',
      boxSizing: 'border-box',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
        <p style={{
          fontSize: '0.6875rem', color: C.gray, margin: 0,
          textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600,
        }}>
          {label}
        </p>
        {Icon && (
          <span style={{
            background: `${color}15`, borderRadius: '50%',
            width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            <Icon size={18} style={{ color }} />
          </span>
        )}
      </div>
      <p style={{ fontSize: '2rem', fontWeight: 700, color: C.text, margin: 0, lineHeight: 1.1 }}>{value}</p>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.375rem' }}>
        {sub && <p style={{ fontSize: '0.75rem', color: C.muted, margin: 0 }}>{sub}</p>}
        {delta && (
          <span style={{
            display: 'flex', alignItems: 'center', gap: 2,
            fontSize: '0.6875rem', fontWeight: 600,
            color: delta.val >= 0 ? C.green : C.red,
          }}>
            {delta.val >= 0
              ? <ArrowUpRight size={12} />
              : <ArrowDownRight size={12} />}
            {delta.label}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <section style={{ marginBottom: '1.875rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <h3 style={{
          fontSize: '0.875rem', fontWeight: 600, color: C.text, margin: 0,
          textTransform: 'uppercase', letterSpacing: '0.06em',
        }}>
          {title}
        </h3>
        {action}
      </div>
      {children}
    </section>
  )
}

// ── Tier bar ──────────────────────────────────────────────────────────────────

function TierBar({ data }: { data: TrafficResponse['by_tier'] }) {
  const tiers = ['simple', 'medium', 'complex', 'reasoning']
  const total = Object.values(data).reduce((s, v) => s + v.requests, 0)
  if (total === 0) return (
    <div style={{
      background: C.surface, borderRadius: 8, padding: '1.25rem',
      color: C.gray, fontSize: '0.875rem', textAlign: 'center',
    }}>
      No requests yet — send a call to{' '}
      <code style={{ background: '#e0e0e0', padding: '1px 6px', borderRadius: 3, fontSize: '0.8125rem' }}>/v1/chat/completions</code>
      {' '}to see live tier distribution.
    </div>
  )
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 8, padding: '1.25rem' }}>
      {/* Segmented bar */}
      <div style={{ display: 'flex', height: 12, borderRadius: 8, overflow: 'hidden', gap: 2, marginBottom: '1rem' }}>
        {tiers.map(t => {
          const pct = ((data[t]?.requests ?? 0) / total) * 100
          return pct > 0 ? (
            <div
              key={t}
              title={`${t}: ${pct.toFixed(1)}%`}
              style={{
                width: `${pct}%`, background: TIER_COLORS[t], borderRadius: 3,
                transition: 'width 0.4s ease',
              }}
            />
          ) : null
        })}
      </div>
      {/* Legend pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
        {tiers.filter(t => data[t]?.requests).map(t => {
          const pct = ((data[t]?.requests ?? 0) / total) * 100
          return (
            <div key={t} style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              background: `${TIER_COLORS[t]}12`,
              border: `1px solid ${TIER_COLORS[t]}40`,
              borderRadius: 20, padding: '5px 12px',
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: TIER_COLORS[t], flexShrink: 0 }} />
              <TierBadge tier={t} />
              <span style={{ fontSize: '0.8125rem', color: C.muted, fontWeight: 500 }}>
                {pct.toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.75rem', color: C.gray }}>
                {data[t].requests.toLocaleString()} req · ${data[t].cost_usd.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Provider card ─────────────────────────────────────────────────────────────

function ProviderCard({ name, data }: {
  name: string
  data: { requests: number; cost_usd: number; input_tokens: number; output_tokens: number }
}) {
  const accent = PROVIDER_ACCENT[name] ?? C.blue
  const totalTok = data.input_tokens + data.output_tokens
  return (
    <div style={{
      background: C.white,
      border: `1px solid ${C.border}`,
      borderLeft: `4px solid ${accent}`,
      borderRadius: 8,
      padding: '1rem 1.25rem',
      flex: '1 1 200px',
      minWidth: 0,
      transition: 'box-shadow 0.2s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%', background: accent, flexShrink: 0,
        }} />
        <p style={{ fontWeight: 700, fontSize: '0.9375rem', margin: 0, color: C.text }}>{name}</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.375rem 1rem' }}>
        <div>
          <p style={{ fontSize: '0.6875rem', color: C.gray, margin: '0 0 2px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Requests</p>
          <p style={{ fontSize: '1.125rem', fontWeight: 700, color: C.text, margin: 0 }}>{data.requests.toLocaleString()}</p>
        </div>
        <div>
          <p style={{ fontSize: '0.6875rem', color: C.gray, margin: '0 0 2px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Cost</p>
          <p style={{ fontSize: '1.125rem', fontWeight: 700, color: C.text, margin: 0 }}>${data.cost_usd.toFixed(4)}</p>
        </div>
      </div>
      <p style={{ fontSize: '0.75rem', color: C.gray, margin: '0.625rem 0 0' }}>
        {(totalTok / 1000).toFixed(1)}k tokens · {(data.input_tokens / 1000).toFixed(1)}k in / {(data.output_tokens / 1000).toFixed(1)}k out
      </p>
    </div>
  )
}

// ── System status strip ───────────────────────────────────────────────────────

function StatusStrip({ health }: { health: HealthResponse }) {
  const allHealthy = health.healthy_deployments === health.total_deployments
  const statusColor = allHealthy ? C.green : C.yellow
  const bgColor     = allHealthy ? '#defbe6' : '#fef3cd'
  const borderColor = allHealthy ? '#a7f0ba' : '#f1c21b'
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap',
      background: bgColor,
      border: `1px solid ${borderColor}`,
      borderLeft: `4px solid ${statusColor}`,
      borderRadius: '0 8px 8px 0',
      padding: '0.625rem 1rem',
      marginBottom: '1.5rem',
    }}>
      {allHealthy
        ? <CircleFilled size={16} style={{ color: C.green, flexShrink: 0 }} />
        : <WatsonHealthStatusPartialFail size={16} style={{ color: C.yellow, flexShrink: 0 }} />}
      <span style={{ fontSize: '0.875rem', fontWeight: 600, color: C.text }}>
        {allHealthy
          ? `All ${health.total_deployments} deployments healthy`
          : `${health.healthy_deployments} / ${health.total_deployments} deployments healthy`}
      </span>
      <span style={{ width: 1, height: 16, background: borderColor, flexShrink: 0 }} />
      <span style={{ fontSize: '0.8125rem', color: C.muted }}>
        Mode: <strong style={{ color: C.text }}>{health.routing_mode}</strong>
      </span>
      <span style={{ fontSize: '0.75rem', color: C.gray, marginLeft: 'auto' }}>
        Buffer {health.buffer_size.toLocaleString()} / {health.buffer_capacity.toLocaleString()}
      </span>
    </div>
  )
}

// ── Deployment pool table ─────────────────────────────────────────────────────

const depHeaders = [
  { key: 'name',              header: 'Model' },
  { key: 'tier',              header: 'Tier' },
  { key: 'provider',          header: 'Provider' },
  { key: 'input_per_mtok_usd', header: 'Input $/1M' },
  { key: 'context_window',    header: 'Context' },
  { key: 'healthy',           header: 'Health' },
]

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const qc      = useQueryClient()
  const traffic = useQuery({ queryKey: ['traffic'], queryFn: () => api.traffic(24), refetchInterval: 10_000 })
  const health  = useQuery({ queryKey: ['health'],  queryFn: api.health,            refetchInterval: 10_000 })
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: () => api.catalog.list(), refetchInterval: 30_000 })

  const t = traffic.data
  const h = health.data

  const byProviderEntries = t ? Object.entries(t.by_provider).filter(([, v]) => v.requests > 0) : []

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ['traffic'] })
    qc.invalidateQueries({ queryKey: ['health'] })
    qc.invalidateQueries({ queryKey: ['catalog'] })
  }

  return (
    <div style={{ padding: '1.75rem 2rem', maxWidth: 1400 }}>
      {/* Page heading */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: C.text, margin: 0 }}>Dashboard</h2>
          <p style={{ fontSize: '0.8125rem', color: C.gray, margin: '0.25rem 0 0' }}>
            Live system overview · auto-refreshes every 10 s
          </p>
        </div>
        <Button
          kind="ghost" size="sm"
          renderIcon={Renew}
          iconDescription="Refresh"
          onClick={refreshAll}
          style={{ marginTop: 4 }}
        >
          Refresh
        </Button>
      </div>

      {(traffic.error || health.error) && (
        <ErrorState message={String(traffic.error ?? health.error)} />
      )}

      {/* Status strip */}
      {/* {h ? (
        <StatusStrip health={h} />
      ) : (
        <div style={{ height: 48, background: C.surface, borderRadius: 8, marginBottom: '1.5rem' }} />
      )} */}

      {/* Stat cards */}
      <Section title="24-hour summary">
        {(traffic.isLoading || health.isLoading) ? (
          <Grid narrow>
            {[0,1,2,3].map(i => (
              <Column key={i} sm={2} md={2} lg={4}><SkeletonCard /></Column>
            ))}
          </Grid>
        ) : (t && h) ? (
          <Grid narrow>
            <Column sm={2} md={2} lg={4}>
              <StatCard
                label="Total Requests"
                value={t.total_requests.toLocaleString()}
                icon={Activity}
                accent={C.blue}
                sub="last 24 hours"
              />
            </Column>
            <Column sm={2} md={2} lg={4}>
              <StatCard
                label="Total Cost"
                value={`$${t.total_cost_usd.toFixed(4)}`}
                icon={Money}
                accent={C.green}
                sub="last 24 hours"
              />
            </Column>
            <Column sm={2} md={2} lg={4}>
              <StatCard
                label="Tokens Processed"
                value={((t.total_input_tokens + t.total_output_tokens) / 1_000).toFixed(1) + 'k'}
                icon={MachineLearning}
                accent={C.purple}
                sub={`${(t.total_input_tokens/1_000).toFixed(1)}k in · ${(t.total_output_tokens/1_000).toFixed(1)}k out`}
              />
            </Column>
            <Column sm={2} md={2} lg={4}>
              <StatCard
                label="Healthy Deployments"
                value={`${h.healthy_deployments} / ${h.total_deployments}`}
                icon={ServerProxy}
                accent={h.healthy_deployments === h.total_deployments ? C.green : C.yellow}
                sub={`Mode: ${h.routing_mode}`}
              />
            </Column>
          </Grid>
        ) : null}
      </Section>

      {/* Tier distribution */}
      {t && (
        <Section title="Tier distribution">
          <TierBar data={t.by_tier} />
        </Section>
      )}

      {/* Provider split */}
      {t && byProviderEntries.length > 0 && (
        <Section title="Provider split">
          <div style={{ display: 'flex', gap: '0.875rem', flexWrap: 'wrap' }}>
            {byProviderEntries.map(([p, v]) => (
              <ProviderCard key={p} name={p} data={v} />
            ))}
          </div>
        </Section>
      )}

      {/* Deployment pool */}
      <Section title="Deployment pool">
        {catalog.isLoading && (
          <div style={{ background: C.surface, borderRadius: 8, padding: '1rem', color: C.gray, fontSize: '0.875rem' }}>
            Loading deployments…
          </div>
        )}
        {catalog.error && <ErrorState message={String(catalog.error)} />}
        {catalog.data && (
          <div style={{
            background: C.white, border: `1px solid ${C.border}`,
            borderRadius: 8, overflow: 'hidden',
          }}>
            <DataTable rows={catalog.data.deployments.map((d: any) => ({ ...d, id: d.name }))} headers={depHeaders}>
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
                      {rows.map((row: any) => (
                        <TableRow {...getRowProps({ row })} key={row.id}>
                          {row.cells.map((cell: any) => (
                            <TableCell key={cell.id}>
                              {cell.info.header === 'tier' ? (
                                <TierBadge tier={cell.value} />
                              ) : cell.info.header === 'healthy' ? (
                                <Tag type={cell.value ? 'green' : 'red'} size="sm">
                                  {cell.value ? '● healthy' : '● degraded'}
                                </Tag>
                              ) : cell.info.header === 'input_per_mtok_usd' ? (
                                <span style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>
                                  ${Number(cell.value ?? 0).toFixed(2)}
                                </span>
                              ) : cell.info.header === 'context_window' ? (
                                cell.value > 0 ? `${Math.round((cell.value ?? 0) / 1000)}k` : '—'
                              ) : (
                                cell.value ?? '—'
                              )}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </DataTable>
          </div>
        )}
      </Section>
    </div>
  )
}
