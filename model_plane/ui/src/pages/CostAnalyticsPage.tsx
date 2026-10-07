/**
 * Cost Analytics — stat cards, horizontal bar charts, task cards, raw table.
 */
import React, { useState } from 'react'
import {
  Select,
  SelectItem,
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  Tag,
} from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { api, CostRow } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

const WINDOWS = [
  { value: 1,   label: 'Last 1 hour' },
  { value: 24,  label: 'Last 24 hours' },
  { value: 168, label: 'Last 7 days' },
  { value: 720, label: 'Last 30 days' },
]

const rowHeaders = [
  { key: 'provider',      header: 'Provider' },
  { key: 'tier',          header: 'Tier' },
  { key: 'task_type',     header: 'Task type' },
  { key: 'requests',      header: 'Requests' },
  { key: 'cost_usd',      header: 'Cost (USD)' },
  { key: 'input_tokens',  header: 'Input' },
  { key: 'output_tokens', header: 'Output' },
]

const TIER_COLORS: Record<string, string> = {
  simple: '#24a148',
  medium: '#0f62fe',
  complex: '#8a3ffc',
  reasoning: '#ff832b',
}

const PROVIDER_COLORS = [
  '#0f62fe', '#8a3ffc', '#ff832b', '#24a148', '#da1e28',
  '#0072c3', '#6929c4', '#9f1853', '#198038',
]

// ── helpers ───────────────────────────────────────────────────────────────────

function groupSum(rows: CostRow[], key: keyof CostRow): [string, number][] {
  const acc: Record<string, number> = {}
  for (const r of rows) {
    const k = String(r[key])
    acc[k] = (acc[k] ?? 0) + r.cost_usd
  }
  return Object.entries(acc).sort(([, a], [, b]) => b - a)
}

function fmt(v: number): string {
  if (v === 0) return '$0'
  if (v < 0.0001) return `$${v.toExponential(2)}`
  if (v < 0.01) return `$${v.toFixed(5)}`
  if (v < 1) return `$${v.toFixed(4)}`
  return `$${v.toFixed(2)}`
}

function fmtNum(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return String(v)
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: string
}) {
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #e0e0e0',
      borderRadius: 8,
      padding: '1rem 1.25rem',
      borderTop: accent ? `3px solid ${accent}` : '1px solid #e0e0e0',
    }}>
      <p style={{ fontSize: '0.6875rem', color: '#525252', margin: '0 0 0.375rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
        {label}
      </p>
      <p style={{ fontSize: '1.625rem', fontWeight: 700, color: '#161616', margin: 0, lineHeight: 1.1 }}>
        {value}
      </p>
      {sub && <p style={{ fontSize: '0.75rem', color: '#6f6f6f', margin: '0.25rem 0 0' }}>{sub}</p>}
    </div>
  )
}

// ── Horizontal bar ─────────────────────────────────────────────────────────────

function HorizBar({ label, value, max, color = '#0f62fe', badge }: {
  label: string; value: number; max: number; color?: string; badge?: React.ReactNode
}) {
  const pct = max > 0 ? Math.max((value / max) * 100, 1.5) : 0
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '130px 1fr 72px', alignItems: 'center', gap: '0.625rem', marginBottom: '0.5rem' }}>
      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {badge ?? <span style={{ fontSize: '0.8125rem', color: '#161616' }}>{label}</span>}
      </div>
      <div style={{ height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <span style={{ fontSize: '0.8125rem', color: '#525252', textAlign: 'right' }}>
        {fmt(value)}
      </span>
    </div>
  )
}

// ── Section card wrapper ───────────────────────────────────────────────────────

function SectionCard({ title, children, style }: {
  title: string; children: React.ReactNode; style?: React.CSSProperties
}) {
  return (
    <div style={{
      background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
      overflow: 'hidden', ...style,
    }}>
      <div style={{ padding: '0.875rem 1.25rem', borderBottom: '1px solid #f4f4f4' }}>
        <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, margin: 0, color: '#161616', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {title}
        </h4>
      </div>
      <div style={{ padding: '1rem 1.25rem' }}>
        {children}
      </div>
    </div>
  )
}

// ── Empty state ────────────────────────────────────────────────────────────────

function EmptyChart() {
  return (
    <p style={{ fontSize: '0.8125rem', color: '#8d8d8d', fontStyle: 'italic', margin: 0 }}>
      No data for this window
    </p>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function CostAnalyticsPage() {
  const [windowHours, setWindowHours] = useState(24)

  const { data, isLoading, error } = useQuery({
    queryKey: ['cost', windowHours],
    queryFn: () => api.cost(windowHours),
    refetchInterval: 30_000,
  })

  const rows = data?.rows ?? []

  // Correct aggregation: reduce into a map, then sort
  const byProvider = groupSum(rows, 'provider')
  const byTier     = groupSum(rows, 'tier')

  const totalRequests = rows.reduce((s, r) => s + r.requests, 0)
  const totalTokensIn  = rows.reduce((s, r) => s + r.input_tokens, 0)
  const totalTokensOut = rows.reduce((s, r) => s + r.output_tokens, 0)

  const maxProviderCost = byProvider[0]?.[1] ?? 0.000001
  const maxTierCost     = byTier[0]?.[1] ?? 0.000001

  const tableRows = rows.map((r, i) => ({ ...r, id: String(i) }))

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: 1280, boxSizing: 'border-box' }}>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: '#161616' }}>
          Cost Analytics
        </h2>
        <div style={{ width: 160, flexShrink: 0 }}>
          <Select
            id="cost-window"
            labelText="" hideLabel
            value={String(windowHours)}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setWindowHours(Number(e.target.value))}
            size="sm"
          >
            {WINDOWS.map(w => <SelectItem key={w.value} value={String(w.value)} text={w.label} />)}
          </Select>
        </div>
      </div>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <>
          {/* ── Stat cards — responsive: 2 on small, up to 5 on wide ── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginBottom: '1.25rem' }}>
            <StatCard
              label="Total cost"
              value={fmt(data.total_cost_usd)}
              accent="#0f62fe"
            />
            <StatCard
              label="Requests"
              value={fmtNum(totalRequests)}
              sub={totalRequests > 0 ? `avg ${fmt(data.total_cost_usd / totalRequests)} / req` : undefined}
              accent="#8a3ffc"
            />
            <StatCard
              label="Input tokens"
              value={fmtNum(totalTokensIn)}
              accent="#24a148"
            />
            <StatCard
              label="Output tokens"
              value={fmtNum(totalTokensOut)}
              accent="#ff832b"
            />
            <StatCard
              label="Providers"
              value={String(byProvider.length)}
              sub={byProvider.length > 0 ? byProvider.map(([p]) => p).join(', ') : undefined}
              accent="#da1e28"
            />
          </div>

          {/* ── Two-col breakdown ── */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <SectionCard title="By provider">
              {byProvider.length === 0 ? <EmptyChart /> : byProvider.map(([prov, cost], i) => (
                <HorizBar
                  key={prov}
                  label={prov}
                  value={cost}
                  max={maxProviderCost}
                  color={PROVIDER_COLORS[i % PROVIDER_COLORS.length]}
                />
              ))}
            </SectionCard>

            <SectionCard title="By tier">
              {byTier.length === 0 ? <EmptyChart /> : byTier.map(([tier, cost]) => (
                <HorizBar
                  key={tier}
                  label={tier}
                  value={cost}
                  max={maxTierCost}
                  color={TIER_COLORS[tier] ?? '#6f6f6f'}
                  badge={<TierBadge tier={tier} />}
                />
              ))}
            </SectionCard>
          </div>

          {/* ── Task type cards ── */}
          {Object.keys(data.by_task_type).length > 0 && (
            <SectionCard title="By task type" style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.625rem' }}>
                {Object.entries(data.by_task_type)
                  .sort(([, a], [, b]) => b.cost_usd - a.cost_usd)
                  .map(([tt, v]) => (
                    <div key={tt} style={{
                      border: '1px solid #e0e0e0', borderRadius: 6, padding: '0.75rem',
                      background: '#f9f9f9',
                    }}>
                      <p style={{ fontSize: '0.6875rem', color: '#6f6f6f', margin: '0 0 0.25rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {tt}
                      </p>
                      <p style={{ fontSize: '1.125rem', fontWeight: 700, margin: 0, color: '#161616' }}>
                        {fmt(v.cost_usd)}
                      </p>
                      <p style={{ fontSize: '0.75rem', color: '#6f6f6f', margin: '0.125rem 0 0' }}>
                        {v.requests.toLocaleString()} req
                      </p>
                    </div>
                  ))
                }
              </div>
            </SectionCard>
          )}

          {/* ── Raw data table ── */}
          {tableRows.length > 0 && (
            <SectionCard title="Raw data">
              <DataTable rows={tableRows} headers={rowHeaders}>
                {({ rows: rs, headers: hs, getTableProps, getHeaderProps, getRowProps }: any) => (
                  <TableContainer style={{ margin: '-1rem -1.25rem -1rem' }}>
                    <Table {...getTableProps()} size="sm">
                      <TableHead>
                        <TableRow>
                          {hs.map((h: any) => (
                            <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {rs.map((row: any) => (
                          <TableRow {...getRowProps({ row })} key={row.id}>
                            {row.cells.map((cell: any) => (
                              <TableCell key={cell.id}>
                                {cell.info.header === 'tier' ? (
                                  <TierBadge tier={cell.value} />
                                ) : cell.info.header === 'cost_usd' ? (
                                  fmt(Number(cell.value ?? 0))
                                ) : cell.info.header === 'input_tokens' || cell.info.header === 'output_tokens' ? (
                                  fmtNum(Number(cell.value ?? 0))
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
            </SectionCard>
          )}

          {/* Empty state for zero data */}
          {tableRows.length === 0 && (
            <div style={{
              background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
              padding: '3rem 2rem', textAlign: 'center',
            }}>
              <p style={{ fontSize: '0.9375rem', color: '#525252', margin: '0 0 0.5rem', fontWeight: 600 }}>
                No requests in this window
              </p>
              <p style={{ fontSize: '0.8125rem', color: '#8d8d8d', margin: 0 }}>
                Cost data will appear here once requests are routed through Model Plane.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
