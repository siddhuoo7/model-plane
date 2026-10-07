/**
 * Request Explorer — filterable, paginated table of routing decisions.
 * Score breakdown + security metadata available via expandable row drawer.
 */
import React, { useState } from 'react'
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
} from '@carbon/react'
import { Search, Reset, Security } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import { api, RequestRecord } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

const PAGE_SIZES = [25, 50, 100]

const SENSITIVITY_COLOR: Record<string, string> = {
  public: '#24a148', internal: '#0f62fe', confidential: '#f1c21b', restricted: '#da1e28',
}

function SensChip({ value }: { value: string | undefined }) {
  if (!value || value === 'disabled') return <span style={{ color: '#a8a8a8', fontSize: '0.75rem' }}>—</span>
  const color = SENSITIVITY_COLOR[value] ?? '#6f6f6f'
  return (
    <span style={{
      display: 'inline-block', fontSize: '0.6875rem', fontWeight: 700,
      padding: '1px 7px', borderRadius: 10,
      background: color + '15', color, border: `1px solid ${color}40`,
    }}>
      {value}
    </span>
  )
}

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

function formatTime(ts: number | undefined | null) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return isNaN(d.getTime()) ? '—' : d.toLocaleTimeString()
}

export default function RequestExplorerPage() {
  const [provider, setProvider] = useState('')
  const [tier, setTier] = useState('')
  const [taskType, setTaskType] = useState('')
  const [sensitivity, setSensitivity] = useState('')
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

  // Filter out skeleton/test records that have no meaningful routing data
  // Also apply client-side sensitivity filter
  const allRecords: RequestRecord[] = (data?.records ?? []).filter(
    (r: any) => (r.deployment || r.provider || r.tier)
      && (!sensitivity || r.security_sensitivity === sensitivity)
  )
  const totalItems = allRecords.length
  const start = (page - 1) * pageSize
  const pageRecords = allRecords.slice(start, start + pageSize)

  function reset() {
    setProvider('')
    setTier('')
    setTaskType('')
    setSensitivity('')
    setPage(1)
  }

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 1rem', color: '#161616' }}>Request Explorer</h2>

      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <TextInput
          id="filter-provider"
          labelText="Provider"
          placeholder="e.g. openai"
          value={provider}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setProvider(e.target.value); setPage(1) }}
          size="sm"
          style={{ maxWidth: 160 }}
        />
        <Select
          id="filter-tier"
          labelText="Tier"
          value={tier}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => { setTier(e.target.value); setPage(1) }}
          size="sm"
          style={{ maxWidth: 160 }}
        >
          <SelectItem value="" text="All tiers" />
          {['simple', 'medium', 'complex', 'reasoning'].map(t => (
            <SelectItem key={t} value={t} text={t} />
          ))}
        </Select>
        <TextInput
          id="filter-task"
          labelText="Task type"
          placeholder="e.g. code_generation"
          value={taskType}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setTaskType(e.target.value); setPage(1) }}
          size="sm"
          style={{ maxWidth: 200 }}
        />
        <Select
          id="filter-sensitivity"
          labelText="Sensitivity"
          value={sensitivity}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => { setSensitivity(e.target.value); setPage(1) }}
          size="sm"
          style={{ maxWidth: 160 }}
        >
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
          <p style={{ fontSize: '0.75rem', color: '#525252', marginBottom: '0.5rem' }}>
            Showing {pageRecords.length} of {totalItems} (buffer: {data.total_buffered})
          </p>
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
                                       title={`Context reuse score: ${Number(cell.value).toFixed(4)}\n${
                                         Number(cell.value) >= 0.8 ? 'Strong cache affinity — session routed to warm model'
                                         : Number(cell.value) >= 0.5 ? 'Moderate cache affinity — reuse bonus applied'
                                         : Number(cell.value) > 0 ? 'Weak cache signal — base scoring dominated'
                                         : 'No session or cold start'
                                       }`}
                                       style={{
                                         display: 'inline-flex', alignItems: 'center', gap: 4,
                                         fontSize: '0.75rem', fontFamily: 'monospace',
                                         cursor: 'help',
                                       }}
                                     >
                                       <span style={{
                                         display: 'inline-block', width: 28, height: 6, borderRadius: 3,
                                         background: `linear-gradient(90deg, #6929c4 ${Math.round(Number(cell.value) * 100)}%, #393939 ${Math.round(Number(cell.value) * 100)}%)`,
                                       }} />
                                       {Number(cell.value).toFixed(2)}
                                     </span>
                                   ) : <span style={{ color: '#8d8d8d', fontSize: '0.75rem' }}>—</span>
                                 ) : cell.info.header === 'timestamp' ? (
                                   formatTime(cell.value)
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
                            <div style={{ padding: '0.75rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                              {/* Score breakdown */}
                              {rec?.score_breakdown && (
                                <div style={{ minWidth: 220 }}>
                                  <h5 style={{ marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 700 }}>Score breakdown</h5>
                                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                    {Object.entries(rec.score_breakdown).map(([k, v]) => (
                                      <span key={k} style={{ fontSize: '0.75rem', background: '#f4f4f4', padding: '2px 6px', borderRadius: 3 }}>
                                        {k}: {typeof v === 'number' ? v.toFixed(3) : String(v)}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {/* Security info */}
                              <div style={{ minWidth: 240 }}>
                                <h5 style={{ marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                                  <Security size={12} /> Security
                                </h5>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                  <div style={{ fontSize: '0.75rem' }}>
                                    <span style={{ color: '#525252' }}>Sensitivity: </span>
                                    <SensChip value={rec?.security_sensitivity} />
                                  </div>
                                  {rec?.context_reuse_score != null && (
                                    <div style={{ fontSize: '0.75rem' }}>
                                      <span style={{ color: '#525252' }}>Context reuse score: </span>
                                      <span style={{ fontWeight: 600 }}>{rec.context_reuse_score.toFixed(3)}</span>
                                    </div>
                                  )}
                                  {rec?.security_blocked && rec.security_blocked.length > 0 ? (
                                    <div style={{ fontSize: '0.75rem' }}>
                                      <span style={{ color: '#525252' }}>Blocked: </span>
                                      <span style={{ color: '#da1e28', fontWeight: 600 }}>{rec.security_blocked.join(', ')}</span>
                                    </div>
                                  ) : (
                                    <div style={{ fontSize: '0.75rem', color: '#24a148' }}>No providers blocked</div>
                                  )}
                                </div>
                              </div>
                            </div>
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
              setPage(p)
              setPageSize(ps)
            }}
          />
        </>
      )}
    </div>
  )
}
