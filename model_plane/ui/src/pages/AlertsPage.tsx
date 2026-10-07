/**
 * Alerts page — CRUD for alert thresholds.
 */
import React, { useState } from 'react'
import {
  Button,
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  Modal,
  TextInput,
  NumberInput,
  Select,
  SelectItem,
  InlineNotification,
} from '@carbon/react'
import { Add, Edit, TrashCan } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, AlertConfig } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

const METRICS = [
  'cost_per_hour',
  'error_rate',
  'p99_latency_ms',
  'requests_per_minute',
  'tier_reasoning_pct',
]

const headers = [
  { key: 'alert_id', header: 'Alert ID' },
  { key: 'metric', header: 'Metric' },
  { key: 'threshold', header: 'Threshold' },
  { key: 'window_minutes', header: 'Window' },
  { key: 'notify_url', header: 'Notify URL' },
  { key: 'actions', header: '' },
]

function emptyAlert(): Partial<AlertConfig> {
  return { alert_id: '', metric: 'cost_per_hour', threshold: 0, window_minutes: 60, notify_url: null }
}

export default function AlertsPage() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({ queryKey: ['alerts'], queryFn: api.alerts.list })
  const [editOpen, setEditOpen] = useState(false)
  const [editMode, setEditMode] = useState<'create' | 'edit'>('create')
  const [form, setForm] = useState<Partial<AlertConfig>>(emptyAlert())
  const [notice, setNotice] = useState<string | null>(null)

  const createMut = useMutation({
    mutationFn: (a: Partial<AlertConfig>) => api.alerts.create(a as any),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alerts'] }); setEditOpen(false); setNotice('Alert created.') },
  })
  const updateMut = useMutation({
    mutationFn: ({ id, a }: { id: string; a: Partial<AlertConfig> }) => api.alerts.update(id, a),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alerts'] }); setEditOpen(false); setNotice('Alert updated.') },
  })
  const deleteMut = useMutation({
    mutationFn: (id: string) => api.alerts.delete(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alerts'] }); setNotice('Alert deleted.') },
  })

  function openEdit(a?: AlertConfig) {
    setForm(a ? { ...a } : emptyAlert())
    setEditMode(a ? 'edit' : 'create')
    setEditOpen(true)
  }

  function save() {
    if (editMode === 'create') createMut.mutate(form)
    else updateMut.mutate({ id: form.alert_id!, a: form })
  }

  const rows = (data?.alerts ?? []).map(a => ({ ...a, id: a.alert_id, actions: '' }))

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 0.25rem', color: '#161616' }}>Alerts</h2>
      <p style={{ fontSize: '0.875rem', color: '#525252', marginBottom: '1.25rem' }}>
        Configure alert thresholds. When a metric exceeds its threshold within the window, a webhook is fired.
      </p>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}
      {notice && <InlineNotification kind="success" title={notice} hideCloseButton={false} onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />}

      <div style={{ marginBottom: '0.75rem', display: 'flex', justifyContent: 'flex-end' }}>
        <Button renderIcon={Add} size="sm" onClick={() => openEdit()}>Add alert</Button>
      </div>

      {data && (
        <DataTable rows={rows} headers={headers}>
          {({ rows: rs, headers: hs, getTableProps, getHeaderProps, getRowProps }: any) => (
            <TableContainer>
              <Table {...getTableProps()} size="sm">
                <TableHead>
                  <TableRow>
                    {hs.map((h: any) => <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>)}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {rs.map((row: any) => {
                    const alrt = data.alerts.find(a => a.alert_id === row.id)!
                    return (
                      <TableRow {...getRowProps({ row })} key={row.id}>
                        {row.cells.map((cell: any) => (
                          <TableCell key={cell.id}>
                            {cell.info.header === 'window_minutes' ? `${cell.value} min`
                              : cell.info.header === 'actions' ? (
                                <div style={{ display: 'flex', gap: '0.25rem' }}>
                                  <Button kind="ghost" size="sm" iconDescription="Edit" renderIcon={Edit} hasIconOnly onClick={() => openEdit(alrt)} />
                                  <Button kind="danger--ghost" size="sm" iconDescription="Delete" renderIcon={TrashCan} hasIconOnly
                                    onClick={() => { if (confirm(`Delete alert ${alrt.alert_id}?`)) deleteMut.mutate(alrt.alert_id) }} />
                                </div>
                              ) : cell.value ?? '—'}
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

      <Modal open={editOpen} modalHeading={editMode === 'create' ? 'Add alert' : `Edit ${form.alert_id}`}
        primaryButtonText={editMode === 'create' ? 'Create' : 'Save'}
        secondaryButtonText="Cancel"
        onRequestClose={() => setEditOpen(false)}
        onRequestSubmit={save}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem 0' }}>
          <TextInput id="alrt-id" labelText="Alert ID *" value={form.alert_id ?? ''} disabled={editMode === 'edit'}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, alert_id: e.target.value }))} />
          <Select id="alrt-metric" labelText="Metric" value={form.metric ?? 'cost_per_hour'}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setForm(f => ({ ...f, metric: e.target.value }))}>
            {METRICS.map(m => <SelectItem key={m} value={m} text={m} />)}
          </Select>
          <NumberInput id="alrt-threshold" label="Threshold" value={form.threshold ?? 0}
            onChange={(_: any, state: any) => setForm(f => ({ ...f, threshold: Number(state?.value ?? 0) }))} />
          <NumberInput id="alrt-window" label="Window (minutes)" value={form.window_minutes ?? 60}
            onChange={(_: any, state: any) => setForm(f => ({ ...f, window_minutes: Number(state?.value ?? 60) }))} />
          <TextInput id="alrt-url" labelText="Notify URL (optional)" value={form.notify_url ?? ''}
            placeholder="https://…/webhook"
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, notify_url: e.target.value || null }))} />
        </div>
      </Modal>
    </div>
  )
}
