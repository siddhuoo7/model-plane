/**
 * Tenant Policies page — CRUD for per-tenant routing rules.
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
  Select,
  SelectItem,
  InlineNotification,
  Tag,
} from '@carbon/react'
import { Add, Edit, TrashCan } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, Tenant } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

const TIERS = ['', 'simple', 'medium', 'complex', 'reasoning']

const headers = [
  { key: 'tenant_id', header: 'Tenant ID' },
  { key: 'min_tier', header: 'Min tier' },
  { key: 'max_tier', header: 'Max tier' },
  { key: 'preferred_deployments', header: 'Preferred' },
  { key: 'blocked_deployments', header: 'Blocked' },
  { key: 'actions', header: '' },
]

function emptyTenant(): Partial<Tenant> {
  return { tenant_id: '', min_tier: null, max_tier: null, preferred_deployments: [], blocked_deployments: [] }
}

export default function TenantsPage() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({ queryKey: ['tenants'], queryFn: api.tenants.list })
  const [editOpen, setEditOpen] = useState(false)
  const [editMode, setEditMode] = useState<'create' | 'edit'>('create')
  const [form, setForm] = useState<Partial<Tenant>>(emptyTenant())
  const [notice, setNotice] = useState<string | null>(null)

  const createMut = useMutation({
    mutationFn: (t: Partial<Tenant>) => api.tenants.create(t as any),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tenants'] }); setEditOpen(false); setNotice('Tenant created.') },
  })
  const updateMut = useMutation({
    mutationFn: ({ id, t }: { id: string; t: Partial<Tenant> }) => api.tenants.update(id, t),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tenants'] }); setEditOpen(false); setNotice('Tenant updated.') },
  })
  const deleteMut = useMutation({
    mutationFn: (id: string) => api.tenants.delete(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tenants'] }); setNotice('Tenant deleted.') },
  })

  function openEdit(t?: Tenant) {
    setForm(t ? { ...t } : emptyTenant())
    setEditMode(t ? 'edit' : 'create')
    setEditOpen(true)
  }

  function save() {
    if (editMode === 'create') createMut.mutate(form)
    else updateMut.mutate({ id: form.tenant_id!, t: form })
  }

  const rows = (data?.tenants ?? []).map(t => ({
    ...t,
    id: t.tenant_id,
    preferred_deployments: t.preferred_deployments.join(', ') || '—',
    blocked_deployments: t.blocked_deployments.join(', ') || '—',
    actions: '',
  }))

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 1rem', color: '#161616' }}>Tenant Policies</h2>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}
      {notice && <InlineNotification kind="success" title={notice} hideCloseButton={false} onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />}

      <div style={{ marginBottom: '0.75rem', display: 'flex', justifyContent: 'flex-end' }}>
        <Button renderIcon={Add} size="sm" onClick={() => openEdit()}>Add tenant</Button>
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
                    const ten = data.tenants.find(t => t.tenant_id === row.id)!
                    return (
                      <TableRow {...getRowProps({ row })} key={row.id}>
                        {row.cells.map((cell: any) => (
                          <TableCell key={cell.id}>
                            {cell.info.header === 'min_tier' || cell.info.header === 'max_tier'
                              ? (cell.value ? <TierBadge tier={cell.value} /> : <Tag type="gray" size="sm">none</Tag>)
                              : cell.info.header === 'actions' ? (
                                <div style={{ display: 'flex', gap: '0.25rem' }}>
                                  <Button kind="ghost" size="sm" iconDescription="Edit" renderIcon={Edit} hasIconOnly onClick={() => openEdit(ten)} />
                                  <Button kind="danger--ghost" size="sm" iconDescription="Delete" renderIcon={TrashCan} hasIconOnly
                                    onClick={() => { if (confirm(`Delete tenant ${ten.tenant_id}?`)) deleteMut.mutate(ten.tenant_id) }} />
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

      <Modal open={editOpen} modalHeading={editMode === 'create' ? 'Add tenant' : `Edit ${form.tenant_id}`}
        primaryButtonText={editMode === 'create' ? 'Create' : 'Save'}
        secondaryButtonText="Cancel"
        onRequestClose={() => setEditOpen(false)}
        onRequestSubmit={save}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem 0' }}>
          <TextInput id="ten-id" labelText="Tenant ID *" value={form.tenant_id ?? ''} disabled={editMode === 'edit'}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, tenant_id: e.target.value }))} />
          <Select id="ten-min" labelText="Min tier" value={form.min_tier ?? ''}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setForm(f => ({ ...f, min_tier: e.target.value || null }))}>
            {TIERS.map(t => <SelectItem key={t} value={t} text={t || '— none —'} />)}
          </Select>
          <Select id="ten-max" labelText="Max tier" value={form.max_tier ?? ''}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setForm(f => ({ ...f, max_tier: e.target.value || null }))}>
            {TIERS.map(t => <SelectItem key={t} value={t} text={t || '— none —'} />)}
          </Select>
          <TextInput id="ten-pref" labelText="Preferred deployments (comma-separated)"
            value={(form.preferred_deployments ?? []).join(', ')}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({
              ...f, preferred_deployments: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
            }))} />
          <TextInput id="ten-blocked" labelText="Blocked deployments (comma-separated)"
            value={(form.blocked_deployments ?? []).join(', ')}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({
              ...f, blocked_deployments: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
            }))} />
        </div>
      </Modal>
    </div>
  )
}
