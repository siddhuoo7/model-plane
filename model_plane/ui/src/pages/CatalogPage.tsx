/**
 * Model Catalog page — full CRUD for deployments.
 * Two-step "Add deployment" wizard: pick provider → confirm/override.
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
  TableToolbar,
  TableToolbarContent,
  TableToolbarSearch,
  Modal,
  TextInput,
  Select,
  SelectItem,
  NumberInput,
  Tag,
  InlineNotification,
  Toggle,
} from '@carbon/react'
import { Add, Edit, TrashCan, WarningAlt, Power } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, DeploymentConfig } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

const PROVIDERS = ['openai', 'anthropic', 'watsonx', 'bedrock', 'azure', 'vertex_ai', 'cohere', 'mistral', 'vllm', 'local']
const TIERS = ['simple', 'medium', 'complex', 'reasoning']

const headers = [
  { key: 'name', header: 'Name' },
  { key: 'provider', header: 'Provider' },
  { key: 'tier', header: 'Tier' },
  { key: 'litellm_model', header: 'LiteLLM model' },
  { key: 'input_per_mtok_usd', header: 'Input $/1M' },
  { key: 'context_window', header: 'Context' },
  { key: 'healthy', header: 'Health' },
  { key: 'actions', header: '' },
]

type EditState = Partial<DeploymentConfig> & { name: string }

function emptyEdit(): EditState {
  return { name: '', litellm_model: '', provider: 'openai', tier: 'simple', max_tokens: 4096, context_window: 8192 }
}

export default function CatalogPage() {
  const qc = useQueryClient()
  const [showAll, setShowAll] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['catalog', showAll],
    queryFn: () => api.catalog.list(showAll),
    refetchInterval: 6_000,        // poll every 6 s to pick up startup validation
    refetchOnWindowFocus: true,
  })

  const [editOpen, setEditOpen] = useState(false)
  const [editMode, setEditMode] = useState<'create' | 'edit'>('create')
  const [form, setForm] = useState<EditState>(emptyEdit())
  const [notice, setNotice] = useState<string | null>(null)
  const [filterText, setFilterText] = useState('')

  // wizard step for "add" flow
  const [wizardStep, setWizardStep] = useState<1 | 2>(1)
  const [wizardProvider, setWizardProvider] = useState('openai')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)

  const createMut = useMutation({
    mutationFn: (d: Partial<DeploymentConfig>) => api.catalog.create(d),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['catalog'] }); setEditOpen(false); setNotice('Deployment created.') },
  })
  const updateMut = useMutation({
    mutationFn: ({ name, d }: { name: string; d: Partial<DeploymentConfig> }) => api.catalog.update(name, d),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['catalog'] }); setEditOpen(false); setNotice('Deployment updated.') },
  })
  const deleteMut = useMutation({
    mutationFn: (name: string) => api.catalog.delete(name),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['catalog'] }); setNotice('Deployment deleted.') },
  })
  const toggleMut = useMutation({
    mutationFn: ({ name, healthy }: { name: string; healthy: boolean }) =>
      api.catalog.setHealthy(name, healthy),
    onSuccess: (_d, { healthy }) => {
      qc.invalidateQueries({ queryKey: ['catalog'] })
      setNotice(healthy ? 'Deployment enabled.' : 'Deployment disabled.')
    },
  })

  function openCreate() {
    setForm(emptyEdit())
    setEditMode('create')
    setWizardStep(1)
    setAvailableModels([])
    setEditOpen(true)
  }

  function openEdit(dep: DeploymentConfig) {
    setForm({ ...dep })
    setEditMode('edit')
    setEditOpen(true)
  }

  async function loadModels() {
    setLoadingModels(true)
    try {
      const res = await api.providers.available(wizardProvider) as { models?: string[] }
      setAvailableModels(res?.models ?? [])
    } catch {
      setAvailableModels([])
    } finally {
      setLoadingModels(false)
      setWizardStep(2)
      setForm(f => ({ ...f, provider: wizardProvider }))
    }
  }

  function save() {
    if (editMode === 'create') {
      createMut.mutate(form)
    } else {
      updateMut.mutate({ name: form.name, d: form })
    }
  }

  const rows = (data?.deployments ?? [])
    .filter(d => !filterText || d.name.toLowerCase().includes(filterText.toLowerCase()))
    .map(d => ({ ...d, id: d.name, actions: '' }))

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 1rem', color: '#161616' }}>Model Catalog</h2>

      {notice && (
        <InlineNotification kind="success" title={notice} hideCloseButton={false} onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />
      )}

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <DataTable rows={rows} headers={headers}>
          {({ rows: rs, headers: hs, getTableProps, getHeaderProps, getRowProps, onInputChange }: any) => (
            <TableContainer>
              <TableToolbar>
                     <TableToolbarContent>
                       <TableToolbarSearch
                         onChange={(e: any) => {
                           const val = typeof e === 'string' ? e : e?.target?.value ?? ''
                           setFilterText(val)
                           onInputChange(e)
                         }}
                         placeholder="Search deployments…"
                       />
                       <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0 0.75rem' }}>
                         <Toggle
                           id="show-all-toggle"
                           labelText=""
                           hideLabel
                           size="sm"
                           labelA="Active only"
                           labelB="All configured"
                           toggled={showAll}
                           onToggle={(v: boolean) => setShowAll(v)}
                         />
                         {data && !showAll && data.uncredentialed_hidden > 0 && (
                           <span style={{ fontSize: '0.75rem', color: '#6f6f6f' }}>
                             +{data.uncredentialed_hidden} need credentials
                           </span>
                         )}
                       </div>
                       <Button renderIcon={Add} size="sm" onClick={openCreate}>Add deployment</Button>
                     </TableToolbarContent>
                   </TableToolbar>
              <Table {...getTableProps()} size="sm">
                <TableHead>
                  <TableRow>
                    {hs.map((h: any) => <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>)}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {rs.map((row: any) => {
                    const dep = data.deployments.find(d => d.name === row.id)
                    // dep can be undefined while DataTable re-renders after a delete —
                    // skip the row rather than crashing on dep.healthy
                    if (!dep) return null
                    const credOk = dep.credential_ok !== false  // undefined → assume ok
                    return (
                      <TableRow
                        {...getRowProps({ row })}
                        key={row.id}
                        style={!credOk ? { opacity: 0.6 } : undefined}
                      >
                        {row.cells.map((cell: any) => (
                          <TableCell key={cell.id}>
                            {cell.info.header === 'tier' ? <TierBadge tier={cell.value} />
                              : cell.info.header === 'healthy' ? (
                                !credOk ? (
                                  <Tag type="warm-gray" size="sm">
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                      <WarningAlt size={12} />
                                      needs credentials
                                    </span>
                                  </Tag>
                                ) : (
                                  <Tag type={cell.value ? 'green' : 'red'} size="sm">{cell.value ? 'ok' : 'down'}</Tag>
                                )
                              ) : cell.info.header === 'input_per_mtok_usd' ? `$${Number(cell.value ?? 0).toFixed(2)}`
                              : cell.info.header === 'context_window' ? `${((cell.value ?? 0) / 1000).toFixed(0)}k`
                              : cell.info.header === 'actions' ? (
                                <div style={{ display: 'flex', gap: '0.25rem' }}>
                                  <Button
                                    kind="ghost" size="sm"
                                    iconDescription={dep.healthy ? 'Disable' : 'Enable'}
                                    renderIcon={Power} hasIconOnly
                                    style={{ color: dep.healthy ? '#24a148' : '#6f6f6f' }}
                                    disabled={toggleMut.isPending && (toggleMut.variables as any)?.name === dep.name}
                                    onClick={() => toggleMut.mutate({ name: dep.name, healthy: !dep.healthy })}
                                  />
                                  <Button kind="ghost" size="sm" iconDescription="Edit" renderIcon={Edit} hasIconOnly onClick={() => openEdit(dep)} />
                                  <Button kind="danger--ghost" size="sm" iconDescription="Delete" renderIcon={TrashCan} hasIconOnly
                                    onClick={() => { if (confirm(`Delete ${dep.name}?`)) deleteMut.mutate(dep.name) }} />
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

      {/* Add / Edit modal */}
      <Modal
        open={editOpen}
        modalHeading={editMode === 'create'
          ? (wizardStep === 1 ? 'Add deployment — Step 1: Choose provider' : 'Add deployment — Step 2: Configure')
          : `Edit ${form.name}`}
        primaryButtonText={editMode === 'create' ? (wizardStep === 1 ? 'Next' : 'Create') : 'Save'}
        secondaryButtonText={editMode === 'create' && wizardStep === 2 ? 'Back' : 'Cancel'}
        onRequestClose={() => setEditOpen(false)}
        onRequestSubmit={() => {
          if (editMode === 'create' && wizardStep === 1) { loadModels(); return }
          if (editMode === 'create' && wizardStep === 2 && createMut.isPending) return
          if (editMode === 'edit' && updateMut.isPending) return
          if (editMode === 'create' && wizardStep === 2) { save(); return }
          save()
        }}
        onSecondarySubmit={() => {
          if (editMode === 'create' && wizardStep === 2) { setWizardStep(1); return }
          setEditOpen(false)
        }}
      >
        {editMode === 'create' && wizardStep === 1 ? (
          <div style={{ padding: '1rem 0' }}>
            <Select id="wizard-provider" labelText="Provider" value={wizardProvider}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setWizardProvider(e.target.value)}>
              {PROVIDERS.map(p => <SelectItem key={p} value={p} text={p} />)}
            </Select>
            {loadingModels && <p style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>Loading models…</p>}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem 0' }}>
            {availableModels.length > 0 && editMode === 'create' && (
              <Select id="form-litellm-from-list" labelText="Select model from prices.yaml"
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                  const v = e.target.value
                  setForm(f => ({ ...f, litellm_model: v, name: f.name || v.split('/').pop() || v }))
                }}>
                <SelectItem value="" text="— choose —" />
                {availableModels.map(m => <SelectItem key={m} value={m} text={m} />)}
              </Select>
            )}
            <TextInput id="form-name" labelText="Deployment name *" value={form.name}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, name: e.target.value }))} />
            <TextInput id="form-model" labelText="LiteLLM model string" value={form.litellm_model ?? ''}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, litellm_model: e.target.value }))} />
            <Select id="form-provider" labelText="Provider" value={form.provider ?? 'openai'}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setForm(f => ({ ...f, provider: e.target.value }))}>
              {PROVIDERS.map(p => <SelectItem key={p} value={p} text={p} />)}
            </Select>
            <Select id="form-tier" labelText="Tier" value={form.tier ?? 'simple'}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setForm(f => ({ ...f, tier: e.target.value }))}>
              {TIERS.map(t => <SelectItem key={t} value={t} text={t} />)}
            </Select>
            <NumberInput id="form-ctx" label="Context window" value={form.context_window ?? 8192}
              onChange={(_: any, state: any) => setForm(f => ({ ...f, context_window: Number(state?.value ?? 8192) }))} />
            <NumberInput id="form-maxtok" label="Max tokens" value={form.max_tokens ?? 4096}
              onChange={(_: any, state: any) => setForm(f => ({ ...f, max_tokens: Number(state?.value ?? 4096) }))} />
          </div>
        )}
      </Modal>
    </div>
  )
}
