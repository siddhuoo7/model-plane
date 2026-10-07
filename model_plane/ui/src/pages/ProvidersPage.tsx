/**
 * Provider Management page.
 *
 * Credentials are saved to SQLite (primary) with .env as fallback.
 * No restart needed — changes activate immediately in the running process.
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
  Tag,
  InlineNotification,
  InlineLoading,
} from '@carbon/react'
import {
  Checkmark,
  Warning,
  Edit,
  TrashCan,
  CheckmarkFilled,
  WarningFilled,
} from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

// ── Field definitions ─────────────────────────────────────────────────────────

interface FieldMeta {
  label: string
  type?: 'password' | 'text' | 'url'
  placeholder?: string
  required?: boolean
  hint?: string
}

type ProviderFields = Record<string, FieldMeta>

const PROVIDER_FIELDS: Record<string, ProviderFields> = {
  openai: {
    OPENAI_API_KEY: { label: 'API key', type: 'password', required: true, placeholder: 'sk-...' },
  },
  anthropic: {
    ANTHROPIC_API_KEY: { label: 'API key', type: 'password', required: true, placeholder: 'sk-ant-...' },
  },
  watsonx: {
    WATSONX_APIKEY:     { label: 'API key', type: 'password', required: true },
    WATSONX_PROJECT_ID: { label: 'Project ID', required: true, placeholder: '00000000-0000-0000-0000-000000000000' },
    WATSONX_URL:        { label: 'Endpoint URL', type: 'url', placeholder: 'https://us-south.ml.cloud.ibm.com' },
  },
  bedrock: {
    AWS_BEARER_TOKEN_BEDROCK: {
      label: 'Bedrock API key', type: 'password', required: true,
      hint: 'Your AWS Bedrock API key (AWS Console → Bedrock → API keys). No IAM access key needed.',
    },
    AWS_REGION_NAME: {
      label: 'AWS region', required: true, placeholder: 'us-east-1',
      hint: 'The region where your Bedrock models are deployed.',
    },
  },
  azure: {
    AZURE_API_KEY:     { label: 'API key', type: 'password', required: true },
    AZURE_API_BASE:    { label: 'Endpoint URL', type: 'url', required: true, placeholder: 'https://<resource>.openai.azure.com' },
    AZURE_API_VERSION: { label: 'API version', placeholder: '2024-02-01' },
  },
  vertex_ai: {
    VERTEXAI_PROJECT:  { label: 'GCP project ID', required: true },
    VERTEXAI_LOCATION: { label: 'Location', placeholder: 'us-central1' },
  },
  cohere: {
    COHERE_API_KEY: { label: 'API key', type: 'password', required: true },
  },
  mistral: {
    MISTRAL_API_KEY: { label: 'API key', type: 'password', required: true },
  },
  local: {
    LOCAL_VLLM_API_BASE: { label: 'API base URL', type: 'url', required: true, placeholder: 'http://localhost:8000/v1' },
    LOCAL_VLLM_API_KEY:  { label: 'API key (optional)', type: 'password' },
  },
}

const PROVIDER_DISPLAY: Record<string, string> = {
  openai: 'OpenAI', anthropic: 'Anthropic', watsonx: 'watsonx',
  bedrock: 'AWS Bedrock', azure: 'Azure OpenAI', vertex_ai: 'Vertex AI',
  cohere: 'Cohere', mistral: 'Mistral', local: 'Local / vLLM',
}

const headers = [
  { key: 'name',             header: 'Provider' },
  { key: 'configured',       header: 'Status' },
  { key: 'available_models', header: 'Catalog models' },
  { key: 'actions',          header: '' },
]

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ProvidersPage() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({ queryKey: ['providers'], queryFn: api.providers.list })
  const { data: credData } = useQuery({ queryKey: ['provider-creds'], queryFn: api.providers.credStatus, retry: false })

  const [formProvider, setFormProvider] = useState<string | null>(null)
  const [formValues, setFormValues]     = useState<Record<string, string>>({})
  const [loadingForm, setLoadingForm]   = useState(false)
  const [notice, setNotice] = useState<{ kind: 'success' | 'error' | 'info'; msg: string } | null>(null)

  // Per-modal test state
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string; missing?: string[] } | null>(null)
  const [testPending, setTestPending] = useState(false)

  // Unset confirm state
  const [unsetConfirm, setUnsetConfirm] = useState(false)

  async function runTest(name: string, currentFormValues?: Record<string, string>) {
    setTestResult(null)
    setTestPending(true)
    try {
      // Strip masked placeholder values (the "••••••••" the server sends back for
      // sensitive fields that are already set). Sending them would overwrite the real
      // credential in os.environ with the placeholder string, causing the test to fail.
      const MASK = '••••••••'
      const cleanValues = currentFormValues
        ? Object.fromEntries(
            Object.entries(currentFormValues).filter(([, v]) => v && v !== MASK)
          )
        : undefined

      const res = await api.providers.test(name, cleanValues) as {
        healthy: boolean
        error?: string
        live_model_count?: number
        available?: string[]
        missing?: string[]
        yaml_models?: string[]
      }
      if (res.healthy) {
        const count = res.live_model_count ?? 0
        const avail = res.available?.length ?? 0
        const miss = res.missing?.length ?? 0
        let msg = `Connection OK — ${count} model${count !== 1 ? 's' : ''} from provider`
        if (avail > 0 || miss > 0) {
          msg += ` · ${avail} in catalog active`
          if (miss > 0) msg += `, ${miss} not found on provider`
        }
        setTestResult({ ok: true, msg, missing: res.missing ?? [] })
      } else {
        setTestResult({ ok: false, msg: res.error ?? 'Test failed.', missing: [] })
      }
      qc.invalidateQueries({ queryKey: ['providers'] })
      qc.invalidateQueries({ queryKey: ['catalog'] })
    } catch (e) {
      setTestResult({ ok: false, msg: String(e), missing: [] })
    } finally {
      setTestPending(false)
    }
  }

  const saveMut = useMutation({
    mutationFn: ({ provider, creds }: { provider: string; creds: Record<string, string> }) =>
      api.providers.saveCreds(provider, creds),
    onSuccess: (_, { provider }) => {
      qc.invalidateQueries({ queryKey: ['providers'] })
      qc.invalidateQueries({ queryKey: ['provider-creds'] })
      closeModal()
      setNotice({ kind: 'success', msg: `Credentials for "${PROVIDER_DISPLAY[provider] ?? provider}" saved and active.` })
    },
    onError: (e) => setNotice({ kind: 'error', msg: `Save failed: ${String(e)}` }),
  })

  const clearMut = useMutation({
    mutationFn: (provider: string) => api.providers.clearCreds(provider),
    onSuccess: (_, provider) => {
      qc.invalidateQueries({ queryKey: ['providers'] })
      qc.invalidateQueries({ queryKey: ['provider-creds'] })
      closeModal()
      setNotice({ kind: 'info', msg: `Credentials for "${PROVIDER_DISPLAY[provider] ?? provider}" removed.` })
    },
  })

  function closeModal() {
    setFormProvider(null)
    setFormValues({})
    setTestResult(null)
    setUnsetConfirm(false)
  }

  async function openForm(name: string) {
    setFormProvider(name)
    setTestResult(null)
    setUnsetConfirm(false)
    setLoadingForm(true)
    // Fetch existing values to pre-fill the form
    try {
      const res = await api.providers.credValues(name)
      setFormValues(res.values ?? {})
    } catch {
      setFormValues({})
    } finally {
      setLoadingForm(false)
    }
  }

  const rows = (data?.providers ?? []).map((p: any) => ({
    ...p,
    id: p.name ?? p.provider,
    name: p.name ?? p.provider,
  }))

  const fields = formProvider ? (PROVIDER_FIELDS[formProvider] ?? {}) : {}

  function isProviderConfigured(name: string): boolean {
    if (data?.providers) {
      const p = data.providers.find((x: any) => (x.name ?? x.provider) === name)
      if (p?.configured) return true
    }
    const saved = credData?.providers?.[name]
    if (saved) return Object.values(saved).some(Boolean)
    return false
  }

  const configured = formProvider ? isProviderConfigured(formProvider) : false

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.25rem' }}>Provider Management</h2>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {notice && (
        <InlineNotification
          kind={notice.kind} title={notice.msg}
          hideCloseButton={false} onClose={() => setNotice(null)}
          style={{ marginBottom: '1rem' }}
        />
      )}

      {data && (
        <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
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
                      const cfg = isProviderConfigured(row.id)
                      return (
                        <TableRow {...getRowProps({ row })} key={row.id}>
                          {row.cells.map((cell: any) => (
                            <TableCell key={cell.id}>
                              {cell.info.header === 'name' ? (
                                <strong style={{ fontSize: '0.875rem' }}>
                                  {PROVIDER_DISPLAY[cell.value] ?? cell.value}
                                </strong>
                              ) : cell.info.header === 'configured' ? (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                  {cfg ? (
                                    <><Checkmark size={16} style={{ color: '#24a148' }} /><Tag type="green" size="sm">configured</Tag></>
                                  ) : (
                                    <><Warning size={16} style={{ color: '#f1c21b' }} /><Tag type="cool-gray" size="sm">not set</Tag></>
                                  )}
                                </div>
                              ) : cell.info.header === 'available_models' ? (
                                cell.value > 0
                                  ? <Tag type="blue" size="sm">{cell.value} model{cell.value !== 1 ? 's' : ''}</Tag>
                                  : <span style={{ color: '#6f6f6f', fontSize: '0.8125rem' }}>none in catalog</span>
                              ) : cell.info.header === 'actions' ? (
                                <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                  <Button kind="ghost" size="sm" renderIcon={Edit} onClick={() => openForm(row.id)}>
                                    {cfg ? 'Edit' : 'Configure'}
                                  </Button>
                                  {cfg && (
                                    <Button
                                      kind="ghost" size="sm" renderIcon={TrashCan}
                                      hasIconOnly iconDescription="Unset credentials"
                                      tooltipPosition="left"
                                      onClick={() => {
                                        if (confirm(`Remove saved credentials for ${PROVIDER_DISPLAY[row.id] ?? row.id}?`)) {
                                          clearMut.mutate(row.id)
                                        }
                                      }}
                                      disabled={clearMut.isPending && clearMut.variables === row.id}
                                      style={{ color: '#da1e28' }}
                                    />
                                  )}
                                </div>
                              ) : null}
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
        </div>
      )}

      {/* ── Credentials modal ── */}
      {formProvider && (
        <Modal
          open
          modalHeading={`Configure ${PROVIDER_DISPLAY[formProvider] ?? formProvider}`}
          primaryButtonText={saveMut.isPending ? 'Saving…' : 'Save & activate'}
          secondaryButtonText="Cancel"
          onRequestClose={closeModal}
          onRequestSubmit={() => saveMut.mutate({ provider: formProvider, creds: formValues })}
          primaryButtonDisabled={saveMut.isPending || loadingForm}
        >
          <div style={{ padding: '0.75rem 0 0' }}>
            {loadingForm ? (
              <InlineLoading description="Loading saved values…" style={{ marginBottom: '1rem' }} />
            ) : (
              Object.entries(fields).map(([envKey, meta]) => (
                <div key={envKey} style={{ marginBottom: '1.25rem' }}>
                  <TextInput
                    id={`cred-${envKey}`}
                    labelText={
                      <span>
                        {meta.label}
                        {meta.required && <span style={{ color: '#da1e28' }}> *</span>}
                        {/* Show a "set" badge next to the label if this field has a value */}
                        {formValues[envKey] && (
                          <Tag type="green" size="sm" style={{ marginLeft: 6 }}>set</Tag>
                        )}
                      </span>
                    }
                    placeholder={formValues[envKey] ? '' : (meta.placeholder ?? `Enter ${meta.label.toLowerCase()}`)}
                    type={(meta.type as any) ?? 'text'}
                    value={formValues[envKey] ?? ''}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setFormValues(v => ({ ...v, [envKey]: e.target.value }))
                    }
                    helperText={meta.hint}
                  />
                </div>
              ))
            )}

            {/* ── Test connection ── */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.75rem',
              paddingTop: '0.75rem', marginTop: '0.25rem',
              borderTop: '1px solid #e0e0e0', flexWrap: 'wrap',
            }}>
              <Button kind="tertiary" size="sm" onClick={() => runTest(formProvider, formValues)} disabled={testPending}>
                {testPending ? 'Testing…' : 'Test connection'}
              </Button>
              {testPending && <InlineLoading description="Connecting…" style={{ width: 'auto' }} />}
              {!testPending && testResult && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  <span style={{
                    display: 'flex', alignItems: 'center', gap: '0.375rem',
                    fontSize: '0.8125rem',
                    color: testResult.ok ? '#24a148' : '#da1e28',
                  }}>
                    {testResult.ok ? <CheckmarkFilled size={16} /> : <WarningFilled size={16} />}
                    {testResult.msg}
                  </span>
                  {testResult.ok && testResult.missing && testResult.missing.length > 0 && (
                    <div style={{ fontSize: '0.75rem', color: '#8a6914', marginLeft: 20 }}>
                      Not found on provider: {testResult.missing.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Unset / delete credentials ── */}
            {configured && (
              <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #e0e0e0' }}>
                {!unsetConfirm ? (
                  <Button
                    kind="danger--ghost" size="sm" renderIcon={TrashCan}
                    onClick={() => setUnsetConfirm(true)}
                  >
                    Unset credentials
                  </Button>
                ) : (
                  <div style={{
                    background: '#fff1f1', border: '1px solid #fa4d56',
                    borderRadius: 6, padding: '0.75rem 1rem',
                  }}>
                    <p style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#da1e28', margin: '0 0 0.625rem' }}>
                      Remove saved credentials for {PROVIDER_DISPLAY[formProvider] ?? formProvider}?
                    </p>
                    <p style={{ fontSize: '0.75rem', color: '#6f6f6f', margin: '0 0 0.75rem' }}>
                      This removes the saved credentials and prevents them from being restored on restart, even if values exist in <code>.env</code>.
                    </p>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <Button
                        kind="danger" size="sm" renderIcon={TrashCan}
                        disabled={clearMut.isPending}
                        onClick={() => clearMut.mutate(formProvider)}
                      >
                        {clearMut.isPending ? 'Removing…' : 'Yes, unset'}
                      </Button>
                      <Button kind="ghost" size="sm" onClick={() => setUnsetConfirm(false)}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  )
}
