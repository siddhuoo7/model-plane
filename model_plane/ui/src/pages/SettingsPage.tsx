/**
 * Settings page — API key management + current user info.
 *
 * Sections:
 *   1. Your account (email, name, logout)
 *   2. API Keys — list, generate (shows full key once), delete
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
  TextInput,
  Tag,
  InlineNotification,
  Modal,
} from '@carbon/react'
import { Add, TrashCan, Copy, CheckmarkFilled, Logout } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, ApiKeyEntry } from '../api/client'
import { useAuth } from '../AuthContext'
import { LoadingState, ErrorState } from '../components/States'

// ── Copy button ──────────────────────────────────────────────────────────────

function InlineCopy({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true); setTimeout(() => setCopied(false), 2000)
        })
      }}
      title="Copy"
      style={{
        background: 'none', border: 'none', cursor: 'pointer',
        display: 'inline-flex', alignItems: 'center', gap: 4,
        color: copied ? '#24a148' : '#0f62fe', fontSize: '0.75rem',
      }}
    >
      {copied ? <CheckmarkFilled size={14} /> : <Copy size={14} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const { user, logout }  = useAuth()
  const qc = useQueryClient()

  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [revealedKey, setRevealedKey] = useState<{ id: string; key: string; label: string } | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<ApiKeyEntry | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['api-keys'],
    queryFn: api.settings.listApiKeys,
  })

  const createKey = useMutation({
    mutationFn: () => api.settings.createApiKey(newKeyLabel || undefined),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['api-keys'] })
      setNewKeyLabel('')
      setRevealedKey({ id: res.id, key: res.key, label: res.label })
      setNotice(null)
    },
    onError: (err: Error) => setNotice(err.message),
  })

  const deleteKey = useMutation({
    mutationFn: (id: string) => api.settings.deleteApiKey(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['api-keys'] })
      setDeleteConfirm(null)
    },
    onError: (err: Error) => setNotice(err.message),
  })

  const keyHeaders = [
    { key: 'label',      header: 'Label' },
    { key: 'prefix',     header: 'Key (masked)' },
    { key: 'created_at', header: 'Created' },
    { key: 'actions',    header: '' },
  ]

  const keyRows = (data?.keys ?? []).map(k => ({
    id: k.id,
    label: k.label,
    prefix: k.prefix,
    created_at: k.created_at
      ? new Date(k.created_at * 1000).toLocaleDateString(undefined, { dateStyle: 'medium' })
      : '—',
    _entry: k,
  }))

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: 800 }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.375rem' }}>Settings</h2>
      <p style={{ fontSize: '0.875rem', color: '#525252', marginBottom: '1.75rem' }}>
        Manage your API keys and account.
      </p>

      {notice && (
        <InlineNotification
          kind="error" title={notice}
          onClose={() => setNotice(null)}
          style={{ marginBottom: '1rem' }}
        />
      )}

      {/* ── Account section ── */}
      <section style={{
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
        padding: '1.25rem', marginBottom: '1.75rem',
        boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
      }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.875rem', color: '#161616' }}>
          Your account
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{
            width: 44, height: 44, borderRadius: '50%',
            background: 'linear-gradient(135deg, #0f62fe, #8a3ffc)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#fff', fontWeight: 700, fontSize: '1.125rem', flexShrink: 0,
          }}>
            {(user?.name || user?.email || 'A').charAt(0).toUpperCase()}
          </div>
          <div>
            <p style={{ fontWeight: 600, margin: 0 }}>{user?.name || 'Admin'}</p>
            <p style={{ fontSize: '0.8125rem', color: '#525252', margin: '0.125rem 0 0' }}>{user?.email}</p>
          </div>
          <Button
            kind="ghost"
            size="sm"
            renderIcon={Logout}
            style={{ marginLeft: 'auto' }}
            onClick={logout}
          >
            Sign out
          </Button>
        </div>
      </section>

      {/* ── API keys section ── */}
      <section style={{
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
        padding: '1.25rem', boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.875rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0, color: '#161616' }}>API Keys</h3>
            <p style={{ fontSize: '0.8125rem', color: '#525252', margin: '0.25rem 0 0' }}>
              Keys are used as <code>Authorization: Bearer &lt;key&gt;</code> on all <code>/v1/*</code> requests.
              The full key is shown only once at creation.
            </p>
          </div>
        </div>

        {/* Generate new key */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-end', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 220px' }}>
            <TextInput
              id="new-key-label"
              labelText="Label for new key"
              placeholder="e.g. Production app"
              value={newKeyLabel}
              size="sm"
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewKeyLabel(e.target.value)}
            />
          </div>
          <Button
            size="sm"
            renderIcon={Add}
            onClick={() => createKey.mutate()}
            disabled={createKey.isPending}
          >
            Generate key
          </Button>
        </div>

        {/* Revealed key banner */}
        {revealedKey && (
          <div style={{
            background: '#defbe6', border: '1px solid #a7f0ba', borderRadius: 8,
            padding: '1rem', marginBottom: '1rem',
          }}>
            <p style={{ fontWeight: 600, fontSize: '0.875rem', margin: '0 0 0.375rem', color: '#0e6027' }}>
              ✓ New key created — copy it now, it won't be shown again.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <code style={{
                background: '#fff', border: '1px solid #a7f0ba', borderRadius: 4,
                padding: '4px 10px', fontSize: '0.8125rem', wordBreak: 'break-all',
              }}>
                {revealedKey.key}
              </code>
              <InlineCopy text={revealedKey.key} />
              <button
                onClick={() => setRevealedKey(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6f6f6f', fontSize: '0.75rem', marginLeft: 'auto' }}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {isLoading && <LoadingState />}
        {error && <ErrorState message={String(error)} />}

        {data && keyRows.length === 0 && (
          <p style={{ fontSize: '0.875rem', color: '#6f6f6f', textAlign: 'center', padding: '2rem' }}>
            No API keys yet — generate one above.
          </p>
        )}

        {data && keyRows.length > 0 && (
          <DataTable rows={keyRows} headers={keyHeaders}>
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
                      const entry: ApiKeyEntry = row.cells.find((c: any) => c.info.header === 'label')?.row?._entry
                        ?? data.keys.find((k: ApiKeyEntry) => k.id === row.id)!
                      return (
                        <TableRow {...getRowProps({ row })} key={row.id}>
                          {row.cells.map((cell: any) => (
                            <TableCell key={cell.id}>
                              {cell.info.header === 'actions' ? (
                                <Button
                                  kind="ghost"
                                  size="sm"
                                  renderIcon={TrashCan}
                                  hasIconOnly
                                  iconDescription="Revoke"
                                  tooltipPosition="left"
                                  onClick={() => setDeleteConfirm(entry)}
                                />
                              ) : cell.info.header === 'prefix' ? (
                                <code style={{ fontSize: '0.8125rem', background: '#f4f4f4', padding: '1px 6px', borderRadius: 3 }}>
                                  {cell.value}
                                </code>
                              ) : (
                                cell.value
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

      {/* Delete confirm modal */}
      {deleteConfirm && (
        <Modal
          open
          danger
          modalHeading="Revoke API key"
          primaryButtonText="Revoke"
          secondaryButtonText="Cancel"
          onRequestClose={() => setDeleteConfirm(null)}
          onRequestSubmit={() => deleteKey.mutate(deleteConfirm.id)}
        >
          <p>
            Revoke key <strong>{deleteConfirm.label}</strong> (<code>{deleteConfirm.prefix}</code>)?
            Any application using this key will immediately get 401 errors.
          </p>
        </Modal>
      )}
    </div>
  )
}
