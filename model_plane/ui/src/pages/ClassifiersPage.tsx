/**
 * Classifier Status page.
 *
 * ACTIVE vs ENABLED — explained:
 *
 * ACTIVE  = the classifier that actually runs on every request. There is
 *           always exactly one active classifier. Regex is the always-on
 *           baseline; you can switch to BERT or Laya here.
 *
 * ENABLED = whether a non-regex classifier is allowed to be selected.
 *           A classifier can be enabled but not active (it's ready to use
 *           but not currently running). Disabling it prevents it from being
 *           set as active. Regex cannot be disabled — it is the permanent
 *           fallback.
 *
 * Routing logic (simplified):
 *   1. Regex always runs first → produces a TaskType + ComplexityTier.
 *   2. If active ≠ regex, the active classifier ALSO runs.
 *      If its confidence ≥ min_confidence, it overrides the regex result.
 *   3. If the active classifier fails / is below threshold, regex result is kept.
 */
import React from 'react'
import {
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  Toggle,
  Button,
  Tag,
  InlineNotification,
} from '@carbon/react'
import { CircleFill, RadioButton, Time } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, ClassifierInfo } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

const CLASSIFIER_DESCRIPTIONS: Record<string, string> = {
  regex: 'Always runs. Pure lexical pattern matching — zero dependencies, ~1 ms. The permanent fallback even when another classifier is active.',
  mbert: 'Optional. Semantic NLI re-ranking using a HuggingFace cross-encoder. Overrides regex task_type when confidence ≥ threshold (~120 ms). Requires transformers installed.',
  laya: 'Optional. IBM Laya local CPU inference — answers task_type AND complexity_tier in one call (~250 ms). Requires laya package installed.',
}

export default function ClassifiersPage() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['classifiers'],
    queryFn: api.classifiers.list,
  })

  const [notice, setNotice] = React.useState<string | null>(null)

  const toggle = useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      api.classifiers.update(name, { enabled }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['classifiers'] })
      setNotice('Classifier updated.')
    },
  })

  const activate = useMutation({
    mutationFn: (name: string) => api.classifiers.update(name, { set_active: true }),
    onSuccess: (_res, name) => {
      qc.invalidateQueries({ queryKey: ['classifiers'] })
      setNotice(`"${name}" is now the active classifier. Change takes effect immediately for new requests.`)
    },
  })

  const headers = [
    { key: 'display_name', header: 'Classifier' },
    { key: 'estimated_latency_ms', header: 'Latency' },
    { key: 'active', header: 'Active' },
    { key: 'enabled', header: 'Enabled' },
    { key: 'actions', header: '' },
  ]

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>Classifier Status</h2>

      {/* Active vs Enabled explainer */}
      <div style={{
        background: '#f4f4f4', border: '1px solid #e0e0e0', borderRadius: 4,
        padding: '1rem', marginBottom: '1.5rem',
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.375rem' }}>
            <CircleFill size={16} style={{ color: '#24a148' }} />
            <strong style={{ fontSize: '0.875rem' }}>Active</strong>
          </div>
          <p style={{ fontSize: '0.8125rem', color: '#525252', margin: 0, lineHeight: 1.5 }}>
            The one classifier that runs on every request. Only <em>one</em> can be active at a time.
            Regex is the baseline and always runs; if another classifier is active, it runs <em>in addition</em> and can override the regex result when confident.
          </p>
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.375rem' }}>
            <RadioButton size={16} style={{ color: '#0f62fe' }} />
            <strong style={{ fontSize: '0.875rem' }}>Enabled</strong>
          </div>
          <p style={{ fontSize: '0.8125rem', color: '#525252', margin: 0, lineHeight: 1.5 }}>
            Whether a classifier is <em>available</em> to be activated. You can disable BERT or Laya to prevent them from being switched to. Regex cannot be disabled — it is permanent.
          </p>
        </div>
      </div>

      {notice && (
        <InlineNotification
          kind="success"
          title={notice}
          hideCloseButton={false}
          onClose={() => setNotice(null)}
          style={{ marginBottom: '1rem' }}
        />
      )}

      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <>
          <p style={{ fontSize: '0.875rem', color: '#525252', marginBottom: '0.75rem' }}>
            Active classifier: <Tag type="blue" size="sm">{data.active_classifier}</Tag>
          </p>

          {/* Classifier cards — one per path */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
            {data.classifiers.map((clf: ClassifierInfo) => {
              const isAvailable = clf.available !== false  // undefined → assume available (backward compat)
              return (
                <div key={clf.name} style={{
                  border: `2px solid ${clf.active ? '#0f62fe' : !isAvailable ? '#f1c21b' : '#e0e0e0'}`,
                  borderRadius: 6,
                  padding: '1rem 1.25rem',
                  background: clf.active ? '#edf5ff' : !isAvailable ? '#fffbf0' : '#fff',
                  display: 'flex', alignItems: 'flex-start', gap: '1rem',
                  opacity: !isAvailable && !clf.active ? 0.85 : 1,
                }}>
                  {/* Left: name + description */}
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.375rem', flexWrap: 'wrap' }}>
                      <strong style={{ fontSize: '0.9375rem' }}>{clf.display_name}</strong>
                      {clf.active && <Tag type="blue" size="sm">active</Tag>}
                      {clf.name === 'regex' && <Tag type="gray" size="sm">always on</Tag>}
                      {!isAvailable && <Tag type="warm-gray" size="sm">not installed</Tag>}
                    </div>
                    <p style={{ fontSize: '0.8125rem', color: '#525252', margin: '0 0 0.375rem 0', lineHeight: 1.5 }}>
                      {CLASSIFIER_DESCRIPTIONS[clf.name]}
                    </p>
                    {/* Install hint when package is missing */}
                    {!isAvailable && clf.install_hint && (
                      <p style={{ fontSize: '0.75rem', color: '#8a6914', margin: '0.25rem 0 0.375rem', lineHeight: 1.5 }}>
                        ⚠ Not installed.{' '}
                        <code style={{ background: '#fef3cd', padding: '1px 6px', borderRadius: 3, fontFamily: 'monospace' }}>
                          {clf.install_hint}
                        </code>
                      </p>
                    )}
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: '#6f6f6f' }}>
                      <span><Time size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />~{clf.estimated_latency_ms} ms</span>
                      <span>Source: <code style={{ background: '#f4f4f4', padding: '0 4px', borderRadius: 2 }}>{clf.model_source}</code></span>
                    </div>
                  </div>

                  {/* Right: controls */}
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.75rem', minWidth: 160 }}>
                    {clf.name !== 'regex' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.75rem', color: '#525252' }}>Enabled</span>
                        <Toggle
                          id={`toggle-${clf.name}`}
                          labelText=""
                          hideLabel
                          size="sm"
                          toggled={clf.enabled}
                          onToggle={(checked: boolean) => toggle.mutate({ name: clf.name, enabled: checked })}
                          disabled={!isAvailable}
                        />
                      </div>
                    )}
                    {!clf.active && clf.enabled && clf.name !== 'regex' && isAvailable && (
                      <Button kind="tertiary" size="sm" onClick={() => activate.mutate(clf.name)} disabled={activate.isPending}>
                        Set as active
                      </Button>
                    )}
                    {!clf.active && clf.enabled && clf.name !== 'regex' && !isAvailable && (
                      <Button kind="secondary" size="sm" disabled>
                        Install first
                      </Button>
                    )}
                    {!clf.active && clf.name === 'regex' && (
                      <Button kind="ghost" size="sm" onClick={() => activate.mutate(clf.name)} disabled={activate.isPending}>
                        Revert to regex
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
