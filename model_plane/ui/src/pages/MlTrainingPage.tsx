/**
 * ML & Training page — two model cards (tier + task), dataset viewer, and file upload.
 */
import React, { useEffect, useRef, useState } from 'react'
import {
  Button,
  Tile,
  Tag,
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  ProgressBar,
  InlineNotification,
  Select,
  SelectItem,
  Toggle,
  FileUploader,
} from '@carbon/react'
import { Renew, MachineLearning, Upload, DataBase } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import { api, MlStatusResponse, MlModelInfo, TrainingDataResponse, UploadResult } from '../api/client'
import { LoadingState, ErrorState } from '../components/States'

type SseEvent = {
  event?: string
  progress?: number
  message?: string
  result?: { accuracy?: number; n_samples?: number; mode?: string }
  error?: string
}

// ── Inline bar chart ─────────────────────────────────────────────────────────

function DistributionBar({ label, counts }: { label: string; counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  const max = Math.max(...Object.values(counts), 1)
  const sorted = Object.entries(counts).sort(([, a], [, b]) => b - a)
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <p style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.4rem' }}>{label}</p>
      {sorted.map(([name, cnt]) => (
        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: 3 }}>
          <span style={{ fontSize: '0.7rem', minWidth: 190, color: '#525252' }}>{name}</span>
          <div style={{ flex: 1, height: 8, background: '#e0e0e0', borderRadius: 3 }}>
            <div style={{ width: `${(cnt / max) * 100}%`, height: '100%', background: '#0f62fe', borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: '0.7rem', minWidth: 56, textAlign: 'right', color: '#525252' }}>
            {cnt} ({((cnt / total) * 100).toFixed(0)}%)
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Feature importance bar ───────────────────────────────────────────────────

function FeatureImportanceBar({ importances }: { importances: Record<string, number> }) {
  const max = Math.max(...Object.values(importances), 0.0001)
  const sorted = Object.entries(importances).sort(([, a], [, b]) => b - a).slice(0, 10)
  return (
    <div style={{ marginTop: '0.75rem' }}>
      <p style={{ fontSize: '0.75rem', color: '#525252', marginBottom: '0.4rem' }}>Feature importances</p>
      {sorted.map(([feat, val]) => (
        <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: 3 }}>
          <span style={{ fontSize: '0.7rem', minWidth: 180, color: '#525252' }}>{feat}</span>
          <div style={{ flex: 1, height: 7, background: '#e0e0e0', borderRadius: 3 }}>
            <div style={{ width: `${(val / max) * 100}%`, height: '100%', background: '#8a3ffc', borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: '0.7rem', minWidth: 44, textAlign: 'right' }}>{(val * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

// ── Model card ───────────────────────────────────────────────────────────────

function ModelCard({
  title,
  info,
  mode,
  onRetrain,
  retraining,
  extraControls,
}: {
  title: string
  info: MlModelInfo | undefined
  mode: 'tier' | 'task' | 'deployment'
  onRetrain: (mode: 'tier' | 'task' | 'deployment') => void
  retraining: boolean
  extraControls?: React.ReactNode
}) {
  const loaded = info?.model_loaded ?? false
  return (
    <Tile style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ flex: 2, minWidth: 220 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <MachineLearning size={16} />
            <strong style={{ fontSize: '0.9rem' }}>{title}</strong>
            <Tag type={loaded ? 'green' : 'gray'} size="sm">{loaded ? 'ready' : 'not loaded'}</Tag>
          </div>
          <p style={{ fontSize: '0.72rem', color: '#525252' }}>Path</p>
          <p style={{ fontFamily: 'monospace', fontSize: '0.72rem', marginBottom: '0.3rem' }}>{info?.model_path ?? '—'}</p>
          {info?.last_modified_utc && (
            <>
              <p style={{ fontSize: '0.72rem', color: '#525252' }}>Last trained</p>
              <p style={{ fontSize: '0.8rem' }}>{new Date(info.last_modified_utc).toLocaleString()}</p>
            </>
          )}
          {info?.error && (
            <p style={{ fontSize: '0.72rem', color: '#da1e28', marginTop: '0.3rem' }}>Error: {info.error}</p>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 130 }}>
          {info?.accuracy != null && (
            <>
              <p style={{ fontSize: '0.72rem', color: '#525252' }}>Accuracy</p>
              <p style={{ fontSize: '1.4rem', fontWeight: 700 }}>{(info.accuracy * 100).toFixed(1)}%</p>
            </>
          )}
          {info?.training_rows != null && (
            <>
              <p style={{ fontSize: '0.72rem', color: '#525252', marginTop: '0.3rem' }}>Training rows</p>
              <p style={{ fontSize: '0.85rem' }}>{info.training_rows.toLocaleString()}</p>
            </>
          )}
          {(info?.class_names?.length ?? 0) > 0 && (
            <>
              <p style={{ fontSize: '0.72rem', color: '#525252', marginTop: '0.3rem' }}>
                Classes ({info?.class_names?.length ?? 0})
              </p>
              <p style={{ fontSize: '0.7rem', color: '#525252' }}>{info?.class_names?.join(' · ')}</p>
            </>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'flex-end' }}>
          {extraControls}
          <Button kind="secondary" size="sm" renderIcon={Renew} onClick={() => onRetrain(mode)} disabled={retraining}>
            {retraining ? 'Retraining…' : 'Retrain'}
          </Button>
        </div>
      </div>

      {info?.feature_importances && Object.keys(info.feature_importances).length > 0 && (
        <FeatureImportanceBar importances={info.feature_importances} />
      )}
    </Tile>
  )
}

// ── Dataset panel ─────────────────────────────────────────────────────────────

const FEATURE_COLS = [
  'reasoning_markers','code_presence','simple_indicators','multi_step_patterns',
  'technical_terms','token_count_signal','creative_markers','question_complexity',
  'constraint_count','imperative_verbs','output_format','domain_specificity',
  'reference_complexity','negation_complexity',
]
const TASK_TYPES = [
  'code_generation','code_editing','code_debugging','repository_search',
  'mathematical_reasoning','technical_reasoning','long_context_synthesis',
  'tool_call_interpretation','structured_extraction','summarization',
  'translation','creative_writing','planning','simple_qa','unknown',
]
const TIER_VALUES = ['simple','medium','complex','reasoning']

function SchemaCard() {
  return (
    <Tile style={{ marginBottom: '1rem', background: '#f4f4f4' }}>
      <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.5rem' }}>
        What does the dataset need to contain?
      </p>
      <p style={{ fontSize: '0.78rem', color: '#525252', marginBottom: '0.5rem' }}>
        Both models train from the <em>same</em> CSV — <code>training_data.csv</code>.
        The <strong>task classifier</strong> reads <code>task_type</code>;
        the <strong>tier classifier</strong> reads <code>tier</code>.
        The <code>query</code> column holds the raw prompt text — the trainer extracts
        14 boolean regex signals from it automatically to augment the 14 numeric features.
      </p>

      <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <p style={{ fontSize: '0.72rem', fontWeight: 600, marginBottom: '0.25rem' }}>Required columns</p>
          <code style={{ display: 'block', fontSize: '0.72rem', marginBottom: 2 }}>task_type</code>
          <p style={{ fontSize: '0.68rem', color: '#525252', marginBottom: '0.4rem' }}>
            {TASK_TYPES.join(' · ')}
          </p>
          <code style={{ display: 'block', fontSize: '0.72rem', marginBottom: 2 }}>tier</code>
          <p style={{ fontSize: '0.68rem', color: '#525252' }}>{TIER_VALUES.join(' · ')}</p>
        </div>

        <div style={{ flex: 2, minWidth: 280 }}>
          <p style={{ fontSize: '0.72rem', fontWeight: 600, marginBottom: '0.25rem' }}>
            Feature columns (float 0–1, all optional)
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
            {FEATURE_COLS.map(c => (
              <code key={c} style={{ fontSize: '0.68rem', background: '#e0e0e0', padding: '1px 4px', borderRadius: 2 }}>{c}</code>
            ))}
          </div>
        </div>
      </div>

      {/* How the model learns */}
      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e0e0e0' }}>
        <p style={{ fontSize: '0.72rem', fontWeight: 600, marginBottom: '0.25rem' }}>
          How does the model learn from training data?
        </p>
        <p style={{ fontSize: '0.72rem', color: '#525252' }}>
          The trainer uses <strong>28 features per row</strong>: the 14 numeric float columns
          (<code>reasoning_markers</code>, <code>code_presence</code>, etc.) plus 14 boolean
          regex signals extracted live from the <code>query</code> text at training time.
          At inference the same signals are computed from the incoming prompt — no manual float
          entry is needed if you include the <code>query</code> column.
        </p>
      </div>

      {/* Why the model files differ */}
      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e0e0e0' }}>
        <p style={{ fontSize: '0.72rem', fontWeight: 600, marginBottom: '0.25rem' }}>
          Why do the two model files look different (.json vs .joblib)?
        </p>
        <p style={{ fontSize: '0.72rem', color: '#525252' }}>
          <strong>Task classifier</strong> (<code>classifier_task.joblib</code>) — Gradient Boosting (sklearn). Accuracy ~98%.<br />
          <strong>Tier classifier</strong> (<code>router_ml_tier_v1.joblib</code>) — Random Forest (sklearn). Accuracy ~99%.<br />
          Portable JSON backups can be trained via the <em>tier-json / task-json</em> modes if sklearn-free inference is needed.
        </p>
      </div>

      {/* Sample downloads */}
      <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.75rem' }}>
        <a
          href={api.ml.sampleDataUrl('csv')}
          download="sample_training_data.csv"
          style={{ fontSize: '0.78rem', color: '#0f62fe', textDecoration: 'none' }}
        >
          ↓ Download sample CSV
        </a>
        <a
          href={api.ml.sampleDataUrl('jsonl')}
          download="sample_training_data.jsonl"
          style={{ fontSize: '0.78rem', color: '#0f62fe', textDecoration: 'none' }}
        >
          ↓ Download sample JSONL
        </a>
      </div>
    </Tile>
  )
}

function DatasetPanel() {
  const { data, isLoading, error, refetch } = useQuery<TrainingDataResponse>({
    queryKey: ['ml-training-data'],
    queryFn: () => api.ml.trainingData(200, 0),
  })

  const [mergeMode, setMergeMode] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) return
    setUploading(true)
    setUploadError(null)
    setUploadResult(null)
    try {
      const result = await api.ml.uploadTrainingData(files[0], mergeMode)
      setUploadResult(result)
      refetch()
    } catch (e) {
      setUploadError(String(e))
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div>
      <SchemaCard />

      {/* Upload zone */}
      <Tile style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <Upload size={16} />
          <strong>Upload training data</strong>
          <span style={{ fontSize: '0.72rem', color: '#525252', marginLeft: '0.5rem' }}>
            — appends to <code>training_data.csv</code>, used by both models
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: '0.72rem', color: '#525252', marginBottom: '0.25rem' }}>Upload mode</p>
            <Toggle
              id="merge-toggle"
              labelText=""
              labelA="Replace"
              labelB="Append"
              toggled={mergeMode}
              onToggle={(v: boolean) => setMergeMode(v)}
            />
            <p style={{ fontSize: '0.68rem', color: '#525252', marginTop: '0.2rem' }}>
              {mergeMode ? 'New rows appended — existing data kept' : '⚠ Replaces entire dataset'}
            </p>
          </div>

          <div style={{ flex: 1, minWidth: 260 }}>
            <label
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
                padding: '0.5rem 1rem', border: '1px solid #8d8d8d',
                borderRadius: 2, cursor: uploading ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem', background: uploading ? '#e0e0e0' : 'transparent',
              }}
            >
              <Upload size={14} />
              {uploading ? 'Uploading…' : 'Choose file (.csv or .jsonl)'}
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.jsonl,.ndjson"
                style={{ display: 'none' }}
                disabled={uploading}
                onChange={(e) => handleUpload(e.target.files)}
              />
            </label>
            <p style={{ fontSize: '0.7rem', color: '#525252', marginTop: '0.25rem' }}>
              After uploading, go to the <strong>Models</strong> tab and click <strong>Retrain</strong> on each model.
            </p>
          </div>
        </div>

        {uploadResult && (
          <InlineNotification
            kind="success"
            title={`Uploaded ${uploadResult.accepted_rows.toLocaleString()} rows. Dataset total: ${uploadResult.total_rows.toLocaleString()} rows`}
            subtitle="Now retrain both classifiers from the Models tab."
            hideCloseButton={false}
            onClose={() => setUploadResult(null)}
            style={{ marginTop: '0.75rem' }}
          />
        )}
        {uploadError && (
          <InlineNotification
            kind="error"
            title={uploadError}
            hideCloseButton={false}
            onClose={() => setUploadError(null)}
            style={{ marginTop: '0.75rem' }}
          />
        )}
      </Tile>

      {/* Current dataset summary */}
      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <Tile>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <DataBase size={16} />
            <strong>Current dataset</strong>
            <Tag type="blue" size="sm">{data.total_rows.toLocaleString()} rows</Tag>
            <span style={{ fontSize: '0.7rem', color: '#525252', marginLeft: 'auto', fontFamily: 'monospace' }}>
              {data.file_path}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 280 }}>
              <DistributionBar
                label="task_type distribution (task classifier trains on this)"
                counts={data.task_type_counts}
              />
            </div>
            <div style={{ flex: 1, minWidth: 200 }}>
              <DistributionBar
                label="tier distribution (tier classifier trains on this)"
                counts={data.tier_counts}
              />
            </div>
          </div>

          <p style={{ fontSize: '0.7rem', color: '#525252', marginTop: '0.5rem' }}>
            All columns ({data.columns.length}): {data.columns.join(', ')}
          </p>
        </Tile>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function MlTrainingPage() {
  const { data, isLoading, error, refetch } = useQuery<MlStatusResponse>({
    queryKey: ['ml-status'],
    queryFn: api.ml.status,
  })

  const [tierAlgo, setTierAlgo] = useState<'gradient_boost' | 'random_forest'>('gradient_boost')
  const [activeMode, setActiveMode] = useState<'tier' | 'task' | 'deployment' | null>(null)
  const [progress, setProgress] = useState(0)
  const [events, setEvents] = useState<SseEvent[]>([])
  const [notice, setNotice] = useState<{ kind: 'success' | 'error'; msg: string } | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => { esRef.current?.close() }, [])

  async function startRetrain(mode: 'tier' | 'task' | 'deployment') {
    setActiveMode(mode)
    setProgress(0)
    setEvents([])
    setNotice(null)
    try {
      const res = await api.ml.retrain(mode, tierAlgo)
      const es = new EventSource(`/admin/api/ml/retrain/${res.job_id}/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          const payload: SseEvent = JSON.parse(e.data)
          setEvents(prev => [...prev, payload])
          if (payload.progress !== undefined) setProgress(payload.progress)
          if (payload.event === 'done') {
            setActiveMode(null)
            const acc = payload.result?.accuracy
            const n = payload.result?.n_samples
            setNotice({
              kind: 'success',
              msg: `${mode === 'task' ? 'Task' : 'Tier'} classifier retrained.`
                + (acc != null ? ` Accuracy: ${(acc * 100).toFixed(1)}%` : '')
                + (n != null ? `  ·  ${n.toLocaleString()} samples` : ''),
            })
            refetch()
            es.close()
          }
          if (payload.event === 'error') {
            setActiveMode(null)
            setNotice({ kind: 'error', msg: payload.error ?? 'Retrain failed.' })
            es.close()
          }
        } catch { /* ignore */ }
      }
      es.onerror = () => {
        setActiveMode(null)
        setNotice({ kind: 'error', msg: 'SSE connection lost.' })
        es.close()
      }
    } catch (e) {
      setActiveMode(null)
      setNotice({ kind: 'error', msg: String(e) })
    }
  }

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 0.25rem', color: '#161616' }}>ML &amp; Training</h2>
      <p style={{ fontSize: '0.85rem', color: '#525252', marginBottom: '1.25rem' }}>
        Two models trained from <code>training_data.csv</code>:{' '}
        <strong>tier classifier</strong> (simple/medium/complex/reasoning) and{' '}
        <strong>task classifier</strong> (code_generation, planning, …).
      </p>

      {notice && (
        <InlineNotification kind={notice.kind} title={notice.msg} hideCloseButton={false}
          onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />
      )}

      <Tabs>
        <TabList aria-label="ML training tabs">
          <Tab renderIcon={MachineLearning}>Models</Tab>
          <Tab renderIcon={DataBase}>Training data</Tab>
        </TabList>
        <TabPanels>

          {/* ── Models tab ── */}
          <TabPanel>
            {isLoading && <LoadingState />}
            {error && <ErrorState message={String(error)} />}

            {data && (
              <>
                <ModelCard
                  title="Tier Classifier — complexity_tier (joblib, sklearn)"
                  info={data.tier_model}
                  mode="tier"
                  onRetrain={startRetrain}
                  retraining={activeMode === 'tier'}
                  extraControls={
                    <Select
                      id="tier-algo"
                      labelText="Algorithm"
                      size="sm"
                      value={tierAlgo}
                      onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                        setTierAlgo(e.target.value as 'gradient_boost' | 'random_forest')
                      }
                      style={{ minWidth: 180 }}
                    >
                      <SelectItem value="gradient_boost" text="Gradient Boosting" />
                      <SelectItem value="random_forest" text="Random Forest" />
                    </Select>
                  }
                />

                <ModelCard
                  title="Task Classifier — task_type (JSON logistic regression)"
                  info={data.task_model}
                  mode="task"
                  onRetrain={startRetrain}
                  retraining={activeMode === 'task'}
                />
              </>
            )}

            {activeMode && (
              <div style={{ marginBottom: '1rem', maxWidth: 520 }}>
                <p style={{ fontSize: '0.8rem', color: '#525252', marginBottom: '0.25rem' }}>
                  Retraining <strong>{activeMode}</strong> classifier…
                </p>
                <ProgressBar label="" value={progress} max={100} />
              </div>
            )}

            {events.length > 0 && (
              <div style={{
                background: '#161616', color: '#f4f4f4', fontFamily: 'monospace',
                fontSize: '0.72rem', padding: '0.75rem', borderRadius: 4,
                maxHeight: 160, overflow: 'auto',
              }}>
                {events.map((ev, i) => (
                  <div key={i}>[{ev.event ?? 'msg'}] {ev.message ?? ''}{ev.progress != null ? ` (${ev.progress}%)` : ''}</div>
                ))}
              </div>
            )}
          </TabPanel>

          {/* ── Training data tab ── */}
          <TabPanel>
            <DatasetPanel />
          </TabPanel>

        </TabPanels>
      </Tabs>
    </div>
  )
}
