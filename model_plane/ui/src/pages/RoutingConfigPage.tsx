/**
 * Routing & Classification — two tabs:
 *   1. Classifiers    — active/enabled classifiers  (first)
 *   2. Routing Rules  — scorer weights + task overrides (second)
 */
import React, { useEffect, useRef, useState } from 'react'
import {
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  Button,
  TextInput,
  Select,
  SelectItem,
  Modal,
  InlineNotification,
  Tag,
  Toggle,
  Slider,
  InlineLoading,
} from '@carbon/react'
import {
  Add,
  Save,
  TrashCan,
  Time,
  Information,
  ChevronRight,
  Checkmark,
} from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, RoutingConfig, ClassifierInfo, SecurityFlagsResponse } from '../api/client'
import { TierBadge } from '../components/TierBadge'
import { LoadingState, ErrorState } from '../components/States'

// ── Constants ─────────────────────────────────────────────────────────────────

const TIERS = ['simple', 'medium', 'complex', 'reasoning']

const DEFAULT_WEIGHTS: Record<string, number> = {
  cost: 0.30,
  tier_match: 0.25,
  capability_match: 0.20,
  context_fit: 0.15,
  health_bonus: 0.10,
}

const WEIGHT_META: Record<string, { label: string; hint: string; color: string; what: string }> = {
  cost: {
    label: 'Cost',
    hint: 'How much to prefer cheaper models over expensive ones.',
    what: 'Raise this to aggressively prefer the lowest-cost eligible model. Lower it when quality matters more than price.',
    color: '#24a148',
  },
  tier_match: {
    label: 'Tier match',
    hint: 'Prefer models whose tier exactly matches the classified complexity.',
    what: 'Controls how strictly the router enforces the classifier\'s tier output. High = always respect the tier; low = allow cross-tier selection.',
    color: '#0f62fe',
  },
  capability_match: {
    label: 'Capability match',
    hint: 'Prefer models that support features the request needs (tool use, vision, etc.).',
    what: 'Raise this if requests frequently use tools or images and you want guaranteed-capable model selection.',
    color: '#8a3ffc',
  },
  context_fit: {
    label: 'Context fit',
    hint: 'Prefer models whose context window fits the prompt without truncation.',
    what: 'Raise this for long-document or multi-turn workloads where context overflow would cause silent failures.',
    color: '#ff832b',
  },
  health_bonus: {
    label: 'Health bonus',
    hint: 'Boost models that are currently healthy and penalise flapping backends.',
    what: 'Raise this if you have unreliable model endpoints and want the router to bias away from them faster.',
    color: '#0072c3',
  },
}

const TIER_META: Record<string, { hint: string; color: string; bg: string; border: string }> = {
  simple: {
    hint: 'Short Q&A, single-turn, summarisation. Fastest and cheapest models.',
    color: '#198038', bg: '#defbe6', border: '#24a14850',
  },
  medium: {
    hint: 'Multi-turn chat, light reasoning, standard completions.',
    color: '#0043ce', bg: '#edf5ff', border: '#0f62fe50',
  },
  complex: {
    hint: 'Code generation, tool use, long-context, structured output.',
    color: '#6929c4', bg: '#f6f2ff', border: '#8a3ffc50',
  },
  reasoning: {
    hint: 'Chain-of-thought, hard math, multi-step planning. Strongest models only.',
    color: '#ba4e00', bg: '#fff2e8', border: '#ff832b50',
  },
}

// ── Inline tooltip ────────────────────────────────────────────────────────────

function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <Information
        size={14}
        style={{ color: '#6f6f6f', cursor: 'help', flexShrink: 0 }}
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      />
      {show && (
        <span style={{
          position: 'absolute', bottom: '100%', left: '50%', transform: 'translateX(-50%)',
          background: '#161616', color: '#f4f4f4',
          fontSize: '0.75rem', lineHeight: 1.5,
          padding: '0.5rem 0.75rem', borderRadius: 4,
          width: 260, zIndex: 999,
          boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
          marginBottom: 6, whiteSpace: 'normal', textAlign: 'left',
          pointerEvents: 'none',
        }}>
          {text}
        </span>
      )}
    </span>
  )
}

// ── Custom range slider (no Carbon number input clutter) ──────────────────────

function WeightSlider({
  id, value, color, onChange,
}: { id: string; value: number; color: string; onChange: (v: number) => void }) {
  const pct = value  // 0–100
  return (
    <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center' }}>
      {/* Track */}
      <div style={{
        position: 'absolute', left: 0, right: 0,
        height: 4, background: '#e0e0e0', borderRadius: 2,
      }} />
      {/* Fill */}
      <div style={{
        position: 'absolute', left: 0,
        width: `${pct}%`, height: 4,
        background: color, borderRadius: 2,
        transition: 'width 0.1s',
      }} />
      {/* Native input (transparent, sits on top) */}
      <input
        id={id}
        type="range"
        min={0} max={100} step={1}
        value={pct}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          position: 'absolute', left: 0, right: 0,
          width: '100%', opacity: 0, cursor: 'pointer',
          height: 20, margin: 0, padding: 0,
        }}
      />
      {/* Thumb visual */}
      <div style={{
        position: 'absolute', left: `${pct}%`, transform: 'translateX(-50%)',
        width: 14, height: 14, borderRadius: '50%',
        background: '#fff', border: `2px solid ${color}`,
        boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
        pointerEvents: 'none', transition: 'left 0.1s',
      }} />
    </div>
  )
}

// ── Weight total bar ──────────────────────────────────────────────────────────

function WeightTotalBar({ weights }: { weights: Record<string, number> }) {
  // Only sum query-dimension weights (exclude cost weights like cost_input_weight / cost_output_weight)
  const featureWeights = Object.entries(weights).filter(([k]) => !k.startsWith('cost_'))
  const sum = featureWeights.reduce((a, [, b]) => a + b, 0)
  const ok = Math.abs(sum - 1.0) < 0.01
  const over = sum > 1.005
  const color = ok ? '#24a148' : over ? '#da1e28' : '#f1c21b'
  return (
    <div style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: '0.75rem', color: '#525252' }}>
          Total weight — must equal 100%
        </span>
        <span style={{ fontSize: '0.8125rem', fontWeight: 700, color }}>
          {(sum * 100).toFixed(0)}%{ok ? ' ✓' : over ? ' — reduce other weights' : ' — increase to reach 100%'}
        </span>
      </div>
      <div style={{ height: 5, background: '#e0e0e0', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.min(sum * 100, 100)}%`,
          background: color, borderRadius: 3, transition: 'width 0.2s, background 0.2s',
        }} />
      </div>
    </div>
  )
}

// ── Routing tab ───────────────────────────────────────────────────────────────

function RoutingTab() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['routing-config'], queryFn: api.routing.getConfig,
  })

  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS)
  const [overrides, setOverrides] = useState<Record<string, string>>({})
  const [notice, setNotice] = useState<string | null>(null)
  const [preview, setPreview] = useState<any>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [newTask, setNewTask] = useState('')
  const [newTier, setNewTier] = useState('simple')

  useEffect(() => {
    if (data) {
      setWeights(data.scorer_weights ?? DEFAULT_WEIGHTS)
      setOverrides(data.task_overrides ?? {})
    }
  }, [data])

  const saveMut = useMutation({
    mutationFn: (cfg: Partial<RoutingConfig>) => api.routing.setConfig(cfg),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['routing-config'] }); setNotice('Configuration saved.') },
  })
  const previewMut = useMutation({
    mutationFn: (cfg: Partial<RoutingConfig>) => api.routing.preview(cfg),
    onSuccess: (res) => setPreview(res),
  })

  function addOverride() {
    if (!newTask.trim()) return
    setOverrides(o => ({ ...o, [newTask.trim()]: newTier }))
    setNewTask('')
    setAddOpen(false)
  }

  return (
    <div style={{ paddingTop: '1.25rem' }}>
      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}
      {notice && (
        <InlineNotification kind="success" title={notice} hideCloseButton={false}
          onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />
      )}

      {/* ── How tiers work ── */}
      <div style={{
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
        marginBottom: '1rem', overflow: 'hidden',
      }}>
        <div style={{ padding: '0.875rem 1rem 0.5rem', borderBottom: '1px solid #f4f4f4' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#161616' }}>Tier routing</span>
            <Tooltip text="Each request is classified into a tier. The router selects a model matching that tier. Add task overrides to force specific task types to a fixed tier — bypassing the classifier entirely." />
          </div>
          <p style={{ fontSize: '0.75rem', color: '#525252', margin: '0.25rem 0 0' }}>
            Requests are classified into one of four tiers. Add overrides to pin specific task types to a tier.
          </p>
        </div>
        {TIERS.map((tier, idx) => {
          const m = TIER_META[tier]
          const pins = Object.entries(overrides).filter(([, t]) => t === tier).map(([k]) => k)
          return (
            <div key={tier} style={{
              display: 'flex', alignItems: 'stretch',
              borderBottom: idx < TIERS.length - 1 ? '1px solid #f4f4f4' : undefined,
            }}>
              {/* Coloured left stripe — just the badge, no description text */}
              <div style={{
                width: 110, flexShrink: 0,
                background: m.bg,
                borderRight: `3px solid ${m.color}`,
                padding: '0.5rem 0.75rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <TierBadge tier={tier} size="sm" />
              </div>
              {/* Override pills */}
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center',
                padding: '0.5rem 0.875rem', gap: '0.375rem', flexWrap: 'wrap', minHeight: 48,
              }}>
                {pins.length > 0
                  ? pins.map(d => (
                    <span key={d} style={{
                      display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                      background: m.bg, border: `1px solid ${m.border}`,
                      borderRadius: 12, padding: '2px 8px 2px 6px',
                      fontSize: '0.75rem', color: m.color, fontWeight: 500,
                    }}>
                      <ChevronRight size={10} />{d}
                    </span>
                  ))
                  : <span style={{ fontSize: '0.75rem', color: '#a8a8a8', fontStyle: 'italic' }}>
                      No overrides — classifier decides
                    </span>
                }
              </div>
              {/* Step # */}
              <div style={{
                width: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '0.625rem', color: '#c6c6c6', fontWeight: 700, flexShrink: 0,
              }}>
                {idx + 1}
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Scorer weights ── */}
      <div style={{
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
        marginBottom: '1rem', overflow: 'hidden',
      }}>
        <div style={{ padding: '0.875rem 1rem 0.75rem', borderBottom: '1px solid #f4f4f4' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#161616' }}>Scorer weights</span>
            <Tooltip text="The router scores every eligible model on five dimensions and picks the highest scorer. Weights control how much each dimension counts. They must sum to 100%." />
          </div>
          <p style={{ fontSize: '0.75rem', color: '#525252', margin: '0.25rem 0 0.5rem' }}>
            Drag the sliders to shift routing priority between cost, quality, and reliability.
          </p>
          <WeightTotalBar weights={weights} />
        </div>

        {Object.entries(weights)
          .filter(([key]) => !key.startsWith('cost_'))
          .map(([key, val], idx, arr) => {
          const m = WEIGHT_META[key] ?? { label: key.replace(/_/g, ' '), hint: '', what: '', color: '#525252' }
          const pct = Math.round(val * 100)
          return (
            <div key={key} style={{
              padding: '0.625rem 1rem',
              borderBottom: idx < arr.length - 1 ? '1px solid #f4f4f4' : undefined,
            }}>
              {/* Label + % on one line, slider below */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                    background: m.color, flexShrink: 0,
                  }} />
                  <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#161616' }}>{m.label}</span>
                  <Tooltip text={`${m.hint}  ${m.what}`} />
                </div>
                <span style={{ fontSize: '0.875rem', fontWeight: 700, color: m.color, minWidth: 36, textAlign: 'right' }}>
                  {pct}%
                </span>
              </div>
              {/* Slider — no hint paragraph */}
              <WeightSlider
                id={`w-${key}`}
                value={pct}
                color={m.color}
                onChange={v => setWeights(w => ({ ...w, [key]: v / 100 }))}
              />
            </div>
          )
        })}
      </div>

      {/* ── Task overrides ── */}
      <div style={{
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
        marginBottom: '1rem', overflow: 'hidden',
      }}>
        <div style={{
          padding: '0.875rem 1rem',
          borderBottom: Object.keys(overrides).length > 0 ? '1px solid #f4f4f4' : undefined,
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#161616' }}>Task type overrides</span>
              <Tooltip text="Override the classifier for specific task types. When a request's task_type matches an entry here, it skips the scorer and goes directly to the mapped tier. Leave empty to use the classifier for all requests." />
            </div>
            <p style={{ fontSize: '0.75rem', color: '#525252', margin: '0.25rem 0 0' }}>
              Force specific task types to a fixed tier, bypassing the classifier.
            </p>
          </div>
          <Button kind="primary" size="sm" renderIcon={Add} onClick={() => setAddOpen(true)}
            style={{ flexShrink: 0 }}>
            Add override
          </Button>
        </div>

        {Object.keys(overrides).length === 0 ? (
          <div style={{ padding: '1.5rem 1rem', textAlign: 'center' }}>
            <p style={{ fontSize: '0.8125rem', color: '#6f6f6f', margin: 0 }}>
              No overrides — the classifier handles all routing decisions
            </p>
            <p style={{ fontSize: '0.75rem', color: '#a8a8a8', margin: '0.25rem 0 0' }}>
              Add an override to pin a task type (e.g. <code style={{ background: '#f4f4f4', padding: '0 3px', borderRadius: 2 }}>code_generation</code>) to a specific tier
            </p>
          </div>
        ) : (
          Object.entries(overrides).map(([task, tier], idx, arr) => {
            const tm = TIER_META[tier]
            return (
              <div key={task} style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.5rem 1rem',
                borderBottom: idx < arr.length - 1 ? '1px solid #f4f4f4' : undefined,
                background: tm?.bg ?? '#fff',
              }}>
                <ChevronRight size={12} style={{ color: tm?.color ?? '#525252', flexShrink: 0 }} />
                <code style={{
                  flex: 1, fontSize: '0.8125rem', color: '#161616',
                  background: 'transparent', fontFamily: 'monospace',
                }}>
                  {task}
                </code>
                <TierBadge tier={tier} size="sm" />
                <Button kind="ghost" size="sm" hasIconOnly renderIcon={TrashCan}
                  iconDescription="Remove" style={{ color: '#da1e28' }}
                  onClick={() => setOverrides(o => { const n = { ...o }; delete n[task]; return n })} />
              </div>
            )
          })
        )}
      </div>

      {/* ── Save + Preview bar ── */}
      <div style={{
        display: 'flex', gap: '0.75rem', alignItems: 'center',
        padding: '0.875rem 1rem',
        background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
      }}>
        <Button kind="primary" renderIcon={Save} size="sm"
          onClick={() => saveMut.mutate({ scorer_weights: weights, task_overrides: overrides })}
          disabled={saveMut.isPending}>
          {saveMut.isPending ? 'Saving…' : 'Save configuration'}
        </Button>
        <Button kind="ghost" size="sm"
          onClick={() => previewMut.mutate({ scorer_weights: weights, task_overrides: overrides })}
          disabled={previewMut.isPending}>
          {previewMut.isPending ? 'Running…' : 'Preview routing diff'}
        </Button>
        <span style={{ fontSize: '0.75rem', color: '#8d8d8d' }}>
          Changes take effect immediately after saving
        </span>
      </div>

      {/* Preview result */}
      {preview && (
        <div style={{
          background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
          marginTop: '1rem', overflow: 'hidden',
        }}>
          <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid #f4f4f4' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#161616' }}>
              Preview — tier distribution shift
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
            {['tier_distribution_before', 'tier_distribution_after'].map((key, ki) => (
              <div key={key} style={{
                padding: '0.75rem 1rem',
                borderRight: ki === 0 ? '1px solid #f4f4f4' : undefined,
              }}>
                <p style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#525252', margin: '0 0 0.5rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  {ki === 0 ? 'Before' : 'After'}
                </p>
                {Object.entries((preview as any)[key] ?? {}).map(([tier, count]) => (
                  <div key={tier} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <TierBadge tier={tier} size="sm" />
                    <span style={{ fontSize: '0.8125rem', color: '#161616', fontWeight: 600 }}>{String(count)} req</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          {(preview as any).replayed && (
            <div style={{ padding: '0.5rem 1rem', borderTop: '1px solid #f4f4f4' }}>
              <span style={{ fontSize: '0.75rem', color: '#525252' }}>
                Replayed {(preview as any).replayed} recent requests
              </span>
            </div>
          )}
        </div>
      )}

      {/* Add override modal */}
      <Modal open={addOpen} modalHeading="Add task type override"
        primaryButtonText="Add" secondaryButtonText="Cancel"
        onRequestClose={() => setAddOpen(false)} onRequestSubmit={addOverride}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem 0' }}>
          <p style={{ fontSize: '0.8125rem', color: '#525252', margin: 0 }}>
            Enter the exact <code style={{ background: '#f4f4f4', padding: '0 4px', borderRadius: 2 }}>task_type</code> string
            returned by your classifier, and choose the tier it should always route to.
          </p>
          <TextInput id="new-task" labelText="Task type"
            placeholder="e.g. code_generation"
            helperText="Must match the task_type value your classifier emits"
            value={newTask}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewTask(e.target.value)} />
          <Select id="new-tier" labelText="Route to tier" value={newTier}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setNewTier(e.target.value)}>
            {TIERS.map(t => (
              <SelectItem key={t} value={t}
                text={`${t.charAt(0).toUpperCase() + t.slice(1)} — ${TIER_META[t].hint}`} />
            ))}
          </Select>
        </div>
      </Modal>
    </div>
  )
}

// ── Classifiers tab ───────────────────────────────────────────────────────────

const CLASSIFIER_META: Record<string, { icon: string; speed: string; detail: string; steps: string[] }> = {
  regex: {
    icon: '⚡',
    speed: '~1 ms',
    detail: 'Pure lexical pattern matching — scans for question words, code blocks, reasoning markers, and tool signatures. Zero external dependencies, zero latency cost.',
    steps: [
      'Always runs on every request as the baseline',
      'Produces a task_type and complexity_tier from pattern rules',
      'Cannot be disabled — it is the permanent fallback',
    ],
  },
  mbert: {
    icon: '🧠',
    speed: '~120 ms',
    detail: 'Semantic NLI re-ranking using a multilingual BERT cross-encoder from HuggingFace. Much more accurate than regex for ambiguous or mixed-language prompts.',
    steps: [
      'Runs after Regex when set as active',
      'Computes a confidence score for each candidate class',
      'Overrides the regex result only when confidence ≥ threshold',
      'Requires: pip install transformers torch',
    ],
  },
  laya: {
    icon: '🔬',
    speed: '~250 ms',
    detail: 'IBM Laya is a compact model fine-tuned for instruction classification. It predicts both task_type and complexity_tier in a single forward pass on CPU.',
    steps: [
      'Runs after Regex when set as active',
      'Returns task_type AND tier in one call — most complete output',
      'Most accurate for IBM-domain workloads',
      'Requires: pip install laya',
    ],
  },
}

function ClassifiersTab() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['classifiers'], queryFn: api.classifiers.list,
  })
  const [notice, setNotice] = useState<string | null>(null)

  // Single action: "Set as active" both enables and activates in one click.
  // No separate Enabled toggle needed.
  const activate = useMutation({
    mutationFn: (name: string) => api.classifiers.update(name, { set_active: true, enabled: true }),
    onSuccess: (_r, name) => {
      qc.invalidateQueries({ queryKey: ['classifiers'] })
      setNotice(`"${name}" set as active classifier.`)
    },
  })

  return (
    <div style={{ paddingTop: '1.25rem' }}>
      {notice && (
        <InlineNotification kind="success" title={notice} hideCloseButton={false}
          onClose={() => setNotice(null)} style={{ marginBottom: '1rem' }} />
      )}
      {isLoading && <LoadingState />}
      {error && <ErrorState message={String(error)} />}

      {data && (
        <>

          {/* Classifier cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {data.classifiers.map((clf: ClassifierInfo) => {
              const isAvailable = clf.available !== false
              const cm = CLASSIFIER_META[clf.name] ?? { icon: '•', speed: '', detail: '', steps: [] }
              const borderColor = clf.active ? '#0f62fe' : !isAvailable ? '#f1c21b' : '#e0e0e0'
              const bg = clf.active ? '#edf5ff' : '#fff'

              return (
                <div key={clf.name} style={{
                  border: `2px solid ${borderColor}`,
                  borderRadius: 8, background: bg, overflow: 'hidden',
                }}>
                  {/* ── Top bar ── */}
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '0.75rem',
                    padding: '0.75rem 1rem',
                    borderBottom: '1px solid #e0e0e040',
                  }}>
                    {/* Icon */}
                    <div style={{
                      width: 32, height: 32, borderRadius: 6, flexShrink: 0,
                      background: clf.active ? '#0f62fe' : '#f4f4f4',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '1rem',
                    }}>
                      {cm.icon}
                    </div>

                    {/* Name + badges + meta */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                        <strong style={{ fontSize: '0.9375rem', color: '#161616' }}>{clf.display_name}</strong>
                        {clf.active && <Tag type="blue" size="sm"><Checkmark size={10} style={{ marginRight: 2 }} />active</Tag>}
                        {clf.name === 'regex' && <Tag type="gray" size="sm">always on</Tag>}
                        {!isAvailable && <Tag type="warm-gray" size="sm">not installed</Tag>}
                      </div>
                      <div style={{ display: 'flex', gap: '0.875rem', marginTop: '0.125rem', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '0.6875rem', color: '#525252', display: 'flex', alignItems: 'center', gap: 3 }}>
                          <Time size={11} />{cm.speed}
                        </span>
                        <span style={{ fontSize: '0.6875rem', color: '#525252' }}>
                          Source: <code style={{ background: '#e0e0e0', padding: '0 3px', borderRadius: 2 }}>
                            {clf.model_source}
                          </code>
                        </span>
                      </div>
                    </div>

                    {/* Controls — single action only */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', flexShrink: 0 }}>
                      {!clf.active && clf.name !== 'regex' && isAvailable && (
                        <Button kind="primary" size="sm"
                          onClick={() => activate.mutate(clf.name)} disabled={activate.isPending}>
                          Set as active
                        </Button>
                      )}
                      {!clf.active && clf.name === 'regex' && (
                        <Button kind="ghost" size="sm"
                          onClick={() => activate.mutate(clf.name)} disabled={activate.isPending}>
                          Revert to regex
                        </Button>
                      )}
                      {!clf.active && clf.name !== 'regex' && !isAvailable && (
                        <Tag type="warm-gray" size="sm">Install to enable</Tag>
                      )}
                    </div>
                  </div>

                  {/* ── Detail body — description only, no bullet steps ── */}
                  <div style={{ padding: '0.625rem 1rem' }}>
                    <p style={{ fontSize: '0.8125rem', color: '#525252', margin: 0, lineHeight: 1.6 }}>
                      {cm.detail}
                    </p>
                  </div>

                  {/* Install hint */}
                  {!isAvailable && clf.install_hint && (
                    <div style={{
                      padding: '0.5rem 1rem',
                      borderTop: '1px solid #f1c21b40',
                      background: '#fef3cd',
                      display: 'flex', alignItems: 'center', gap: '0.5rem',
                    }}>
                      <span style={{ fontSize: '0.75rem', color: '#8a6914' }}>Install with:</span>
                      <code style={{ fontSize: '0.75rem', color: '#8a6914', fontFamily: 'monospace' }}>
                        {clf.install_hint}
                      </code>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

// ── Smart Routing tab (security + context-reuse flags) ────────────────────────

function SmartRoutingTab() {
  const qc = useQueryClient()
  const { data: flags, isLoading, error } = useQuery({
    queryKey: ['security-flags'],
    queryFn: api.security.flags,
  })

  // Local draft — mirrors flags from server; user edits are applied here first.
  const [draft, setDraft] = useState<Partial<SecurityFlagsResponse>>({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const saveMut = useMutation({
    mutationFn: async (patch: Record<string, unknown>) => {
      setSaving(true)
      setSaved(false)
      const res = await fetch('/admin/api/security/flags', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('mp_token') || ''}`
        },
        body: JSON.stringify(patch),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['security-flags'] })
      setDraft({})
      setSaving(false)
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    },
    onError: () => setSaving(false),
  })

  function applyPatch(patch: Partial<SecurityFlagsResponse>) {
    setDraft(prev => ({ ...prev, ...patch }))
  }

  function commitSave() {
    saveMut.mutate({ ...flags, ...draft } as Record<string, unknown>)
  }

  const hasDraft = Object.keys(draft).length > 0

  if (isLoading) return <LoadingState />
  if (error) return <ErrorState message={String(error)} />
  if (!flags) return null

  // Merge server flags with local draft edits for rendering
  const merged: SecurityFlagsResponse = { ...flags, ...draft }

  const SectionHeader = ({ children }: { children: React.ReactNode }) => (
    <h4 style={{
      fontSize: '0.8125rem', fontWeight: 700, color: '#525252',
      textTransform: 'uppercase', letterSpacing: '0.06em',
      margin: '1.5rem 0 0.75rem', borderTop: '1px solid #e0e0e0', paddingTop: '1.25rem',
    }}>{children}</h4>
  )

  const Hint = ({ children }: { children: React.ReactNode }) => (
    <p style={{ margin: '0.25rem 0 0', fontSize: '0.75rem', color: '#525252', lineHeight: 1.5 }}>{children}</p>
  )

  return (
    <div style={{ paddingTop: '1.25rem', maxWidth: 760 }}>

      {/* Save bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.75rem',
        background: hasDraft ? '#fff8e1' : '#f4f4f4',
        border: `1px solid ${hasDraft ? '#f0c800' : '#e0e0e0'}`,
        borderRadius: 6, padding: '0.625rem 1rem', marginBottom: '1.25rem',
      }}>
        <span style={{ flex: 1, fontSize: '0.8125rem', color: hasDraft ? '#6b4e00' : '#525252' }}>
          {hasDraft ? 'You have unsaved changes.' : saved ? '✓ Saved.' : 'Adjust settings below and save.'}
        </span>
        {saving && <InlineLoading description="Saving…" />}
        {!saving && (
          <Button size="sm" kind={hasDraft ? 'primary' : 'ghost'} onClick={commitSave} disabled={!hasDraft}>
            Save changes
          </Button>
        )}
      </div>

      {/* ── Cache-Aware Routing ──────────────────────────────────────────── */}
      <SectionHeader>Cache-Aware Routing</SectionHeader>
      <p style={{ margin: '0 0 0.75rem', fontSize: '0.8125rem', color: '#525252' }}>
        Controls how the gateway preserves KV-cache locality when selecting deployments. Two levels
        are available. <strong>Session affinity (L1)</strong> — the highest ROI approach — routes
        all requests from the same session to the same endpoint once enough context has accumulated,
        eliminating repeated prefill costs. <strong>Content-hash routing (L3)</strong> concentrates
        requests sharing the same retrieved-document prefix on one endpoint, turning independent
        cache misses into cache hits for RAG workloads.
      </p>

      <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: '1rem', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>Session affinity (Level 1)</div>
            <Hint>
              Routes every request from the same session to its warm deployment once accumulated
              tokens exceed the threshold. Eliminates prefill cost on all turns after the first.
            </Hint>
          </div>
          <Toggle
            id="cache-session-affinity-enabled"
            labelA="Off" labelB="On"
            toggled={merged.cache_session_affinity_enabled ?? true}
            onToggle={(v: boolean) => applyPatch({ cache_session_affinity_enabled: v })}
            size="sm"
          />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.25rem', color: '#525252' }}>
            Min tokens before affinity engages
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Slider
              id="cache-affinity-tokens"
              min={0} max={4000} step={100}
              value={merged.cache_session_affinity_min_tokens ?? 500}
              onChange={({ value }: { value: number }) => applyPatch({ cache_session_affinity_min_tokens: value })}
              hideTextInput
              disabled={!(merged.cache_session_affinity_enabled ?? true)}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: 600, minWidth: 48 }}>
              {merged.cache_session_affinity_min_tokens ?? 500}
            </span>
          </div>
          <Hint>
            Requests with fewer accumulated session tokens proceed through normal tier scoring.
            Below 200 is not recommended — short sessions have low cache value.
          </Hint>
        </div>
      </div>

      <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: '1rem', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>Content-hash routing (Level 3)</div>
            <Hint>
              For RAG workloads: routes requests sharing the same retrieved-document hash to the
              same deployment so the document prefix cache is shared across all queries.
            </Hint>
          </div>
          <Toggle
            id="cache-content-hash-enabled"
            labelA="Off" labelB="On"
            toggled={merged.cache_content_hash_routing_enabled ?? true}
            onToggle={(v: boolean) => applyPatch({ cache_content_hash_routing_enabled: v })}
            size="sm"
          />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.25rem', color: '#525252' }}>
            Max sticky sessions per content hash
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Slider
              id="cache-hash-max-sticky"
              min={1} max={32} step={1}
              value={merged.cache_content_hash_max_sticky ?? 8}
              onChange={({ value }: { value: number }) => applyPatch({ cache_content_hash_max_sticky: value })}
              hideTextInput
              disabled={!(merged.cache_content_hash_routing_enabled ?? true)}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: 600, minWidth: 48 }}>
              {merged.cache_content_hash_max_sticky ?? 8}
            </span>
          </div>
          <Hint>
            Once this many sessions are sticky to one deployment for the same hash, new sessions
            fall through to normal routing. Prevents one hot document from starving other candidates.
          </Hint>
        </div>
      </div>

      {/* ── Context Reuse Scoring ────────────────────────────────────────── */}
      <SectionHeader>Context Reuse Scoring</SectionHeader>
      <p style={{ margin: '0 0 0.75rem', fontSize: '0.8125rem', color: '#525252' }}>
        After affinity routing, remaining candidates are soft-re-ranked by estimated KV/prefix cache
        locality. This is a soft signal — it boosts cache-warm candidates but cannot override the
        security hard-filter or affinity routing.
      </p>

      <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: '1rem', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>Context reuse re-ranking</div>
            <Hint>Boost cache-warm candidates in the final sort before strategy selection.</Hint>
          </div>
          <Toggle
            id="context-reuse-enabled"
            labelA="Off" labelB="On"
            toggled={merged.context_reuse_enabled ?? true}
            onToggle={(v: boolean) => applyPatch({ context_reuse_enabled: v })}
            size="sm"
          />
        </div>
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.25rem', color: '#525252' }}>
            Reuse weight (max score bonus)
          </div>
          <Slider
            id="context-reuse-weight"
            min={0} max={0.5} step={0.01}
            value={merged.context_reuse_weight ?? 0.2}
            onChange={({ value }: { value: number }) => applyPatch({ context_reuse_weight: value })}
            hideTextInput
            disabled={!(merged.context_reuse_enabled ?? true)}
          />
          <Hint>
            0.10–0.30 is the typical range. Higher values prefer cache locality more aggressively
            (current: <strong>{(merged.context_reuse_weight ?? 0.2).toFixed(2)}</strong>).
          </Hint>
        </div>
        <div>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.25rem', color: '#525252' }}>
            Token saturation threshold
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Slider
              id="context-reuse-token-threshold"
              min={500} max={8000} step={250}
              value={merged.context_reuse_token_threshold ?? 2000}
              onChange={({ value }: { value: number }) => applyPatch({ context_reuse_token_threshold: value })}
              hideTextInput
              disabled={!(merged.context_reuse_enabled ?? true)}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: 600, minWidth: 48 }}>
              {merged.context_reuse_token_threshold ?? 2000}
            </span>
          </div>
          <Hint>
            Accumulated tokens above this threshold give the full reuse bonus. Below it, the bonus
            scales proportionally.
          </Hint>
        </div>
      </div>

      {/* ── Cache Mode ───────────────────────────────────────────────────── */}
      <SectionHeader>Cache Economics Mode</SectionHeader>
      <p style={{ margin: '0 0 0.75rem', fontSize: '0.8125rem', color: '#525252' }}>
        Controls how the gateway weighs the cost of switching away from the session's warm model.
      </p>
      <Select
        id="cache-mode-select"
        labelText="Cache mode"
        value={merged.cache_mode ?? 'soft_preference'}
        onChange={(e: React.ChangeEvent<HTMLSelectElement>) => applyPatch({ cache_mode: e.target.value })}
      >
        <SelectItem value="soft_preference" text="Soft preference — stay if context > 2 000 tokens (heuristic)" />
        <SelectItem value="switch_cost" text="Switch cost — USD-based economics (penalises switching based on token price)" />
      </Select>
    </div>
  )
}

export default function RoutingConfigPage() {
  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 0.25rem', color: '#161616' }}>
          Routing &amp; Classification
        </h2>
        <p style={{ fontSize: '0.875rem', color: '#525252', margin: 0 }}>
          Choose how requests are classified and configure the model selection rules.
        </p>
      </div>

      <Tabs>
        <TabList aria-label="Routing and classification tabs">
          <Tab>Classifiers</Tab>
          <Tab>Routing Rules</Tab>
          <Tab>Smart Routing</Tab>
        </TabList>
        <TabPanels>
          <TabPanel><ClassifiersTab /></TabPanel>
          <TabPanel><RoutingTab /></TabPanel>
          <TabPanel><SmartRoutingTab /></TabPanel>
        </TabPanels>
      </Tabs>
    </div>
  )
}
