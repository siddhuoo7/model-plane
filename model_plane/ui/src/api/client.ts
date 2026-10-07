/**
 * Typed API client for the Model Plane Admin API.
 * All calls are relative to /admin/api/ so they work behind both the
 * FastAPI static mount and the Vite dev-server proxy.
 *
 * Session token is stored in localStorage under 'mp_token'.
 * All authenticated calls inject it as Authorization: Bearer <token>.
 */

const BASE = '/admin/api'

// ── Session token helpers ──────────────────────────────────────────────────

export function getSessionToken(): string | null {
  return typeof localStorage !== 'undefined' ? localStorage.getItem('mp_token') : null
}

export function setSessionToken(token: string): void {
  localStorage.setItem('mp_token', token)
}

export function clearSessionToken(): void {
  localStorage.removeItem('mp_token')
}

// ── Fetch wrapper ──────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit, withAuth = true): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (withAuth) {
    const token = getSessionToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  const res = await fetch(`${BASE}${path}`, {
    headers: { ...headers, ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  // 204 No Content
  if (res.status === 204) return undefined as unknown as T
  return res.json() as Promise<T>
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string
  timestamp: number
  routing_mode: string
  ml_routing_enabled: boolean
  total_deployments: number
  healthy_deployments: number
  buffer_size: number
  buffer_capacity: number
}

export interface TrafficResponse {
  window_hours: number
  total_requests: number
  total_cost_usd: number
  total_input_tokens: number
  total_output_tokens: number
  by_provider: Record<string, { requests: number; cost_usd: number; input_tokens: number; output_tokens: number }>
  by_tier: Record<string, { requests: number; cost_usd: number }>
}

export interface RequestRecord {
  id: string
  timestamp: number
  deployment: string
  provider: string
  tier: string
  task_type: string
  tenant_id: string | null
  input_tokens: number
  output_tokens: number
  cost_usd: number
  latency_ms: number
  score_breakdown?: Record<string, number>
  // security fields (filled by Stage 0.5 SecurityGuard)
  security_sensitivity?: string
  security_blocked?: string[]
  context_reuse_score?: number
}

export interface SecuritySummaryResponse {
  enabled: boolean
  policy_matrix: Array<Record<string, boolean | string>>
  provider_trust_map: Record<string, string>
  trust_overrides: Record<string, string>
  sensitivity_counts: Record<string, number>
  blocked_counts: Record<string, number>
  recent_events: Array<{
    timestamp: number | null
    request_id: string | null
    sensitivity: string
    blocked: string[]
    provider: string | null
    deployment: string | null
    context_reuse_score: number | null
  }>
  total_requests_sampled: number
}

export interface SecurityFlagsResponse {
  security_routing_enabled: boolean
  context_reuse_enabled: boolean
  context_reuse_weight: number
  context_reuse_token_threshold: number
  cache_mode: string
  compression_enabled: boolean
  compression_token_threshold: number
  // Cache-aware routing level flags
  cache_session_affinity_enabled: boolean
  cache_session_affinity_min_tokens: number
  cache_content_hash_routing_enabled: boolean
  cache_content_hash_max_sticky: number
}

export interface RequestsResponse {
  count: number
  total_buffered: number
  records: RequestRecord[]
}

export interface CostRow {
  provider: string
  tier: string
  task_type: string
  requests: number
  cost_usd: number
  input_tokens: number
  output_tokens: number
}

export interface CostResponse {
  window_hours: number
  total_cost_usd: number
  rows: CostRow[]
  by_task_type: Record<string, { requests: number; cost_usd: number }>
}

export interface DeploymentConfig {
  name: string
  litellm_model: string
  provider: string
  tier: string
  max_tokens: number
  context_window: number
  capabilities: string[]
  input_per_mtok_usd: number
  output_per_mtok_usd: number
  healthy: boolean
  credential_ok?: boolean
}

export interface CatalogResponse {
  deployments: DeploymentConfig[]
  count: number
  total_in_yaml: number
  uncredentialed_hidden: number
}

export interface RoutingConfig {
  scorer_weights: Record<string, number>
  task_overrides: Record<string, string>
  task_deployment_overrides: Record<string, unknown>
  tenant_policies: Record<string, unknown>
  ml_guards: Record<string, number>
  laya_tier_thresholds: Record<string, number>
}

export interface SimulateResponse {
  dry_run: boolean
  classifier_warning: string | null
  router: string
  request_id: string
  selected_model: string | null
  tier: string | null
  task_type: string | null
  routing_stage: string | null
  score: number | null
  pipeline_steps: {
    features?: {
      total_tokens: number
      user_messages: number
      has_tools: boolean
      has_images: boolean
      language_hint: string | null
      dimensions: Record<string, number>
    }
    classifier?: {
      task_type: string | null
      classifier_tier: string | null
      confidence: number | null
      source: string
      fired_signals: Record<string, number>
      all_signals: Record<string, number>
    }
    scorer?: {
      raw_score: number | null
      scorer_tier: string | null
      scorer_confidence: number | null
      dimension_scores: Record<string, number>
    }
    blend?: {
      scorer_tier: string | null
      classifier_tier: string | null
      blended_tier: string | null
    }
    ml?: {
      enabled: boolean
      predicted_tier: string | null
      confidence: number
      active: boolean
      threshold: number
    }
    policy_applied?: {
      task_override_used: boolean
      resolved_tier: string | null
    }
    winner?: {
      model: string | null
      tier: string | null
      stage: string | null
      confidence: number
      reasoning: string | null
    }
  }
  candidates: {
    model: string
    provider: string
    tier: string
    input_per_mtok: number
    capabilities: string[]
  }[]
  cost: {
    input_per_mtok_usd: number
    estimated_input_cost_usd: number
  }
  features: {
    last_message_chars: number
    estimated_tokens: number
    message_count: number
    has_tools: boolean
  }
  // Phase 7 — security, cache, compression traces
  security?: {
    enabled: boolean
    sensitivity: string
    pii_types: string[]
    secret_types: string[]
    blocked_providers: string[]
    audit_summary: string
    candidates_removed: number
  }
  cache_routing?: {
    enabled: boolean
    session_id: string | null
    context_reuse_score: number
    turn_index: number
    cache_warm: boolean
    prefix_hash: string | null
    routing_reason: string
  }
  compression?: {
    profile: string
    tokens_before: number
    tokens_after: number
    savings_pct: number
    cache_aware_skip: boolean
    cache_aware_threshold: number
  }
}

export interface ClassifierInfo {
  name: string
  display_name: string
  model_source: string
  estimated_latency_ms: number
  description: string
  active: boolean
  enabled: boolean
  available: boolean
  install_hint: string | null
}

export interface ClassifiersResponse {
  classifiers: ClassifierInfo[]
  active_classifier: string
}

export interface MlModelInfo {
  model_path: string
  model_loaded: boolean
  model_type: string | null
  class_names: string[]
  training_rows: number | null
  feature_count: number | null
  accuracy: number | null
  feature_importances?: Record<string, number>
  last_modified_utc: string | null
  last_trained?: string | null
  error?: string
}

export interface TrainingDataResponse {
  total_rows: number
  columns: string[]
  task_type_counts: Record<string, number>
  tier_counts: Record<string, number>
  rows: Record<string, string>[]
  file_path: string
}

export interface UploadResult {
  accepted_rows: number
  total_rows: number
  merge: boolean
  task_type_counts: Record<string, number>
  tier_counts: Record<string, number>
  file_path: string
}

export interface MlStatusResponse {
  // backward-compat flat fields (tier model)
  model_type: string
  model_path: string
  last_trained: string | null
  accuracy: number | null
  training_rows: number | null
  feature_importances: Record<string, number> | null
  status: string
  // per-model detail
  tier_model: MlModelInfo
  task_model: MlModelInfo
}

export interface Tenant {
  tenant_id: string
  min_tier: string | null
  max_tier: string | null
  preferred_deployments: string[]
  blocked_deployments: string[]
}

export interface AlertConfig {
  alert_id: string
  metric: string
  threshold: number
  window_minutes: number
  notify_url: string | null
}

export interface ProviderInfo {
  name: string
  configured: boolean
  available_models: number
  cred_source?: string | null
  endpoint?: string | null
  health?: string
}

// ── API calls ──────────────────────────────────────────────────────────────

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),

  traffic: (windowHours = 24) =>
    apiFetch<TrafficResponse>(`/traffic?window_hours=${windowHours}`),

  requests: (params?: { limit?: number; provider?: string; tier?: string; task_type?: string; tenant_id?: string }) => {
    const q = new URLSearchParams()
    if (params?.limit) q.set('limit', String(params.limit))
    if (params?.provider) q.set('provider', params.provider)
    if (params?.tier) q.set('tier', params.tier)
    if (params?.task_type) q.set('task_type', params.task_type)
    if (params?.tenant_id) q.set('tenant_id', params.tenant_id)
    return apiFetch<RequestsResponse>(`/requests?${q}`)
  },

  cost: (windowHours = 24) =>
    apiFetch<CostResponse>(`/cost?window_hours=${windowHours}`),

  catalog: {
    list: (all?: boolean) => apiFetch<CatalogResponse>(`/catalog${all ? '?all=true' : ''}`),
    create: (dep: Partial<DeploymentConfig>) =>
      apiFetch<DeploymentConfig>('/catalog', { method: 'POST', body: JSON.stringify(dep) }),
    update: (name: string, dep: Partial<DeploymentConfig>) =>
      apiFetch<DeploymentConfig>(`/catalog/${name}`, { method: 'PUT', body: JSON.stringify(dep) }),
    // Dedicated toggle — sends only { name, healthy } so no other fields are touched
    setHealthy: (name: string, healthy: boolean) =>
      apiFetch<DeploymentConfig>(`/catalog/${name}/healthy`, {
        method: 'PATCH',
        body: JSON.stringify({ healthy }),
      }),
    delete: (name: string) => apiFetch<void>(`/catalog/${name}`, { method: 'DELETE' }),
    reload: () => apiFetch<unknown>('/catalog/reload', { method: 'POST' }),
  },

  routing: {
    // Server wraps the config in { config: <yaml dict> } — unwrap so callers
    // always work with the flat RoutingConfig dict (no double-nesting on save).
    getConfig: () =>
      apiFetch<{ config: RoutingConfig }>('/routing/config').then((r) => r.config ?? (r as unknown as RoutingConfig)),
    setConfig: (cfg: Partial<RoutingConfig>) =>
      apiFetch<{ status: string }>('/routing/config', { method: 'PUT', body: JSON.stringify(cfg) }),
    preview: (cfg: Partial<RoutingConfig>) =>
      apiFetch<unknown>('/routing/preview', { method: 'POST', body: JSON.stringify(cfg) }),
  },

  simulate: (messages: { role: string; content: string }[], options?: Record<string, unknown>) =>
    apiFetch<SimulateResponse>('/simulate', {
      method: 'POST',
      body: JSON.stringify({ messages, options }),
    }),

  classifiers: {
    list: () => apiFetch<ClassifiersResponse>('/classifiers'),
    update: (name: string, body: { enabled?: boolean; set_active?: boolean }) =>
      apiFetch<unknown>(`/classifiers/${name}`, { method: 'PUT', body: JSON.stringify(body) }),
  },

  ml: {
    status: () => apiFetch<MlStatusResponse>('/ml/status'),
    retrain: (mode: 'tier' | 'task' | 'deployment' = 'tier', modelType = 'gradient_boost') =>
      apiFetch<{ job_id: string; status: string }>(
        `/ml/retrain?mode=${mode}&model_type=${modelType}`,
        { method: 'POST' },
      ),
    activate: (modelPath: string) =>
      apiFetch<unknown>(`/ml/activate?model_path=${encodeURIComponent(modelPath)}`, { method: 'POST' }),
    trainingData: (limit = 200, offset = 0) =>
      apiFetch<TrainingDataResponse>(`/ml/training-data?limit=${limit}&offset=${offset}`),
    uploadTrainingData: (file: File, merge = true) => {
      const form = new FormData()
      form.append('file', file)
      return apiFetch<UploadResult>(`/ml/training-data/upload?merge=${merge}`, {
        method: 'POST',
        body: form,
        headers: {},   // let browser set multipart/form-data with boundary
      })
    },
    sampleDataUrl: (fmt: 'csv' | 'jsonl') =>
      `${BASE}/ml/training-data/sample?fmt=${fmt}`,
  },

  tenants: {
    list: () => apiFetch<{ tenants: Tenant[] }>('/tenants'),
    create: (t: Omit<Tenant, 'tenant_id'> & { tenant_id?: string }) =>
      apiFetch<Tenant>('/tenants', { method: 'POST', body: JSON.stringify(t) }),
    update: (id: string, t: Partial<Tenant>) =>
      apiFetch<Tenant>(`/tenants/${id}`, { method: 'PUT', body: JSON.stringify(t) }),
    delete: (id: string) => apiFetch<void>(`/tenants/${id}`, { method: 'DELETE' }),
  },

  alerts: {
    list: () => apiFetch<{ alerts: AlertConfig[] }>('/alerts'),
    create: (a: Omit<AlertConfig, 'alert_id'> & { alert_id?: string }) =>
      apiFetch<AlertConfig>('/alerts', { method: 'POST', body: JSON.stringify(a) }),
    update: (id: string, a: Partial<AlertConfig>) =>
      apiFetch<AlertConfig>(`/alerts/${id}`, { method: 'PUT', body: JSON.stringify(a) }),
    delete: (id: string) => apiFetch<void>(`/alerts/${id}`, { method: 'DELETE' }),
  },

  providers: {
    list: () => apiFetch<{ providers: ProviderInfo[] }>('/providers'),
    test: (name: string, creds?: Record<string, string>) =>
      apiFetch<unknown>(
        `/providers/${name}/test`,
        { method: 'POST', body: JSON.stringify(creds ?? {}) },
        true,
      ),
    available: (provider: string) => apiFetch<unknown>(`/providers/${provider}/available`),
    // Credential management (auth-gated via session token)
    credStatus: () => apiFetch<{ providers: Record<string, Record<string, boolean>> }>('/settings/providers', undefined, true),
    credValues: (provider: string) =>
      apiFetch<{ provider: string; values: Record<string, string> }>(`/settings/providers/${provider}/values`, undefined, true),
    saveCreds: (provider: string, creds: Record<string, string>) =>
      apiFetch<{ provider: string; configured: Record<string, boolean> }>(
        `/settings/providers/${provider}`,
        { method: 'PUT', body: JSON.stringify(creds) },
        true,
      ),
    clearCreds: (provider: string) =>
      apiFetch<void>(`/settings/providers/${provider}`, { method: 'DELETE' }, true),
  },

  auth: {
    hasUsers:  () => apiFetch<{ has_users: boolean }>('/auth/has-users'),
    signup: (email: string, password: string, name?: string) =>
      apiFetch<{ token: string; email: string; name: string }>(
        '/auth/signup', { method: 'POST', body: JSON.stringify({ email, password, name }) },
      ),
    login: (email: string, password: string) =>
      apiFetch<{ token: string; email: string; name: string }>(
        '/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) },
      ),
    me: () => apiFetch<{ email: string; name: string; id: string }>('/auth/me', undefined, true),
  },

  settings: {
    listApiKeys: () =>
      apiFetch<{ keys: ApiKeyEntry[] }>('/settings/api-keys', undefined, true),
    createApiKey: (label?: string) =>
      apiFetch<{ id: string; label: string; key: string; prefix: string }>(
        '/settings/api-keys',
        { method: 'POST', body: JSON.stringify({ label }) },
        true,
      ),
    deleteApiKey: (id: string) =>
      apiFetch<void>(`/settings/api-keys/${id}`, { method: 'DELETE' }, true),
  },

  security: {
    summary: () => apiFetch<SecuritySummaryResponse>('/security/summary'),
    flags:   () => apiFetch<SecurityFlagsResponse>('/security/flags'),
  },

  compression: {
    getConfig: () => apiFetch<CompressionConfig>('/compression/config'),
    setConfig: (body: Partial<CompressionConfig>) =>
      apiFetch<{ status: string; config: CompressionConfig }>(
        '/compression/config', { method: 'PUT', body: JSON.stringify(body) }
      ),
    metrics: () => apiFetch<CompressionMetrics>('/compression/metrics'),
  },
}

export interface CompressionConfig {
  enabled: boolean
  profile: string
  token_threshold: number
  cache_aware_gate_enabled: boolean
  cache_aware_threshold: number
  available_profiles?: string[]
}

export interface CompressionMetrics {
  total_requests_sampled: number
  compressed_count: number
  cache_skip_count: number
  total_tokens_before: number
  total_tokens_after: number
  total_savings_tokens: number
  overall_savings_pct: number
  profile_distribution: Record<string, number>
}

export interface ApiKeyEntry {
  id: string
  label: string
  prefix: string
  created_at: number
}
