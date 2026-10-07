# Model Plane

A production-grade **model control plane and smart router** built on [LiteLLM](https://docs.litellm.ai/). Sits between your agent/harness and any LLM provider, adding a 7-step intelligent routing pipeline, tier classifier, multi-provider support, cache switch-cost economics, and a full Carbon Design System operator UI — all in a single Python service.

```
Claude Code / Cursor / Custom Agent
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Model Plane — FastAPI (port 8081)                     │
   │                                                        │
   │  POST /v1/chat/completions          ← OpenAI-compat    │
   │  POST /v1/messages                  ← Anthropic-compat │
   │  POST /v1/routing/explain           ← dry-run trace    │
   │  POST /v1/routing/feedback          ← feedback loop    │
   │  GET  /v1/routing/metrics           ← live dashboard   │
   │  GET  /admin/                       ← React Admin UI   │
   │  GET  /admin/api/*                  ← Admin REST API   │
   └────────────────────────────────────────────────────────┘
         │         │        │         │        │        │
     watsonx    OpenAI   Anthropic  Bedrock   Azure  Vertex AI
```

---

## Quick Start

### One command (API + UI dev server together)

```bash
# 1. Install Python deps
pip install -e ".[dev]"

# 2. Install JS deps (root + UI)
pnpm install

# 3. Copy and fill env
cp .env.example .env
# Minimum: set one provider key — watsonx is the default:
#   WATSONX_APIKEY=<key>
#   WATSONX_PROJECT_ID=<id>
#   WATSONX_URL=https://us-south.ml.cloud.ibm.com
#   MODEL_PLANE_API_KEYS=my-secret-key

# 4. Start everything
pnpm start
```

This starts two processes in parallel:
- **API** → `http://localhost:8081` (FastAPI with hot-reload)
- **UI dev server** → `http://localhost:5173/admin/` (Vite HMR, proxies API calls to :8081)

### Individual commands

| Command | What it does |
|---|---|
| `pnpm start` | Start API + UI dev server together |
| `pnpm run api` | Start only the FastAPI backend (port 8081) |
| `pnpm run ui` | Start only the Vite dev server (port 5173) |
| `pnpm run build` | Build the React UI bundle into `model_plane/ui/dist/` |
| `pnpm test` | Run all Python tests with coverage |
| `pnpm run test:fast` | Run Python tests without coverage (faster) |

### API only (no pnpm)

```bash
uvicorn model_plane.app:app --reload --port 8081
# or
model-plane --reload
```

---

## Admin UI

Open `http://localhost:8081/admin/` (production) or `http://localhost:5173/admin/` (dev) for the operator dashboard.

| Screen | URL | What it shows |
|---|---|---|
| Dashboard | `/admin/` | RPS, cost, tier distribution, deployment health grid — auto-refreshes every 10 s |
| Request Explorer | `/admin/requests` | Filterable, paginated routing decisions with score breakdown drawer |
| Cost Analytics | `/admin/cost` | Cost by provider / tier / task-type; configurable time window |
| Classifiers | `/admin/classifiers` | Enable / disable / switch active classifier (Regex / BERT / Laya) |
| Model Catalog | `/admin/catalog` | Full CRUD; 2-step "Add deployment" wizard with prices.yaml lookup |
| Routing Config | `/admin/routing` | Smart Router Rules view; scorer weight sliders; task override CRUD; live preview diff |
| Provider Management | `/admin/providers` | Credential forms for all 9 providers; connectivity test button |
| Security & Governance | `/admin/security` | Data-sensitivity matrix (Public/Internal/Confidential/Restricted), PII & secret detection |
| Compression | `/admin/compression` | Prompt compression profiles, token savings metrics, cache-aware gating |
| ML & Training | `/admin/ml` | Model status card; SSE retrain progress bar; feature importance chart |
| Tenant Policies | `/admin/tenants` | Per-tenant min/max tier, preferred/blocked deployment CRUD |
| Alerts | `/admin/alerts` | Threshold-based alert CRUD with webhook URL |
| Settings & Keys | `/admin/settings` | API key lifecycle management, runtime configuration, user profiles |

The Routing Simulator is accessible from every page via the **RouterWifi** button in the header — paste any prompt to see the full 7-step pipeline trace.

To rebuild the UI bundle after changing UI source:

```bash
pnpm run build
```

---

## Features

| Layer | What it does |
|---|---|
| **7-step routing pipeline** | Features → Classify → Score → Blend → ML → Policy → Strategy |
| **Tier classifier** | 3 backends: Regex (default), BERT NLI (`CLASSIFIER_MODEL=mbert`), Laya (`CLASSIFIER_MODEL=laya`), ML logistic regression (`CLASSIFIER_MODEL=ml_task`) |
| **15-dim scorer** | 14 base dims + optional `task_type_tier_signal` (dim 15); weighted sum → `simple / medium / complex / reasoning` tier |
| **Tier blending (Step 3.5)** | `max(scorer_tier, classifier_tier)` — prevents silent under-routing |
| **Active ML routing (Step 3.6)** | GradientBoosting tier model; escalates blended tier when confidence ≥ threshold |
| **ML task classifier (4.3)** | Separate logistic-regression model on 14 regex signals → task type; portable JSON weights, no sklearn at inference |
| **Multi-provider support** | 9 providers: watsonx, OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, Google Vertex AI, Cohere, Mistral, vLLM/local |
| **Price registry** | `config/prices.yaml` — auto-enriches `input_per_mtok_usd` / `output_per_mtok_usd` at catalog load time |
| **Cache switch-cost economics (4.1)** | USD reprefill cost model; re-ranks candidates by `(stay_cost − switch_cost)`; capped at ±0.30; flag-gated (`CACHE_MODE=switch_cost`) |
| **Security & Data Governance** | Hard-eligibility filter blocking high-sensitivity/PII requests from untrusted providers |
| **Context Reuse & Prefix Caching** | Level 1 session affinity & Level 3 content-hash routing for RAG/prompt caching optimization |
| **Policy engine** | Per-tenant provider allow/deny, require_local, regulated task overrides, token budgets |
| **Compression** | `tool_output_compaction`, `conversation_summary`, `code_context_reduction`, `auto` |
| **Validation + escalation** | JSON schema + tool-call checks + quality floor → retry/escalate/fallback |
| **Admin REST API** | 30+ endpoints: catalog CRUD, routing config, simulate, provider management, ML retrain + SSE, tenant/alert CRUD |
| **React Admin UI** | Carbon Design System; 10 pages; TanStack Query; SideNav shell; Simulator panel |
| **Observability** | Prometheus counters/histograms; structlog JSON; cost accumulator; request ring buffer |

---

## API Endpoints

### Routing (client-facing)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | ✓ | OpenAI-compatible chat completions |
| `POST` | `/v1/messages` | ✓ | Anthropic-compatible messages |
| `GET`  | `/v1/models` | — | List all catalog deployments |
| `GET`  | `/v1/routing/status` | — | Routing engine health snapshot |
| `POST` | `/v1/routing/explain` | ✓ | Dry-run pipeline trace (no LLM call) |
| `POST` | `/v1/routing/feedback` | ✓ | Submit user labels → similarity + CSV |
| `GET`  | `/v1/routing/metrics` | — | Live dashboard from training CSV |
| `GET`  | `/healthz` | — | Liveness probe |
| `GET`  | `/readyz` | — | Readiness probe |
| `GET`  | `/metrics` | — | Prometheus scrape endpoint |

### Admin API (`/admin/api/*`, optional Bearer auth)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/api/health` | Status + buffer size |
| `GET` | `/admin/api/traffic` | RPS, cost, tier/provider breakdown (windowed) |
| `GET` | `/admin/api/requests` | Ring-buffer routing decisions (filterable) |
| `GET` | `/admin/api/cost` | Per-provider/tier/task-type cost breakdown |
| `GET/POST/PUT/DELETE` | `/admin/api/catalog` | Deployment CRUD |
| `POST` | `/admin/api/catalog/reload` | Hot-reload catalog from YAML |
| `GET/PUT` | `/admin/api/routing/config` | Read/write routing config |
| `POST` | `/admin/api/routing/preview` | Preview config diff on buffered traffic |
| `POST` | `/admin/api/simulate` | Full pipeline trace for a test prompt |
| `GET` | `/admin/api/providers` | Provider status grid |
| `POST` | `/admin/api/providers/{name}/test` | Test provider connectivity |
| `GET` | `/admin/api/providers/{p}/available` | Models from prices.yaml for provider |
| `GET` | `/admin/api/classifiers` | Classifier status + active flag |
| `PUT` | `/admin/api/classifiers/{name}` | Enable/disable or set active classifier |
| `GET` | `/admin/api/ml/status` | Model path, accuracy, feature importances |
| `POST` | `/admin/api/ml/retrain` | Trigger background retrain; returns `job_id` |
| `GET` | `/admin/api/ml/retrain/{id}/stream` | SSE progress stream |
| `POST` | `/admin/api/ml/activate` | Swap active model path |
| `GET/POST/PUT/DELETE` | `/admin/api/tenants` | Tenant policy CRUD |
| `GET/POST/PUT/DELETE` | `/admin/api/alerts` | Alert config CRUD |

### Request headers

| Header | Purpose |
|---|---|
| `Authorization: Bearer <key>` | API key auth (routing endpoints) |
| `X-Tenant-Id` | Activates per-tenant policy rules |
| `X-Session-Id` | Enables cache-continuity / switch-cost routing |
| `X-Agent-Type` | Logged for observability |

### Response headers

| Header | Value |
|---|---|
| `X-Request-Id` | Unique request identifier |
| `X-Deployment` | Logical deployment alias selected |
| `X-Task-Type` | Classified task type |
| `X-Routing-Source` | `custom_plugin` / `ml_active` / `similarity` / `cache_anchor` / `default` |
| `X-Routing-Decision` | Human-readable routing explanation (inline) |

---

## Routing Pipeline (7 Steps)

| Step | Module | What it does |
|---|---|---|
| 1 | [`classifier/features.py`](model_plane/classifier/features.py) | Extract 14 (or 15) signals from raw messages |
| 2 | [`classifier/classifier.py`](model_plane/classifier/classifier.py) | Regex → `TaskType` + `ComplexityTier`; optionally augmented by BERT / Laya / ml_task |
| 2.5 | [`routing/similarity_router.py`](model_plane/routing/similarity_router.py) | Cosine fast-path (opt-in) |
| 3 | [`scorer/scorer.py`](model_plane/scorer/scorer.py) | 14-dim weighted sum → `raw_score` → tier |
| 3.5 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | Tier blending — `max(scorer_tier, classifier_tier)` |
| 3.6 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | Active ML — GradientBoosting escalates tier if confidence ≥ threshold |
| 4 | [`registry/catalog.py`](model_plane/registry/catalog.py) | Build healthy candidate pool |
| 5 | [`policy/engine.py`](model_plane/policy/engine.py) | Tenant allow/deny/regulated/budget filter |
| 6 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | Tier filter → task override → **switch-cost re-rank** (flag-gated) |
| 7 | [`routing/strategy.py`](model_plane/routing/strategy.py) | 5-criteria deployment scorer → winner |

---

## Configuration

### Environment Variables

All settings are `MODEL_PLANE_` prefixed unless noted. See [`.env.example`](.env.example) for the full list.

```bash
# Core
MODEL_PLANE_PORT=8081
MODEL_PLANE_ROUTING_MODE=custom_plugin   # fixed|auto_router|custom_plugin|hybrid
MODEL_PLANE_DEFAULT_MODEL=watsonx-granite-small

# ML routing
MODEL_PLANE_ML_ROUTING_ENABLED=true
MODEL_PLANE_ML_CONFIDENCE_THRESHOLD=0.65
MODEL_PLANE_ML_MODEL_PATH=models/router_ml_tier_v1.joblib
MODEL_PLANE_ML_OUTPUT_MODE=tier          # tier (default) | deployment (legacy)

# Classifier
MODEL_PLANE_CLASSIFIER_MODEL=regex       # regex | ml_task | mbert | laya
MODEL_PLANE_CLASSIFIER_MIN_CONFIDENCE=0.60
MODEL_PLANE_ML_TASK_CLASSIFIER_PATH=models/classifier_ml.json  # for ml_task mode

# Cache economics (Sub-Task 4.1)
MODEL_PLANE_CACHE_MODE=soft_preference   # soft_preference (default) | switch_cost

# Admin UI auth (leave unset for open dev mode)
ADMIN_API_KEY=my-admin-secret

# Provider credentials
WATSONX_APIKEY=<key>                     # note: APIKEY not API_KEY
WATSONX_PROJECT_ID=<id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
OPENAI_API_KEY=<key>
ANTHROPIC_API_KEY=<key>
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_REGION_NAME=us-east-1
AZURE_API_KEY=<key>
AZURE_API_BASE=https://<resource>.openai.azure.com
AZURE_API_VERSION=2024-02-01
COHERE_API_KEY=<key>
MISTRAL_API_KEY=<key>
```

### `config/models.yaml` — Model Catalog

```yaml
deployments:
  - name: watsonx-granite-large
    litellm_model: watsonx/meta-llama/llama-3-3-70b-instruct
    provider: watsonx
    tier: complex               # simple | medium | complex | reasoning
    capabilities: [text, code, reasoning, function-calling]
    context_window: 128000
    max_tokens: 4096
```

Credentials go in env vars, never in YAML. Provider adapters are auto-registered for: `watsonx`, `openai`, `anthropic`, `bedrock`, `azure`, `vertex_ai`, `cohere`, `mistral`, `vllm`, `local`.

### `config/routing.yaml` — Routing Rules

Key sections:

| Section | Purpose |
|---|---|
| `scorer_weights` | 14-dim weights (must sum to 1.0) |
| `task_overrides` | Task type → tier overrides (highest priority) |
| `task_deployment_overrides` | Task type → preferred deployment list |
| `ml_guards` | `upgrade_cap_conf`, `downgrade_veto_conf`, `skip_ml_classifier_conf` |
| `laya_tier_thresholds` | Float buckets for Laya score → tier mapping |
| `tenant_policies` | Per-tenant provider allow/deny and tier caps |

### `config/prices.yaml` — Price Registry

Copied from the strata reference. Used to auto-enrich `input_per_mtok_usd` / `output_per_mtok_usd` on catalog entries at load time. Powers the "Add deployment" wizard in the Admin UI.

---

## ML Models

### Tier Recommender (`models/router_ml_tier_v1.joblib`)

Predicts `ComplexityTier` from the 15-dim feature vector.

```bash
# Retrain from CLI
python -m model_plane.ml.trainer \
  --data model_plane/data/training_data.csv \
  --output models/router_ml_tier_v1.joblib \
  --model gradient_boost

# Or via Admin API
curl -X POST http://localhost:8081/admin/api/ml/retrain \
  -H "Authorization: Bearer $ADMIN_API_KEY"

# Watch SSE progress
curl http://localhost:8081/admin/api/ml/retrain/<job_id>/stream
```

### ML Task Classifier (`models/classifier_ml.json`)

Lightweight logistic regression on the 14 regex signal booleans → `TaskType`. Pure-Python at inference (no sklearn). Enable with `CLASSIFIER_MODEL=ml_task`.

```bash
# Train from Python
from model_plane.classifier.ml_task_classifier import train_and_save
train_and_save("model_plane/data/training_data.csv", "models/classifier_ml.json")
```

---

## Project Layout

```text
model-plane/
│
├── package.json                  ← root scripts: pnpm start / build / test
│
├── model_plane/
│   ├── app.py                    FastAPI factory; lifespan; static /admin mount
│   ├── config.py                 pydantic-settings (MODEL_PLANE_ prefix); production validation
│   ├── db.py                     SQLite persistence for API keys, users, and cost buckets
│   ├── runtime_overrides.py      Dynamic in-memory configuration overrides
│   ├── provider_creds.py         Provider credential management
│   ├── main.py                   CLI entry point
│   │
│   ├── adapters/
│   │   ├── chat_router.py        /v1/chat/completions; X-Routing-Decision header
│   │   ├── messages_router.py    /v1/messages (Anthropic-compat)
│   │   ├── admin_router.py       /v1/* legacy endpoints
│   │   ├── admin_api_router.py   /admin/api/* — 30+ admin endpoints
│   │   ├── settings_router.py    /admin/api/auth & /admin/api/settings endpoints
│   │   ├── executor.py           LiteLLM acompletion + fallback chain
│   │   ├── auth.py               API key / JWT
│   │   └── providers/            Provider adapter registry (watsonx, openai, bedrock, azure, vertex, vllm, cohere, mistral)
│   │       ├── __init__.py         _REGISTRY + get_adapter()
│   │       ├── base.py             ProviderAdapter protocol
│   │       ├── default.py          DefaultAdapter
│   │       ├── watsonx.py
│   │       ├── bedrock.py
│   │       ├── azure_openai.py
│   │       ├── vertex_ai.py
│   │       ├── cohere.py
│   │       ├── mistral.py
│   │       └── vllm.py
│   │
│   ├── classifier/
│   │   ├── features.py           RequestFeatures + extract_features(); 15 dims
│   │   ├── classifier.py         classify(); priority regex → TaskType
│   │   ├── taxonomy.py           TaskType enum, ComplexityTier enum
│   │   ├── base.py               ClassifierAdapter protocol
│   │   ├── factory.py            get_classifier(); regex|ml_task|mbert|laya
│   │   ├── bert_classifier.py    BertClassifierAdapter (ModernBERT cross-encoder)
│   │   ├── laya_classifier.py    LayaClassifier (local CPU inference)
│   │   └── ml_task_classifier.py MLTaskClassifier (logistic regression, JSON weights)
│   │
│   ├── scorer/
│   │   └── scorer.py             14/15-dim weighted scorer; exponential cost decay; tier-aware bonus
│   │
│   ├── routing/
│   │   ├── pipeline.py           run_routing_pipeline(); Steps 1–7; switch-cost re-rank
│   │   ├── security.py           SecurityGuard hard eligibility filter (sensitivity matrix)
│   │   ├── context_reuse.py      Context reuse & prefix caching score booster
│   │   ├── tier_resolver.py      TierResolver — picks best deployment for ML tier
│   │   ├── strategy.py           select_deployment(); 5-criteria scoring
│   │   ├── context.py            RoutingContext + RoutingDecision
│   │   └── similarity_router.py  SimilarityRouter (opt-in cosine index)
│   │
│   ├── ml/
│   │   ├── recommender.py        LocalMLRecommender; tier guard rails; blended tier
│   │   └── trainer.py            train() / train_model(); GradientBoosting + isotonic
│   │
│   ├── cache/
│   │   ├── session.py            SessionCache; warm_model_ids; accumulated_tokens
│   │   └── switch_cost.py        compute_switch_cost(); cache_score_adjustment(); ±0.30 cap
│   │
│   ├── hooks/
│   │   └── litellm_hooks.py      PostCallHook → ring buffer + cost accumulator
│   │
│   ├── observability/
│   │   ├── request_buffer.py     RequestRingBuffer (admin_buffer_size entries)
│   │   ├── cost_accumulator.py   CostAccumulator (hourly buckets)
│   │   └── metrics.py            Prometheus counters/histograms
│   │
│   ├── registry/
│   │   └── catalog.py            ModelCatalog; price enrichment from prices.yaml
│   │
│   ├── policy/
│   │   └── engine.py             PolicyEngine; per-tenant enforcement
│   │
│   ├── compression/
│   │   └── processor.py          4 compression profiles + cache-aware gate
│   │
│   ├── validation/
│   │   └── validator.py          JSON schema + tool-call + EscalationPolicy
│   │
│   └── ui/                       React Admin UI (Vite + Carbon Design System)
│       ├── src/
│       │   ├── App.tsx             SideNav shell + React Router
│       │   ├── AuthContext.tsx     JWT Auth state
│       │   ├── api/client.ts       Typed API client for all /admin/api/* calls
│       │   ├── components/         TierBadge, SimulatorPanel, LoadingState, ErrorState
│       │   └── pages/              12 pages (Dashboard, Requests, Cost, Classifiers,
│       │                           Catalog, Routing, Providers, Security, Compression,
│       │                           ML, Tenants, Alerts, Settings)
│       └── dist/                  Built bundle (served by FastAPI at /admin/)
│
├── config/
│   ├── models.yaml               Deployment catalog & trust levels
│   ├── routing.yaml              Scorer weights, task overrides, guard rails, tenant policies
│   └── prices.yaml               Per-model input/output $/1M token prices
│
├── models/
│   ├── router_ml_tier_v1.joblib  Active tier classifier (GradientBoosting, 4-class)
│   └── router_ml.joblib          Legacy deployment-name predictor (kept for rollback)
│
├── model_plane/data/
│   └── training_data.csv         Seed + live traffic rows for ML retraining
│
├── tests/unit/                   358 tests, all passing
│   ├── test_switch_cost.py         4.1 cache economics
│   ├── test_ml_task_classifier.py  4.3 ML task classifier
│   ├── test_admin_api_ml.py        3.3 ML admin endpoints
│   ├── test_admin_api_catalog.py   3.2 catalog/routing endpoints
│   ├── test_admin_api.py           3.1 admin foundation
│   ├── test_price_registry.py      3.0 prices.yaml integration
│   ├── test_tier_task_overrides.py 2.5 tier task overrides
│   ├── test_tier_resolver.py       2.3 tier resolver
│   ├── test_recommender.py         2.2 + 2.4 recommender + guard rails
│   ├── test_pipeline.py            core routing pipeline
│   └── ...                         (classifier, scorer, policy, compression, …)
│
└── docs/
    └── tier-classifier-implementation-plan.md   Implementation plan (all phases)
```

---

## Running Tests

```bash
pnpm test                     # full suite with coverage
pnpm run test:fast            # no coverage (faster CI)
python -m pytest tests/unit/test_switch_cost.py -v     # 4.1 only
python -m pytest tests/unit/test_ml_task_classifier.py # 4.3 only
```

Current: **306 tests, 0 failures**, 69% coverage.

---

## Phase Roadmap

| Phase | Sub-task | Status | What's in it |
|---|---|---|---|
| 1 | 1.1 | ✅ | Provider adapter registry (`providers/` package, `get_adapter()`) |
| 1 | 1.2 | ✅ | New provider adapters: Bedrock, Azure OpenAI, Vertex AI, Cohere, Mistral, vLLM |
| 1 | 1.3 | ✅ | 15th feature dimension (`task_type_tier_signal`) |
| 1 | 1.4 | ✅ | Exponential cost decay in scorer |
| 1 | 1.5 | ✅ | Tier-aware local/vLLM bonus in scorer |
| 1 | 1.6 | ✅ | `X-Routing-Decision` inline explanation header |
| 2 | 2.1 | ✅ | Classifier adapter protocol + BERT / Laya / factory |
| 2 | 2.2 | ✅ | Tier recommender retrain (output mode = tier) |
| 2 | 2.3 | ✅ | Tier resolver (ML tier → best deployment) |
| 2 | 2.4 | ✅ | ML guard rails (upgrade cap, downgrade veto, skipMl) |
| 2 | 2.5 | ✅ | Tier-based task overrides in pipeline + routing.yaml |
| 3 | 3.0 | ✅ | Price registry (`config/prices.yaml`, cost enrichment) |
| 3 | 3.1 | ✅ | Admin API foundation (ring buffer, cost accumulator, /admin/api/health, traffic, requests, cost) |
| 3 | 3.2 | ✅ | Catalog + routing config APIs, simulate, provider endpoints |
| 3 | 3.3 | ✅ | ML admin API: status, retrain, SSE stream, activate; tenant + alert CRUD |
| 3 | 3.4 | ✅ | React Admin UI — Dashboard, Request Explorer, Cost Analytics, Classifiers, Simulator |
| 3 | 3.5 | ✅ | React Admin UI — Catalog, Routing Config, Providers, ML & Training, Tenants, Alerts |
| 4 | 4.1 | ✅ | Cache switch-cost economics (`switch_cost.py`; `CACHE_MODE=switch_cost`) |
| 4 | 4.2 | ⏳ | Laya production cut-over *(requires 1 week shadow data — operational task, no code)* |
| 4 | 4.3 | ✅ | Separate ML task classifier (logistic regression, JSON weights, `CLASSIFIER_MODEL=ml_task`) |

---

## Deployment (Docker)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8081
CMD ["model-plane"]
```

```bash
docker build -t model-plane .
docker run -p 8081:8081 \
  -e WATSONX_APIKEY=<key> \
  -e WATSONX_PROJECT_ID=<id> \
  -e WATSONX_URL=https://us-south.ml.cloud.ibm.com \
  -e MODEL_PLANE_API_KEYS=my-key \
  model-plane
```

The built React UI bundle (`model_plane/ui/dist/`) is included in the image and served by FastAPI at `/admin/`.

---

## Architecture Decisions

- **No LiteLLM internals modified.** All extension via public hooks (`CustomLogger`), executor wrapper, and YAML config.
- **Feature-flagged everything.** Every behavioural change ships with a rollback env var; defaults restore prior behaviour.
- **ML is conservative.** Tier recommender can escalate but never downgrade. Shadow mode is the safe default until sufficient training data is accumulated.
- **Cache switch-cost is opt-in.** Default `CACHE_MODE=soft_preference` preserves the existing heuristic; `switch_cost` activates the USD model.
- **Admin API is auth-optional.** Leave `ADMIN_API_KEY` unset for open local development; set it in production.
- **React UI is pre-built.** `model_plane/ui/dist/` is committed so the service works without Node.js at runtime.
