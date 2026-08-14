# Model Plane

A production-grade **model control plane and smart router** built on [LiteLLM](https://docs.litellm.ai/). Sits between your agent/harness and any LLM provider, adding intelligent routing, token compression, session cache continuity, tenant policy enforcement, and Prometheus observability — all in a single Python service.

```
Claude Code / Cursor / Custom Agent
              │
              ▼
   ┌──────────────────────────┐
   │  Model Plane (FastAPI)   │
   │                          │
   │  Protocol adapter        │
   │  Feature extractor       │
   │  Task classifier         │  ← 14-task taxonomy
   │  14-dim weighted scorer  │
   │  Policy engine           │  ← per-tenant rules
   │  Routing strategy        │  ← custom + ML shadow
   │  Compression hooks       │  ← tool/conv/code
   │  Validator + escalation  │
   │  LiteLLM executor        │
   │  Prometheus metrics      │
   └──────────────────────────┘
         │          │          │
      OpenAI   Anthropic   watsonx.ai / local vLLM
```

## Features

| Layer | What it does |
|---|---|
| **Task classifier** | 14 task types (code_debugging, summarization, mathematical_reasoning, …) from lexical signals |
| **14-dim scorer** | Weighted complexity score → `simple / medium / complex / reasoning` tier |
| **Custom routing strategy** | Scores every candidate deployment on tier fit, capability affinity, cost, context usage, cache warmth |
| **Policy engine** | Per-tenant provider allow/deny, require_local, regulated task overrides, token budgets |
| **ML shadow (Phase 4)** | scikit-learn classifier trained on `data/training_data.csv`; shadow mode until `ml_routing_enabled=true` |
| **Cache/session plugin (Phase 5)** | Tracks session prefix hash; avoids breaking warm cache on model switches |
| **Compression (Phase 6)** | `tool_output_compaction`, `conversation_summary`, `code_context_reduction`, `auto` |
| **Validation + escalation (Phase 7)** | JSON schema check, tool-call check, quality floor → retry/escalate/fallback |
| **Observability** | Prometheus counters/histograms, structured JSON logs (structlog), training CSV recorder |

## Quick start

```bash
# 1. Clone and install
git clone <repo>
cd model-plane
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Fill in at least one provider key, e.g.:
#   OPENAI_API_KEY=sk-...
#   MODEL_PLANE_API_KEYS=my-secret-key

# 3. Run
model-plane
# or
uvicorn model_plane.app:app --reload --port 8080
```

The server exposes an **OpenAI-compatible** API at `http://localhost:8080`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `POST` | `/v1/messages` | Anthropic-compatible messages |
| `GET` | `/v1/models` | List all deployments |
| `GET` | `/v1/routing/status` | Routing engine status |
| `POST` | `/v1/admin/reload` | Hot-reload model catalog |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe (requires ≥1 healthy deployment) |
| `GET` | `/metrics` | Prometheus metrics |

### Request headers (optional)

| Header | Purpose |
|---|---|
| `Authorization: Bearer <key>` | API key authentication |
| `X-Tenant-Id` | Activates per-tenant policy rules |
| `X-Session-Id` | Enables cache-continuity routing |
| `X-Agent-Type` | Logged for observability |

### Response headers

| Header | Value |
|---|---|
| `X-Request-Id` | Unique request identifier |
| `X-Deployment` | Logical deployment alias selected |
| `X-Task-Type` | Classified task type |
| `X-Routing-Source` | `custom_plugin` / `cache_anchor` / `default` |

## Configuration

### `config/models.yaml` — model catalog

```yaml
deployments:
  - name: local-small
    litellm_model: openai/gpt-3.5-turbo
    provider: local
    api_base: "${LOCAL_SMALL_API_BASE}"
    tier: small
    capabilities: [text]
    fallback_to: [local-medium]
    context_limit: 8192
    cost_per_1k_input: 0.0
    cost_per_1k_output: 0.0
```

Put credentials in environment variables, never in YAML. The `api_key_env` field names the env var to read.

### `config/routing.yaml` — routing rules

Key sections:

- `scorer_weights` — override the 14 dimension weights
- `tier_defaults` — default deployment per tier
- `task_overrides` — preferred deployments for specific task types
- `tenant_policies` — per-tenant provider allow/deny, regulated tasks, cost caps
- `compression_profiles` — task-type → compression profile mapping
- `escalation` — max retries and fallback chain for validation failures

### Environment variables

All settings are prefixed `MODEL_PLANE_`. See [`.env.example`](.env.example) for the full list.

Key variables:

```bash
MODEL_PLANE_ROUTING_MODE=custom_plugin   # fixed|auto_router|custom_plugin|hybrid
MODEL_PLANE_COMPRESSION_ENABLED=true
MODEL_PLANE_ML_ROUTING_ENABLED=false     # enable after training
MODEL_PLANE_DEFAULT_MODEL=local-medium
```

## Project layout

```
model_plane/
├── adapters/           # FastAPI routers, LiteLLM executor, auth
│   ├── chat_router.py      /v1/chat/completions
│   ├── messages_router.py  /v1/messages (Anthropic compat)
│   ├── admin_router.py     /v1/models, /healthz, /readyz
│   ├── executor.py         LiteLLM acompletion + retry + fallback
│   └── auth.py             Bearer token validation
├── classifier/         # Feature extraction + task classification
│   ├── features.py         14-signal RequestFeatures extractor
│   ├── classifier.py       Rule-based task → TaskType
│   └── taxonomy.py         TaskType enum + ComplexityTier enum
├── scorer/             # 14-dimension weighted scorer
│   └── scorer.py           → ScorerResult (raw_score, tier, confidence)
├── policy/             # Tenant policy engine
│   └── engine.py           Provider allow/deny, regulated tasks, token budgets
├── routing/            # Routing pipeline
│   ├── context.py          RoutingContext dataclass (shared request state)
│   ├── pipeline.py         Orchestrates all stages → RoutingDecision
│   └── strategy.py         Per-candidate scoring and selection
├── ml/                 # Local ML recommender (Phase 4)
│   ├── recommender.py      scikit-learn wrapper + TrainingDataRecorder
│   └── trainer.py          Offline training script
├── cache/              # Session + cache continuity (Phase 5)
│   └── session.py          SessionCache (in-proc or Redis), CacheAwareRouter
├── compression/        # Token compression (Phase 6)
│   └── processor.py        4 profiles + auto-selector
├── validation/         # Response validation + escalation (Phase 7)
│   └── validator.py        JSON schema, tool-call, quality checks
├── hooks/              # LiteLLM pre/post-call hooks
│   └── litellm_hooks.py    PreCallHook (compression), PostCallHook (metrics/training)
├── observability/      # Prometheus metrics
│   └── metrics.py
├── registry/           # Model catalog loader
│   └── catalog.py          ModelCatalog, DeploymentConfig
├── config.py           # Pydantic settings (env + .env file)
├── logging_setup.py    # structlog JSON/console config
├── app.py              # FastAPI app factory + lifespan
└── main.py             # CLI entry point (uvicorn)
config/
├── models.yaml         # Deployment catalog (15 example deployments)
└── routing.yaml        # Routing rules, weights, policies, compression
tests/unit/             # 37 unit tests (scorer, classifier, compression,
                        #   validation, catalog, policy)
```

## Phase roadmap

| Phase | Status | What's enabled |
|---|---|---|
| 0 — Vanilla gateway | ✅ | FastAPI proxy, `/v1/chat/completions`, `/v1/messages`, auth, streaming |
| 1 — Model catalog | ✅ | YAML catalog, 4 providers, deployment pools, health tracking, fallback |
| 2 — Auto Router baseline | ✅ | Complexity tiers, tier mapping, routing source logging |
| 3 — Custom routing plugin | ✅ | Task classifier, 14-dim scorer, policy engine, custom strategy |
| 4 — Local ML plugin | ✅ | scikit-learn recommender, TrainingDataRecorder, offline trainer, shadow mode |
| 5 — Cache/session plugin | ✅ | Session state, prefix hash, cache-warm bonus, model-switch penalty |
| 6 — Compression | ✅ | tool_output_compaction, conversation_summary, code_context_reduction, auto |
| 7 — Validation + cascades | ✅ | JSON schema, tool-call, quality floor, escalation policy |
| 8 — Continuous optimisation | ✅ | Prometheus metrics, training CSV, shadow traffic fraction config |

## Training the ML recommender

Once you have collected routing data:

```bash
# Data is written automatically to data/training_data.csv
# by the post-call hook. When you have >= 20 rows:

python -m model_plane.ml.trainer \
  --data data/training_data.csv \
  --output models/router_ml.joblib \
  --model random_forest

# Enable in .env:
MODEL_PLANE_ML_ROUTING_ENABLED=true
```

## Running tests

```bash
pytest tests/unit/ --no-cov      # fast, no coverage
pytest tests/unit/               # with coverage report
```

## Deployment (Docker)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8080
CMD ["model-plane"]
```

```bash
docker build -t model-plane .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_PLANE_API_KEYS=my-key \
  model-plane
```

## Architecture decisions preserved

- **No LiteLLM internals modified.** All extension via public hooks (`CustomLogger`), executor wrapper, and YAML config.
- **Module boundaries are enforced**: `protocol adapter → feature extractor → task classifier → policy → routing strategy → compression → LiteLLM executor`.
- **ML is shadow-mode by default** (`ml_routing_enabled=false`) until you have sufficient training data and confidence.
- **Compression is pass-through by default** until routing decisions and evaluation logs are stable.
- **Session cache is opt-in** via `X-Session-Id` header — no global state without explicit session tracking.
