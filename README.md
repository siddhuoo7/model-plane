# Model Plane

A production-grade **model control plane and smart router** built on [LiteLLM](https://docs.litellm.ai/). Sits between your agent/harness and any LLM provider, adding a 7-step intelligent routing pipeline, token compression, session cache continuity, tenant policy enforcement, active ML routing, and Prometheus observability — all in a single Python service.

```
Claude Code / Cursor / Custom Agent
              │
              ▼
   ┌──────────────────────────────────────┐
   │  Model Plane — FastAPI (port 8081)   │
   │                                      │
   │  POST /v1/chat/completions           │  ← OpenAI-compatible
   │  POST /v1/messages                   │  ← Anthropic-compatible
   │  POST /v1/routing/explain            │  ← dry-run pipeline trace
   │  POST /v1/routing/feedback           │  ← user labels → ML/similarity
   │  GET  /v1/routing/metrics            │  ← live dashboard from CSV
   │  GET  /v1/routing/status             │
   │  POST /v1/admin/reload               │  ← hot-reload catalog
   │  POST /v1/admin/retrain              │  ← background ML retrain
   └──────────────────────────────────────┘
         │              │              │
     watsonx.ai      OpenAI       Anthropic / local vLLM
```

---

## Features

| Layer | What it does |
|---|---|
| **7-step routing pipeline** | Feature extraction → classification → scoring → tier blending → active ML → policy → strategy |
| **Task classifier** | 14 task types (code_debugging, summarization, mathematical_reasoning, …) from lexical signals |
| **14-dim scorer** | Weighted complexity score → `simple / medium / complex / reasoning` tier |
| **Tier blending (Step 3.5)** | `max(scorer_tier, classifier_tier)` fixes two-brain disagreement on keyword-sparse prompts |
| **Active ML routing (Step 3.6)** | GradientBoosting + isotonic calibration; escalates tier when confidence ≥ threshold |
| **Similarity fast-path (Step 2.5)** | Cosine index (sentence-transformers); short-circuits to best past deployment when enabled |
| **Custom routing strategy** | 5-criteria per-candidate scorer: tier fit, capability affinity, cost, context usage, cache warmth |
| **Policy engine** | Per-tenant provider allow/deny, require_local, regulated task overrides, token budgets |
| **Cache/session continuity** | Tracks session prefix hash; avoids breaking warm cache on model switches |
| **Compression** | `tool_output_compaction`, `conversation_summary`, `code_context_reduction`, `auto` |
| **Validation + escalation** | JSON schema check, tool-call check, quality floor → retry/escalate/fallback |
| **Feedback loop** | `POST /v1/routing/feedback` updates similarity index + appends labelled rows to training CSV |
| **Live metrics dashboard** | `GET /v1/routing/metrics` — request counts, latency, cost, validation/escalation/fallback rates |
| **Background retrain** | `POST /v1/admin/retrain` spawns a daemon thread; hot-swaps the in-process ML model on completion |
| **Observability** | Prometheus counters/histograms, structlog JSON logs, auto-appending training CSV |

---

## Quick start

```bash
# 1. Clone and install
git clone <repo>
cd model-plane
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Fill in at least one provider key — watsonx is the default provider:
#   WATSONX_APIKEY=<key>          ← note: WATSONX_APIKEY, not WATSONX_API_KEY
#   WATSONX_PROJECT_ID=<id>
#   WATSONX_URL=https://us-south.ml.cloud.ibm.com
#   MODEL_PLANE_API_KEYS=my-secret-key

# 3. Run
model-plane --reload
# or
uvicorn model_plane.app:app --reload --port 8081
```

The server exposes an **OpenAI-compatible** API at `http://localhost:8081`.

---

## API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | ✓ | OpenAI-compatible chat completions |
| `POST` | `/v1/messages` | ✓ | Anthropic-compatible messages |
| `GET`  | `/v1/models` | — | List all catalog deployments |
| `GET`  | `/v1/routing/status` | — | Routing engine health snapshot |
| `POST` | `/v1/routing/explain` | ✓ | Dry-run pipeline trace (no LLM call) |
| `POST` | `/v1/routing/feedback` | ✓ | Submit user labels → similarity + CSV |
| `GET`  | `/v1/routing/metrics` | — | Live dashboard from training CSV |
| `POST` | `/v1/admin/reload` | ✓ | Hot-reload model catalog from YAML |
| `POST` | `/v1/admin/retrain` | ✓ | Trigger background ML retrain |
| `GET`  | `/healthz` | — | Liveness probe |
| `GET`  | `/readyz` | — | Readiness probe (requires ≥1 healthy deployment) |
| `GET`  | `/metrics` | — | Prometheus scrape endpoint |

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
| `X-Routing-Source` | `custom_plugin` / `ml_active` / `similarity` / `cache_anchor` / `default` |

---

## Routing pipeline (7 steps)

| Step | Module | What it does |
|------|--------|---|
| 1 | [`classifier/features.py`](model_plane/classifier/features.py) | Extract 14 regex + tiktoken signals from raw messages |
| 2 | [`classifier/classifier.py`](model_plane/classifier/classifier.py) | Priority if-else regex → `TaskType` + `ComplexityTier` |
| 2.5 | [`routing/similarity_router.py`](model_plane/routing/similarity_router.py) | Fast-path cosine index (sentence-transformers, opt-in) |
| 3 | [`scorer/scorer.py`](model_plane/scorer/scorer.py) | 14-dim weighted sum → raw_score → tier |
| 3.5 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | **Tier blending** — `max(scorer_tier, classifier_tier)` |
| 3.6 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | **Active ML** — GradientBoosting escalates tier if confidence ≥ threshold |
| 4 | [`registry/catalog.py`](model_plane/registry/catalog.py) | Build pool of `catalog.all_healthy()` candidates |
| 5 | [`policy/engine.py`](model_plane/policy/engine.py) | Tenant allow/deny/regulated/budget filter |
| 6 | [`routing/pipeline.py`](model_plane/routing/pipeline.py) | Tier filter → task_override (highest priority) |
| 7 | [`routing/strategy.py`](model_plane/routing/strategy.py) | 5-criteria deployment scorer → winner |

### Tier blending (Step 3.5)

Fixes the "two-brain" problem: a short `"write a BST class"` prompt scores `SIMPLE` on the 14-dim scorer (no code blocks) but classifies as `CODE_GENERATION` (`COMPLEX`). Taking `max(scorer, classifier)` prevents silent under-routing.

### Active ML (Step 3.6)

When `MODEL_PLANE_ML_ROUTING_ENABLED=true` and a trained `.joblib` exists, the GradientBoosting + isotonic-calibrated classifier predicts a tier from the 14 feature dimensions. It can **escalate** the blended tier but never downgrade — conservative by design. The current shipped model (`models/router_ml.joblib`) achieves **99.4% accuracy** on the 800-row seed dataset.

### Similarity fast-path (Step 2.5)

When `MODEL_PLANE_SIMILARITY_ROUTING_ENABLED=true`, a sentence-transformers (`all-MiniLM-L6-v2`) cosine index is consulted before the full pipeline. If the top-k retrieval returns a high-confidence match (≥ threshold), routing short-circuits directly to that deployment. The index auto-updates every request and persists to `data/similarity_index.npy`.

---

## Configuration

### `config/models.yaml` — model catalog

Active watsonx deployments (verified against the live API):

| Name | LiteLLM model | Tier |
|------|--------------|------|
| `watsonx-granite-small` | `watsonx/ibm/granite-4-h-small` | small |
| `watsonx-granite-medium` | `watsonx/ibm/granite-4-h-small` | medium |
| `watsonx-granite-large` | `watsonx/meta-llama/llama-3-3-70b-instruct` | complex |
| `watsonx-mistral-medium` | `watsonx/mistralai/mistral-small-3-1-24b-instruct-2503` | medium |
| `watsonx-mistral-large` | `watsonx/mistral-large-2512` | complex |

OpenAI (`gpt-4o-mini`, `gpt-4o`, `o1-mini`) and Anthropic (`claude-3-haiku`, `claude-3-5-sonnet`, `claude-3-7-sonnet`) deployments are pre-configured and activate when their respective `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars are set. Local vLLM entries are commented out and ready to uncomment.

**YAML schema (key fields):**

```yaml
deployments:
  - name: watsonx-granite-large
    litellm_model: watsonx/meta-llama/llama-3-3-70b-instruct
    provider: watsonx
    api_key_env: WATSONX_APIKEY      # ← env var name, not value
    context_limit: 128000
    cost_per_1k_input: 0.0009
    cost_per_1k_output: 0.0009
    tier: complex                    # small | medium | complex | reasoning
    capabilities: [text, code, reasoning, function-calling, long-context]
    tags: [watsonx, sovereign]
    fallback_to: [watsonx-granite-medium]
    max_parallel_requests: 20
```

Put credentials in environment variables, never in YAML.

### `config/routing.yaml` — routing rules

Key sections:

- `scorer_weights` — override the 14 dimension weights (must sum to 1.0)
- `tier_defaults` — default deployment per tier
- `preferred_providers` — tie-break order (current: `[watsonx, local, anthropic, openai]`)
- `task_overrides` — highest-priority preferred deployments for specific task types
- `thresholds` — `ml_confidence_min`, `routing_confidence_low`, complexity score boundaries
- `tenant_policies` — per-tenant provider allow/deny, regulated tasks, cost caps
- `compression_profiles` — task-type → compression profile mapping
- `escalation` — max retries and ordered fallback chain for validation failures

### Environment variables

All settings are prefixed `MODEL_PLANE_`. See [`.env.example`](.env.example) for the full list.

```bash
# Core
MODEL_PLANE_PORT=8081
MODEL_PLANE_ROUTING_MODE=custom_plugin   # fixed|auto_router|custom_plugin|hybrid
MODEL_PLANE_DEFAULT_MODEL=watsonx-granite-small

# ML routing
MODEL_PLANE_ML_ROUTING_ENABLED=true      # false = shadow mode only
MODEL_PLANE_ML_CONFIDENCE_THRESHOLD=0.65 # below this, ML vote is ignored
MODEL_PLANE_ML_MODEL_PATH=models/router_ml.joblib

# Similarity routing (requires sentence-transformers installed)
MODEL_PLANE_SIMILARITY_ROUTING_ENABLED=false
MODEL_PLANE_SIMILARITY_CONFIDENCE_THRESHOLD=0.55
MODEL_PLANE_SIMILARITY_TOP_K=5
MODEL_PLANE_SIMILARITY_MIN_HITS=3

# Compression
MODEL_PLANE_COMPRESSION_ENABLED=true
MODEL_PLANE_COMPRESSION_TOKEN_THRESHOLD=6000

# Session cache
MODEL_PLANE_CACHE_TTL_SECONDS=3600
# MODEL_PLANE_REDIS_URL=redis://localhost:6379/0

# watsonx (note: WATSONX_APIKEY, not WATSONX_API_KEY)
WATSONX_APIKEY=<key>
WATSONX_PROJECT_ID=<id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **watsonx gotcha:** LiteLLM reads `WATSONX_APIKEY` (no underscore between API and KEY). `project_id` must also be passed as a direct kwarg to `acompletion()`.

---

## Project layout

```
model_plane/
├── app.py                    ← FastAPI app factory; _sync_provider_env() at startup
├── config.py                 ← pydantic-settings; AliasChoices for WATSONX_APIKEY
├── main.py                   ← CLI entry point (model-plane --reload)
├── adapters/
│   ├── chat_router.py        /v1/chat/completions; validation loop; shadow traffic
│   ├── messages_router.py    /v1/messages (Anthropic-compatible)
│   ├── admin_router.py       /v1/models, /routing/explain, /routing/feedback,
│   │                           /routing/metrics, /routing/status,
│   │                           /admin/reload, /admin/retrain, /healthz, /readyz
│   ├── executor.py           LiteLLM acompletion; fallback chain; shadow _shadow_call()
│   └── auth.py               API key / JWT verification
├── classifier/
│   ├── features.py           RequestFeatures + extract_features(); 14 regex signals
│   ├── classifier.py         classify(); priority regex → TaskType
│   └── taxonomy.py           TaskType enum, ComplexityTier enum, TASK_DEFAULT_TIER
├── scorer/
│   └── scorer.py             14-dim weighted scorer; ScorerResult
├── routing/
│   ├── pipeline.py           run_routing_pipeline(); Steps 1-7; tier blending; active ML
│   ├── strategy.py           select_deployment(); 5-criteria scoring; provider preference
│   ├── context.py            RoutingContext + RoutingDecision dataclasses
│   └── similarity_router.py  SimilarityRouter (cosine index; auto-updates per request)
├── ml/
│   ├── recommender.py        LocalMLRecommender; TrainingDataRecorder; provider-ordered tier lookup
│   └── trainer.py            train() / train_model(); GradientBoosting + isotonic calibration
├── cache/
│   └── session.py            SessionCache (in-proc or Redis); CacheAwareRouter
├── compression/
│   └── processor.py          4 profiles: passthrough / tool_compaction / summary / code_reduction
├── validation/
│   └── validator.py          JSON schema + tool-call checks + EscalationPolicy
├── hooks/
│   └── litellm_hooks.py      PreCallHook (compression) + PostCallHook (metrics/CSV/session)
├── observability/
│   └── metrics.py            Prometheus counters/histograms
├── policy/
│   └── engine.py             Per-tenant allow/deny/regulated/budget enforcement
└── registry/
    └── catalog.py            ModelCatalog loaded from YAML; by_tier(); mark_unhealthy()

config/
├── models.yaml               11 deployments (5 watsonx active, 3 OpenAI, 3 Anthropic)
└── routing.yaml              Scorer weights, tier_defaults, preferred_providers,
                                task_overrides (6 tasks), tenant_policies, escalation chain

models/
└── router_ml.joblib          GradientBoosting + isotonic; 99.4% accuracy; 800-row seed

model_plane/data/
└── training_data.csv         800 seed rows; real traffic auto-appended by PostCallHook

tests/unit/                   74 tests, all passing, 59% coverage
  test_catalog.py, test_classifier.py, test_compression.py, test_policy.py,
  test_scorer.py, test_validation.py, test_routing_explain.py,
  test_pipeline.py, test_strategy.py, test_recommender.py,
  test_shadow_traffic.py, test_metrics_endpoint.py
```

---

## Training the ML recommender

Training data is written automatically to `model_plane/data/training_data.csv` by `PostCallHook`. Once you have ≥ 20 rows:

```bash
python -m model_plane.ml.trainer \
  --data model_plane/data/training_data.csv \
  --output models/router_ml.joblib \
  --model gradient_boost          # gradient_boost | random_forest

# Enable active routing:
MODEL_PLANE_ML_ROUTING_ENABLED=true
```

Or trigger a live retrain (hot-swaps the in-process model on completion):

```bash
curl -X POST http://localhost:8081/v1/admin/retrain \
  -H "Authorization: Bearer $MODEL_PLANE_API_KEYS"
```

The shipped model (`models/router_ml.joblib`) was trained on 800 seed rows with GradientBoosting + isotonic calibration and achieves **99.4% cross-validated accuracy**.

---

## Routing explain (dry-run)

Send any chat body to `/v1/routing/explain` to see the full pipeline trace without triggering a real LLM call:

```bash
curl -X POST http://localhost:8081/v1/routing/explain \
  -H "Authorization: Bearer $MODEL_PLANE_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Debug this Python stack trace"}]}'
```

Response includes: selected deployment, routing source, confidence, task type, complexity tier, all 14 scorer dimensions, candidate list, fallback chain, cost rates, ML recommendation, and similarity routing status.

---

## Live metrics dashboard

```bash
curl http://localhost:8081/v1/routing/metrics
```

Returns (aggregated from training CSV):

- `total_requests`, `success_rate`
- `validation_failure_rate`, `escalation_rate`, `fallback_rate`
- `by_deployment` — requests, mean latency (ms), mean/total cost (USD) per deployment
- `by_task_type`, `by_tier`, `by_routing_source`

---

## Feedback loop

```bash
curl -X POST http://localhost:8081/v1/routing/feedback \
  -H "Authorization: Bearer $MODEL_PLANE_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "mp-abc123",
    "prompt": "Write a Python class for a BST",
    "correct_deployment": "watsonx-granite-large",
    "routing_was_correct": false,
    "task_type": "code_generation",
    "quality_score": 0.2
  }'
```

Each feedback call:
1. Updates the similarity cosine index (reinforces correct labels)
2. Appends a labelled row to the training CSV for the next retrain

---

## Similarity routing (optional)

Install `sentence-transformers` and enable:

```bash
pip install sentence-transformers
MODEL_PLANE_SIMILARITY_ROUTING_ENABLED=true
```

The `all-MiniLM-L6-v2` (384-dim) encoder builds an in-memory index of past prompt → deployment outcomes. At query time it runs cosine top-k retrieval with a weighted vote. The index persists to `data/similarity_index.npy` and is capped at 10,000 entries (oldest evicted). It must accumulate `SIMILARITY_MIN_HITS=3` entries before activating.

---

## Running tests

```bash
pytest tests/unit/ --no-cov      # fast
pytest tests/unit/               # with coverage report (59% current)
```

74 tests, all passing, lint clean (`ruff`).

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

---

## Phase roadmap

| Phase | Status | What's enabled |
|---|---|---|
| 0 — Vanilla gateway | ✅ | FastAPI proxy, `/v1/chat/completions`, `/v1/messages`, auth, streaming |
| 1 — Model catalog | ✅ | YAML catalog, 4 providers, deployment pools, health tracking, fallback |
| 2 — Auto Router baseline | ✅ | Complexity tiers, tier mapping, routing source logging |
| 3 — Custom routing plugin | ✅ | Task classifier, 14-dim scorer, policy engine, custom strategy |
| 4 — Local ML plugin | ✅ | GradientBoosting recommender, TrainingDataRecorder, offline trainer, active mode |
| 5 — Cache/session plugin | ✅ | Session state, prefix hash, cache-warm bonus, model-switch penalty |
| 6 — Compression | ✅ | tool_output_compaction, conversation_summary, code_context_reduction, auto |
| 7 — Validation + cascades | ✅ | JSON schema, tool-call, quality floor, escalation policy |
| 8 — Continuous optimisation | ✅ | Prometheus metrics, training CSV, shadow traffic, feedback loop, live retrain |
| 9 — Similarity routing | ✅ | sentence-transformers cosine index, opt-in fast-path, index persistence |

---

## Architecture decisions

- **No LiteLLM internals modified.** All extension via public hooks (`CustomLogger`), executor wrapper, and YAML config.
- **Module boundaries are enforced**: `protocol adapter → feature extractor → task classifier → policy → routing strategy → compression → LiteLLM executor`.
- **ML is conservative**: it can escalate tier but never downgrade. Shadow mode (`ml_routing_enabled=false`) is the safe default until you have sufficient training data.
- **Similarity routing is additive**: it only activates when the index has enough history and cosine similarity clears the threshold — it never degrades routing for cold-start cases.
- **Session cache is opt-in** via `X-Session-Id` header — no global state without explicit session tracking.
- **`RoutingContext` passes through LiteLLM hooks** as `metadata={"__routing_ctx": ctx}` — never as a top-level kwarg.
