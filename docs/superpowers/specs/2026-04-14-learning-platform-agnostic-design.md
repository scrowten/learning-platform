# Learning Platform — Domain-Agnostic Architecture Design

**Date:** 2026-04-14
**Status:** Approved

---

## Overview

Redesign the learning platform from a DS/ML-only static repo into a domain-agnostic, API-first learning platform. Content stays in git (markdown + notebooks); a separate platform API repo serves it to consumers (existing Flask+React webapp, agentic-ai backend, future clients).

---

## Scope

### Domains (v1)
- Data Science & ML Engineering (existing, keep and expand)
- Economics — Macro & Micro
- Stocks & Trading
- Software Engineering
- Finance & Accounting
- Open/pluggable — new domains added without code changes

### AI Features (v1)
- **Concept Q&A** — RAG over module content via agentic-ai model router
- **Auto-Generated Quizzes** — LLM generates MCQ from concepts.md, stored and served
- **Progress Tracking** — per-user module status + domain coverage dashboard
- **SOTA Alerts** — periodic worker flags stale sota.md content, drafts updates

Deferred to v2: Adaptive learning paths (personalized module sequencing).

---

## Architecture

### Two-Repo Split

**Repo 1: `learning-platform` (this repo) — Content**
- Source of truth for all learning content
- Human-authored markdown + Jupyter notebooks
- Domain registry and module frontmatter define the schema
- Git history is the audit log for content changes

**Repo 2: `learning-platform-api` (new) — Platform Engine**
- FastAPI serving content + AI features
- Sync worker (git → PostgreSQL)
- AI layer (RAG, quiz generation, SOTA checks)
- DB migrations and models
- Docker Compose for local development

### Data Flow

```
git push (content repo)
  → CI webhook
  → sync worker (learning-platform-api)
  → PostgreSQL (index + embeddings)
  → FastAPI
  → Flask+React webapp / agentic-ai backend
```

---

## Content Schema

### domain-registry.yaml (repo root)

Declares all available domains. Adding a new domain requires only a new entry here — no code changes in the platform repo.

```yaml
domains:
  - id: ds-ml
    name: "Data Science & ML Engineering"
    path: domains/ds-ml
    color: "#3b82f6"
    icon: "🧠"
    tags: [quantitative, programming, math]

  - id: economics
    name: "Economics"
    path: domains/economics
    color: "#10b981"
    icon: "📈"
    tags: [theory, policy, markets]

  - id: trading
    name: "Stocks & Trading"
    path: domains/trading
    color: "#f59e0b"
    icon: "📊"
    tags: [quantitative, markets, finance]

  - id: software-eng
    name: "Software Engineering"
    path: domains/software-eng
    color: "#8b5cf6"
    icon: "⚙️"
    tags: [programming, systems, architecture]

  - id: finance
    name: "Finance & Accounting"
    path: domains/finance
    color: "#ef4444"
    icon: "💰"
    tags: [theory, markets, valuation]
```

### Module README.md Frontmatter

Every module in every domain uses the same frontmatter schema. The sync worker is domain-blind — it reads fields, not folder names.

```yaml
---
id: supervised-learning
domain: ds-ml
title: "Supervised Learning"
difficulty: beginner-intermediate   # beginner | beginner-intermediate | intermediate | advanced | research
tags: [regression, classification, sklearn]
prerequisites:
  - linear-algebra
  - probability-statistics
estimated_hours: 8
last_reviewed: 2026-04-01
sota_topics:
  - gradient boosting dominance on tabular data
  - AutoML (H2O, AutoSklearn)
---
```

### Module Directory Structure (Universal Template)

```
<module-slug>/
├── README.md          ← frontmatter lives here
├── concepts.md        ← theory, models, derivations
├── cheatsheet.md      ← quick reference
├── sota.md            ← current field state / active debates
└── notebooks/         ← optional (quantitative domains)
    ├── 01_model.ipynb
    └── 02_real_data.ipynb
```

Notebooks are optional per domain. Economics/finance modules may use case studies or problem sets instead. The directory template is consistent; file types adapt to the domain.

### Repo Directory Structure (Post-Migration)

```
learning-platform/
├── domain-registry.yaml
├── domains/
│   ├── ds-ml/           ← migrated from existing 00–04 structure
│   │   ├── prerequisites/
│   │   ├── ml-fundamentals/
│   │   ├── deep-learning/
│   │   ├── llms-genai/
│   │   └── mlops/
│   ├── economics/
│   ├── trading/
│   ├── software-eng/
│   └── finance/
├── docs/
│   └── superpowers/specs/
├── frameworks/          ← keep as-is
├── projects/            ← keep as-is
└── README.md            ← update to reflect multi-domain platform
```

---

## Database Schema (PostgreSQL)

### Sync-Populated Tables

**`domains`** — from domain-registry.yaml
```sql
id           VARCHAR PRIMARY KEY
name         VARCHAR NOT NULL
path         VARCHAR NOT NULL
color        VARCHAR
icon         VARCHAR
tags         TEXT[]
created_at   TIMESTAMPTZ DEFAULT now()
```

**`modules`** — from README.md frontmatter
```sql
id               VARCHAR PRIMARY KEY  -- slug
domain_id        VARCHAR REFERENCES domains(id)
title            VARCHAR NOT NULL
difficulty       VARCHAR
tags             TEXT[]
estimated_hours  INT
git_path         VARCHAR
last_reviewed    DATE
sota_topics      TEXT[]
synced_at        TIMESTAMPTZ DEFAULT now()
```

**`module_prerequisites`** — prerequisite graph
```sql
module_id   VARCHAR REFERENCES modules(id)
prereq_id   VARCHAR REFERENCES modules(id)
PRIMARY KEY (module_id, prereq_id)
```

**`content_chunks`** — for RAG (Concept Q&A)
```sql
id           UUID PRIMARY KEY DEFAULT gen_random_uuid()
module_id    VARCHAR REFERENCES modules(id)
source_file  VARCHAR        -- 'concepts.md', 'cheatsheet.md', etc.
chunk_index  INT
content      TEXT
embedding    VECTOR(1536)   -- pgvector; dimension matches chosen embedding model
synced_at    TIMESTAMPTZ DEFAULT now()
```

### Runtime Tables

**`quiz_items`** — auto-generated quizzes
```sql
id             UUID PRIMARY KEY DEFAULT gen_random_uuid()
module_id      VARCHAR REFERENCES modules(id)
question       TEXT NOT NULL
options        JSONB          -- { "a": "...", "b": "...", "c": "...", "d": "..." }
correct_answer VARCHAR
explanation    TEXT
difficulty     VARCHAR
generated_at   TIMESTAMPTZ DEFAULT now()
```

**`user_progress`** — progress tracking
```sql
user_id       VARCHAR
module_id     VARCHAR REFERENCES modules(id)
status        VARCHAR        -- not_started | in_progress | done | skipped
started_at    TIMESTAMPTZ
completed_at  TIMESTAMPTZ
quiz_score    FLOAT
PRIMARY KEY (user_id, module_id)
```

### Ops Table

**`sota_reviews`** — SOTA alert tracking
```sql
id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
module_id       VARCHAR REFERENCES modules(id)
status          VARCHAR        -- current | stale | flagged
flagged_topics  TEXT[]
checked_at      TIMESTAMPTZ DEFAULT now()
draft_update    TEXT           -- AI-drafted suggestion for content owner
```

---

## API Surface (learning-platform-api)

All responses use the standard envelope:
```json
{ "success": true, "data": {}, "error": null, "meta": { "total": 0, "page": 1, "limit": 20 } }
```

### Content
```
GET  /domains                                → list all domains
GET  /domains/:id/modules                    → modules in domain (filter: difficulty, tags)
GET  /modules/:id                            → module detail + prerequisites
GET  /modules/:id/prerequisites              → prerequisite graph
GET  /search?q=&domain=&difficulty=&tags=    → full-text + tag search
```

### Concept Q&A
```
POST /modules/:id/ask                        → { question } → RAG answer
```

### Quizzes
```
GET  /modules/:id/quiz                       → quiz items for module
POST /modules/:id/quiz/generate              → trigger async quiz generation
POST /modules/:id/quiz/:item_id/answer       → { answer } → { correct, score, explanation }
```

### Progress
```
GET  /users/:user_id/progress                → all module statuses
PUT  /users/:user_id/progress/:module_id     → { status } → update
GET  /users/:user_id/dashboard               → domain coverage + streaks
```

### SOTA
```
GET  /modules/:id/sota-status                → { status, flagged_topics, draft_update }
POST /admin/sota/check                       → trigger SOTA review worker
```

### Sync
```
POST /sync/trigger                           → webhook from content repo CI
GET  /sync/status                            → last sync time + errors
```

---

## Integration Points

### agentic-ai backend
- Concept Q&A (`POST /modules/:id/ask`) delegates to agentic-ai's model router for RAG inference
- Quiz generation (`POST /modules/:id/quiz/generate`) uses agentic-ai for LLM calls
- SOTA worker uses agentic-ai for summarisation and draft generation
- Embedding generation during sync uses agentic-ai's embedding endpoint

### Flask+React webapp
- Queries `learning-platform-api` for content, progress, and AI features
- Passes `user_id` from its own auth system — the platform has no auth of its own
- No changes needed to the webapp's auth layer

---

## Implementation Phases

### Phase 1 — Content Migration
- Restructure this repo: `domains/` layout + `domain-registry.yaml`
- Add frontmatter to all existing DS/ML modules
- Scaffold empty domain directories for economics, trading, software-eng, finance
- Update README to reflect multi-domain platform

### Phase 2 — Platform API Repo
- Create `learning-platform-api` repo
- DB schema + migrations (Alembic)
- Sync worker: clone content repo → parse frontmatter → upsert to DB
- Basic content endpoints: domains, modules, search, prerequisites

### Phase 3 — AI Features
- pgvector + embedding pipeline in sync worker
- Concept Q&A endpoint (RAG via agentic-ai)
- Quiz generation + answer scoring
- SOTA review worker (periodic, cron-triggered)

### Phase 4 — Progress & Dashboard
- `user_progress` table + endpoints
- Dashboard endpoint (domain coverage, streaks)
- Integration test with Flask+React webapp

---

## Open Questions
- Embedding model: use agentic-ai's configured model (Claude or local), or a dedicated embedding model (text-embedding-3-small)?
- CI trigger: GitHub Actions webhook, or polling on a schedule?
- SOTA worker frequency: daily? weekly per domain?
