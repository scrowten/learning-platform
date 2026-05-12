# Implementation Plan: Learning Platform UI + Full Local Testing

**Status:** In Progress  
**Created:** 2026-05-12  
**Scope:** Phase 0 (API enhancements) → Phase 7 (Testing)

---

## Architecture Overview

```
┌─────────────────────┐         ┌─────────────────────┐
│  learning-platform  │ git     │ learning-platform-  │
│  (content / md)     │◄────────┤ api (FastAPI)       │
│  /content-repo      │  reads  │ :8080               │
└─────────────────────┘         └──────────┬──────────┘
                                           │ HTTP
                                ┌──────────▼──────────┐
                                │ learning-platform-  │
                                │ ui (Next.js 15)     │
                                │ :3000               │
                                └─────────────────────┘
```

---

## Phase 0 — API Enhancements ⬜

Prerequisite for all UI work. Changes to `learning-platform-api`.

| Change | File | Why |
|--------|------|-----|
| `GET /modules/{id}/content?file=readme\|concepts\|cheatsheet` | `app/routers/content.py` | Stream raw markdown to UI |
| `GET /modules/{id}/notebooks` | `app/routers/content.py` | List `.ipynb` files in module dir |
| Add `category` field on Module response (derived from `git_path`) | `app/routers/content.py` | Group modules by `deep-learning`, `mlops`, etc. |
| CORS middleware (`localhost:3000`, `localhost:8080`) | `app/main.py` | Local dev outside Docker |
| Path traversal guard on content endpoint | content router | Security — whitelist filenames only |
| pytest for new endpoints | `tests/test_content.py` | 80%+ coverage |

---

## Phase 1 — UI Scaffold ⬜

**Repo:** `/home/rizkyagung/self/learning-platform-ui`

**Stack:**
| Layer | Choice |
|-------|--------|
| Framework | Next.js 15 (App Router) + React 19 |
| Language | TypeScript (strict) |
| Styling | Tailwind CSS + `@tailwindcss/typography` |
| Markdown | `react-markdown` + `remark-gfm` + `rehype-katex` + `rehype-pretty-code` (Shiki) |
| Math | KaTeX via `rehype-katex` |
| Icons | `lucide-react` |
| Theme | System-default with persisted user toggle |

**File structure:**
```
learning-platform-ui/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                # Home: domain dashboard
│   ├── domains/[id]/page.tsx   # Domain → grouped module list
│   ├── modules/[id]/page.tsx   # Module detail with tabs
│   └── search/page.tsx
├── lib/
│   ├── api.ts                  # Typed API client + envelope unwrap
│   ├── types.ts
│   └── markdown.tsx            # MDX render component (KaTeX + Shiki)
├── components/
│   ├── DomainCard.tsx
│   ├── ModuleCard.tsx
│   ├── DifficultyBadge.tsx
│   ├── PrereqGraph.tsx
│   ├── TagPill.tsx
│   ├── ProgressTracker.tsx
│   └── Tabs.tsx
└── Dockerfile
```

---

## Phase 2 — Browse Pages ⬜

### `/` — Home Dashboard
- Stats strip: domains · modules · total hours
- Domain cards (color/icon from registry); "Coming soon" for empty domains
- "Recently reviewed" row (top 4 by `last_reviewed`)

### `/domains/[id]` — Domain Page
- Modules **grouped by category** (deep-learning, mlops, etc.)
- Sticky filter bar: difficulty chips, tag search, hours range
- Module cards: title, difficulty badge, tags, hours, prereq count, last reviewed

---

## Phase 3 — Module Detail Page ⬜

### `/modules/[id]`

**Layout:**
- Breadcrumb: Domain › Category › Title
- **Tabs:** Overview / Theory / Cheatsheet / Notebooks / SOTA
  - Overview → `README.md` (minus frontmatter)
  - Theory → `concepts.md`
  - Cheatsheet → `cheatsheet.md`
  - Notebooks → list `.ipynb` files with GitHub links
  - SOTA → `sota_topics` bullet list
- **Sticky sidebar:** prerequisites (linked chips), "required by" reverse lookup, mini prereq DAG, mark-as-complete

**Markdown pipeline:** `react-markdown` → `remark-gfm` + `remark-math` + `rehype-katex` + `rehype-pretty-code`

---

## Phase 4 — Search & Discovery ⬜

- `/search` with debounced input + domain/difficulty/tag filters
- Prerequisite graph on module page using `@xyflow/react` (2-level DAG)
- "What's next?" — modules that have this one as a prereq

---

## Phase 5 — Local Progress Tracking ⬜

- `localStorage` schema: `{ moduleId: { status, startedAt, completedAt } }`
- Progress bar per domain
- "Continue learning" widget on home
- Export/import progress as JSON

---

## Phase 6 — Docker Compose Integration ⬜

Add `ui` service to `learning-platform-api/docker-compose.yml`:

```yaml
ui:
  build:
    context: ../learning-platform-ui
    target: ${UI_TARGET:-development}
  ports:
    - "3000:3000"
  environment:
    - NEXT_PUBLIC_API_URL=http://localhost:8080
    - API_URL_INTERNAL=http://api:8080
  volumes:
    - ../learning-platform-ui:/app
    - /app/node_modules
    - /app/.next
  depends_on:
    api:
      condition: service_started
  networks:
    - platform
```

---

## Phase 7 — Testing ⬜

### API (pytest, 80%+ coverage)
- `test_content_endpoint.py` — read markdown files, 404 on missing, path traversal blocked
- `test_notebooks_endpoint.py` — list notebooks
- `test_cors.py` — OPTIONS returns correct headers
- Gap-fill: `test_sync.py`, `test_search.py`, `test_domains.py`

### UI (Vitest + Playwright)
- Vitest: component unit tests (DomainCard, ModuleCard, Tabs, ProgressTracker)
- Playwright E2E flows:
  1. Home → 5 domain cards visible
  2. Click ds-ml → 19 modules in 5 categories
  3. Filter difficulty=advanced → correct subset
  4. Click attention-transformers → tabs render, math renders, prereqs linked
  5. Switch Theory tab → concepts.md with code highlighting
  6. Search "transformer" → attention-transformers in results
  7. Mark complete → progress persists on refresh

---

## Effort Estimates

| Phase | Estimate |
|-------|----------|
| 0. API enhancements | 1.5 h |
| 1. UI scaffold | 1 h |
| 2. Browse pages | 2 h |
| 3. Module detail | 3-4 h |
| 4. Search + prereq graph | 2 h |
| 5. Progress tracking | 1 h |
| 6. Docker Compose | 0.5 h |
| 7. Testing | 2-3 h |
| **Total** | **~13-15 h** |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Path traversal on content endpoint | **HIGH** | Whitelist filenames; verify path stays within content repo |
| KaTeX bundle size | LOW | Lazy-load on module detail pages only |
| Hot reload through Docker on WSL2 | MEDIUM | `WATCHPACK_POLLING=true` |
| Empty domains create dead links | LOW | "Coming soon" card; disable click |
| CORS with credentials in future | LOW | Explicit `allow_origins`, no wildcard |

---

## Out of Scope

- Auth / user accounts
- RAG / semantic search (Phase 3 roadmap)
- Notebook execution in browser
- Mobile-first design (functional, not optimized)
- i18n
