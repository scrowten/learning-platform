# Learning Platform

A modular, domain-agnostic learning platform. Pick any domain, pick any module, learn at your own pace.

**Philosophy:**
- Domain-agnostic — same module structure across DS/ML, Economics, Trading, Software Engineering, Finance, and more
- Non-linear — every module declares its prerequisites; skip freely if you already know the material
- Each module: Theory (`concepts.md`) → Practice (`notebooks/`) → Quick Reference (`cheatsheet.md`) → Field State (`sota.md`)
- Difficulty-tagged: `[Beginner]` `[Intermediate]` `[Advanced]` `[Research]`

---

## Domains

| Domain | Path | Status |
|---|---|---|
| Data Science & ML Engineering | `domains/ds-ml/` | Active |
| Economics | `domains/economics/` | Coming soon |
| Stocks & Trading | `domains/trading/` | Coming soon |
| Software Engineering | `domains/software-eng/` | Coming soon |
| Finance & Accounting | `domains/finance/` | Coming soon |

Full domain registry: [`domain-registry.yaml`](./domain-registry.yaml)

---

## DS/ML Module Map

```
domains/ds-ml/
├── prerequisites/
│   ├── linear-algebra/          [Beginner]
│   ├── probability-statistics/  [Beginner]
│   └── python-numpy-pandas/     [Beginner]
├── ml-fundamentals/
│   ├── supervised/              [Beginner–Intermediate]
│   ├── unsupervised/            [Intermediate]
│   └── ensemble-methods/        [Intermediate]
├── deep-learning/
│   ├── foundations/             [Intermediate]
│   ├── cnns/                    [Intermediate]
│   ├── rnns-lstms/              [Intermediate]
│   └── attention-transformers/  [Advanced]
├── llms-genai/
│   ├── language-modeling/       [Advanced]
│   ├── finetuning/              [Advanced]
│   ├── rag/                     [Advanced]
│   ├── agents/                  [Advanced]
│   └── multimodal/              [Research]
└── mlops/
    ├── experiment-tracking/     [Intermediate]
    ├── pipelines/               [Intermediate]
    ├── serving/                 [Advanced]
    └── monitoring/              [Advanced]
```

---

## Module Structure (Universal Template)

Every module in every domain follows this template:

```
<module-slug>/
├── README.md          ← YAML frontmatter (id, domain, difficulty, prerequisites…) + overview
├── concepts.md        ← Theory, math, models, derivations
├── cheatsheet.md      ← Quick reference card
├── sota.md            ← Current field state / active debates
└── notebooks/         ← Optional; quantitative domains
    ├── 01_model.ipynb
    └── 02_real_data.ipynb
```

---

## How to Use

### Jump in anywhere
Every module `README.md` has a **Prerequisites** section and **Skip Guide**. Read it first, then dive in.

### Linear path (DS/ML — beginner to practitioner)
`prerequisites/` → `ml-fundamentals/` → `deep-learning/foundations` → `deep-learning/attention-transformers` → `llms-genai/language-modeling` → `mlops/serving`

### Research catch-up
Read all `sota.md` files → identify gaps → fill in with `concepts.md` + notebooks as needed.

---

## Validating Content

```bash
python scripts/validate_content.py
```

Checks domain-registry.yaml, all module frontmatter, and prerequisite graph integrity.

---

## Adding a New Domain

1. Add an entry to `domain-registry.yaml`
2. Create `domains/<domain-id>/README.md`
3. Add modules following the universal template
4. Run `python scripts/validate_content.py`

No changes needed anywhere else.
