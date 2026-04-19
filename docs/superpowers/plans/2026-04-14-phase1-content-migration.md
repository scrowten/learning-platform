# Phase 1 — Content Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the learning-platform repo from a DS/ML-only layout into a domain-agnostic `domains/` structure with YAML frontmatter on every module, validated by a script that CI can run.

**Architecture:** Content stays markdown + notebooks in git. A `domain-registry.yaml` at root declares all domains. Every module README carries YAML frontmatter (id, domain, title, difficulty, tags, prerequisites, estimated_hours, last_reviewed, sota_topics). A validation script (`scripts/validate_content.py`) enforces the schema — it is the test harness for this migration.

**Tech Stack:** Python 3 (stdlib + pyyaml), shell for directory moves, markdown for content files.

**Note on scope:** This plan covers Phase 1 only (this repo). Phases 2–4 (the `learning-platform-api` repo: FastAPI, sync worker, DB, AI features) are a separate plan written when that repo is created. See spec: `docs/superpowers/specs/2026-04-14-learning-platform-agnostic-design.md`.

---

## File Map

```
CREATED:
  domain-registry.yaml                              ← domain declarations
  scripts/validate_content.py                       ← validation / CI test harness
  domains/ds-ml/prerequisites/linear-algebra/README.md
  domains/ds-ml/prerequisites/probability-statistics/README.md
  domains/ds-ml/prerequisites/python-numpy-pandas/README.md
  domains/ds-ml/ml-fundamentals/supervised/README.md
  domains/ds-ml/ml-fundamentals/unsupervised/README.md
  domains/ds-ml/ml-fundamentals/ensemble-methods/README.md
  domains/ds-ml/mlops/experiment-tracking/README.md
  domains/ds-ml/mlops/pipelines/README.md
  domains/ds-ml/mlops/serving/README.md
  domains/ds-ml/mlops/monitoring/README.md
  domains/ds-ml/llms-genai/language-modeling/README.md
  domains/ds-ml/llms-genai/finetuning/README.md
  domains/ds-ml/llms-genai/rag/README.md
  domains/ds-ml/llms-genai/agents/README.md
  domains/ds-ml/llms-genai/multimodal/README.md
  domains/ds-ml/deep-learning/cnns/README.md
  domains/ds-ml/deep-learning/rnns-lstms/README.md
  domains/economics/README.md
  domains/trading/README.md
  domains/software-eng/README.md
  domains/finance/README.md

MOVED (content preserved, paths change):
  00_prerequisites/linear_algebra/     → domains/ds-ml/prerequisites/linear-algebra/
  00_prerequisites/probability_statistics/ → domains/ds-ml/prerequisites/probability-statistics/
  00_prerequisites/python_numpy_pandas/ → domains/ds-ml/prerequisites/python-numpy-pandas/
  01_ml_fundamentals/supervised/       → domains/ds-ml/ml-fundamentals/supervised/
  01_ml_fundamentals/unsupervised/     → domains/ds-ml/ml-fundamentals/unsupervised/
  01_ml_fundamentals/ensemble_methods/ → domains/ds-ml/ml-fundamentals/ensemble-methods/
  02_deep_learning/foundations/        → domains/ds-ml/deep-learning/foundations/
  02_deep_learning/cnns/               → domains/ds-ml/deep-learning/cnns/
  02_deep_learning/rnns_lstms/         → domains/ds-ml/deep-learning/rnns-lstms/
  02_deep_learning/attention_transformers/ → domains/ds-ml/deep-learning/attention-transformers/
  03_llms_genai/language_modeling/     → domains/ds-ml/llms-genai/language-modeling/
  03_llms_genai/finetuning/            → domains/ds-ml/llms-genai/finetuning/
  03_llms_genai/rag/                   → domains/ds-ml/llms-genai/rag/
  03_llms_genai/agents/                → domains/ds-ml/llms-genai/agents/
  03_llms_genai/multimodal/            → domains/ds-ml/llms-genai/multimodal/
  04_mlops/experiment_tracking/        → domains/ds-ml/mlops/experiment-tracking/
  04_mlops/pipelines/                  → domains/ds-ml/mlops/pipelines/
  04_mlops/serving/                    → domains/ds-ml/mlops/serving/
  04_mlops/monitoring/                 → domains/ds-ml/mlops/monitoring/

MODIFIED:
  README.md                            ← rewritten for multi-domain platform
  ROADMAP.md                           ← updated paths to match new structure
```

---

## Task 1: Write the Validation Script

This script is the test harness for all subsequent tasks. Run it after every task — it goes from all-fail to all-pass as the migration completes.

**Files:**
- Create: `scripts/validate_content.py`

- [ ] **Step 1: Install pyyaml if not already present**

```bash
pip show pyyaml || pip install pyyaml
```

Expected: version printed or "Successfully installed"

- [ ] **Step 2: Create the validation script**

Create `scripts/validate_content.py`:

```python
#!/usr/bin/env python3
"""
Validates domain-registry.yaml and all module README.md frontmatter.
Exit 0 = all checks pass. Exit 1 = failures found.
Run from repo root: python scripts/validate_content.py
"""
import sys
import re
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "domain-registry.yaml"
DOMAINS_DIR = REPO_ROOT / "domains"

REQUIRED_REGISTRY_FIELDS = {"id", "name", "path", "tags"}
REQUIRED_MODULE_FIELDS = {
    "id", "domain", "title", "difficulty",
    "tags", "prerequisites", "estimated_hours",
    "last_reviewed", "sota_topics",
}
VALID_DIFFICULTIES = {
    "beginner", "beginner-intermediate", "intermediate",
    "intermediate-advanced", "advanced", "research",
}

errors = []


def err(msg: str) -> None:
    errors.append(msg)
    print(f"  ERROR: {msg}")


def extract_frontmatter(path: Path) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        err(f"{path}: invalid YAML frontmatter — {e}")
        return None


def validate_registry() -> list[str]:
    """Returns list of valid domain IDs."""
    print("\n[1] Validating domain-registry.yaml")
    if not REGISTRY_PATH.exists():
        err(f"domain-registry.yaml not found at {REGISTRY_PATH}")
        return []

    try:
        data = yaml.safe_load(REGISTRY_PATH.read_text())
    except yaml.YAMLError as e:
        err(f"domain-registry.yaml: invalid YAML — {e}")
        return []

    if "domains" not in data or not isinstance(data["domains"], list):
        err("domain-registry.yaml: missing or invalid 'domains' list")
        return []

    domain_ids = []
    for domain in data["domains"]:
        missing = REQUIRED_REGISTRY_FIELDS - set(domain.keys())
        if missing:
            err(f"domain '{domain.get('id', '?')}': missing fields {missing}")
            continue
        domain_path = REPO_ROOT / domain["path"]
        if not domain_path.is_dir():
            err(f"domain '{domain['id']}': path '{domain['path']}' does not exist")
        else:
            print(f"  OK: domain '{domain['id']}' → {domain['path']}")
        domain_ids.append(domain["id"])

    return domain_ids


def validate_modules(valid_domain_ids: list[str]) -> list[str]:
    """Validates all module READMEs. Returns list of valid module IDs."""
    print("\n[2] Validating module frontmatter")
    if not DOMAINS_DIR.exists():
        err(f"domains/ directory not found at {DOMAINS_DIR}")
        return []

    module_ids = []
    readmes = sorted(DOMAINS_DIR.rglob("README.md"))

    # Skip domain-level READMEs (direct children of domains/)
    module_readmes = [
        r for r in readmes
        if r.parent.parent != DOMAINS_DIR
    ]

    if not module_readmes:
        err("No module README.md files found under domains/")
        return []

    for readme in module_readmes:
        fm = extract_frontmatter(readme)
        if fm is None:
            err(f"{readme.relative_to(REPO_ROOT)}: missing frontmatter block")
            continue

        missing = REQUIRED_MODULE_FIELDS - set(fm.keys())
        if missing:
            err(f"{readme.relative_to(REPO_ROOT)}: missing fields {missing}")
            continue

        if fm["difficulty"] not in VALID_DIFFICULTIES:
            err(
                f"{readme.relative_to(REPO_ROOT)}: invalid difficulty "
                f"'{fm['difficulty']}' — must be one of {VALID_DIFFICULTIES}"
            )

        if fm["domain"] not in valid_domain_ids:
            err(
                f"{readme.relative_to(REPO_ROOT)}: domain '{fm['domain']}' "
                f"not in domain-registry.yaml"
            )

        print(f"  OK: {fm['id']} ({fm['domain']})")
        module_ids.append(fm["id"])

    return module_ids


def validate_prerequisites(module_ids: list[str]) -> None:
    """Checks all prerequisite IDs reference valid module IDs."""
    print("\n[3] Validating prerequisite references")
    id_set = set(module_ids)

    for readme in sorted(DOMAINS_DIR.rglob("README.md")):
        if readme.parent.parent == DOMAINS_DIR:
            continue
        fm = extract_frontmatter(readme)
        if not fm or "prerequisites" not in fm:
            continue
        prereqs = fm.get("prerequisites") or []
        for prereq in prereqs:
            if prereq not in id_set:
                err(
                    f"{readme.relative_to(REPO_ROOT)}: "
                    f"prerequisite '{prereq}' not found in any module"
                )

    if not errors:
        print("  OK: all prerequisite references are valid")


def main() -> None:
    print("=" * 60)
    print("Learning Platform — Content Validation")
    print("=" * 60)

    valid_domain_ids = validate_registry()
    module_ids = validate_modules(valid_domain_ids)
    validate_prerequisites(module_ids)

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s) found")
        sys.exit(1)
    else:
        print(f"PASSED — {len(module_ids)} modules validated across {len(valid_domain_ids)} domains")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run validation — confirm it fails as expected**

```bash
cd /home/rizkyagung/self/learning-platform && python scripts/validate_content.py
```

Expected output (failures are correct at this point):
```
ERROR: domain-registry.yaml not found at .../domain-registry.yaml
FAILED — 1 error(s) found
```

- [ ] **Step 4: Commit the validation script**

```bash
git add scripts/validate_content.py
git commit -m "feat: add content validation script for domain-agnostic migration"
```

---

## Task 2: Create domain-registry.yaml and domains/ Skeleton

**Files:**
- Create: `domain-registry.yaml`
- Create: `domains/ds-ml/` (directory)
- Create: `domains/economics/README.md` (stub)
- Create: `domains/trading/README.md` (stub)
- Create: `domains/software-eng/README.md` (stub)
- Create: `domains/finance/README.md` (stub)

- [ ] **Step 1: Create domain-registry.yaml at repo root**

Create `domain-registry.yaml`:

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

- [ ] **Step 2: Create the domains/ directory skeleton**

```bash
mkdir -p domains/ds-ml domains/economics domains/trading domains/software-eng domains/finance
```

- [ ] **Step 3: Create stub READMEs for new domains**

Create `domains/economics/README.md`:
```markdown
# Economics

Domain covering macro and microeconomics — supply/demand, market structures, monetary policy, behavioral economics, and more.

**Status:** Modules coming soon. See [domain-registry.yaml](../../domain-registry.yaml) for domain metadata.
```

Create `domains/trading/README.md`:
```markdown
# Stocks & Trading

Domain covering technical analysis, fundamental analysis, options, portfolio theory, and quantitative trading strategies.

**Status:** Modules coming soon. See [domain-registry.yaml](../../domain-registry.yaml) for domain metadata.
```

Create `domains/software-eng/README.md`:
```markdown
# Software Engineering

Domain covering system design, distributed systems, algorithms, data structures, and backend architecture.

**Status:** Modules coming soon. See [domain-registry.yaml](../../domain-registry.yaml) for domain metadata.
```

Create `domains/finance/README.md`:
```markdown
# Finance & Accounting

Domain covering financial statements, valuation, corporate finance, and risk management.

**Status:** Modules coming soon. See [domain-registry.yaml](../../domain-registry.yaml) for domain metadata.
```

- [ ] **Step 4: Run validation — registry checks should now pass, module checks still fail**

```bash
python scripts/validate_content.py
```

Expected: `domain-registry.yaml` section passes (5 domains found), module section fails with "No module README.md files found".

- [ ] **Step 5: Commit**

```bash
git add domain-registry.yaml domains/
git commit -m "feat: add domain-registry.yaml and scaffold domain directories"
```

---

## Task 3: Migrate Prerequisites → domains/ds-ml/prerequisites/

**Files:**
- Move: `00_prerequisites/linear_algebra/` → `domains/ds-ml/prerequisites/linear-algebra/`
- Move: `00_prerequisites/probability_statistics/` → `domains/ds-ml/prerequisites/probability-statistics/`
- Move: `00_prerequisites/python_numpy_pandas/` → `domains/ds-ml/prerequisites/python-numpy-pandas/`
- Create/update: frontmatter in each module's `README.md`

- [ ] **Step 1: Move the directories**

```bash
mkdir -p domains/ds-ml/prerequisites
mv 00_prerequisites/linear_algebra domains/ds-ml/prerequisites/linear-algebra
mv 00_prerequisites/probability_statistics domains/ds-ml/prerequisites/probability-statistics
mv 00_prerequisites/python_numpy_pandas domains/ds-ml/prerequisites/python-numpy-pandas
rmdir 00_prerequisites
```

- [ ] **Step 2: Add frontmatter to linear-algebra README.md**

The directory may not have a README yet. Create `domains/ds-ml/prerequisites/linear-algebra/README.md`:

```markdown
---
id: linear-algebra
domain: ds-ml
title: "Linear Algebra"
difficulty: beginner
tags: [linear-algebra, math, vectors, matrices, eigenvalues, svd]
prerequisites: []
estimated_hours: 6
last_reviewed: 2026-04-14
sota_topics:
  - Matrix decompositions in ML (SVD, PCA, NMF)
  - Tensors and automatic differentiation
---

# Linear Algebra `[Beginner]`

Vectors, matrices, eigenvalues, SVD — the mathematical foundation for all of ML.

## Who needs this?

If you're comfortable with matrix operations and eigendecomposition, skip this module. Come back if you hit a gap in a later module.

## Contents

| Topic | Description |
|---|---|
| Vectors & dot products | Geometric intuition, inner products |
| Matrix multiplication | Composition of linear maps |
| Eigenvalues & eigenvectors | PCA foundation, stability analysis |
| SVD | Dimensionality reduction, pseudoinverse |

## Skip guide

| Background | Recommendation |
|---|---|
| STEM degree (math/CS/physics) | Skip entirely |
| Self-taught, shaky on matrix math | Work through everything |

## Prerequisites

None — this is a starting point.
```

- [ ] **Step 3: Add frontmatter to probability-statistics README.md**

Create `domains/ds-ml/prerequisites/probability-statistics/README.md`:

```markdown
---
id: probability-statistics
domain: ds-ml
title: "Probability & Statistics"
difficulty: beginner
tags: [probability, statistics, distributions, bayesian, mle, hypothesis-testing]
prerequisites: []
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - Bayesian deep learning and uncertainty quantification
  - Calibration in modern classifiers
---

# Probability & Statistics `[Beginner]`

Probability rules, distributions, MLE, Bayesian inference, and hypothesis testing.

## Who needs this?

If you understand probability distributions, MLE, and can derive Bayes' theorem, skip this.

## Contents

| Topic | Description |
|---|---|
| Probability rules | Conditional probability, independence, Bayes |
| Distributions | Gaussian, Bernoulli, Poisson, Beta |
| MLE & MAP | Parameter estimation |
| Hypothesis testing | p-values, confidence intervals |

## Skip guide

| Background | Recommendation |
|---|---|
| Took a stats course | Skim — focus on Bayesian sections |
| Self-taught | Work through everything |

## Prerequisites

None — no prior probability knowledge required.
```

- [ ] **Step 4: Add frontmatter to python-numpy-pandas README.md**

Create `domains/ds-ml/prerequisites/python-numpy-pandas/README.md`:

```markdown
---
id: python-numpy-pandas
domain: ds-ml
title: "Python, NumPy & Pandas"
difficulty: beginner
tags: [python, numpy, pandas, data-manipulation, matplotlib]
prerequisites: []
estimated_hours: 5
last_reviewed: 2026-04-14
sota_topics:
  - Polars as a faster pandas alternative
  - NumPy 2.0 changes and compatibility
---

# Python, NumPy & Pandas `[Beginner]`

Python essentials, numpy broadcasting, pandas DataFrames, and matplotlib basics.

## Who needs this?

If you use numpy and pandas daily, skip this entirely.

## Contents

| Topic | Description |
|---|---|
| Python essentials | Functions, classes, list comprehensions |
| NumPy | Arrays, broadcasting, vectorized ops |
| Pandas | DataFrames, groupby, merge, time series |
| Matplotlib | Line plots, scatter, histograms |

## Skip guide

| Background | Recommendation |
|---|---|
| Python developer (no numpy) | Do numpy + pandas sections only |
| Complete beginner | Work through everything |

## Prerequisites

None — basic programming literacy helpful but not required.
```

- [ ] **Step 5: Run validation — 3 modules should pass**

```bash
python scripts/validate_content.py
```

Expected: 3 modules validated (`linear-algebra`, `probability-statistics`, `python-numpy-pandas`). Prerequisite check passes (all have empty prerequisites).

- [ ] **Step 6: Commit**

```bash
git add domains/ds-ml/prerequisites/ && git rm -r 00_prerequisites/ 2>/dev/null || true
git commit -m "feat: migrate prerequisites to domains/ds-ml/prerequisites with frontmatter"
```

---

## Task 4: Migrate ML Fundamentals → domains/ds-ml/ml-fundamentals/

**Files:**
- Move: `01_ml_fundamentals/supervised/` → `domains/ds-ml/ml-fundamentals/supervised/`
- Move: `01_ml_fundamentals/unsupervised/` → `domains/ds-ml/ml-fundamentals/unsupervised/`
- Move: `01_ml_fundamentals/ensemble_methods/` → `domains/ds-ml/ml-fundamentals/ensemble-methods/`
- Create: `README.md` with frontmatter in each module

- [ ] **Step 1: Move the directories**

```bash
mkdir -p domains/ds-ml/ml-fundamentals
mv 01_ml_fundamentals/supervised domains/ds-ml/ml-fundamentals/supervised
mv 01_ml_fundamentals/unsupervised domains/ds-ml/ml-fundamentals/unsupervised
mv 01_ml_fundamentals/ensemble_methods domains/ds-ml/ml-fundamentals/ensemble-methods
mv 01_ml_fundamentals/sota.md domains/ds-ml/ml-fundamentals/
rmdir 01_ml_fundamentals
```

- [ ] **Step 2: Add frontmatter to supervised README.md**

Create `domains/ds-ml/ml-fundamentals/supervised/README.md`:

```markdown
---
id: supervised-learning
domain: ds-ml
title: "Supervised Learning"
difficulty: beginner-intermediate
tags: [regression, classification, svm, decision-trees, knn, naive-bayes, sklearn]
prerequisites:
  - linear-algebra
  - probability-statistics
  - python-numpy-pandas
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - Gradient boosting dominance on tabular data
  - AutoML (H2O, AutoSklearn)
  - TabPFN for small tabular datasets
---

# Supervised Learning `[Beginner–Intermediate]`

Linear/logistic regression, SVMs, decision trees, k-NN, naïve Bayes — the classical ML toolkit.

## Who needs this?

If you've trained classifiers in a production setting and understand the bias-variance tradeoff, skim `concepts.md` and do the from-scratch notebooks as a refresher.

## Contents

| Topic | Description |
|---|---|
| Linear regression | OLS, gradient descent, regularization |
| Logistic regression | Binary/multiclass, cross-entropy |
| SVMs | Margin maximization, kernels |
| Decision trees | Information gain, pruning |
| k-NN | Distance metrics, curse of dimensionality |

## Prerequisites

- Linear algebra (matrix multiply, dot products)
- Probability basics (distributions, MLE)
- Python + numpy/pandas
```

- [ ] **Step 3: Add frontmatter to unsupervised README.md**

Create `domains/ds-ml/ml-fundamentals/unsupervised/README.md`:

```markdown
---
id: unsupervised-learning
domain: ds-ml
title: "Unsupervised Learning"
difficulty: intermediate
tags: [clustering, dimensionality-reduction, pca, kmeans, dbscan, anomaly-detection]
prerequisites:
  - linear-algebra
  - probability-statistics
  - python-numpy-pandas
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - Self-supervised representation learning
  - UMAP vs t-SNE for high-dimensional visualization
  - Contrastive learning (SimCLR, MoCo)
---

# Unsupervised Learning `[Intermediate]`

k-means, DBSCAN, hierarchical clustering, PCA, ICA, and anomaly detection.

## Who needs this?

If you're familiar with dimensionality reduction and clustering in practice, skim `sota.md` and go straight to the real dataset notebook.

## Contents

| Topic | Description |
|---|---|
| k-means | Lloyd's algorithm, k-means++ initialization |
| DBSCAN | Density-based clustering, noise handling |
| PCA | Eigendecomposition, explained variance |
| Anomaly detection | Isolation Forest, LOF |

## Prerequisites

- Linear algebra (SVD, eigenvalues)
- Python + numpy/pandas
```

- [ ] **Step 4: Add frontmatter to ensemble-methods README.md**

Create `domains/ds-ml/ml-fundamentals/ensemble-methods/README.md`:

```markdown
---
id: ensemble-methods
domain: ds-ml
title: "Ensemble Methods"
difficulty: intermediate
tags: [random-forest, boosting, xgboost, lightgbm, stacking, bagging]
prerequisites:
  - supervised-learning
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - XGBoost/LightGBM still dominate tabular Kaggle competitions
  - TabNet and deep tabular models
  - CatBoost for categorical features
---

# Ensemble Methods `[Intermediate]`

Bagging, random forests, AdaBoost, gradient boosting, XGBoost, LightGBM, and stacking.

## Who needs this?

If you regularly use gradient boosting in production and understand how it differs from bagging, you can skip to `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Bagging | Bootstrap aggregating, variance reduction |
| Random forests | Feature subsampling, OOB error |
| AdaBoost | Sequential reweighting |
| Gradient boosting | GBDT, XGBoost, LightGBM |
| Stacking | Meta-learners, cross-val stacking |

## Prerequisites

- `supervised-learning` (decision trees, bias-variance tradeoff)
```

- [ ] **Step 5: Run validation — 6 modules should pass**

```bash
python scripts/validate_content.py
```

Expected: 6 modules validated. Prerequisite check passes.

- [ ] **Step 6: Commit**

```bash
git add domains/ds-ml/ml-fundamentals/ && git rm -r 01_ml_fundamentals/ 2>/dev/null || true
git commit -m "feat: migrate ml-fundamentals to domains/ds-ml/ml-fundamentals with frontmatter"
```

---

## Task 5: Migrate Deep Learning → domains/ds-ml/deep-learning/

**Files:**
- Move: `02_deep_learning/foundations/` → `domains/ds-ml/deep-learning/foundations/`
- Move: `02_deep_learning/cnns/` → `domains/ds-ml/deep-learning/cnns/`
- Move: `02_deep_learning/rnns_lstms/` → `domains/ds-ml/deep-learning/rnns-lstms/`
- Move: `02_deep_learning/attention_transformers/` → `domains/ds-ml/deep-learning/attention-transformers/`
- Prepend frontmatter to existing `foundations/README.md` and `attention_transformers/README.md`
- Create `README.md` for `cnns/` and `rnns-lstms/`

- [ ] **Step 1: Move the directories**

```bash
mkdir -p domains/ds-ml/deep-learning
mv 02_deep_learning/foundations domains/ds-ml/deep-learning/foundations
mv 02_deep_learning/cnns domains/ds-ml/deep-learning/cnns
mv 02_deep_learning/rnns_lstms domains/ds-ml/deep-learning/rnns-lstms
mv 02_deep_learning/attention_transformers domains/ds-ml/deep-learning/attention-transformers
mv 02_deep_learning/sota.md domains/ds-ml/deep-learning/
rmdir 02_deep_learning
```

- [ ] **Step 2: Prepend frontmatter to foundations/README.md**

The existing `README.md` starts at line 1 with `# MLP & Backpropagation`. Prepend the frontmatter block. Open `domains/ds-ml/deep-learning/foundations/README.md` and insert at the very top (before the existing `#` heading):

```markdown
---
id: dl-foundations
domain: ds-ml
title: "MLP & Backpropagation"
difficulty: intermediate
tags: [neural-networks, backpropagation, mlp, optimization, numpy, regularization]
prerequisites:
  - linear-algebra
  - probability-statistics
  - python-numpy-pandas
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - AdamW and Lion optimizers
  - Batch norm vs layer norm tradeoffs
  - Weight initialization for deep networks
---

```

(Leave all existing content unchanged below this block.)

- [ ] **Step 3: Prepend frontmatter to attention-transformers/README.md**

Open `domains/ds-ml/deep-learning/attention-transformers/README.md` and insert at the very top:

```markdown
---
id: attention-transformers
domain: ds-ml
title: "Attention & Transformers"
difficulty: advanced
tags: [attention, transformer, bert, gpt, vit, self-attention, multi-head-attention]
prerequisites:
  - dl-foundations
  - rnns-lstms
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - Flash Attention 2 and 3
  - Mixture of Experts (MoE) architectures
  - Efficient attention variants (linear, sparse)
---

```

- [ ] **Step 4: Create cnns/README.md**

Create `domains/ds-ml/deep-learning/cnns/README.md`:

```markdown
---
id: convolutional-networks
domain: ds-ml
title: "Convolutional Neural Networks"
difficulty: intermediate
tags: [cnn, convolutions, resnet, image-classification, transfer-learning, batch-norm]
prerequisites:
  - dl-foundations
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - Vision Transformers (ViT) challenging CNNs on image tasks
  - EfficientNet and compound scaling
  - ConvNeXt as a modernized CNN
---

# Convolutional Neural Networks `[Intermediate]`

Convolutions, pooling, batch normalization, ResNet, and transfer learning.

## Who needs this?

If you can explain how a convolution operation works mathematically and have implemented ResNet, skip to `attention_transformers/`.

## Contents

| Topic | Description |
|---|---|
| Convolution operation | Filters, feature maps, padding, stride |
| Pooling | Max/average pooling, spatial downsampling |
| Batch normalization | Training stability, internal covariate shift |
| ResNet | Skip connections, deep network training |
| Transfer learning | Fine-tuning pretrained models |

## Prerequisites

- `dl-foundations` (backprop, MLPs, optimizers)
```

- [ ] **Step 5: Create rnns-lstms/README.md**

Create `domains/ds-ml/deep-learning/rnns-lstms/README.md`:

```markdown
---
id: rnns-lstms
domain: ds-ml
title: "RNNs & LSTMs"
difficulty: intermediate
tags: [rnn, lstm, gru, sequence-modeling, vanishing-gradient, bptt]
prerequisites:
  - dl-foundations
estimated_hours: 7
last_reviewed: 2026-04-14
sota_topics:
  - Mamba and state space models (SSMs) replacing RNNs
  - RWKV architecture (RNN-speed Transformer)
  - xLSTM revisiting the architecture
---

# RNNs & LSTMs `[Intermediate]`

Recurrent neural networks, LSTMs, GRUs, vanishing gradients, and sequence-to-sequence models.

## Who needs this?

If you understand how LSTMs gate information and have implemented BPTT, you can go straight to `attention_transformers/`.

## Contents

| Topic | Description |
|---|---|
| Vanilla RNN | Hidden state, BPTT, vanishing gradients |
| LSTM | Input/forget/output gates, cell state |
| GRU | Simplified gating, fewer parameters |
| Seq2seq | Encoder-decoder, teacher forcing |

## Prerequisites

- `dl-foundations` (backprop, computational graphs, numpy)
```

- [ ] **Step 6: Run validation — 10 modules should pass**

```bash
python scripts/validate_content.py
```

Expected: 10 modules validated. All prerequisite IDs reference valid modules.

- [ ] **Step 7: Commit**

```bash
git add domains/ds-ml/deep-learning/ && git rm -r 02_deep_learning/ 2>/dev/null || true
git commit -m "feat: migrate deep-learning to domains/ds-ml/deep-learning with frontmatter"
```

---

## Task 6: Migrate LLMs/GenAI → domains/ds-ml/llms-genai/

**Files:**
- Move: `03_llms_genai/language_modeling/` → `domains/ds-ml/llms-genai/language-modeling/`
- Move: `03_llms_genai/finetuning/` → `domains/ds-ml/llms-genai/finetuning/`
- Move: `03_llms_genai/rag/` → `domains/ds-ml/llms-genai/rag/`
- Move: `03_llms_genai/agents/` → `domains/ds-ml/llms-genai/agents/`
- Move: `03_llms_genai/multimodal/` → `domains/ds-ml/llms-genai/multimodal/`
- Create: `README.md` with frontmatter in each module

- [ ] **Step 1: Move the directories**

```bash
mkdir -p domains/ds-ml/llms-genai
mv 03_llms_genai/language_modeling domains/ds-ml/llms-genai/language-modeling
mv 03_llms_genai/finetuning domains/ds-ml/llms-genai/finetuning
mv 03_llms_genai/rag domains/ds-ml/llms-genai/rag
mv 03_llms_genai/agents domains/ds-ml/llms-genai/agents
mv 03_llms_genai/multimodal domains/ds-ml/llms-genai/multimodal
mv 03_llms_genai/sota.md domains/ds-ml/llms-genai/
rmdir 03_llms_genai
```

- [ ] **Step 2: Create language-modeling/README.md**

Create `domains/ds-ml/llms-genai/language-modeling/README.md`:

```markdown
---
id: language-modeling
domain: ds-ml
title: "Language Modeling"
difficulty: advanced
tags: [llm, pretraining, tokenization, bpe, scaling-laws, causal-lm, gpt]
prerequisites:
  - attention-transformers
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - Chinchilla scaling laws (compute-optimal training)
  - Byte-level tokenization (BBPE)
  - Long-context scaling (Llama 3, Mistral)
---

# Language Modeling `[Advanced]`

Pretraining objectives, tokenization (BPE/WordPiece), scaling laws, and causal language modeling.

## Who needs this?

If you can explain next-token prediction, BPE tokenization, and Chinchilla scaling laws, skip to `finetuning/`.

## Contents

| Topic | Description |
|---|---|
| Pretraining objectives | Causal LM, masked LM (BERT), span prediction |
| Tokenization | BPE, WordPiece, SentencePiece |
| Scaling laws | Compute-optimal training, Chinchilla |
| Positional encoding | Sinusoidal, RoPE, ALiBi |

## Prerequisites

- `attention-transformers` (full Transformer architecture)
```

- [ ] **Step 3: Create finetuning/README.md**

Create `domains/ds-ml/llms-genai/finetuning/README.md`:

```markdown
---
id: llm-finetuning
domain: ds-ml
title: "LLM Fine-Tuning"
difficulty: advanced
tags: [sft, rlhf, dpo, lora, qlora, peft, alignment]
prerequisites:
  - language-modeling
estimated_hours: 12
last_reviewed: 2026-04-14
sota_topics:
  - DPO replacing PPO for alignment (simpler, more stable)
  - ORPO and SimPO as DPO alternatives
  - LoRA rank selection and merging strategies
---

# LLM Fine-Tuning `[Advanced]`

Supervised fine-tuning (SFT), RLHF, PPO, DPO, LoRA, QLoRA, and PEFT techniques.

## Who needs this?

If you've fine-tuned an LLM end-to-end with LoRA and understand the difference between SFT and RLHF, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| SFT | Instruction following, chat formatting |
| RLHF | Reward model, PPO loop |
| DPO | Direct preference optimization |
| LoRA/QLoRA | Parameter-efficient fine-tuning |

## Prerequisites

- `language-modeling` (pretraining, scaling)
```

- [ ] **Step 4: Create rag/README.md**

Create `domains/ds-ml/llms-genai/rag/README.md`:

```markdown
---
id: rag
domain: ds-ml
title: "Retrieval-Augmented Generation"
difficulty: advanced
tags: [rag, embeddings, vector-databases, retrieval, reranking, chunking]
prerequisites:
  - language-modeling
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - HyDE and advanced retrieval (query expansion)
  - Reranking models (Cohere Rerank, BGE)
  - GraphRAG for structured knowledge
---

# Retrieval-Augmented Generation `[Advanced]`

Embeddings, vector databases, chunking strategies, retrieval, and reranking.

## Who needs this?

If you've built a production RAG pipeline with evaluation, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Embeddings | Semantic similarity, embedding models |
| Vector databases | Pinecone, Qdrant, pgvector, FAISS |
| Chunking | Fixed, semantic, hierarchical |
| Retrieval | Sparse (BM25), dense, hybrid |
| Reranking | Cross-encoder rerankers |

## Prerequisites

- `language-modeling` (LLM inference, context windows)
```

- [ ] **Step 5: Create agents/README.md**

Create `domains/ds-ml/llms-genai/agents/README.md`:

```markdown
---
id: ai-agents
domain: ds-ml
title: "AI Agents"
difficulty: advanced
tags: [agents, tool-use, function-calling, react, multi-agent, planning]
prerequisites:
  - language-modeling
  - rag
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - OpenAI function calling and tool use standards
  - Multi-agent frameworks (AutoGen, CrewAI, LangGraph)
  - Agent evaluation (tau-bench, SWE-bench)
---

# AI Agents `[Advanced]`

Tool use, function calling, ReAct framework, and multi-agent systems.

## Who needs this?

If you've built and evaluated multi-agent systems in production, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Tool use | Function calling, structured outputs |
| ReAct | Reasoning + acting loop |
| Planning | Chain-of-thought, tree-of-thought |
| Multi-agent | Orchestrator/worker patterns |

## Prerequisites

- `language-modeling` (LLM inference)
- `rag` (retrieval tools in agent loops)
```

- [ ] **Step 6: Create multimodal/README.md**

Create `domains/ds-ml/llms-genai/multimodal/README.md`:

```markdown
---
id: multimodal
domain: ds-ml
title: "Multimodal Models"
difficulty: research
tags: [clip, llava, vision-language, diffusion, multimodal, image-generation]
prerequisites:
  - language-modeling
  - convolutional-networks
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - GPT-4o and Claude 3 native vision capabilities
  - Diffusion model improvements (SDXL, Flux)
  - Video generation (Sora, Kling, Runway Gen-3)
---

# Multimodal Models `[Research]`

CLIP, LLaVA, vision-language models, audio models, and diffusion-based generation.

## Who needs this?

If you've trained or fine-tuned a vision-language model and understand diffusion, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| CLIP | Contrastive image-text pretraining |
| LLaVA | Visual instruction tuning |
| Diffusion models | Score matching, DDPM, DDIM |
| Audio models | Whisper, audio tokenization |

## Prerequisites

- `language-modeling` (Transformer architecture)
- `convolutional-networks` (image feature extraction)
```

- [ ] **Step 7: Run validation — 15 modules should pass**

```bash
python scripts/validate_content.py
```

Expected: 15 modules validated. Prerequisite check passes.

- [ ] **Step 8: Commit**

```bash
git add domains/ds-ml/llms-genai/ && git rm -r 03_llms_genai/ 2>/dev/null || true
git commit -m "feat: migrate llms-genai to domains/ds-ml/llms-genai with frontmatter"
```

---

## Task 7: Migrate MLOps → domains/ds-ml/mlops/

**Files:**
- Move: `04_mlops/experiment_tracking/` → `domains/ds-ml/mlops/experiment-tracking/`
- Move: `04_mlops/pipelines/` → `domains/ds-ml/mlops/pipelines/`
- Move: `04_mlops/serving/` → `domains/ds-ml/mlops/serving/`
- Move: `04_mlops/monitoring/` → `domains/ds-ml/mlops/monitoring/`
- Create: `README.md` with frontmatter in each module

- [ ] **Step 1: Move the directories**

```bash
mkdir -p domains/ds-ml/mlops
mv 04_mlops/experiment_tracking domains/ds-ml/mlops/experiment-tracking
mv 04_mlops/pipelines domains/ds-ml/mlops/pipelines
mv 04_mlops/serving domains/ds-ml/mlops/serving
mv 04_mlops/monitoring domains/ds-ml/mlops/monitoring
mv 04_mlops/sota.md domains/ds-ml/mlops/
rmdir 04_mlops
```

- [ ] **Step 2: Create experiment-tracking/README.md**

Create `domains/ds-ml/mlops/experiment-tracking/README.md`:

```markdown
---
id: experiment-tracking
domain: ds-ml
title: "Experiment Tracking"
difficulty: intermediate
tags: [mlflow, wandb, experiment-tracking, reproducibility, artifact-management]
prerequisites:
  - supervised-learning
estimated_hours: 5
last_reviewed: 2026-04-14
sota_topics:
  - MLflow 2.x unified tracking API
  - W&B Weave for LLM evaluation tracking
  - DVC for data versioning
---

# Experiment Tracking `[Intermediate]`

MLflow, W&B, experiment reproducibility, and artifact management.

## Who needs this?

If you already track experiments with MLflow or W&B and version your datasets, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| MLflow | Run tracking, model registry, artifact store |
| W&B | Sweeps, dashboards, collaboration |
| Reproducibility | Seeds, environment capture, DVC |

## Prerequisites

- `supervised-learning` (you have a model to track)
```

- [ ] **Step 3: Create pipelines/README.md**

Create `domains/ds-ml/mlops/pipelines/README.md`:

```markdown
---
id: ml-pipelines
domain: ds-ml
title: "ML Pipelines"
difficulty: intermediate
tags: [airflow, prefect, zenml, feature-stores, data-pipelines, orchestration]
prerequisites:
  - experiment-tracking
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - Feature stores (Feast, Tecton)
  - ZenML vs Prefect for ML-specific orchestration
  - dbt for feature transformation
---

# ML Pipelines `[Intermediate]`

Data pipelines, feature stores, and orchestration with Airflow/Prefect/ZenML.

## Who needs this?

If you've built end-to-end ML pipelines with a feature store in production, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Data pipelines | Ingestion, transformation, validation |
| Feature stores | Online/offline stores, point-in-time correctness |
| Orchestration | DAGs, Airflow, Prefect, ZenML |

## Prerequisites

- `experiment-tracking` (artifacts, versioning)
```

- [ ] **Step 4: Create serving/README.md**

Create `domains/ds-ml/mlops/serving/README.md`:

```markdown
---
id: model-serving
domain: ds-ml
title: "Model Serving"
difficulty: advanced
tags: [fastapi, torchserve, onnx, triton, latency, throughput, quantization]
prerequisites:
  - ml-pipelines
estimated_hours: 8
last_reviewed: 2026-04-14
sota_topics:
  - vLLM for high-throughput LLM serving
  - Continuous batching and paged attention
  - Model optimization (GPTQ, AWQ quantization)
---

# Model Serving `[Advanced]`

FastAPI serving, TorchServe, ONNX, model optimization, and latency/throughput trade-offs.

## Who needs this?

If you've deployed models behind APIs with SLOs and optimized for latency, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| FastAPI serving | REST endpoints, async, batching |
| TorchServe | Handlers, multi-model server |
| ONNX/TensorRT | Model export and optimization |
| Latency tuning | Profiling, caching, quantization |

## Prerequisites

- `ml-pipelines` (you have a trained model to deploy)
```

- [ ] **Step 5: Create monitoring/README.md**

Create `domains/ds-ml/mlops/monitoring/README.md`:

```markdown
---
id: ml-monitoring
domain: ds-ml
title: "ML Monitoring"
difficulty: advanced
tags: [drift-detection, model-decay, evidently, retraining, alerting, observability]
prerequisites:
  - model-serving
estimated_hours: 6
last_reviewed: 2026-04-14
sota_topics:
  - LLM evaluation and monitoring (Langsmith, Braintrust, Weave)
  - Evidently AI cloud for drift dashboards
  - Continuous training triggers
---

# ML Monitoring `[Advanced]`

Data drift detection, model decay, alerting, and automated retraining triggers.

## Who needs this?

If you run monitoring in production with automated alerting and retraining pipelines, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Data drift | PSI, KL divergence, Kolmogorov-Smirnov |
| Model decay | Performance degradation over time |
| Evidently | Drift dashboards, test suites |
| Retraining triggers | Threshold-based, scheduled, continuous |

## Prerequisites

- `model-serving` (deployed model to monitor)
```

- [ ] **Step 6: Run validation — all 19 modules should pass**

```bash
python scripts/validate_content.py
```

Expected:
```
PASSED — 19 modules validated across 5 domains
```

- [ ] **Step 7: Commit**

```bash
git add domains/ds-ml/mlops/ && git rm -r 04_mlops/ 2>/dev/null || true
git commit -m "feat: migrate mlops to domains/ds-ml/mlops with frontmatter"
```

---

## Task 8: Update Root README.md and ROADMAP.md

**Files:**
- Modify: `README.md` (rewrite to reflect multi-domain platform)
- Modify: `ROADMAP.md` (update all paths from old `00_–04_` structure to new `domains/ds-ml/` paths)

- [ ] **Step 1: Replace README.md content**

Overwrite `README.md` with:

```markdown
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
```

- [ ] **Step 2: Update ROADMAP.md paths**

Open `ROADMAP.md` and replace all old path references with the new `domains/ds-ml/` structure. The module dependency map section should reference new paths. Replace the Module Index section with:

```markdown
## Module Index

### Prerequisites `[Beginner]`

| Module | Path | Skip if... |
|---|---|---|
| Linear Algebra | `domains/ds-ml/prerequisites/linear-algebra/` | Comfortable with matrix ops |
| Probability & Statistics | `domains/ds-ml/prerequisites/probability-statistics/` | Know probability theory |
| Python, NumPy & Pandas | `domains/ds-ml/prerequisites/python-numpy-pandas/` | You use these daily |

### ML Fundamentals `[Beginner–Intermediate]`

| Module | Path | Skip if... |
|---|---|---|
| Supervised Learning | `domains/ds-ml/ml-fundamentals/supervised/` | You've trained classifiers before |
| Unsupervised Learning | `domains/ds-ml/ml-fundamentals/unsupervised/` | Familiar with dimensionality reduction |
| Ensemble Methods | `domains/ds-ml/ml-fundamentals/ensemble-methods/` | You know ensemble methods |

### Deep Learning `[Intermediate–Advanced]`

| Module | Path | Skip if... |
|---|---|---|
| MLP & Backpropagation | `domains/ds-ml/deep-learning/foundations/` | You can derive backprop |
| CNNs | `domains/ds-ml/deep-learning/cnns/` | You know ConvNets well |
| RNNs & LSTMs | `domains/ds-ml/deep-learning/rnns-lstms/` | You know recurrent models |
| Attention & Transformers | `domains/ds-ml/deep-learning/attention-transformers/` | You understand Transformer architecture |

### LLMs & Generative AI `[Advanced–Research]`

| Module | Path | Skip if... |
|---|---|---|
| Language Modeling | `domains/ds-ml/llms-genai/language-modeling/` | You know how LLMs are pretrained |
| LLM Fine-Tuning | `domains/ds-ml/llms-genai/finetuning/` | You've fine-tuned a model |
| RAG | `domains/ds-ml/llms-genai/rag/` | You've built a RAG pipeline |
| AI Agents | `domains/ds-ml/llms-genai/agents/` | You've built agents |
| Multimodal | `domains/ds-ml/llms-genai/multimodal/` | Familiar with multimodal models |

### MLOps `[Intermediate–Advanced]`

| Module | Path | Skip if... |
|---|---|---|
| Experiment Tracking | `domains/ds-ml/mlops/experiment-tracking/` | You track experiments already |
| ML Pipelines | `domains/ds-ml/mlops/pipelines/` | You build ML pipelines |
| Model Serving | `domains/ds-ml/mlops/serving/` | You've deployed models |
| ML Monitoring | `domains/ds-ml/mlops/monitoring/` | You monitor models in prod |
```

- [ ] **Step 3: Run validation — still 19 modules, just confirming nothing broke**

```bash
python scripts/validate_content.py
```

Expected: `PASSED — 19 modules validated across 5 domains`

- [ ] **Step 4: Commit**

```bash
git add README.md ROADMAP.md
git commit -m "docs: update README and ROADMAP for multi-domain platform"
```

---

## Task 9: Final Cleanup and Validation

- [ ] **Step 1: Confirm old numbered directories are fully removed**

```bash
ls -la | grep -E "^d.*0[0-9]_"
```

Expected: no output (all `00_–04_` directories should be gone).

- [ ] **Step 2: Run full validation one last time**

```bash
python scripts/validate_content.py
```

Expected:
```
============================================================
Learning Platform — Content Validation
============================================================

[1] Validating domain-registry.yaml
  OK: domain 'ds-ml' → domains/ds-ml
  OK: domain 'economics' → domains/economics
  OK: domain 'trading' → domains/trading
  OK: domain 'software-eng' → domains/software-eng
  OK: domain 'finance' → domains/finance

[2] Validating module frontmatter
  OK: linear-algebra (ds-ml)
  OK: probability-statistics (ds-ml)
  OK: python-numpy-pandas (ds-ml)
  OK: supervised-learning (ds-ml)
  OK: unsupervised-learning (ds-ml)
  OK: ensemble-methods (ds-ml)
  OK: dl-foundations (ds-ml)
  OK: convolutional-networks (ds-ml)
  OK: rnns-lstms (ds-ml)
  OK: attention-transformers (ds-ml)
  OK: language-modeling (ds-ml)
  OK: llm-finetuning (ds-ml)
  OK: rag (ds-ml)
  OK: ai-agents (ds-ml)
  OK: multimodal (ds-ml)
  OK: experiment-tracking (ds-ml)
  OK: ml-pipelines (ds-ml)
  OK: model-serving (ds-ml)
  OK: ml-monitoring (ds-ml)

[3] Validating prerequisite references
  OK: all prerequisite references are valid

============================================================
PASSED — 19 modules validated across 5 domains
```

- [ ] **Step 3: Add .gitignore entry for brainstorm session files**

```bash
echo ".superpowers/" >> .gitignore
git add .gitignore
git commit -m "chore: add .superpowers to .gitignore"
```

- [ ] **Step 4: Final commit tagging Phase 1 complete**

```bash
git add -A
git status  # verify nothing unexpected
git commit -m "feat: complete Phase 1 — domain-agnostic content migration

- Restructured from numbered directories (00-04) to domains/ layout
- Added domain-registry.yaml with 5 domains (ds-ml, economics, trading, software-eng, finance)
- Added YAML frontmatter to all 19 ds-ml modules
- Wrote validate_content.py for CI enforcement
- Scaffolded 4 new domain directories for future content
- Updated README to reflect multi-domain platform

Next: Phase 2 — create learning-platform-api repo (see spec)"
```

---

## What's Next (Phase 2 Plan)

Once Phase 1 is merged, the next plan covers `learning-platform-api`:
- PostgreSQL schema (7 tables from spec)
- Sync worker: reads `domain-registry.yaml` + module frontmatter → upserts to DB
- FastAPI: content endpoints (domains, modules, search, prerequisites)
- Docker Compose for local dev

Write the Phase 2 plan in the new `learning-platform-api` repo once it's created.
