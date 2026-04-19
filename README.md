# DS/AI-ML Self-Study & Open Source Course

A modular, from-scratch learning resource for data science and machine learning — from math prerequisites to LLMs and MLOps.

**Philosophy:**
- Pick any module independently (no forced linear path)
- Each module: Theory → From-scratch implementation → Real dataset challenge
- Skip-friendly: each module has a difficulty tag and explicit skip guide
- Research-aware: `sota.md` in each section covers current papers and directions
- Framework-agnostic core (numpy/pandas/sklearn); optional PyTorch/TF/JAX tracks under `/frameworks/`

---

## Who is this for?

| You are... | Start here |
|---|---|
| Complete beginner | `00_prerequisites` → `01_ml_fundamentals` |
| Know classical ML, rusty on deep learning | `02_deep_learning/foundations` |
| Know DL, want to catch up on LLMs | `03_llms_genai` |
| Practicing engineer wanting production ML | `04_mlops` |
| Researcher wanting framework comparisons | `/frameworks/` |

---

## Course Map

```
00_prerequisites/          Math, stats, Python basics     [Beginner]
01_ml_fundamentals/        Classical ML                   [Beginner–Intermediate]
02_deep_learning/          MLP → CNNs → Transformers      [Intermediate–Advanced]
03_llms_genai/             LLMs, fine-tuning, RAG, agents [Advanced–Research]
04_mlops/                  Production ML systems          [Intermediate–Advanced]
frameworks/                PyTorch / TF / JAX tracks      [varies]
projects/                  End-to-end capstone projects   [varies]
```

See [ROADMAP.md](./ROADMAP.md) for the full dependency map with difficulty tags.

---

## How to Use This Repo

### Option A — Linear path (recommended for beginners)
Follow the numbered directories in order. Each module builds conceptual depth but is still self-contained.

### Option B — Jump in anywhere
Every module has a `README.md` with a **Prerequisites / Who can skip this** section. Read that first, then dive into `concepts.md`.

### Option C — Research catch-up
Go straight to the `sota.md` files to see what's current in each area, then work backwards into fundamentals as needed.

---

## Module Structure

Every subdirectory follows this template:

```
module/
├── README.md              # Overview, difficulty tag, skip guide, prerequisites
├── concepts.md            # Theory — math, intuition, derivations
├── notebooks/
│   ├── 01_from_scratch.ipynb    # Implement the algorithm from scratch
│   └── 02_real_dataset.ipynb   # Apply to a real/kaggle/HuggingFace dataset
└── cheatsheet.md          # Quick reference card
```

**Difficulty tags:** `[Beginner]` `[Intermediate]` `[Advanced]` `[Research]`

---

## Setup

```bash
# Clone
git clone <repo-url>
cd self_repo

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install core deps
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel

# Optional: framework tracks
pip install torch torchvision          # PyTorch
pip install tensorflow                 # TensorFlow
pip install jax jaxlib flax optax      # JAX
```

---

## Validating Notebooks

Each notebook should run clean:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_from_scratch.ipynb
```

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) *(coming in Phase 3)*.

---

## Roadmap

- **Phase 1 (current):** Bootstrap — structure + first complete module
- **Phase 2:** Fill out all core modules
- **Phase 3:** Framework tracks, open source polish, CI, rendered docs site
