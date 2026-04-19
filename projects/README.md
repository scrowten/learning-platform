# Capstone Projects

End-to-end projects that integrate multiple modules. Each project is self-contained and deployable.

## Projects

### `classification_pipeline/`
**Prereqs:** `01_ml_fundamentals`
Full ML pipeline: data ingestion → feature engineering → model selection → evaluation → simple serving API.
Dataset: your choice (Titanic, house prices, fraud detection, etc.)

### `llm_from_scratch/`
**Prereqs:** `02_deep_learning/attention_transformers/` + `03_llms_genai/language_modeling/`
Train a small GPT (character-level or BPE) from scratch. Understand every line of a pretraining loop.
Reference: Karpathy's nanoGPT

### `rag_system/`
**Prereqs:** `03_llms_genai/rag/`
Production-quality RAG pipeline: document ingestion → chunking → embedding → vector DB → retrieval → generation → evaluation with RAGAS.

### `mlops_full_stack/`
**Prereqs:** `04_mlops/`
Full MLOps stack: training pipeline → experiment tracking → model registry → serving API → monitoring dashboard.

## Project Template

Each project includes:
- `README.md` — goal, architecture, setup instructions
- `data/` — data download/generation scripts (no raw data in repo)
- `src/` — Python modules (not just notebooks)
- `notebooks/` — exploration and analysis
- `tests/` — unit tests for core logic
- `Dockerfile` (for serving projects)
