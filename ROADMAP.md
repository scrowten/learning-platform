# Course Roadmap

Module dependency map with difficulty tags and estimated time.

---

## Difficulty Scale

| Tag | Description |
|---|---|
| `[Beginner]` | No prior ML knowledge needed; high school math |
| `[Intermediate]` | Comfortable with Python, basic calculus/linear algebra |
| `[Advanced]` | Familiar with classical ML and neural networks |
| `[Research]` | Active research area; assumes strong fundamentals |

---

## Module Map

```
                        ┌─────────────────────┐
                        │   00_prerequisites   │  [Beginner]
                        │  Math · Stats · Py   │
                        └──────────┬──────────┘
                                   │  (skippable if you know the basics)
                        ┌──────────▼──────────┐
                        │  01_ml_fundamentals  │  [Beginner–Intermediate]
                        │  Regression · Trees  │
                        │  Clustering · PCA    │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
        ┌──────────▼──────┐        │      ┌────────▼───────┐
        │ 02_deep_learning │        │      │   04_mlops     │
        │ [Intermediate–  │        │      │ [Intermediate– │
        │   Advanced]     │        │      │   Advanced]    │
        └──────────┬──────┘        │      └────────────────┘
                   │               │
        ┌──────────▼──────┐        │
        │  03_llms_genai  │        │
        │  [Advanced–     │        │
        │   Research]     │        │
        └─────────────────┘        │
                                   │
                        ┌──────────▼──────────┐
                        │     frameworks/      │
                        │  PyTorch · TF · JAX  │
                        └─────────────────────┘
```

---

## Module Index

### 00 — Prerequisites `[Beginner]`

| Module | Topics | Skip if... |
|---|---|---|
| `linear_algebra/` | Vectors, matrices, eigenvalues, SVD | Comfortable with matrix ops |
| `probability_statistics/` | Probability, distributions, MLE, Bayes | Know probability theory |
| `python_numpy_pandas/` | Python, numpy broadcasting, pandas | You use these daily |

### 01 — ML Fundamentals `[Beginner–Intermediate]`

| Module | Topics | Skip if... |
|---|---|---|
| `supervised/` | Linear/logistic regression, SVMs, decision trees, k-NN | You've trained classifiers before |
| `unsupervised/` | k-means, DBSCAN, PCA, anomaly detection | Familiar with dimensionality reduction |
| `ensemble_methods/` | Bagging, random forests, boosting (XGBoost/LightGBM), stacking | You know ensemble methods |

### 02 — Deep Learning `[Intermediate–Advanced]`

| Module | Topics | Skip if... |
|---|---|---|
| `foundations/` | MLP, backprop, optimizers, regularization from scratch | You can derive backprop |
| `cnns/` | Conv layers, pooling, ResNet, image classification | You know ConvNets well |
| `rnns_lstms/` | RNN, LSTM, GRU, sequence modeling | You know recurrent models |
| `attention_transformers/` | Attention, self-attention, Transformer, BERT, ViT | You understand Transformer architecture |

### 03 — LLMs & Generative AI `[Advanced–Research]`

| Module | Topics | Skip if... |
|---|---|---|
| `language_modeling/` | Pretraining, tokenization (BPE), scaling laws, next-token prediction | You know how LLMs are pretrained |
| `finetuning/` | SFT, RLHF, PPO, DPO, LoRA, QLoRA | You've fine-tuned a model |
| `rag/` | Vector DBs, embeddings, retrieval strategies, chunking | You've built a RAG pipeline |
| `agents/` | Tool use, function calling, ReAct, multi-agent systems | You've built agents |
| `multimodal/` | Vision-language (CLIP, LLaVA), audio, diffusion models | Familiar with multimodal models |

### 04 — MLOps `[Intermediate–Advanced]`

| Module | Topics | Skip if... |
|---|---|---|
| `experiment_tracking/` | MLflow, W&B, experiment reproducibility | You track experiments already |
| `pipelines/` | Data pipelines, feature stores, Airflow/Prefect patterns | You build ML pipelines |
| `serving/` | Model APIs, FastAPI, TorchServe, latency/throughput | You've deployed models |
| `monitoring/` | Data drift, model decay, alerting, retraining triggers | You monitor models in prod |

### Framework Tracks `[varies]`

| Track | Notes |
|---|---|
| `frameworks/pytorch/` | PyTorch equivalents of core notebooks |
| `frameworks/tensorflow/` | TF/Keras equivalents |
| `frameworks/jax/` | JAX/Flax for research-style code |

### Capstone Projects `[varies]`

| Project | Description | Prereqs |
|---|---|---|
| `classification_pipeline/` | End-to-end classical ML pipeline | 01_ml_fundamentals |
| `llm_from_scratch/` | Train a small GPT from scratch | 02_deep_learning + 03_llms_genai/language_modeling |
| `rag_system/` | Production RAG with evaluation | 03_llms_genai/rag |
| `mlops_full_stack/` | Full MLOps stack for a real problem | 04_mlops |

---

## Suggested Learning Paths

### Path A — Complete Beginner to Practitioner
`00` → `01` → `02/foundations` → `02/cnns` → `02/attention_transformers` → `03/language_modeling` → `04/serving`

### Path B — ML Engineer catching up on LLMs
`02/attention_transformers` (refresh) → `03/language_modeling` → `03/finetuning` → `03/rag` → `03/agents`

### Path C — Research catch-up sprint
Read all `sota.md` files → identify gaps → fill in with `concepts.md` + notebooks as needed

### Path D — Production focus
`01` (light) → `04/pipelines` → `04/serving` → `04/monitoring` → `projects/mlops_full_stack`
