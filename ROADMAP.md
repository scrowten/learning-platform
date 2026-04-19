# Roadmap

Module dependency map with difficulty tags and estimated time.

---

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

---

## Phase 2: Platform API

Once content migration is complete, Phase 2 builds `learning-platform-api`:
- PostgreSQL schema (domains, modules, prerequisites, progress, embeddings)
- Sync worker: reads `domain-registry.yaml` + frontmatter → upserts to DB
- FastAPI: content endpoints + AI features (Concept Q&A, Auto Quizzes, SOTA Alerts)
- Docker Compose for local dev / self-hosted deployment

See spec: `docs/superpowers/specs/2026-04-14-learning-platform-agnostic-design.md`
