# State of the Field — MLOps (2025)

## The Stack Has Consolidated

The MLOps tooling space was fragmented in 2020–2022 but has consolidated around a few winners:

- **Experiment tracking**: W&B and MLflow (MLflow for open-source control, W&B for teams)
- **Orchestration**: Airflow (legacy, still dominant in enterprise), Prefect/ZenML (modern Python-native)
- **Feature stores**: Feast (open source), Tecton (managed), Hopsworks
- **Serving**: FastAPI (custom), BentoML, TorchServe, Ray Serve, vLLM (LLMs specifically)
- **Model registry**: MLflow, W&B, HuggingFace Hub
- **Monitoring**: Evidently AI, Whylogs, Arize, Fiddler

## LLMOps Emerging as Sub-Discipline

LLMs have unique operational challenges:
- **Prompt management**: versioning, A/B testing prompts
- **Evaluation pipelines**: LLM-as-judge, human eval, automated evals (RAGAS for RAG)
- **Cost tracking**: token usage, latency, provider costs
- **Guardrails**: input/output filtering, PII detection, hallucination detection
- **Tools**: LangSmith, Braintrust, Phoenix (Arize)

## Key Trends

- **Feature store adoption** growing; real-time feature serving is harder than batch
- **ML platform teams** standardizing on internal platforms (Uber Michelangelo-style)
- **Kubeflow / Ray** for large-scale distributed training pipelines
- **vLLM + PagedAttention** — de facto standard for high-throughput LLM serving
- **Continuous training** patterns: not retraining from scratch but fine-tuning on new data

## Key Papers / Resources

1. **"Machine Learning Operations (MLOps): Overview, Definition, and Architecture" (Kreuzberger et al., 2022)** — good survey
2. **"Scaling Machine Learning at Uber with Michelangelo" (Uber Eng Blog)** — real-world ML platform design
3. **"vLLM: Efficient Memory Management for LLM Serving" (Kwon et al., 2023)**
4. **"Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015, NeurIPS)** — still essential

## Resources

- [Made With ML](https://madewithml.com/) — practical MLOps course
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Community Slack](https://mlops.community/)
- [Evidently AI Blog](https://www.evidentlyai.com/blog) — monitoring in depth
