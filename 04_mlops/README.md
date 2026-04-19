# 04 — MLOps `[Intermediate–Advanced]`

## Overview

Building ML models is 20% of the work; getting them to production and keeping them working is the other 80%. This module covers the full MLOps lifecycle.

## Contents

| Subdirectory | Topics | Difficulty |
|---|---|---|
| `experiment_tracking/` | MLflow, W&B, experiment reproducibility, artifact management | Intermediate |
| `pipelines/` | Data pipelines, feature stores, Airflow/Prefect/ZenML patterns | Intermediate–Advanced |
| `serving/` | FastAPI serving, TorchServe, model optimization (ONNX, TRT), latency/throughput | Advanced |
| `monitoring/` | Data drift detection, model decay, alerting, retraining triggers, Evidently | Advanced |

## Skip guide

| Background | Recommendation |
|---|---|
| Software engineer moving into ML | Full module |
| ML researcher going to production | `serving/` + `monitoring/` |
| Already run ML in production | Skim `sota.md`, fill gaps |

## Prerequisites

- `01_ml_fundamentals/` or equivalent (you have a model to deploy)
- Python + Docker basics
- Basic cloud familiarity helpful (AWS/GCP/Azure) but not required

## State of the Field

See [`sota.md`](./sota.md).
