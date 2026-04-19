---
id: ml-monitoring
domain: ds-ml
title: "ML Monitoring"
difficulty: advanced
tags: [drift-detection, model-decay, evidently, retraining, alerting, observability]
prerequisites:
  - model-serving
estimated_hours: 6
last_reviewed: 2026-04-19
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
