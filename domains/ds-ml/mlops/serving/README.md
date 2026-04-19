---
id: model-serving
domain: ds-ml
title: "Model Serving"
difficulty: advanced
tags: [fastapi, torchserve, onnx, triton, latency, throughput, quantization]
prerequisites:
  - ml-pipelines
estimated_hours: 8
last_reviewed: 2026-04-19
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
