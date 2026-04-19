# 03 — LLMs & Generative AI `[Advanced–Research]`

## Overview

The current frontier of AI. This module covers the full stack of large language models — from pretraining fundamentals to fine-tuning, RAG, agents, and multimodal systems.

## Contents

| Subdirectory | Topics | Difficulty |
|---|---|---|
| `language_modeling/` | Pretraining objectives, tokenization (BPE/WordPiece), scaling laws, causal LM | Advanced |
| `finetuning/` | Supervised fine-tuning (SFT), RLHF, PPO, DPO, LoRA, QLoRA, PEFT | Advanced–Research |
| `rag/` | Embeddings, vector databases, chunking strategies, retrieval, reranking | Advanced |
| `agents/` | Tool use, function calling, ReAct framework, multi-agent systems, reasoning | Advanced–Research |
| `multimodal/` | CLIP, LLaVA, vision-language models, audio models, diffusion | Advanced–Research |

## Skip guide

| Background | Recommendation |
|---|---|
| Strong DL background, new to LLMs | Start at `language_modeling/` |
| Know how GPT works, want practical skills | `finetuning/` + `rag/` |
| Know fine-tuning, building applications | `rag/` + `agents/` |
| Research background | `sota.md` first, then dig into gaps |

## Prerequisites

- `02_deep_learning/attention_transformers/` or equivalent
- Understanding of the Transformer architecture
- Python + PyTorch or equivalent framework (for the fine-tuning notebooks)

## State of the Field

See [`sota.md`](./sota.md) — this is the most active area in ML. Updated more frequently than other sections.
