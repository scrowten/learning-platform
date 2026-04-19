---
id: language-modeling
domain: ds-ml
title: "Language Modeling"
difficulty: advanced
tags: [llm, pretraining, tokenization, bpe, scaling-laws, causal-lm, gpt]
prerequisites:
  - attention-transformers
estimated_hours: 10
last_reviewed: 2026-04-19
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
