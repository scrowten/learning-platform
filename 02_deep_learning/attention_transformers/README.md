# Attention & Transformers `[Advanced]`

## Overview

The Transformer architecture (Vaswani et al., 2017) is the foundation of modern AI. This module builds it from scratch — every matrix multiply, every attention head — so you understand what's actually happening when you call `model.forward()`.

## What You'll Learn

- Why attention solves problems that RNNs couldn't
- The math of scaled dot-product attention and multi-head attention
- How the Transformer encoder/decoder architecture works end-to-end
- BERT (encoder-only) and GPT (decoder-only) architecture variants
- Vision Transformer (ViT) — attention applied to image patches

## Difficulty

`[Advanced]` — comfortable with backprop, matrix calculus, and PyTorch/numpy

## Prerequisites

- `02_deep_learning/foundations/` or equivalent (you know MLPs and backprop)
- `02_deep_learning/rnns_lstms/` helpful but not required
- Linear algebra: matrix multiply, softmax, layer norm

## Skip Guide

| You know... | Action |
|---|---|
| You can implement scaled dot-product attention from memory | Do the from-scratch notebook as a speed run, then read `sota.md` |
| You've used BERT/GPT but haven't implemented attention | Read `concepts.md` carefully, then do both notebooks |
| You're completely new to sequence models | Do `rnns_lstms/` first |

## Contents

```
attention_transformers/
├── README.md           ← this file
├── concepts.md         ← full theory: attention math, Transformer architecture
├── notebooks/
│   ├── 01_from_scratch.ipynb   ← build Transformer in numpy/PyTorch from scratch
│   └── 02_real_dataset.ipynb   ← fine-tune or apply to a real NLP task
└── cheatsheet.md       ← quick reference: key equations and architecture diagram
```

## Estimated Time

- `concepts.md`: 2–3 hours (go slow, work through the math)
- `01_from_scratch.ipynb`: 3–5 hours
- `02_real_dataset.ipynb`: 2–3 hours
