# 02 — Deep Learning `[Intermediate–Advanced]`

## Overview

From the perceptron to the Transformer. This module covers the full arc of deep learning — understanding what happens mathematically, implementing it from scratch, and applying it to real tasks.

## Contents

| Subdirectory | Topics | Difficulty |
|---|---|---|
| `foundations/` | Perceptron, MLP, forward pass, backprop, SGD, Adam, regularization | Intermediate |
| `cnns/` | Convolutions, pooling, batch norm, ResNet, transfer learning | Intermediate |
| `rnns_lstms/` | RNN, LSTM, GRU, vanishing gradients, sequence-to-sequence | Intermediate–Advanced |
| `attention_transformers/` | Attention mechanism, multi-head attention, full Transformer, BERT, ViT | Advanced |

## Skip guide

| Background | Recommendation |
|---|---|
| Implemented backprop before | Start from `cnns/` or `attention_transformers/` |
| Know CNNs, unfamiliar with Transformers | Go straight to `attention_transformers/` |
| Solid DL background, want LLMs | Skip to `03_llms_genai/` |

## Prerequisites

- Python + numpy
- Linear algebra (matrix multiply, chain rule)
- `01_ml_fundamentals/supervised/` or equivalent (gradient descent intuition)

## State of the Field

See [`sota.md`](./sota.md).
