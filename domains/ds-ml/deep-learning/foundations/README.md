---
id: dl-foundations
domain: ds-ml
title: "MLP & Backpropagation"
difficulty: intermediate
tags: [neural-networks, backpropagation, mlp, optimization, numpy, regularization]
prerequisites:
  - linear-algebra
  - probability-statistics
  - python-numpy-pandas
estimated_hours: 10
last_reviewed: 2026-04-14
sota_topics:
  - AdamW and Lion optimizers
  - Batch norm vs layer norm tradeoffs
  - Weight initialization for deep networks
---

# MLP & Backpropagation `[Intermediate]`

## Overview

This module builds the foundation of deep learning from first principles. You will implement
a multi-layer perceptron (MLP) using only numpy, derive backpropagation by hand, and verify
your gradients numerically. Everything that follows in this course — Transformers, LLMs, CNNs
— is built on these ideas.

## What You'll Learn

- How a single neuron computes a prediction and how to train it
- How to compose neurons into layers and layers into an MLP (forward pass)
- Why activation functions matter, and which ones are used in practice today
- How loss functions measure prediction error and guide learning
- The chain rule of calculus applied to computational graphs (backpropagation)
- Full gradient derivation for a 2-layer MLP — every delta written out explicitly
- Gradient descent variants: SGD, momentum, RMSProp, Adam
- Weight initialization strategies (Xavier, He) and why they matter
- Regularization: L1, L2, dropout, batch normalization
- Training dynamics: learning rate schedules, overfitting/underfitting, early stopping

## Difficulty

`[Intermediate]` — comfortable with calculus (chain rule), basic linear algebra (matrix
multiply, transpose), and Python/numpy

## Prerequisites

- Linear algebra: matrix multiply, dot products, transpose
- Calculus: partial derivatives, chain rule
- Python and numpy (indexing, broadcasting, vectorized ops)
- No prior deep learning knowledge required

## Skip Guide

| You know... | Action |
|---|---|
| You can derive backprop for a 2-layer MLP from memory | Skip to `notebooks/02_real_dataset.ipynb` or go straight to `attention_transformers/` |
| You understand forward pass but have never derived backprop | Read `concepts.md` sections 4–5 carefully, then do notebook 01 |
| You are new to neural networks | Work through everything in order — this is the right starting point |

## Contents

```
foundations/
├── README.md           ← this file
├── concepts.md         ← full theory: perceptron, MLP, backprop, optimizers, regularization
├── notebooks/
│   ├── 01_from_scratch.ipynb   ← build MLP + backprop in numpy from scratch
│   └── 02_real_dataset.ipynb   ← train MLP on MNIST/Fashion-MNIST with PyTorch
└── cheatsheet.md       ← quick reference: equations, update rules, training checklist
```

## Estimated Time

- `concepts.md`: 2–3 hours (work through the derivations with pen and paper)
- `01_from_scratch.ipynb`: 3–5 hours (implementing backprop fully is the key exercise)
- `02_real_dataset.ipynb`: 2–3 hours
