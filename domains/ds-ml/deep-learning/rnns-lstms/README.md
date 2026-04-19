---
id: rnns-lstms
domain: ds-ml
title: "RNNs & LSTMs"
difficulty: intermediate
tags: [rnn, lstm, gru, sequence-modeling, vanishing-gradient, bptt]
prerequisites:
  - dl-foundations
estimated_hours: 7
last_reviewed: 2026-04-14
sota_topics:
  - Mamba and state space models (SSMs) replacing RNNs
  - RWKV architecture (RNN-speed Transformer)
  - xLSTM revisiting the architecture
---

# RNNs & LSTMs `[Intermediate]`

Recurrent neural networks, LSTMs, GRUs, vanishing gradients, and sequence-to-sequence models.

## Who needs this?

If you understand how LSTMs gate information and have implemented BPTT, you can go straight to `attention_transformers/`.

## Contents

| Topic | Description |
|---|---|
| Vanilla RNN | Hidden state, BPTT, vanishing gradients |
| LSTM | Input/forget/output gates, cell state |
| GRU | Simplified gating, fewer parameters |
| Seq2seq | Encoder-decoder, teacher forcing |

## Prerequisites

- `dl-foundations` (backprop, computational graphs, numpy)
