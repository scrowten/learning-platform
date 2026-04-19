# Attention & Transformers — Cheatsheet

## Core Equations

```
# Scaled dot-product attention
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

# Multi-head attention
MultiHead(Q, K, V) = concat(head_1, ..., head_h) @ W_O
  where head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)

# FFN (position-wise)
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

# Layer norm
LN(x) = gamma * (x - mean) / std + beta
```

## Architecture Variants

| Variant | Attention type | Pretraining | Use case |
|---|---|---|---|
| BERT | Bidirectional | MLM + NSP | Understanding (classification, NER, QA) |
| GPT | Causal | Next-token | Generation |
| T5 | Enc-Dec | Span masking | Seq2seq (translation, summarization) |
| ViT | Bidirectional | Supervised / DINO | Image classification |
| LLaMA | Causal + RoPE | Next-token | Generation (open weights) |

## Key Hyperparameters

| Name | Symbol | Typical value (BERT-base) |
|---|---|---|
| Model dim | `d_model` | 768 |
| Num heads | `h` | 12 |
| Head dim | `d_k = d_model / h` | 64 |
| FFN dim | `d_ff` | 3072 (4×) |
| Num layers | `N` | 12 |
| Max seq len | — | 512 |
| Vocab size | — | 30522 |

## Transformer Block (Pre-LN variant, modern default)

```
x → LN → MHA → + x →
  → LN → FFN → + x → output
```

Original paper used Post-LN (`x + LN(sublayer(x))`); Pre-LN is more stable.

## Attention Complexity

| | Time | Space |
|---|---|---|
| Standard attention | O(n² · d) | O(n²) |
| Flash Attention | O(n² · d) | O(n) | ← same compute, less memory |
| Linear attention approx | O(n · d²) | O(d²) |

## Positional Encoding Options

| Method | Notes |
|---|---|
| Sinusoidal (fixed) | Original paper; extrapolates to longer sequences |
| Learned absolute | BERT; simple but doesn't extrapolate |
| Relative PE (Shaw et al.) | Encodes relative distances |
| RoPE | Modern standard; rotates Q/K in complex space |
| ALiBi | Bias attention scores by distance; good extrapolation |

## Masking

```python
# Causal mask (decoder)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))

# Padding mask
# Set attention score to -inf where key is a padding token
```

## Quick Implementation Checklist

- [ ] Token embeddings
- [ ] Positional encoding (add to embeddings)
- [ ] Q, K, V linear projections (d_model → d_k, per head)
- [ ] Scaled dot-product attention
- [ ] Optional causal mask
- [ ] Concat heads + output projection W_O
- [ ] Residual + LayerNorm (around attention sublayer)
- [ ] FFN with expansion ratio 4
- [ ] Residual + LayerNorm (around FFN sublayer)
- [ ] Stack N blocks
- [ ] Task head (classifier / LM head)
