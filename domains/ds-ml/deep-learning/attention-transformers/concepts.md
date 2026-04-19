# Attention & Transformers — Theory

## 1. The Problem with RNNs

Recurrent models process sequences token-by-token. This creates two fundamental problems:

1. **Sequential bottleneck**: token at position `t` can only "see" position `t-1`. Long-range dependencies require information to travel through every intermediate hidden state.
2. **Vanishing gradients**: gradients from early positions are multiplied through many timesteps, shrinking to near-zero. LSTMs help but don't eliminate this.

**The attention insight**: what if every position could directly attend to every other position in one operation?

---

## 2. Scaled Dot-Product Attention

The core operation. Given:

- **Q** (Query matrix): what am I looking for?
- **K** (Key matrix): what do I have to offer?
- **V** (Value matrix): what do I actually return?

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### Step by step

**Step 1 — Compute similarity scores:**
```
scores = Q @ K.T          # shape: (seq_len, seq_len)
```
Each element `scores[i, j]` = dot product between query at position `i` and key at position `j`.

**Step 2 — Scale by sqrt(d_k):**
```
scores = scores / sqrt(d_k)
```
Without scaling, dot products grow large with dimension, pushing softmax into regions of near-zero gradients.

**Step 3 — Apply softmax (over the key dimension):**
```
weights = softmax(scores, dim=-1)   # shape: (seq_len, seq_len)
```
Each row sums to 1. `weights[i, j]` = how much position `i` attends to position `j`.

**Step 4 — Weighted sum of values:**
```
output = weights @ V                # shape: (seq_len, d_v)
```
Each output position is a weighted combination of all value vectors.

### Why Q, K, V?

Q/K/V are learned linear projections of the input. Same input `X` is projected three ways:
```
Q = X @ W_Q     # (seq_len, d_model) → (seq_len, d_k)
K = X @ W_K
V = X @ W_V
```
This lets the model learn *what to look for*, *what to match against*, and *what to return* independently.

---

## 3. Masking

### Padding mask
For batched sequences of different lengths, pad shorter sequences with zeros and mask them so attention is not applied to padding tokens.

### Causal (autoregressive) mask
In decoder (GPT-style), each position should only attend to previous positions — not future ones. Achieved by setting future attention scores to `-inf` before softmax:

```
mask = upper_triangular_matrix_of_ones
scores.masked_fill(mask, float('-inf'))
```

---

## 4. Multi-Head Attention

Single attention learns one type of relationship. Multi-head attention runs `h` attention operations in parallel, each with different learned projections:

```
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
MultiHead(Q, K, V) = concat(head_1, ..., head_h) @ W_O
```

Each head can specialize:
- One head might learn syntactic relationships
- Another might learn coreference
- Another might learn positional patterns

Typical config: `d_model = 512`, `h = 8`, `d_k = d_v = d_model / h = 64`

---

## 5. Positional Encoding

Attention is permutation-equivariant — shuffling input tokens gives shuffled output, with no position information. We add positional encodings to inject order.

### Fixed sinusoidal (original paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension oscillates at a different frequency. Two positions always have a unique encoding, and the encoding generalizes to longer sequences than seen in training.

### Learned absolute positions
Simple: add a learned embedding `E[pos]` to each token embedding. Used in BERT.

### Rotary Position Embedding (RoPE)
Modern standard. Encodes position by rotating the Q/K vectors in a way that relative position affects attention scores naturally. Used in LLaMA, GPT-NeoX, most modern LLMs.

---

## 6. Transformer Block

The full block that stacks to make a Transformer:

```
x → LayerNorm → MultiHeadAttention → + (residual) →
  → LayerNorm → FeedForward         → + (residual) → output
```

### Layer Normalization
Normalize activations across features (not batch):
```
LayerNorm(x) = gamma * (x - mean(x)) / std(x) + beta
```
Stabilizes training — critical for deep Transformers.

### Feed-Forward Network (FFN)
Applied position-wise (same MLP at each position independently):
```
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
```
Typical expansion: `d_ff = 4 * d_model` (e.g., 512 → 2048 → 512)

FFN is where most of the "knowledge" is stored — roughly 2/3 of parameters in a Transformer.

### Residual connections
`output = x + sublayer(x)` — essential for gradient flow in deep networks.

---

## 7. Encoder vs. Decoder vs. Encoder-Decoder

### Encoder-only (BERT)
- Bidirectional attention: every token attends to every other
- Used for: classification, NER, embeddings
- Pretraining: Masked Language Modeling (MLM)

### Decoder-only (GPT)
- Causal (autoregressive) attention: each token only attends to past
- Used for: text generation, in-context learning
- Pretraining: next-token prediction

### Encoder-Decoder (original Transformer, T5, BART)
- Encoder: processes source sequence (bidirectional)
- Decoder: generates target sequence (causal), attends to encoder output via cross-attention
- Used for: translation, summarization, seq2seq tasks

---

## 8. Full Transformer: Putting It Together

```python
# Pseudocode for encoder-only Transformer

Input: token_ids  (batch_size, seq_len)

# 1. Embedding + positional encoding
x = token_embedding(token_ids) + positional_encoding(seq_len)
# shape: (batch_size, seq_len, d_model)

# 2. Stack N Transformer blocks
for block in transformer_blocks:
    # Self-attention sublayer
    attn_out = MultiHeadAttention(Q=x, K=x, V=x, mask=padding_mask)
    x = LayerNorm(x + attn_out)

    # FFN sublayer
    ffn_out = FFN(x)
    x = LayerNorm(x + ffn_out)

# 3. Task-specific head
output = classifier_head(x[:, 0, :])   # use [CLS] token for classification
```

---

## 9. BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers (Devlin et al., 2018)

Architecture: encoder-only Transformer (12 or 24 layers)

Pretraining tasks:
1. **Masked Language Modeling (MLM)**: randomly mask 15% of tokens; predict them
2. **Next Sentence Prediction (NSP)**: predict if sentence B follows sentence A

Fine-tuning: add a task head on top of the [CLS] token and fine-tune all weights.

**Key insight**: bidirectional context is better for understanding tasks; you can see both left and right context when predicting a masked token.

---

## 10. Vision Transformer (ViT)

Transformers applied to images (Dosovitskiy et al., 2020).

**Patch embedding**: split image into fixed-size patches (e.g., 16×16 pixels), flatten each to a vector, project to `d_model`. Now each patch is a "token."

```
Image: (H, W, C) → patches: (N, P*P*C) where N = HW / P^2
Projected: (N, d_model)
```

Add positional embeddings + [CLS] token → standard Transformer encoder.

**Why it works**: at large scale, ViT outperforms CNNs. At small scale, CNNs' inductive biases (locality, translation equivariance) give them an edge.

---

## Key Equations Summary

| Operation | Formula |
|---|---|
| Scaled dot-product attention | `softmax(QK^T / sqrt(d_k)) V` |
| Multi-head concat | `concat(head_1,...,head_h) @ W_O` |
| FFN | `max(0, xW_1 + b_1) W_2 + b_2` |
| Layer norm | `gamma * (x - mu) / sigma + beta` |
| Sinusoidal PE | `sin(pos/10000^(2i/d))`, `cos(...)` |

---

## Further Reading

- [Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762) — original paper
- [Devlin et al., "BERT" (2018)](https://arxiv.org/abs/1810.04805)
- [Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020)](https://arxiv.org/abs/2010.11929)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual walkthrough
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — line-by-line implementation
