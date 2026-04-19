# State of the Field — Deep Learning (2025)

## The Transformer Has Won (For Now)

The Transformer architecture (Vaswani et al., 2017) now dominates across modalities — NLP, vision, audio, code, protein folding. The key insight: attention is a general-purpose sequence operation, and with enough scale it generalizes remarkably.

CNNs retain relevance for:
- Edge/mobile inference (efficiency)
- Video understanding (local temporal structure)
- Scientific domains (e.g., molecular simulation, medical imaging)

## Current Architecture Frontiers

### Vision
- **ViT (Vision Transformer)** — patch-based image transformers; scales better than CNNs with more data
- **DINOv2 (Meta, 2023)** — self-supervised ViT; state-of-the-art visual features without labels
- **ConvNeXt V2** — modern CNN that keeps up with ViTs; efficient for constrained settings

### Efficient Transformers
- **Flash Attention 2/3** — IO-aware attention kernel; 2-4x speedup in practice
- **Mamba (Gu & Dao, 2023)** — state space model alternative to Transformers; O(n) instead of O(n²)
- **RWKV** — RNN-Transformer hybrid; efficient inference

### Optimization
- **Adam/AdamW** — still the default; gradient clipping essential for large models
- **Muon optimizer (2024)** — momentum + orthogonalization; strong results on LLM pretraining
- **Learning rate schedules**: cosine decay with warmup is standard

## Key Papers

1. **"Attention Is All You Need" (Vaswani et al., 2017)** — foundational Transformer paper; still essential reading
2. **"An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)** — ViT; Transformers for vision
3. **"DINOv2" (Oquab et al., 2023)** — self-supervised visual foundation models
4. **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)**
5. **"Flash Attention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)**

## Open Problems

- Can we replace attention with something fundamentally cheaper at scale?
- How do we train efficiently on longer sequences (>100k tokens)?
- Interpretability: what do attention heads actually learn?
- Data efficiency: can we train good models on less data?

## Resources

- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — best from-scratch DL course
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — line-by-line Transformer walkthrough
- [Lilian Weng's Blog](https://lilianweng.github.io/) — excellent deep dives on DL topics
- Papers With Code: [paperswithcode.com](https://paperswithcode.com)
