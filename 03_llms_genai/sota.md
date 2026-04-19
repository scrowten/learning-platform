# State of the Field — LLMs & Generative AI (early 2026)

> This is the fastest-moving area of ML. Treat this as a snapshot; check dates before relying on specific numbers.

## Frontier Models (as of early 2026)

| Model | Organization | Notes |
|---|---|---|
| Claude 3.5/3.7 Sonnet, Claude 4 | Anthropic | Strong reasoning, coding, safety |
| GPT-4o, o1, o3 | OpenAI | Multimodal, strong reasoning chain |
| Gemini 1.5/2.0 Pro | Google DeepMind | Long context (1M+ tokens), multimodal |
| Llama 3.x | Meta | Best open-weight family |
| Mistral/Mixtral | Mistral AI | Efficient MoE models |
| DeepSeek-V3, R1 | DeepSeek | Strong open-weight reasoning models |
| Qwen 2.5 | Alibaba | Strong multilingual + math |

## Key Techniques

### Training
- **Pretraining**: next-token prediction at scale (data > architecture for capability)
- **Instruction tuning (SFT)**: teach models to follow instructions
- **RLHF**: human preference alignment via reward model + PPO
- **DPO (Direct Preference Optimization)**: simpler RLHF alternative; no reward model needed
- **Constitutional AI (Anthropic)**: self-critique for alignment

### Efficient Fine-tuning
- **LoRA / QLoRA**: low-rank adapter layers; fine-tune with <1% of parameters
- **PEFT (HuggingFace)**: unified library for parameter-efficient methods
- **Quantization**: 4-bit / 8-bit quantization (bitsandbytes) for inference + training

### Reasoning
- **Chain-of-thought (CoT)**: prompting models to reason step-by-step
- **o1 / o3 / R1 style**: models trained to do extended internal reasoning (test-time compute)
- **Process Reward Models (PRMs)**: reward individual steps, not just final answers

### Retrieval & Grounding
- **RAG (Retrieval-Augmented Generation)**: retrieve relevant context at inference time
- **Long context**: Gemini 1.5 (1M tokens), Claude (200k tokens) reduce need for RAG in some cases
- **GraphRAG**: knowledge-graph-enhanced retrieval

### Agents
- **ReAct**: interleave reasoning + action steps
- **Tool use / function calling**: models invoke external tools/APIs
- **Multi-agent**: orchestrator + specialist agents; debate, critique, refinement
- **Computer use**: models control GUIs directly

## Key Papers (2023–2025)

1. **"Direct Preference Optimization" (Rafailov et al., 2023)** — RLHF without RL
2. **"QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)**
3. **"Scaling Laws for Neural Language Models" (Kaplan et al., 2020)** — foundational; still relevant
4. **"Chinchilla" (Hoffmann et al., 2022)** — compute-optimal training (tokens >> params)
5. **"LLaMA 2/3" (Meta, 2023/2024)** — open-weight model paper; training recipe details
6. **"Mixtral of Experts" (Mistral, 2024)** — sparse MoE for efficient inference
7. **"DeepSeek-R1" (2025)** — reasoning model trained with RL; strong open alternative to o1

## Open Problems / Active Directions

- **Reliable reasoning**: models still hallucinate; test-time compute helps but doesn't solve it
- **Long-context faithfulness**: do models actually use retrieved context?
- **Efficient inference**: speculative decoding, KV cache compression, distillation
- **Multimodality**: native audio/video understanding (not just images)
- **Data quality**: synthetic data pipelines, deduplication, quality filtering at scale
- **Agent reliability**: tool use and planning at >90% success rate in real tasks

## Resources

- [Hugging Face Blog](https://huggingface.co/blog) — practical articles on LLMs
- [Andrej Karpathy's "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Lilian Weng: "Extrinsic Hallucinations in LLMs"](https://lilianweng.github.io/posts/2024-07-07-hallucination/)
- [Chip Huyen's AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) — practical LLM book
- [Alignment Forum](https://www.alignmentforum.org/) — research on safety/alignment
- ICLR, NeurIPS, ICML proceedings — main research venues
