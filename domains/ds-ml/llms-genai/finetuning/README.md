---
id: llm-finetuning
domain: ds-ml
title: "LLM Fine-Tuning"
difficulty: advanced
tags: [sft, rlhf, dpo, lora, qlora, peft, alignment]
prerequisites:
  - language-modeling
estimated_hours: 12
last_reviewed: 2026-04-19
sota_topics:
  - DPO replacing PPO for alignment (simpler, more stable)
  - ORPO and SimPO as DPO alternatives
  - LoRA rank selection and merging strategies
---

# LLM Fine-Tuning `[Advanced]`

Supervised fine-tuning (SFT), RLHF, PPO, DPO, LoRA, QLoRA, and PEFT techniques.

## Who needs this?

If you've fine-tuned an LLM end-to-end with LoRA and understand the difference between SFT and RLHF, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| SFT | Instruction following, chat formatting |
| RLHF | Reward model, PPO loop |
| DPO | Direct preference optimization |
| LoRA/QLoRA | Parameter-efficient fine-tuning |

## Prerequisites

- `language-modeling` (pretraining, scaling)
