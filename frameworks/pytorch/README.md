# Framework Track — PyTorch

This directory mirrors the core modules with PyTorch-native implementations.

Each notebook here assumes you've read the corresponding `concepts.md` and ideally worked through the numpy from-scratch version. The goal is to show how the same math maps to production-quality PyTorch code: training loops, data loaders, checkpointing, mixed precision, etc.

## Contents (planned)

| File | Mirrors |
|---|---|
| `attention_transformer.ipynb` | `02_deep_learning/attention_transformers/` |
| `mlp_backprop.ipynb` | `02_deep_learning/foundations/` |
| `cnn_image_classification.ipynb` | `02_deep_learning/cnns/` |
| `language_model_gpt.ipynb` | `03_llms_genai/language_modeling/` |
| `lora_finetuning.ipynb` | `03_llms_genai/finetuning/` |

## Conventions

- All notebooks use `torch >= 2.0`
- `torch.compile()` used where beneficial
- Mixed precision (`torch.autocast`) shown in training loops
- `DataLoader` with custom collate functions for NLP tasks
