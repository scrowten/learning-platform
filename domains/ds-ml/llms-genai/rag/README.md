---
id: rag
domain: ds-ml
title: "Retrieval-Augmented Generation"
difficulty: advanced
tags: [rag, embeddings, vector-databases, retrieval, reranking, chunking]
prerequisites:
  - language-modeling
estimated_hours: 10
last_reviewed: 2026-04-19
sota_topics:
  - HyDE and advanced retrieval (query expansion)
  - Reranking models (Cohere Rerank, BGE)
  - GraphRAG for structured knowledge
---

# Retrieval-Augmented Generation `[Advanced]`

Embeddings, vector databases, chunking strategies, retrieval, and reranking.

## Who needs this?

If you've built a production RAG pipeline with evaluation, skim `sota.md`.

## Contents

| Topic | Description |
|---|---|
| Embeddings | Semantic similarity, embedding models |
| Vector databases | Pinecone, Qdrant, pgvector, FAISS |
| Chunking | Fixed, semantic, hierarchical |
| Retrieval | Sparse (BM25), dense, hybrid |
| Reranking | Cross-encoder rerankers |

## Prerequisites

- `language-modeling` (LLM inference, context windows)
