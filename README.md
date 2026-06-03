# Regulatory RAG System

Production-grade Retrieval-Augmented Generation (RAG) system for regulatory document intelligence with hybrid retrieval, reranking, evaluation pipelines, citation grounding, and observability.

---

# Project Overview

This project demonstrates how to build a production-oriented AI system capable of:

* Ingesting regulatory PDFs/documents
* Chunking and embedding large documents
* Performing hybrid retrieval (semantic + lexical)
* Applying cross-encoder reranking
* Generating grounded answers with citations
* Running automated evaluation metrics
* Preventing hallucinations using guardrails
* Tracking conversations and caching responses
* Serving everything through a FastAPI application with UI

The system is designed with real-world AI engineering principles:

* modular architecture
* observability
* defensive validation
* evaluation-driven development
* production-focused retrieval pipelines

---

# System Architecture

## High-Level Pipeline

```text
User Query
    ↓
Query Rewriter
    ↓
Hybrid Retriever
(BM25 + Vector Search)
    ↓
Cross Encoder Reranker
    ↓
Guardrails + Validation
    ↓
LLM Answer Generation
    ↓
Citation Extraction
    ↓
Evaluation Engine
    ↓
FastAPI Response + UI Rendering
```

---

# Core Technical Features

## 1. Hybrid Retrieval Pipeline

Implemented a production-style retrieval architecture:

### Semantic Retrieval

* Dense vector embeddings
* Amazon Titan Embeddings
* Vector similarity search

### Lexical Retrieval

* BM25 keyword retrieval
* Exact terminology matching
* Regulatory phrase prioritization

### Retrieval Fusion

* Combined lexical + semantic retrieval
* Improved recall for compliance documents
* Better handling of long-tail terminology

---

## 2. Cross Encoder Reranking

Implemented a second-stage reranking pipeline using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Capabilities:

* semantic relevance scoring
* metadata-aware reranking
* section-aware ranking
* top-k refinement
* fallback handling

This significantly improves final context quality before answer generation.

---

# LLM Orchestration

## Model Layer

### LLM

* Groq-hosted LLM inference
* configurable providers
* temperature-controlled generation

### Embeddings

* Amazon Titan Embeddings
* Bedrock integration

### Prompt Engineering

Custom prompt templates enforce:

* grounded generation
* citation requirements
* anti-hallucination constraints
* concise regulatory responses

---

# Hallucination Prevention & Guardrails

A dedicated guardrails layer validates:

## Citation Validation

* verifies all citations exist
* prevents fake citations
* validates chunk references

## Grounding Validation

* computes answer-context overlap
* rejects unsupported generations
* detects weak evidence alignment

## Multi-document Safety

* prevents unsafe cross-document answers
* enforces single-document grounding

---

# Evaluation System

The project includes a production-style evaluation pipeline.

## Metrics Implemented

### Semantic Grounding Score

Measures similarity between:

* generated answer
* retrieved evidence

using cosine similarity over embeddings.

---

### LLM-as-a-Judge

The system automatically evaluates:

* correctness
* completeness
* supportiveness

using an independent LLM evaluator.

---

### Citation Score

Measures:

* citation coverage
* evidence utilization
* grounding quality

---

### Optional RAGAS Integration

Support for:

* faithfulness
* answer relevancy
* context precision
* context recall

---

# Performance Optimizations

## Context Limiting

* token-safe chunk limiting
* prevents context overflow
* adaptive chunk selection

## In-memory Caching

* TTL cache
* avoids repeated LLM calls
* namespace-aware caching

## Background Evaluation

* asynchronous evaluation execution
* non-blocking API responses

---

# Production Engineering Features

## Structured Logging

Implemented JSON-based structured logging using:

```text
structlog
```

Logs include:

* retrieval metrics
* reranking scores
* grounding failures
* latency metrics
* evaluation metrics
* cache hits

---

## Typed Schema Design

Implemented fully typed Pydantic schemas for:

* retrieval state
* ingestion state
* RAG pipeline state
* API contracts
* chunk metadata
* evaluation outputs

---

## Async Architecture

Built using:

* FastAPI async endpoints
* async evaluation execution
* async DB operations
* async retrieval orchestration

---

# UI Features

Custom frontend interface supports:

* document ingestion
* live querying
* citation visualization
* evaluation metric rendering
* per-document chat sessions
* document filtering
* ingestion state handling

---

# Technology Stack

## Backend

* Python
* FastAPI
* LangChain
* LangGraph
* Pydantic
* AsyncIO

---

## AI/ML Stack

* Groq LLMs
* Amazon Titan Embeddings
* SentenceTransformers
* CrossEncoder Reranking
* RAGAS

---

## Retrieval Stack

* Hybrid Retrieval
* Vector Search
* BM25
* Reranking Pipelines

---

## Infrastructure

* PostgreSQL
* pgvector
* Bedrock
* Structlog
* TTL Cache

---

# Project Structure

```text
REGULATORY-RAG-SYSTEM/
│
├── config/
├── exception/
├── logger/
│
├── src/
│   ├── api/
│   │
│   ├── core/
│   │   ├── cache/
│   │   ├── chunking/
│   │   ├── evaluation/
│   │   ├── generation/
│   │   ├── guardrails/
│   │   ├── ingestion/
│   │   ├── memory/
│   │   ├── parsing/
│   │   ├── pipeline/
│   │   ├── prompts/
│   │   ├── reranking/
│   │   ├── retrieval/
│   │   └── rewrite/
│   │
│   ├── infra/
│   │   ├── config/
│   │   ├── db/
│   │   └── models/
│   │
│   ├── shared/
│   │   └── schemas/
│   │
│   ├── static/
│   └── templates/
│
└── main.py
```

---

# Engineering Challenges Solved

## Retrieval Quality Problems

Solved using:

* hybrid retrieval
* reranking
* metadata-aware scoring

---

## Hallucination Prevention

Solved using:

* grounding validation
* strict prompting
* citation enforcement
* evidence overlap checks

---

## Production Observability

Solved using:

* structured logs
* evaluation metrics
* latency tracking
* tracing metadata

---

## Context Explosion

Solved using:

* token-safe context limiting
* chunk prioritization
* reranking

---

# Example Evaluation Output

```text
Grounding Score: 0.91
Judge Score: 4.5 / 5
Citation Score: 1.00
Latency: 1320 ms
Citations Used: 4
```


# How to Run

## Create Environment

```bash
uv venv
```

## Activate

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Run FastAPI

```bash
uvicorn main:app --reload
```

---

# Interview Talking Points

## Retrieval Engineering

* hybrid retrieval strategies
* dense vs sparse retrieval
* reranking architectures
* chunking strategies
* metadata-aware ranking

---

## LLM Systems Engineering

* grounding enforcement
* hallucination prevention
* prompt constraints
* evaluation-driven development

---

## Production AI Engineering

* observability
* structured logging
* async architecture
* caching
* evaluation pipelines
* schema-driven design

---

## AI Evaluation

* semantic grounding
* LLM-as-a-judge
* citation coverage
* automated QA evaluation

---

# Why This Project Stands Out

Most RAG demos only implement:

* upload PDF
* ask question

This project demonstrates:

* hybrid retrieval
* reranking
* grounding validation
* citation enforcement
* evaluation pipelines
* production observability
* caching
* async architecture
* typed schema design
* guardrails
* real-world AI engineering patterns

---
