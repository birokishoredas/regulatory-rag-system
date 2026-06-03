
# Regulatory RAG System

Production-grade Retrieval-Augmented Generation (RAG) platform for regulatory document intelligence. The system combines structure-aware document ingestion, hybrid retrieval, reranking, grounded answer generation, evaluation, guardrails, caching, observability, and workflow orchestration into a single production-ready architecture.

---

# Project Highlights

This project demonstrates end-to-end AI system engineering rather than a simple "upload PDF and ask questions" workflow.

Key capabilities include:

- Structure-aware document ingestion using LlamaParse
- Regulatory-aware chunking with paragraph and section indexes
- Hybrid retrieval (BM25 + Vector Search)
- Dynamic retrieval weighting
- Cross-encoder reranking
- Grounded answer generation with citations
- Citation and grounding validation
- Conversation-aware query rewriting
- Runtime evaluation framework
- Response caching
- Conversation memory
- LangGraph orchestration
- LangSmith tracing
- Structured observability

---

# System Architecture

```text
                        User
                          │
                          ▼
                    FastAPI Layer
                          │
                          ▼
                     RAGPipeline
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
     ▼                    ▼                    ▼
Conversation       Cache Manager      Query Rewriter
History
     │                    │                    │
     └────────────────────┼────────────────────┘
                          ▼
                    HybridRetriever
                          │
         ┌────────────────┼────────────────┐
         │                                 │
         ▼                                 ▼
     BM25 Search                    Vector Search
         │                                 │
         └────────────────┬────────────────┘
                          ▼
                    Score Fusion
                          ▼
                CrossEncoderReranker
                          ▼
                  AnswerGenerator
                          ▼
                   AnswerGuardrails
                          ▼
                   ConversationStore
                          ▼
                     RAGEvaluator
                          ▼
                     API Response
```

---

# Architecture Documentation

The repository contains detailed engineering documentation inside the `docs/` folder.

| Document | Purpose |
|-----------|----------|
| [Architecture](docs/architecture.md) | Complete system architecture, component interactions, dependency flow, storage architecture, orchestration model, observability and production design. |
| [Ingestion Pipeline](docs/ingestion.md) | LlamaParse integration, document processing workflow, chunk generation, embeddings, deduplication, storage strategy and indexing architecture. |
| [Retrieval System](docs/retrieval.md) | Safe query rewriting, HybridRetriever V2, BM25 retrieval, vector retrieval, fusion, reranking and retrieval safeguards. |
| [Workflow Orchestration](docs/orchestration.md) | LangGraph workflow execution, state propagation, caching, memory integration, fallback mechanisms and pipeline coordination. |
| [Evaluation Framework](docs/evaluation.md) | Grounding metrics, citation scoring, LLM-as-a-Judge, RAGAS integration and runtime evaluation architecture. |

These documents describe the actual implementation, architectural decisions, tradeoffs, scalability considerations, and future evolution of the system.

---

# Core Engineering Components

## Document Ingestion

- LlamaParse-based document parsing
- Structured section extraction
- Metadata lineage preservation
- Regulatory-aware chunking
- Paragraph chunk generation
- Section chunk generation
- Amazon Titan embedding generation
- PostgreSQL persistence
- Hash-based deduplication
- Source-based document replacement

---

## Retrieval Layer

### HybridRetriever V2

Combines:

- PostgreSQL Full Text Search (BM25)
- pgvector similarity search

Features:

- Query expansion
- Dynamic retrieval weighting
- Metadata-aware retrieval
- Document isolation
- Score normalization
- Weighted fusion
- Retrieval fallback mechanisms

---

## Reranking Layer

### CrossEncoderReranker

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Responsibilities:

- Relevance scoring
- Precision improvement
- Metadata-aware ranking
- Context prioritization
- Top-K refinement

---

## Answer Generation

### AnswerGenerator

Capabilities:

- Grounded answer generation
- Citation generation
- Single-document reasoning
- Context-aware prompting
- Hallucination reduction

---

## Guardrails

### AnswerGuardrails

Validation performed before response delivery:

- Citation validation
- Citation filtering
- Grounding validation
- Source consistency enforcement
- Unsupported answer detection

---

## Evaluation Framework

### RAGEvaluator

Metrics:

- Semantic Grounding Score
- Citation Score
- Judge Score
- Optional RAGAS Metrics
- Pipeline Latency Metrics

Evaluation results are persisted for monitoring and quality tracking.

---

# Technology Stack

## Backend

- Python
- FastAPI
- AsyncIO
- Pydantic

## Workflow Orchestration

- LangGraph

## LLM Layer

- Groq
- LangChain

## Embeddings

- Amazon Titan Embeddings
- AWS Bedrock

## Retrieval

- PostgreSQL Full Text Search
- pgvector
- Hybrid Retrieval
- Cross-Encoder Reranking

## Evaluation

- RAGAS (Optional)

## Observability

- Structlog
- LangSmith

## Database

- PostgreSQL
- pgvector

---

# Production Features

## Observability

- Structured JSON logging
- Retrieval diagnostics
- Reranking diagnostics
- Evaluation metrics
- Latency monitoring
- LangSmith traces

## Reliability

- Custom exception framework
- Fault isolation
- Recovery mechanisms
- Retrieval fallback
- Reranker collapse protection

## Performance

- Response caching
- Async execution
- Background evaluation
- Optimized retrieval pipeline

---
# Current Limitations

Current implementation limitations include:

- Single-document grounding strategy
- In-memory caching implementation
- Static query expansion
- Rule-based retrieval weighting
- No multimodal retrieval support
- No distributed ingestion architecture
- No streaming response generation

These tradeoffs were intentionally chosen to prioritize reliability, explainability, and maintainability.

---

# Future Enhancements

Potential future enhancements include:

- Multimodal document understanding
- OCR support for scanned documents
- Graph-based retrieval
- Adaptive query expansion
- Learned retrieval weighting
- Distributed ingestion workers
- Streaming response generation
- Agentic workflows
- Retrieval quality feedback loops

---

# Project Structure

```text
REGULATORY-RAG-SYSTEM/
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── ingestion.md
│   ├── orchestration.md
│   └── retrieval.md
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

# How to Run

## Create Virtual Environment

```bash
uv venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

## Run Application

```bash
uvicorn main:app --reload
```

---

# Design Decisions & Tradeoffs

Key architectural decisions made during development include:

- PostgreSQL + pgvector instead of a dedicated vector database to simplify operations and maintain transactional consistency.
- Hybrid retrieval instead of vector-only retrieval to balance semantic understanding and exact regulatory matching.
- Dual-index chunking (paragraph + section) to improve both retrieval precision and recall.
- Cross-encoder reranking to separate recall optimization from precision optimization.
- LangGraph orchestration to provide stateful workflow execution and future extensibility.
- Runtime evaluation to continuously measure answer quality in production.

Detailed discussions are available in the documentation inside the `docs/` directory.

---

# Engineering Deep Dive

## Retrieval Engineering

- Hybrid retrieval architecture
- BM25 vs vector retrieval
- Query expansion strategies
- Dynamic retrieval weighting
- Dual-index retrieval design

## RAG Architecture

- Structure-aware chunking
- Retrieval-first system design
- Grounded generation
- Citation enforcement
- Guardrail architecture

## AI Platform Engineering

- LangGraph orchestration
- Runtime evaluation
- Conversation memory
- Caching strategy
- Observability design

## Production AI Systems

- Fault tolerance
- Monitoring and tracing
- Structured logging
- Evaluation-driven development
- Scalable document intelligence systems

---

# Why This Project Stands Out

This project demonstrates production-grade AI engineering patterns including:

- End-to-end RAG architecture
- Hybrid retrieval
- Dual-index chunking strategy
- Cross-encoder reranking
- Runtime evaluation
- Citation grounding
- Guardrails
- Workflow orchestration
- Observability
- Caching
- Conversation memory
- Structured system design

The focus is not only on answer generation but on building a reliable, measurable, maintainable, and production-ready AI platform.
