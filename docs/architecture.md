
# System Architecture (Actual Implementation)

## Overview

The system is a production-oriented Regulatory RAG platform built around a centralized LangGraph orchestration layer.

The architecture consists of tightly integrated components responsible for:

- document ingestion
- document parsing
- structure-aware chunking
- embedding generation
- PostgreSQL persistence
- hybrid retrieval
- reranking
- grounded answer generation
- guardrail validation
- evaluation
- observability

The system is designed around document-grounded question answering where every response must originate from a selected document.

The architecture prioritizes:

- retrieval quality
- answer grounding
- traceability
- observability
- production reliability

---

# Architectural Overview

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

# Architectural Layers

The implementation can be divided into seven primary layers.

1. API Layer
2. Ingestion Layer
3. Storage Layer
4. Retrieval Layer
5. Generation Layer
6. Evaluation Layer
7. Observability Layer

---

# API Layer

## Framework

FastAPI

The API layer exposes all user-facing functionality.

Endpoints include:

- /ingest
- /query
- /documents
- /documents/titles
- /documents/{title}
- /chunks/{chunk_id}
- /health

The API layer is responsible for:

- request validation
- response serialization
- exception handling
- orchestration invocation

The API layer does not contain business logic.

Business logic is delegated to pipeline components.

---

# Ingestion Layer

## Core Component

DocumentIngestionPipeline

The ingestion architecture transforms raw documents into retrieval-ready assets.

Major responsibilities:

- document discovery
- structured parsing
- content cleaning
- metadata extraction
- deduplication
- chunk generation
- embedding generation
- database persistence

---

# Parsing Subsystem

## Component

LlamaDocumentParser

The parser uses:

LlamaParse

Configured to:

- preserve hierarchy
- preserve numbering
- preserve section structure

Output includes:

- full document content
- structured sections

The parser becomes the foundation for structure-aware retrieval.

---

# Chunking Subsystem

## Component

RegulatoryChunker

The chunker creates:

- paragraph chunks
- section chunks

Features:

- hierarchical context injection
- sliding overlap
- token-aware chunk sizing
- metadata enrichment

The dual chunk strategy directly supports hybrid retrieval.

---

# Embedding Subsystem

## Component

EmbeddingGenerator

Embedding model:

Amazon Titan Embeddings

Responsibilities:

- chunk embeddings
- query embeddings

Embeddings are generated asynchronously and attached to chunk objects before persistence.

---

# Storage Layer

## Primary Database

PostgreSQL

## Vector Database

pgvector

The architecture intentionally uses PostgreSQL as both:

- transactional datastore
- vector datastore

This eliminates synchronization challenges between multiple databases.

---

# Stored Entities

## Documents

Stores:

- document id
- title
- source
- metadata
- file hash

---

## Chunks

Stores:

- chunk content
- metadata
- embeddings
- token counts

---

## QA Logs

Stores:

- user queries
- generated answers
- citations
- conversation metadata

---

## Evaluation Results

Stores:

- evaluation metrics
- quality measurements
- latency information

---

# Retrieval Layer

## Core Component

HybridRetriever V2

The retrieval architecture combines:

- BM25 retrieval
- vector retrieval

Both retrieval paths execute independently.

Results are later fused into a unified candidate set.

---

# BM25 Retrieval

Uses PostgreSQL Full Text Search.

Retrieves:

- paragraph chunks
- section chunks

Optimized for:

- exact terminology
- clause references
- regulation identifiers

---

# Vector Retrieval

Uses:

- Amazon Titan embeddings
- pgvector similarity search

Retrieves:

- paragraph chunks
- section chunks

Optimized for:

- semantic similarity
- paraphrased questions
- contextual retrieval

---

# Query Expansion

Before vector retrieval:

HybridRetriever expands the query with regulatory context hints.

Purpose:

- improve semantic recall
- increase retrieval coverage

The original query remains unchanged for lexical retrieval.

---

# Dynamic Retrieval Weighting

The retrieval architecture adjusts weighting based on query characteristics.

Factors include:

- numeric references
- query length

This allows retrieval behavior to adapt automatically.

---

# Generation Layer

## Core Component

AnswerGenerator

The generation layer receives reranked chunks and generates grounded answers.

The generator enforces:

- single-document grounding
- citation-based responses
- context limits

Answer generation is intentionally constrained to retrieved evidence.

---

# Prompt Layer

## Component

PromptLibrary

Contains:

- answer generation prompts
- query rewrite prompts

Prompt design explicitly discourages:

- hallucinations
- unsupported claims
- external knowledge usage

---

# Guardrail Layer

## Component

AnswerGuardrails

Responsible for post-generation validation.

Validation includes:

- citation verification
- citation filtering
- grounding enforcement

The guardrail layer operates after generation but before final response construction.

---

# Orchestration Layer

## Core Component

RAGPipeline

The orchestration layer is the central execution engine.

The pipeline coordinates:

- conversation history retrieval
- cache lookup
- query rewriting
- retrieval
- reranking
- answer generation
- validation
- persistence

The pipeline is implemented using LangGraph.

---

# Workflow Graph

The orchestration graph contains:

- rewrite node
- retrieval node
- rerank node
- answer node

All nodes operate on shared RAGState.

This provides deterministic execution flow.

---

# Conversation Layer

## Component

ConversationStore

Responsibilities:

- conversation persistence
- history retrieval
- context management

The architecture supports conversational interactions without exposing retrieval to previous documents.

---

# Caching Layer

## Component

LLMAnswerCacheManager

The cache uses:

- TTLCache
- SHA256 keys

Purpose:

- latency reduction
- inference cost reduction

Caching exists inside the orchestration layer rather than the API layer.

---

# Evaluation Layer

## Component

RAGEvaluator

The evaluator executes after answer generation.

Metrics include:

- semantic grounding score
- citation score
- judge score
- optional RAGAS metrics

Evaluation results are persisted for monitoring and analysis.

---

# Observability Layer

## Structured Logging

The architecture uses:

CustomLogger
Structlog

Every major subsystem emits structured logs.

Captured events include:

- ingestion
- retrieval
- reranking
- generation
- evaluation
- failures

---

# LangSmith Tracing

Several pipeline nodes are instrumented with:

@traceable

Current traces include:

- Retrieval
- Rerank
- Answer Generation

Purpose:

- execution tracing
- debugging
- latency investigation

---

# Exception Architecture

## Component

RegulatoryRAGException

The custom exception framework captures:

- file location
- line number
- stack trace
- error message

The exception architecture provides consistent failure handling across all layers.

---

# Asynchronous Architecture

The entire system is built around asynchronous execution.

Async components include:

- FastAPI endpoints
- ingestion pipeline
- retrieval
- embedding generation
- evaluation
- conversation storage

This allows the architecture to scale beyond simple synchronous request processing.

---

# Dependency Relationships

The architecture follows a strict dependency flow.

```text
API
 ↓
Orchestration
 ↓
Retrieval / Generation
 ↓
Storage
```

Lower layers never depend on higher layers.

This keeps the architecture modular and maintainable.

---

# Technology Stack

## API

- FastAPI
- Pydantic

## Workflow

- LangGraph

## LLM

- Groq

## Embeddings

- Amazon Titan Embeddings

## Retrieval

- PostgreSQL Full Text Search
- pgvector

## Evaluation

- RAGAS (optional)

## Observability

- Structlog
- LangSmith

## Database

- PostgreSQL
- pgvector

---

# Architectural Characteristics

The implementation is designed around:

- document-grounded generation
- hybrid retrieval
- stateful orchestration
- retrieval-first design
- structured observability
- runtime evaluation
- fault isolation
- component modularity

Every major subsystem contributes directly to answer quality, retrieval quality, grounding quality, and production reliability.

---

# Design Decisions & Tradeoffs

## Why PostgreSQL + pgvector Instead of a Dedicated Vector Database

The system uses PostgreSQL as both the transactional database and vector database.

Benefits:

- Single persistence layer
- Strong transactional guarantees
- Simplified operational management
- Native metadata filtering
- Reduced infrastructure complexity

Tradeoff:

- Dedicated vector databases may provide better horizontal scalability for extremely large vector workloads.

The current architecture prioritizes operational simplicity and consistency while maintaining sufficient scalability for large regulatory corpora.

---

## Why Hybrid Retrieval Instead of Vector-Only Retrieval

Regulatory documents contain both semantic information and exact references.

Vector retrieval performs well for:

- paraphrased questions
- conceptual queries
- semantic similarity

BM25 performs well for:

- clause identifiers
- section numbers
- article references
- regulatory terminology

Hybrid retrieval combines both approaches to improve recall while maintaining precision.

---

## Why Paragraph and Section Chunks

The ingestion layer generates two independent chunk indexes.

Paragraph chunks provide:

- high retrieval precision
- fine-grained evidence extraction

Section chunks provide:

- broader context
- long-range reasoning support

The dual-index strategy improves retrieval coverage without sacrificing answer grounding.

---

## Why Cross Encoder Reranking

Retrieval focuses on maximizing recall.

Reranking focuses on maximizing precision.

Separating these responsibilities allows the system to retrieve a larger candidate pool while ensuring only the most relevant evidence reaches answer generation.

---

## Why LangGraph Instead of a Custom Pipeline

LangGraph provides:

- explicit workflow definition
- shared state management
- deterministic execution
- easier extensibility

This simplifies future additions such as:

- agentic workflows
- tool calling
- multimodal retrieval
- multi-agent architectures

---

# Scalability Characteristics

The architecture is designed to support:

- large regulatory document collections
- millions of indexed chunks
- concurrent user workloads
- independent subsystem evolution

Scalability is achieved through separation of concerns across:

- ingestion
- retrieval
- reranking
- generation
- evaluation

This allows individual subsystems to be optimized independently.

---

# Reliability Characteristics

Several safeguards are implemented to improve production reliability.

## Retrieval Safeguards

- metadata validation
- document isolation
- retrieval fallback mechanisms

## Reranking Safeguards

- reranker collapse protection
- fallback to retrieval results

## Generation Safeguards

- citation validation
- grounding validation
- source consistency enforcement

## Evaluation Safeguards

- evaluation failure isolation
- non-blocking quality assessment

---

# Current Limitations

The current implementation intentionally prioritizes reliability and explainability over architectural complexity.

Current limitations include:

- single-document grounding
- in-memory cache implementation
- rule-based retrieval weighting
- static query expansion strategy
- no multimodal retrieval support
- no distributed retrieval architecture
- no streaming answer generation

---

# Future Enhancements

Planned architectural enhancements include:

- multimodal document understanding
- learned retrieval weighting
- adaptive query expansion
- distributed ingestion pipelines
- graph-based retrieval
- agentic workflows
- retrieval feedback optimization
- streaming response generation
- distributed cache infrastructure

---

# What Makes This Architecture Production-Oriented

The architecture extends beyond a traditional RAG implementation through:

- hybrid retrieval
- dual-index chunking
- reranking
- query rewriting
- guardrails
- runtime evaluation
- structured observability
- conversation memory
- response caching
- workflow orchestration

These components collectively address retrieval quality, answer grounding, reliability, monitoring, and maintainability requirements commonly encountered in production AI systems.

---

# Final Outcome

The architecture combines ingestion, retrieval, generation, validation, persistence, evaluation, and observability into a unified RAG platform.

For every query the system:

- retrieves conversation context
- rewrites the query if necessary
- retrieves evidence
- reranks evidence
- generates an answer
- validates citations
- validates grounding
- persists conversation history
- evaluates output quality
- returns a grounded response

This architecture provides a complete production-grade RAG system optimized for regulatory document intelligence.
