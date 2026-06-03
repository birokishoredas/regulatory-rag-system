
# Workflow Orchestration Design (Actual Implementation)

## Overview

The orchestration layer is implemented through the `RAGPipeline` class using LangGraph.

The pipeline coordinates:

- query rewriting
- retrieval
- reranking
- answer generation
- guardrail validation
- citation extraction
- conversation persistence
- caching
- evaluation integration

The orchestration layer is the execution engine responsible for controlling how every query flows through the system.

Unlike a simple retrieve-and-generate workflow, this implementation maintains shared state across multiple stages and applies several recovery mechanisms to improve reliability.

---

# Core Orchestration Component

## Component

RAGPipeline

The pipeline initializes and manages:

- LLMAnswerCacheManager
- AnswerGuardrails
- ConversationStore
- HybridRetriever
- CrossEncoderReranker
- AnswerGenerator
- SafeQueryRewriter

These components are wired together through a LangGraph workflow.

---

# Actual Execution Flow

```text
User Query
      ↓
Load Conversation History
      ↓
Cache Check
      ↓
Query Rewrite Node
      ↓
Retrieval Node
      ↓
Reranking Node
      ↓
Answer Generation Node
      ↓
Guardrail Validation
      ↓
Conversation Persistence
      ↓
Cache Storage
      ↓
Return Result
      ↓
Evaluation Layer
```

---

# LangGraph Implementation

The orchestration layer uses LangGraph StateGraph.

The graph operates on a shared state object.

The graph is responsible for:

- node execution
- state propagation
- execution ordering
- workflow control

The graph allows every stage to operate on a common state structure rather than passing parameters manually between functions.

---

# Shared State Model

## RAGState

The workflow state contains:

- user_query
- filters
- retrieved_chunks
- reranked_chunks
- answer
- citations
- chat_history
- rewritten_query
- previous_chunks

Every node reads from and writes back to this shared state.

---

# Query Entry Point

The orchestration process begins when the `/query` endpoint invokes:

```python
await rag_pipeline.run()
```

The query endpoint performs:

1. request validation
2. document selection validation
3. pipeline execution
4. evaluation execution
5. response construction

The orchestration layer is therefore the central execution path for every user question.

---

# Conversation History Loading

Before retrieval begins, conversation history is loaded.

The pipeline retrieves previous question-answer pairs from:

ConversationStore

History is converted into chat-compatible messages and inserted into workflow state.

Purpose:

- follow-up question support
- query rewriting context
- conversational continuity

---

# Cache Integration

The pipeline integrates directly with:

LLMAnswerCacheManager

The cache key is generated using:

- question
- namespace

The cache uses:

- SHA256 key generation
- TTL expiration
- in-memory storage

Purpose:

- reduce repeated LLM calls
- reduce latency
- lower inference cost

---

# Query Rewrite Node

## Component

SafeQueryRewriter

The rewrite node executes before retrieval.

The node:

1. Inspects the query.
2. Determines whether rewriting is required.
3. Retrieves recent conversational context.
4. Invokes the LLM.
5. Validates rewrite safety.

If rewriting is not required:

- original query is preserved

If rewriting fails:

- original query is preserved

This guarantees retrieval can always proceed.

---

# Retrieval Node

## Component

HybridRetriever

The retrieval node uses:

- BM25 search
- pgvector search

The node executes retrieval using:

- rewritten query if available
- original query otherwise

---

# Document Isolation

After retrieval:

the orchestration layer performs an additional safety filter.

When a document title is selected:

- only chunks from that document remain

Chunks from other documents are removed.

This prevents cross-document contamination.

---

# Retrieval Recovery Mechanism

The retrieval node includes fallback behavior.

When retrieval produces insufficient results:

- previous chunks may be reused

Conditions:

- previous chunks exist
- previous chunks belong to the same document

Purpose:

- improve conversational continuity
- reduce empty retrieval situations

---

# Retrieval Logging

The retrieval node records:

- query
- chunk count
- chunk identifiers
- document sources

This provides traceability and debugging visibility.

---

# Reranking Node

## Component

CrossEncoderReranker

The reranker receives:

- query
- retrieved chunks

The reranker calculates relevance scores and returns the highest ranked chunks.

The node records:

- input chunk count
- output chunk count
- top reranking scores

---

# Reranker Collapse Protection

The orchestration layer contains explicit protection against aggressive reranking.

If reranking leaves too few chunks:

- retrieved chunks are reused

Fallback threshold:

less than three chunks

Purpose:

- preserve context availability
- prevent answer generation failure

---

# Answer Generation Node

## Component

AnswerGenerator

The node attempts answer generation using:

1. reranked chunks
2. retrieval fallback chunks

Generation is attempted in sequence until a valid answer is produced.

---

# Multi-Attempt Strategy

The orchestration layer supports multiple answer attempts.

Execution order:

Primary Attempt:

- reranked chunks

Fallback Attempt:

- retrieved chunks

This improves robustness when reranking quality is poor.

---

# Citation Extraction

After generation:

the pipeline extracts citations directly from generated text.

The implementation searches for:

```text
[1]
[2]
[3]
```

style references.

The referenced chunk indices are mapped back to retrieved chunks.

---

# Citation Validation

## Component

AnswerGuardrails

The orchestration layer validates:

- cited chunk identifiers
- citation integrity

Invalid citations trigger validation failures.

Purpose:

- prevent fabricated citations
- ensure citation consistency

---

# Citation Filtering

The guardrail layer removes citations that do not belong to the active chunk set.

This guarantees only valid evidence can appear in responses.

---

# Grounding Validation

The orchestration layer performs grounding validation.

The validator checks:

- generated answer
- supporting chunks

and verifies sufficient overlap exists.

Purpose:

- detect unsupported answers
- reduce hallucinations

---

# Source Consistency Enforcement

The answer generation layer already restricts answers to a single source document.

The orchestration layer reinforces this behavior through:

- document filtering
- citation validation
- source checking

This ensures answers remain document-grounded.

---

# Conversation Persistence

After successful answer generation:

ConversationStore saves:

- user query
- generated answer
- citations
- metadata

This information becomes available for future conversational turns.

---

# Cache Storage

Successful answers are written back into cache.

Future requests with the same key can bypass generation.

Purpose:

- faster responses
- reduced LLM utilization

---

# LangSmith Integration

Several orchestration nodes are decorated using:

```python
@traceable
```

Current traces include:

- Retrieval
- Rerank
- Answer Generation

Purpose:

- workflow visualization
- latency investigation
- debugging
- performance analysis

---

# Structured Logging

Every orchestration stage emits structured logs.

Captured information includes:

- retrieval metrics
- reranking metrics
- generation metrics
- fallback activation
- citation counts
- failures

The logging layer uses the custom structlog implementation.

---

# Error Handling Strategy

Failures are isolated at the orchestration level.

Recovery mechanisms include:

## Query Rewrite Failure

Fallback:

- original query

---

## Retrieval Weakness

Fallback:

- previous chunks

---

## Reranker Collapse

Fallback:

- retrieval results

---

## Generation Failure

Fallback:

- alternate chunk set

---

## Evaluation Failure

Answer still returned

Evaluation failure does not block responses.

---

# Relationship With Evaluation

The orchestration layer returns:

- answer
- citations

These outputs become inputs to:

RAGEvaluator

The evaluator executes after orchestration completes.

This separation prevents evaluation logic from interfering with answer generation.

---

# Architectural Characteristics

The orchestration layer is designed around:

- stateful workflow execution
- asynchronous operations
- component isolation
- fault recovery
- observability

The workflow is intentionally modular so individual components can evolve independently.

---

# Key Architectural Decisions

1. LangGraph-based workflow execution.
2. Shared state propagation.
3. Safe query rewriting.
4. Hybrid retrieval orchestration.
5. Document isolation.
6. Reranker collapse protection.
7. Multi-attempt answer generation.
8. Citation extraction and validation.
9. Conversation persistence.
10. Cache integration.
11. LangSmith tracing.
12. Evaluation separation.

---

# Final Outcome

The orchestration layer coordinates the entire RAG lifecycle.

For every user query it:

- loads conversational context
- rewrites the query if necessary
- retrieves evidence
- reranks evidence
- generates an answer
- validates citations
- validates grounding
- stores conversation history
- updates cache
- forwards outputs for evaluation

This workflow acts as the central execution engine that connects retrieval, generation, validation, persistence, caching, and evaluation into a single production-ready RAG pipeline.
