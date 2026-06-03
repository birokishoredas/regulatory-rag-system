
# Evaluation System Design (Actual Implementation)

## Overview

The evaluation layer is implemented through the `RAGEvaluator` component.

The evaluator operates on outputs produced by the RAG pipeline and measures:

- semantic grounding
- answer quality
- citation utilization
- retrieval effectiveness
- system latency

Unlike offline benchmarking frameworks, this implementation executes during runtime and can evaluate every production query.

Evaluation results are persisted for monitoring and analysis.

---

# Evaluation Flow

```text
User Query
      ↓
RAG Pipeline
      ↓
Retrieved Chunks
      ↓
Generated Answer
      ↓
RAGEvaluator
      ├── Semantic Grounding
      ├── Citation Scoring
      ├── LLM Judge
      └── Optional RAGAS
      ↓
Structured Evaluation Result
      ↓
Database Storage
      ↓
API Response
```

---

# Evaluation Trigger Points

Evaluation is executed inside the `/query` endpoint.

After the RAG pipeline returns:

- answer
- citations

the evaluator is invoked immediately.

This means evaluation is performed on actual production outputs rather than synthetic benchmark datasets.

---

# Double Execution Prevention

The evaluator contains explicit protection against duplicate RAG execution.

If a precomputed pipeline result is provided:

- retrieval is not executed again
- reranking is not executed again
- answer generation is not executed again

The evaluator reuses the existing result.

This avoids unnecessary latency and compute cost.

---

# Evaluation Inputs

The evaluator receives:

## Query

Original user question.

## Filters

Document filters applied during retrieval.

## Answer

Generated answer returned by the RAG pipeline.

## Citations

Retrieved chunks used as supporting evidence.

---

# Citation Normalization

The evaluator performs defensive normalization.

Every citation is converted into a `RetrievedChunk` object.

This guarantees:

- consistent schema
- predictable processing
- type safety

before evaluation begins.

---

# Core Evaluation Metrics

The implementation currently calculates three primary metrics.

1. Semantic Grounding Score
2. LLM Judge Score
3. Citation Score

---

# Semantic Grounding Score

## Purpose

Measures semantic alignment between:

- generated answer
- retrieved evidence

---

# Implementation

The evaluator:

1. Combines retrieved chunk content.
2. Generates an embedding for the answer.
3. Generates an embedding for the combined evidence.
4. Computes cosine similarity.

The embedding model used is the same embedding infrastructure used elsewhere in the system.

---

# Grounding Calculation

The evaluator computes:

- answer embedding
- context embedding

and calculates cosine similarity.

Higher similarity indicates stronger grounding between generated content and supporting evidence.

---

# Failure Handling

If embedding generation fails:

- grounding score becomes 0.0
- evaluation continues

This prevents evaluation failures from affecting query responses.

---

# LLM Judge Evaluation

## Purpose

Provides a qualitative assessment of answer quality.

---

# Evaluation Input

The judge receives:

- original query
- generated answer
- retrieved evidence

The evaluator limits evidence size by using only a subset of retrieved chunks.

This prevents excessive evaluation latency.

---

# Judge Output

The LLM returns a score representing:

- answer quality
- relevance
- supportiveness

The score becomes the judge metric stored in evaluation results.

---

# Failure Handling

If judge evaluation fails:

- score defaults safely
- evaluation pipeline continues

The system does not block user responses.

---

# Citation Score

## Purpose

Measures whether retrieved evidence was effectively used.

---

# Implementation

The evaluator analyzes:

- answer text
- supporting chunks
- citation presence

The score reflects how effectively retrieved evidence contributed to the generated answer.

---

# Citation Metrics Captured

The evaluator records:

- citation count
- citation utilization
- evidence coverage

This provides visibility into answer grounding quality.

---

# Latency Measurement

The evaluator tracks latency.

The measured latency represents:

```text
Pipeline Start
      ↓
Pipeline Completion
```

The latency value is stored alongside evaluation metrics.

---

# Evaluation Result Structure

The evaluator generates a structured result containing:

- query
- filters
- answer
- latency
- grounding score
- judge score
- citation score
- citation count
- status

This structure becomes the primary evaluation record.

---

# Database Persistence

Every evaluation result is stored.

Persisted information includes:

- query
- answer
- metrics
- status
- metadata

This creates a historical evaluation dataset.

---

# Failure Persistence

Failures are also stored.

Captured information includes:

- query
- filters
- error message
- failure status

This enables debugging and root-cause analysis.

---

# Optional RAGAS Integration

The evaluator supports RAGAS.

RAGAS execution is controlled through:

```text
USE_RAGAS
```

environment configuration.

---

# Runtime Activation Logic

RAGAS executes only when:

1. Environment variable is enabled.
2. RAGAS package is installed.

If either condition is missing:

- evaluation continues
- RAGAS is skipped

This prevents dependency failures from affecting production execution.

---

# Supported RAGAS Metrics

When enabled, the evaluator computes:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Results are added to the evaluation payload.

---

# Batch Evaluation

The evaluator supports batch execution.

Multiple queries can be evaluated sequentially.

For each query:

1. Retrieval executes.
2. Generation executes.
3. Evaluation executes.

Results are aggregated into a summary structure.

---

# Cosine Similarity Implementation

Grounding relies on a custom cosine similarity implementation.

The evaluator calculates:

- vector dot product
- vector magnitudes
- normalized similarity score

Protection exists against:

- empty vectors
- zero magnitude vectors

to avoid numerical errors.

---

# Structured Logging

The evaluator emits logs for:

- evaluation started
- evaluation completed
- grounding score
- judge score
- citation score
- RAGAS failures
- persistence failures
- evaluation exceptions

This supports operational monitoring.

---

# Asynchronous Execution

Evaluation runs asynchronously.

The query endpoint performs:

1. Immediate evaluation.
2. Additional background evaluation logging task.

This architecture enables observability without blocking future pipeline extensions.

---

# Relationship With Retrieval

Evaluation depends heavily on retrieval quality.

Grounding score quality is directly influenced by:

- retrieved chunk relevance
- reranker effectiveness
- evidence completeness

Poor retrieval quality immediately impacts evaluation outcomes.

---

# Relationship With Answer Generation

Evaluation validates answer generation output.

Metrics are designed to determine:

- whether generated answers are grounded
- whether evidence was utilized
- whether answers sufficiently address queries

This creates a quality feedback layer above generation.

---

# Production Characteristics

The evaluation system was designed for:

- continuous quality monitoring
- production observability
- retrieval diagnostics
- answer diagnostics
- regression detection

The evaluator can operate continuously on production traffic.

---

# Key Architectural Decisions

1. Runtime evaluation instead of offline-only evaluation.
2. Embedding-based grounding analysis.
3. LLM-as-a-Judge scoring.
4. Citation quality measurement.
5. Persistent metric storage.
6. Optional RAGAS integration.
7. Failure isolation.
8. Asynchronous execution.
9. Structured observability.
10. Reuse of pipeline outputs to avoid duplicate execution.

---

# Final Outcome

For every completed query, the system produces a structured evaluation record containing:

- semantic grounding quality
- answer quality assessment
- citation quality assessment
- retrieval effectiveness indicators
- latency measurements
- optional RAGAS metrics

These metrics provide continuous visibility into RAG system quality and support monitoring, debugging, regression detection, and future optimization efforts.
