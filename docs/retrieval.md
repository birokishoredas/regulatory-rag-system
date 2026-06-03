
# Retrieval System Design (Actual Implementation)

## Overview

The retrieval layer is implemented through the combination of:

- SafeQueryRewriter
- HybridRetriever V2
- CrossEncoderReranker
- RAGPipeline retrieval graph

The objective of the retrieval layer is not simply to find similar chunks. It is designed to maximize recall while maintaining document-level isolation, metadata safety, and answer grounding.

The retrieval architecture is tightly coupled with the ingestion strategy, which creates both paragraph-level and section-level chunks.

---

# Actual Retrieval Flow

```text
User Query
      ↓
Conversation History Analysis
      ↓
Safe Query Rewriting
      ↓
HybridRetriever V2
 ┌─────────────────────┬─────────────────────┐
 │                     │                     │
 ↓                     ↓
BM25 Search      Vector Search
 │                     │
 └──────────┬──────────┘
            ↓
Dynamic Weighted Fusion
            ↓
Metadata Validation
            ↓
Document Filtering
            ↓
Cross Encoder Reranking
            ↓
Top Chunks
            ↓
Answer Generator
```

---

# Query Rewriting

## Component

SafeQueryRewriter

The system does not rewrite every query.

A rewrite only occurs when the query contains ambiguous references such as:

- this
- that
- it
- they
- section
- clause
- article
- above
- previous

If no ambiguity is detected, the original query is used.

---

# Rewrite Context Construction

The rewriter extracts:

- most recent user question
- most recent assistant response

Only minimal conversational context is used.

This prevents excessive prompt growth and reduces hallucination risk.

---

# Rewrite Safety Controls

The rewriter is intentionally conservative.

The rewritten query must:

- remain close to the original question
- avoid introducing new facts
- avoid answering the question
- avoid speculative expansions

If validation fails, the original query is used.

This guarantees retrieval operates on a safe query representation.

---

# Hybrid Retrieval V2

## Component

HybridRetriever

The retriever combines:

- PostgreSQL full-text search
- pgvector semantic search

Both retrieval strategies execute independently and are later merged.

The implementation prioritizes recall before reranking.

---

# Query Expansion

Before vector retrieval, the query is expanded.

The retriever automatically appends regulatory retrieval hints such as:

- scope
- applicability
- definitions
- requirements
- conditions
- approval criteria
- system description

This expansion is used only for semantic retrieval.

The original query remains unchanged for BM25 retrieval.

The goal is to improve dense retrieval recall for regulatory documents.

---

# BM25 Retrieval

## Data Source

PostgreSQL Full Text Search

The retriever executes BM25 retrieval against:

- paragraph chunks
- section chunks

Both chunk categories are stored during ingestion.

---

# Paragraph Chunk Retrieval

Approximately 70% of BM25 candidates originate from paragraph chunks.

Purpose:

- precise evidence retrieval
- clause-level matching
- detailed factual extraction

---

# Section Chunk Retrieval

Approximately 30% of BM25 candidates originate from section chunks.

Purpose:

- broader contextual retrieval
- long-range reasoning support
- retrieval of complete regulatory sections

---

# Vector Retrieval

## Data Source

pgvector

The retriever generates an Amazon Titan embedding for the expanded query.

The embedding is compared against stored chunk vectors.

Similarity scoring uses pgvector distance calculations.

---

# Paragraph Vector Search

Most vector candidates originate from paragraph chunks.

Purpose:

- high precision semantic matching
- localized evidence retrieval

---

# Section Vector Search

Additional candidates originate from section chunks.

Purpose:

- contextual retrieval
- improved recall
- recovery of broader supporting evidence

---

# Dynamic Weight Selection

The retriever does not use fixed fusion weights.

Weights change based on query characteristics.

## Numeric Queries

When the query contains numbers:

- BM25 receives higher priority
- Vector retrieval receives lower priority

Reason:

Regulatory references often contain:

- clause numbers
- article numbers
- regulation identifiers

Exact matching becomes more important.

---

## Short Queries

For very short queries:

- BM25 influence increases

Reason:

Short queries frequently depend on exact terminology.

---

## Normal Queries

Longer natural language questions use the default balance between:

- lexical retrieval
- semantic retrieval

---

# Fusion Stage

The retriever merges:

- BM25 candidates
- Vector candidates

The system performs score normalization before fusion.

Purpose:

- eliminate score scale differences
- allow fair weighting
- create a unified ranking

The fusion stage creates a larger candidate pool for reranking.

---

# Metadata Validation

After fusion, retrieval results are validated.

The retriever checks whether returned chunks respect requested filters.

If a metadata violation is detected:

- retrieval is aborted
- an exception is raised

This prevents document leakage.

---

# Document Isolation

The RAG pipeline performs an additional document-level safety filter.

When a document title is selected:

- only chunks from that document are allowed
- all other chunks are removed

This guarantees answer generation remains grounded in the selected document.

---

# Retrieval Fallback Strategy

The retrieval node contains fallback logic.

If too few chunks are retrieved:

- previously retrieved chunks may be reused

Conditions:

- previous chunks belong to the same document
- current retrieval quality is insufficient

Purpose:

- reduce empty retrieval scenarios
- improve conversational continuity

---

# Cross Encoder Reranking

## Component

CrossEncoderReranker

Model:

cross-encoder/ms-marco-MiniLM-L-6-v2

---

# Metadata-Aware Reranking

The reranker does not evaluate chunk text alone.

Additional metadata is injected:

- section number
- section title

The model evaluates:

- query
- chunk content
- structural context

This improves ranking quality for regulatory documents.

---

# Score Assignment

Every candidate chunk receives a relevance score.

Chunks are sorted by:

- semantic relevance
- lexical relevance
- structural context relevance

Only the highest-ranked chunks are retained.

---

# Reranker Collapse Protection

The pipeline includes a safeguard against aggressive reranking.

If too few chunks survive reranking:

- retrieved chunks are reused
- answer generation continues

Purpose:

- prevent answer failures
- preserve context availability

---

# Retrieval Output

The reranking stage produces the final context set.

Each retrieved chunk contains:

- chunk id
- document id
- chunk content
- retrieval score
- source document
- metadata

This structure is passed directly to answer generation.

---

# Retrieval Logging

The retrieval layer emits structured logs for:

- query received
- retrieval completed
- reranking completed
- chunk counts
- retrieval scores
- document sources
- fallback activation
- retrieval failures

This enables production monitoring and debugging.

---

# Failure Handling

Protected retrieval stages include:

- query rewriting
- BM25 retrieval
- vector retrieval
- score fusion
- metadata validation
- reranking

Failures are propagated through RegulatoryRAGException.

All failures are logged with contextual information.

---

# Relationship With Ingestion

The retrieval architecture depends heavily on ingestion design.

The ingestion layer creates:

- paragraph chunks
- section chunks
- metadata-rich records
- vector embeddings

Without dual-index chunk generation, HybridRetriever V2 would lose a significant portion of its recall capability.

---

# Key Architectural Decisions

1. Conservative query rewriting.
2. Hybrid retrieval instead of vector-only retrieval.
3. Query expansion for semantic recall.
4. Dual-index retrieval using paragraph and section chunks.
5. Dynamic weighting based on query characteristics.
6. Metadata validation for safety.
7. Document-level isolation.
8. Metadata-aware reranking.
9. Reranker collapse protection.
10. Retrieval fallback mechanisms.

---

# Design Decisions & Tradeoffs

## Why Hybrid Retrieval Instead of Vector-Only Retrieval

The system combines BM25 and vector retrieval because regulatory documents contain both semantic concepts and exact references.

Vector retrieval performs well for natural language questions but can miss exact regulatory identifiers.

BM25 performs well for clause references, article numbers, section identifiers, and regulatory terminology.

Combining both retrieval methods improves recall while preserving precision.

---

## Why Paragraph And Section Chunks

Paragraph chunks improve retrieval precision by targeting specific evidence.

Section chunks improve retrieval recall by preserving broader contextual information.

The dual-index strategy allows retrieval to balance detailed evidence retrieval with long-range contextual reasoning.

---

## Why Dynamic Weighting

Different query types require different retrieval behavior.

Numeric queries benefit from stronger lexical matching.

Natural language questions benefit from stronger semantic retrieval.

Dynamic weighting allows retrieval behavior to adapt automatically to query characteristics.

---

## Why Cross Encoder Reranking

Hybrid retrieval prioritizes recall.

The reranker prioritizes precision.

This two-stage retrieval architecture increases the probability that relevant chunks remain available while ensuring only the most relevant chunks reach answer generation.

---

## Why Metadata Validation

Metadata validation prevents retrieval results from violating document filters.

This ensures document isolation and prevents evidence leakage across documents.

# Current Limitations

- Retrieval is restricted to a selected document.
- Query expansion uses static regulatory hints.
- Retrieval weights are rule-based rather than learned.
- PostgreSQL remains a single-node deployment.
- No distributed retrieval architecture currently exists.
- No multimodal retrieval support.

---

# Future Enhancements

- Learned retrieval weighting.
- Adaptive query expansion.
- Hybrid sparse embedding retrieval.
- Knowledge graph augmentation.
- Multimodal retrieval support.
- Distributed retrieval architecture.
- Retrieval quality feedback loops.

---

# Scalability Characteristics

The retrieval architecture is designed to support:

- Millions of indexed chunks.
- Large regulatory corpora.
- Concurrent retrieval requests.
- Independent scaling of retrieval and generation components.

Current architecture separates:

- Retrieval
- Reranking
- Generation

allowing individual subsystems to evolve independently.

---
# Final Outcome

The retrieval layer returns a small set of highly relevant chunks that have been:

- rewritten safely
- retrieved lexically
- retrieved semantically
- fused
- validated
- filtered
- reranked

The final context set is optimized for grounded answer generation, citation extraction, evaluation, and guardrail validation within the RAG pipeline.