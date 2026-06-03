
# Document Ingestion Pipeline (Actual Implementation)

## Overview

The ingestion pipeline is implemented by `DocumentIngestionPipeline` and is responsible for converting uploaded regulatory documents into retrieval-ready records stored in PostgreSQL + pgvector.

Unlike a generic ingestion workflow, this implementation is tightly integrated with:

- LlamaParse-based document parsing
- Structure-aware regulatory chunking
- Dual-index retrieval preparation
- Amazon Titan embeddings
- PostgreSQL + pgvector storage
- Document deduplication and update handling

The ingestion stage directly impacts retrieval accuracy, reranking quality, grounding scores, citation quality, and final answer generation.

---

# Actual Processing Flow

```text
User Uploads PDF
        ↓
/ingest Endpoint
        ↓
Document Saved To data/
        ↓
DocumentIngestionPipeline
        ↓
Database Initialization
        ↓
LlamaParse Structured Parsing
        ↓
Section Extraction
        ↓
Content Cleaning
        ↓
Title Extraction
        ↓
Metadata Generation
        ↓
File Hash Computation
        ↓
Deduplication Check
        ↓
Paragraph Chunk Generation
        ↓
Section Chunk Generation
        ↓
Embedding Generation
        ↓
PostgreSQL + pgvector Storage
```
---

# API Entry Point

The ingestion process starts from the `/ingest` FastAPI endpoint.

What happens:

1. The uploaded file is saved into the local `data/` directory.
2. A singleton `DocumentIngestionPipeline` instance is invoked.
3. The pipeline processes every supported file in the configured document folder.
4. Results are returned to the API caller.

The endpoint also performs structured logging and exception handling through `RegulatoryRAGException`.

---

# Supported File Types

Current implementation supports:

- PDF
- Markdown (.md)
- Text (.txt)

Unsupported files are ignored during ingestion discovery.

---

# Database Initialization

Before processing any document:

- PostgreSQL connection pool is initialized.
- SSL-enabled asyncpg connection pool is created.
- Pool reuse is enabled to avoid repeated initialization.
- Database lifecycle is managed centrally.

This ensures all downstream operations use the same connection infrastructure.

---

# Document Parsing

## Parser Used

LlamaParse

## Parser Configuration

The parser is configured to:

- Preserve section numbering
- Preserve hierarchy
- Preserve subsection structure
- Return markdown output
- Prevent section merging

Example structures retained:

- 1.
- 1.1
- 1.1.1

This is critical because regulatory retrieval often depends on section-level context.

---

# Structured Parsing Output

The parser returns:

## Content

Complete document text.

## Sections

Structured section objects containing:

- section number
- section title
- section content

The pipeline requests both outputs simultaneously.

```python
{
    "content": "...",
    "sections": [...]
}
```

The section structure becomes the foundation for chunk generation.

---

# Content Cleaning

The pipeline cleans parsed content before indexing.

Operations include:

- whitespace normalization
- formatting cleanup
- parser artifact reduction

The goal is to ensure embedding generation receives clean and consistent text.

---

# Title Extraction

The pipeline derives a document title.

The title is later used in:

- retrieval filtering
- citation generation
- metadata lineage
- source identification

The title is prepended to document content before chunking.

---

# Metadata Extraction

Document metadata is generated before chunk creation.

Metadata is propagated into every chunk.

Typical metadata includes:

- title
- source
- section number
- section title
- chunk type

This metadata becomes available during retrieval, reranking and answer generation.

---

# Deduplication Logic

The ingestion pipeline performs file-level deduplication.

## File Hash Generation

A hash is computed from document content.

## Existing Document Check

The system checks whether the same hash already exists.

If a matching hash is found:

- ingestion is skipped
- duplicate storage is prevented

---

# Document Update Handling

The system also checks for existing documents using source path.

If the same source already exists:

1. Existing document is identified.
2. Existing chunks are removed.
3. New version is ingested.

This enables document replacement without accumulating stale chunks.

---

# Chunking Architecture

The implementation does not use simple fixed-size chunking.

It uses a specialized RegulatoryChunker.

Key capabilities:

- section-aware chunking
- hierarchical context injection
- sliding overlap
- token-aware chunk sizing
- metadata enrichment

---

# Hierarchical Context Injection

Before chunk creation, contextual prefixes are added.

Each chunk contains:

- document title
- section number
- section title

This improves retrieval quality because embeddings contain structural context.

---

# Dual Index Strategy

The pipeline creates two chunk categories.

## Paragraph Chunks

Purpose:

Fine-grained retrieval.

Characteristics:

- paragraph based
- smaller units
- higher precision

Chunk Type:

```text
paragraph
```

---

## Section Chunks

Purpose:

Broader contextual retrieval.

Characteristics:

- entire section content
- larger context window
- supports long-form reasoning

Chunk Type:

```text
section
```

---

# Why Two Chunk Types Exist

Many RAG systems fail because they retrieve only small chunks.

This implementation indexes:

- detailed paragraph evidence
- larger section context

During retrieval both chunk types can participate in ranking.

This improves recall while preserving precision.

---

# Sliding Window Overlap

The chunker retains a configurable percentage of the previous chunk.

Current behavior:

- overlap ratio based
- preserves context continuity
- reduces boundary information loss

This is particularly important for regulatory clauses spanning multiple paragraphs.

---

# Embedding Generation

After chunk generation:

1. Chunk text is collected.
2. Amazon Titan embeddings are generated.
3. Embeddings are attached to chunk objects.
4. Embedding timestamps are recorded in metadata.

Generated vectors become the semantic representation used by pgvector.

---

# Storage Layer

Data is stored in PostgreSQL.

Embeddings are stored using pgvector.

Stored entities include:

## Documents

Stores:

- document id
- title
- source
- file hash
- metadata

## Chunks

Stores:

- chunk content
- metadata
- token count
- embedding vector
- chunk type

---

# Retrieval Preparation

The ingestion stage explicitly prepares data for the retrieval architecture.

The retriever later uses:

- BM25 search
- pgvector similarity search
- paragraph chunks
- section chunks
- metadata filtering

Because ingestion stores both paragraph and section chunks, retrieval can perform multi-granularity search.

---

# Observability

Structured logging is emitted throughout ingestion.

Important events include:

- ingestion started
- parsing started
- parsing completed
- chunking completed
- embedding completed
- document skipped
- document updated
- ingestion completed
- failures

Logs are emitted through the custom structlog-based logger.

---

# Failure Handling

The ingestion layer uses RegulatoryRAGException for error propagation.

Protected stages:

- database initialization
- parsing
- chunking
- embedding generation
- document storage

Failures are logged with:

- source file
- line number
- stack trace
- error message

---

# Scalability Characteristics

The implementation is designed for:

- large regulatory documents
- multi-document corpora
- future parallel ingestion
- dual-index retrieval architectures

The architecture separates:

- parsing
- chunking
- embedding
- storage

allowing individual components to evolve independently.

---

# Key Architectural Decisions

1. Structured parsing instead of raw text extraction.
2. Regulatory-aware chunking instead of fixed chunking.
3. Dual-index chunk generation (paragraph + section).
4. Hash-based deduplication.
5. Source-based document replacement.
6. Metadata propagation into every chunk.
7. Amazon Titan embeddings.
8. PostgreSQL + pgvector storage.
9. Full traceability through metadata lineage.
10. Retrieval-oriented indexing strategy.

---
# Design Decisions & Tradeoffs

## Why LlamaParse Instead of Traditional PDF Extraction

Regulatory documents typically contain:

- hierarchical numbering
- nested sections
- complex formatting
- structured references

Traditional PDF parsers often flatten document structure and lose relationships between sections.

LlamaParse preserves document hierarchy and section boundaries, enabling structure-aware chunking and retrieval.

Tradeoff:

- Higher parsing cost compared to lightweight text extraction libraries.
- Additional external dependency.

The implementation prioritizes retrieval quality over ingestion speed.

---

## Why Regulatory-Aware Chunking Instead of Fixed-Size Chunking

Fixed-size chunking frequently breaks:

- clauses
- definitions
- requirements
- approval conditions

The RegulatoryChunker uses document structure to preserve semantic boundaries.

Benefits:

- improved retrieval precision
- improved answer grounding
- better citation quality

Tradeoff:

- More complex ingestion logic.
- Additional metadata processing.

---

## Why Paragraph And Section Chunks

A single chunking strategy cannot optimize both precision and recall.

Paragraph chunks provide:

- precise evidence retrieval
- fine-grained citations

Section chunks provide:

- broader context
- long-range reasoning support

The dual-index strategy improves retrieval coverage while preserving answer grounding.

---

## Why Metadata Enrichment

Every chunk receives structural metadata including:

- title
- section number
- section title
- chunk type

Benefits:

- retrieval filtering
- reranking improvements
- citation traceability
- document lineage tracking

Tradeoff:

- increased storage footprint

The additional metadata is justified by retrieval and observability gains.

---

## Why PostgreSQL + pgvector Storage

The ingestion pipeline stores both metadata and embeddings inside PostgreSQL.

Benefits:

- single persistence layer
- transactional consistency
- simplified infrastructure
- unified backup strategy

Tradeoff:

- dedicated vector databases may provide stronger horizontal scaling characteristics.

The current architecture prioritizes operational simplicity and maintainability.

---

# Reliability Characteristics

The ingestion pipeline includes several safeguards.

## Duplicate Protection

Hash-based deduplication prevents identical documents from being indexed multiple times.

---

## Update Protection

Source-based replacement ensures stale document versions are removed before re-indexing.

---

## Metadata Consistency

Metadata is generated before chunk creation and propagated to all chunks.

This ensures downstream retrieval and citation systems receive consistent information.

---

## Failure Isolation

Failures occurring in:

- parsing
- chunking
- embedding generation
- persistence

are isolated through RegulatoryRAGException and structured logging.

---

# Current Limitations

Current implementation limitations include:

- PDF-centric ingestion workflow
- no OCR processing pipeline
- no image understanding
- no table-specific indexing strategy
- sequential document processing
- no distributed ingestion architecture
- no incremental chunk-level updates

The implementation prioritizes correctness and retrieval quality over ingestion throughput.

---

# Future Enhancements

Planned ingestion enhancements include:

- OCR support
- multimodal document ingestion
- table-aware chunking
- image extraction and indexing
- distributed ingestion workers
- incremental re-indexing
- ingestion queue architecture
- automatic document classification

---

# Scalability Characteristics

The ingestion architecture is designed to support:

- large regulatory documents
- multi-document corpora
- millions of downstream chunks
- future parallel processing

Scalability is achieved through separation of:

- parsing
- chunking
- embedding generation
- persistence

allowing individual components to evolve independently.

---

# What Makes This Ingestion Pipeline Production-Oriented

The ingestion layer extends beyond basic document parsing through:

- structured parsing
- metadata lineage preservation
- dual-index chunk generation
- deduplication
- update handling
- embedding generation
- retrieval-oriented indexing
- structured observability
- fault isolation

These capabilities ensure documents are transformed into retrieval-ready assets optimized for downstream retrieval, reranking, grounding, citation generation, and evaluation.

---
# Final Outcome

After ingestion completes, every document exists in PostgreSQL as:

- a document record
- multiple paragraph chunks
- multiple section chunks
- embedding vectors
- retrieval metadata

This indexed representation becomes the foundation for hybrid retrieval, reranking, grounded answer generation, citation generation, evaluation, and guardrail validation across the RAG system.
