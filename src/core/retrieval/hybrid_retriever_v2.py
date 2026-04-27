import os
import json
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, START
from src.infra.models.embedder import create_embedder
from src.shared.schemas import RetrievedChunk, RetrievalState
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class HybridRetriever:
    """
    Production-grade hybrid retriever (V2 - structure-aware upgrade).

    Enhancements over base version:
        - Query expansion for improved semantic recall
        - Metadata-aware fusion (section boosting)
        - Annex-aware penalty handling
        - Larger candidate pool for better reranking
        - Fixed reranker execution (sync-safe)

    Combines:
        - BM25 (PostgreSQL full-text search)
        - Dense vector similarity (pgvector)
        - Score-level weighted fusion with dynamic weighting
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        bm25_k: int = 30,
        vector_k: int = 40,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        bm25_timeout: float = 30,
        vector_timeout: float = 30,
    ):
        """
        Initialize the HybridRetriever.

        Parameters:
            top_k (int): Final number of results returned after fusion.
            bm25_k (int): BM25 candidate pool size.
            vector_k (int): Vector search candidate pool size.
            bm25_weight (float): Weight assigned to BM25 scores.
            vector_weight (float): Weight assigned to vector scores.
            bm25_timeout (float): Timeout (seconds) for BM25 retrieval.
            vector_timeout (float): Timeout (seconds) for vector retrieval.

        Raises:
            RuntimeError: If DATABASE_URL is not set.
        """

        self.top_k = top_k
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_timeout = bm25_timeout
        self.vector_timeout = vector_timeout

        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL must be set")

        self.embedder = create_embedder()

        # Optional reranker hook
        self.reranker = None
        self.rerank_k = 20

        self.graph = self._build_graph()

        log.info(
            "hybrid_retriever_v2_initialized",
            top_k=top_k,
            bm25_k=bm25_k,
            vector_k=vector_k,
        )

    # -------------------------------------------------
    # Query Expansion
    # -------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """
        Expands query with regulatory context hints.

        Improves recall for dense vector retrieval.

        Returns:
            str: Expanded query.
        """
        return f"""
        {query}

        Related concepts:
        - definitions
        - requirements
        - thresholds
        - conditions
        """

    # -------------------------------------------------
    # DB Helpers
    # -------------------------------------------------

    async def _get_conn(self):
        try:
            return await asyncpg.connect(self.database_url)
        except Exception as e:
            log.error("db_connection_failed", error=str(e))
            raise

    def _parse_json(self, val):
        return json.loads(val) if isinstance(val, str) else val

    # -------------------------------------------------
    # Score Utilities
    # -------------------------------------------------

    def _min_max_normalize(self, results):
        if not results:
            return {}

        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return {r.chunk_id: 1.0 for r in results}

        return {
            r.chunk_id: (r.score - min_s) / (max_s - min_s)
            for r in results
        }

    def _get_dynamic_weights(self, query: str):
        if any(char.isdigit() for char in query):
            return 0.7, 0.3
        if len(query.split()) <= 3:
            return 0.6, 0.4
        return self.bm25_weight, self.vector_weight

    def _validate_filters(self, results, filters):
        if not filters:
            return

        title_filter = filters.get("title")

        for r in results:
            if title_filter and r.source != title_filter:
                log.error("metadata_leak_detected")
                raise RegulatoryRAGException(
                    "Strict metadata filter violation detected."
                )

    # -------------------------------------------------
    # BM25 Node
    # -------------------------------------------------

    async def _bm25_node(self, state: RetrievalState):
        async def _run():
            conn = await self._get_conn()
            try:
                filters = state.filters or {}
                title_filter = filters.get("title")

                rows = await conn.fetch(
                    """
                    SELECT c.id::text AS chunk_id,
                           c.document_id::text,
                           c.content,
                           ts_rank_cd(
                               to_tsvector('english', c.content),
                               plainto_tsquery('english', $1)
                           ) AS score,
                           d.title AS source,
                           c.metadata
                    FROM public.chunks c
                    JOIN public.documents d ON d.id = c.document_id
                    WHERE to_tsvector('english', c.content)
                          @@ plainto_tsquery('english', $1)
                      AND ($2::text IS NULL OR d.title = $2)
                    ORDER BY score DESC
                    LIMIT $3;
                    """,
                    state.user_query,
                    title_filter,
                    self.bm25_k,
                )

                return {
                    "bm25_results": [
                        RetrievedChunk(
                            chunk_id=r["chunk_id"],
                            document_id=r["document_id"],
                            content=r["content"],
                            score=float(r["score"]),
                            source=r["source"],
                            metadata=self._parse_json(r["metadata"]),
                        )
                        for r in rows
                    ]
                }
            finally:
                await conn.close()

        return await asyncio.wait_for(_run(), timeout=self.bm25_timeout)

    # -------------------------------------------------
    # Vector Node (UPDATED)
    # -------------------------------------------------

    async def _vector_node(self, state: RetrievalState):
        async def _run():
            expanded_query = self._expand_query(state.user_query)

            embedding = await self.embedder.embed_query(expanded_query)
            vector_literal = "[" + ",".join(map(str, embedding)) + "]"

            filters = state.filters or {}
            title_filter = filters.get("title")

            conn = await self._get_conn()
            try:
                rows = await conn.fetch(
                    """
                    SELECT c.id::text AS chunk_id,
                           c.document_id::text,
                           c.content,
                           1 - (c.embedding <=> $1::vector) AS score,
                           d.title AS source,
                           c.metadata
                    FROM public.chunks c
                    JOIN public.documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                      AND ($2::text IS NULL OR d.title = $2)
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $3;
                    """,
                    vector_literal,
                    title_filter,
                    self.vector_k,
                )

                return {
                    "vector_results": [
                        RetrievedChunk(
                            chunk_id=r["chunk_id"],
                            document_id=r["document_id"],
                            content=r["content"],
                            score=float(r["score"]),
                            source=r["source"],
                            metadata=self._parse_json(r["metadata"]),
                        )
                        for r in rows
                    ]
                }
            finally:
                await conn.close()

        return await asyncio.wait_for(_run(), timeout=self.vector_timeout)

    # -------------------------------------------------
    # Fusion Node (UPDATED)
    # -------------------------------------------------

    def _fusion_node(self, state: RetrievalState):
        bm25_results = state.bm25_results or []
        vector_results = state.vector_results or []

        bm25_norm = self._min_max_normalize(bm25_results)
        vector_norm = self._min_max_normalize(vector_results)

        bm25_w, vector_w = self._get_dynamic_weights(state.user_query)

        all_chunks: Dict[str, RetrievedChunk] = {}
        final_scores = {}

        for r in bm25_results + vector_results:
            all_chunks[r.chunk_id] = r

        for chunk_id, chunk in all_chunks.items():

            score = (
                bm25_w * bm25_norm.get(chunk_id, 0.0)
                + vector_w * vector_norm.get(chunk_id, 0.0)
            )

            # Section boost
            if chunk.metadata.get("section"):
                score += 0.1

            # Annex penalty
            if "annex" in str(chunk.metadata.get("section_title", "")).lower():
                score -= 0.05

            final_scores[chunk_id] = score

        fused = sorted(
            all_chunks.values(),
            key=lambda r: final_scores[r.chunk_id],
            reverse=True,
        )[: self.top_k]

        return {"fused_results": fused}

    # -------------------------------------------------
    # Graph
    # -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RetrievalState)

        graph.add_node("bm25", self._bm25_node)
        graph.add_node("vector", self._vector_node)
        graph.add_node("fusion", self._fusion_node)

        graph.add_edge(START, "bm25")
        graph.add_edge(START, "vector")
        graph.add_edge("bm25", "fusion")
        graph.add_edge("vector", "fusion")

        return graph.compile()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes hybrid retrieval pipeline and returns structured chunks.

        Pipeline steps:
            1. Parallel BM25 + vector retrieval
            2. Score normalization and weighted fusion
            3. Optional reranking (if enabled)
            4. Metadata validation (strict filtering)

        Guarantees:
            - Always returns List[RetrievedChunk]
            - No dict leakage (ensures type consistency for downstream pipeline)

        Args:
            query (str): User query.
            filters (Optional[Dict[str, Any]]): Metadata filters (e.g., document title).

        Returns:
            List[RetrievedChunk]: Ranked and filtered chunks.

        Raises:
            RegulatoryRAGException: If retrieval fails.
        """

        log.info("retrieval_started")

        try:
            state = RetrievalState(
                user_query=query,
                filters=filters or {},
            )

            final_state = await self.graph.ainvoke(state)

            raw_results = final_state.get("fused_results") or []

            # -------------------------------------------------
            # Defensive normalization (important)
            # -------------------------------------------------
            results: List[RetrievedChunk] = [
                r if isinstance(r, RetrievedChunk) else RetrievedChunk(**r)
                for r in raw_results
            ]

            # Validate filters (safety)
            self._validate_filters(results, filters)

            # Optional reranking
            if self.reranker and results:
                results = self.reranker.rerank(query, results)

            log.info(
                "retrieval_completed",
                result_count=len(results),
                sample_ids=[r.chunk_id for r in results[:3]],
            )

            return results

        except Exception as e:
            log.error("retrieval_failed", error=str(e))
            raise RegulatoryRAGException(e)