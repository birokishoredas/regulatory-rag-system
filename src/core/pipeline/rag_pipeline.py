from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START
from src.core.retrieval.hybrid_retriever_v2 import HybridRetriever
from src.core.memory.conversation_store import ConversationStore
from src.core.generation.answer_generation import AnswerGenerator
from src.core.reranking.reranker import CrossEncoderReranker
from src.shared.schemas import RAGState
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException
import re
from src.core.cache.cache_manager import LLMAnswerCacheManager
from langsmith import traceable, get_current_run_tree
from src.core.guardrails.guardrails import AnswerGuardrails
from src.core.rewrite.query_rewriter import SafeQueryRewriter


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline using LangGraph.

    Coordinates query rewriting, retrieval, reranking, answer generation,
    caching, guardrails validation, and conversation persistence.

    Attributes:
        cache (LLMAnswerCacheManager): In-memory cache for responses.
        guardrails (AnswerGuardrails): Post-generation validation layer.
        conversation_store (ConversationStore): Stores and retrieves chat history.
        retriever (HybridRetriever): Hybrid retrieval component (vector + BM25).
        reranker (CrossEncoderReranker): Reranks retrieved chunks.
        answer_generator (AnswerGenerator): Generates final answers using LLM.
        graph (CompiledGraph): LangGraph execution graph.
    """
    
    def __init__(self):
        """
        Initializes all pipeline components and builds the execution graph.
        """
        self.cache = LLMAnswerCacheManager(memory_maxsize=1000,
                                           memory_ttl_seconds=300)
        self.guardrails = AnswerGuardrails()
        self.conversation_store = ConversationStore()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.rewriter = SafeQueryRewriter(self.answer_generator.llm)
        self.graph = self._build_graph()

        log.info("rag_pipeline_initialized")

    # -------------------------------------------------
    # LangGraph Nodes
    # ------------------------------------------------

    @traceable(name="Retrieval")
    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Retrieves relevant chunks using hybrid retrieval and applies safety filters.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Retrieved chunks and updated previous chunks.

        Raises:
            Exception: If retrieval fails.
        """
        try:
            query = state.rewritten_query or state.user_query

            chunks = await self.retriever.retrieve(
                query=query,
                filters=state.filters,
            )

            current_doc = (state.filters or {}).get("title")

            if current_doc:
                chunks = [
                    c for c in chunks
                    if c.source == current_doc
                ]

            if len(chunks) < 2 and state.previous_chunks:
                safe_previous_chunks = [
                    c for c in state.previous_chunks
                    if c.source == current_doc
                ]

                if safe_previous_chunks:
                    log.warning("fallback_to_previous_chunks_same_doc")
                    chunks = safe_previous_chunks

            log.info(
                "retrieval_node_completed",
                query=query,
                retrieved_count=len(chunks),
                sample_chunk_ids=[c.chunk_id for c in chunks[:3]],
                sources=list(set([c.source for c in chunks])),
            )

            return {
                "retrieved_chunks": chunks,
                "previous_chunks": chunks,
            }

        except Exception as e:
            log.error("retrieval_node_failed", error=str(e))
            raise
    
    @traceable(name="Rerank")
    async def rerank_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Reranks retrieved chunks based on relevance to the query.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Reranked chunks.
        """
        chunks = state.retrieved_chunks or []

        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=chunks,
        )

        log.info(
            "rerank_node_completed",
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            top_scores=[round(c.score, 3) for c in reranked[:3]],
        )

        return {"reranked_chunks": reranked}
    
    async def rewrite_node(self, state: RAGState):
        """
        Rewrites user query using chat history to improve retrieval quality.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Rewritten query.
            """
        rewritten_query = await self.rewriter.rewrite(
            query=state.user_query,
            chat_history=state.chat_history
        )

        return {"rewritten_query": rewritten_query}

    @traceable(name="Answer_Generation")
    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Generates final answer with validation, retry logic, and guardrails enforcement.

        Fixes:
            - Prevents reranker collapse (ensures minimum context)
            - Adds fallback to retrieved chunks when reranker is too aggressive
        """

        attempts = [
            ("primary", state.reranked_chunks or []),
            ("fallback", state.retrieved_chunks or []),
        ]

        # --------------------------------------------------
        # Prevent reranker collapse
        # --------------------------------------------------
        fixed_attempts = []

        for attempt_name, chunks in attempts:

            if not chunks:
                continue

            # If too few chunks → fallback to retrieved chunks
            if len(chunks) < 3:
                log.warning(
                    "reranker_too_aggressive_fallback",
                    attempt=attempt_name,
                    original_count=len(chunks),
                )

                fallback_chunks = state.retrieved_chunks[:5] if state.retrieved_chunks else []

                fixed_attempts.append((attempt_name, fallback_chunks))
            else:
                fixed_attempts.append((attempt_name, chunks))

        # Replace attempts with fixed version
        attempts = fixed_attempts

        # --------------------------------------------------
        # Answer generation loop
        # --------------------------------------------------
        for attempt_name, chunks in attempts:

            cited_chunks = []

            try:
                log.info(
                    "answer_attempt_started",
                    attempt=attempt_name,
                    chunk_count=len(chunks),
                )

                result = await self.answer_generator.generate(
                    question=state.user_query,
                    retrieved_chunks=chunks,
                    chat_history=state.chat_history,
                )

                answer_text = result.answer

                # --------------------------------------------
                # Extract citations
                # --------------------------------------------
                citation_matches = re.findall(r"\[(\d+)\]", answer_text)

                cited_indices = {
                    int(num) - 1 for num in citation_matches
                }

                raw_cited_chunks = [
                    chunks[idx]
                    for idx in cited_indices
                    if 0 <= idx < len(chunks)
                ]

                # --------------------------------------------
                # Guardrails
                # --------------------------------------------
                cited_chunks = self.guardrails.filter_citations(
                    raw_cited_chunks,
                    chunks,
                )

                self.guardrails.validate_citations(
                    used_chunks=chunks,
                    cited_chunk_ids=[c.chunk_id for c in cited_chunks],
                )

                self.guardrails.enforce_answer_grounded(
                    answer_text,
                    cited_chunks,
                )

                return {
                    "answer": answer_text,
                    "citations": cited_chunks,
                }

            except RegulatoryRAGException as e:
                log.warning(
                    "validation_failed",
                    attempt=attempt_name,
                    reason=str(e),
                    chunk_count=len(chunks),
                    cited_chunks=len(cited_chunks),
                )
                continue

        # --------------------------------------------------
        # Final fallback
        # --------------------------------------------------
        log.warning("all_attempts_failed", query=state.user_query)

        return {
            "answer": "The answer is not clearly supported by the document.",
            "citations": [],
        }

    # -------------------------------------------------
    # Graph wiring
    # -------------------------------------------------

    def _build_graph(self):
        """
        Builds the LangGraph workflow connecting all pipeline nodes.

        Returns:
            CompiledGraph: Executable LangGraph pipeline.
        """
        graph = StateGraph(RAGState)

        graph.add_node("rewrite", self.rewrite_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "answer")

        return graph.compile()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    @traceable(name="RAG_Run")
    async def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline and returns typed results.

        Key Improvements:
            - Ensures citations remain as RetrievedChunk internally
            - Avoids premature serialization (dict conversion)
            - Keeps API layer responsible for output formatting

        Args:
            query (str): User query.
            filters (Optional[Dict[str, Any]]): Retrieval filters.

        Returns:
            Dict[str, Any]:
                - answer (str)
                - citations (List[RetrievedChunk])  ← IMPORTANT CHANGE
        """

        log.info("rag_run_started", has_filters=bool(filters))

        try:
            current_doc = (filters or {}).get("title", "default")
            session_id = f"doc:{current_doc}"

            cache_key = self.cache.make_key(
                question=query,
                namespace=f"rag:{session_id}",
            )

            cached = await self.cache.get(cache_key)

            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata.update({
                    "has_filters": bool(filters),
                    "cache_hit": cached is not None,
                    "query_length": len(query),
                    "document": current_doc,
                })

            if cached:
                log.info("rag_cache_hit", document=current_doc)
                return cached

            chat_history = await self.conversation_store.get_recent_history(
                session_id=session_id,
                limit=4
            )

            previous_chunks = []

            final_state = await self.graph.ainvoke(
                RAGState(
                    user_query=query,
                    filters=filters or {},
                    chat_history=chat_history,
                    previous_chunks=previous_chunks,
                )
            )

            citations = final_state.get("citations", [])

            response = {
                "answer": final_state.get("answer"),
                "citations": citations,
            }

            if response["answer"] and not response["answer"].startswith(
                "The question cannot be answered safely"
            ):
                try:
                    await self.conversation_store.save_qa(
                        query=query,
                        answer=response["answer"],
                        citations=[
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in citations
                        ],
                        metadata={
                            "filters": filters,
                            "session_id": session_id,
                            "document": current_doc,
                        }
                    )
                except Exception as e:
                    log.error("conversation_store_save_failed", error=str(e))

                await self.cache.set(
                    cache_key,
                    {
                        "answer": response["answer"],
                        "citations": [
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in citations
                        ],
                    }
                )

            log.info("rag_run_completed", document=current_doc)

            return response

        except Exception as e:
            log.error("rag_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)