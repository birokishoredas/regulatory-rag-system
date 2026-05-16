import time
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from src.core.pipeline.rag_pipeline import RAGPipeline
from src.infra.db import db_utils
from src.shared.schemas import RetrievedChunk
from logger import GLOBAL_LOGGER as log
load_dotenv()

# -------------------------------------------------
# Optional RAGAS import (graceful fallback)
# -------------------------------------------------

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RAGEvaluator:
    """Unified evaluator with semantic grounding, LLM judge, and optional RAGAS."""

    def __init__(self, pipeline: RAGPipeline, embedder, llm):
        # Read ONLY from .env (default = False)
        use_ragas_env = os.getenv("USE_RAGAS", "false").strip().lower()
        use_ragas = use_ragas_env in {"1", "true", "yes"}

        self.pipeline = pipeline
        self.llm = llm
        self.embedder = embedder

        # Enable only if installed
        self.use_ragas = use_ragas and RAGAS_AVAILABLE

        if use_ragas and not RAGAS_AVAILABLE:
            log.warning("ragas_not_installed")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def evaluate_sample(
        self,
        query: str,
        filters: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query result.

        Key Improvements:
            - Avoids duplicate pipeline execution
            - Accepts precomputed result from query flow
            - Ensures type-safe citation handling
            - Stores structured evaluation in DB

        Args:
            query (str): User query.
            filters (Dict[str, Any]): Query filters.
            result (Optional[Dict]): Precomputed RAG output.

        Returns:
            Dict[str, Any]: Evaluation metrics and metadata.
        """

        start = time.time()

        try:
            # -------------------------------------------------
            # Avoid double execution
            # -------------------------------------------------
            if result is None:
                result = await self.pipeline.run(query=query, filters=filters)

            latency_ms = int((time.time() - start) * 1000)

            answer = result.get("answer", "")

            raw_citations = result.get("citations", [])

            # -------------------------------------------------
            # Defensive normalization
            # -------------------------------------------------
            citations: List[RetrievedChunk] = [
                c if isinstance(c, RetrievedChunk) else RetrievedChunk(**c)
                for c in raw_citations
            ]

            # -------------------------------------------------
            # Core Metrics
            # -------------------------------------------------
            grounding_score = await self._semantic_grounding(answer, citations)
            judge_score = await self._llm_judge(query, answer, citations)
            citation_score = self._citation_score(answer, citations)

            eval_result = {
                "query": query,
                "filters": filters,
                "answer": answer,
                "latency_ms": latency_ms,
                "grounding_score": grounding_score,
                "judge_score": judge_score,
                "citation_score": citation_score,
                "num_citations": len(citations),
                "status": "success",
            }

            # -------------------------------------------------
            # Optional RAGAS
            # -------------------------------------------------
            if self.use_ragas:
                try:
                    ragas_scores = self._run_ragas(query, answer, citations)
                    eval_result["ragas"] = ragas_scores
                except Exception as e:
                    log.warning("ragas_failed", error=str(e))
                    eval_result["ragas_error"] = str(e)

            # -------------------------------------------------
            # FIX 3: Structured DB insert
            # -------------------------------------------------
            await self._store_result(eval_result)

            return eval_result

        except Exception as e:
            log.error("evaluation_failed", error=str(e))

            failure = {
                "query": query,
                "filters": filters,
                "error": str(e),
                "status": "failed",
            }

            await self._store_result(failure)
            return failure

    # -------------------------------------------------
    # Batch Evaluation
    # -------------------------------------------------

    async def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate multiple queries."""

        results = []

        for sample in samples:
            res = await self.evaluate_sample(
                query=sample["query"],
                filters=sample.get("filters", {}),
            )
            results.append(res)

        return self._aggregate(results)

    # -------------------------------------------------
    # Semantic Grounding (Embedding-based)
    # -------------------------------------------------

    async def _semantic_grounding(self, answer: str, chunks: List[RetrievedChunk]) -> float:
        """Compute embedding similarity between answer and context."""

        if not answer or not chunks:
            return 0.0

        try:
            combined_text = " ".join(c.content for c in chunks)

            answer_emb = await self.embedder.embed_query(answer)
            context_emb = await self.embedder.embed_query(combined_text)

            return float(self._cosine_similarity(answer_emb, context_emb))

        except Exception as e:
            log.warning("semantic_grounding_failed", error=str(e))
            return 0.0

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""

        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # -------------------------------------------------
    # LLM-as-Judge
    # -------------------------------------------------

    async def _llm_judge(self, query: str, answer: str, chunks: List[RetrievedChunk]) -> float:
        """Score answer using LLM (1-5 scale)."""

        if not answer:
            return 0.0

        context = "\n\n".join(c.content for c in chunks[:3])

        prompt = f"""
You are an evaluator.

Question:
{query}

Answer:
{answer}

Context:
{context}

Evaluate:
1. Is the answer supported by the context?
2. Is it complete?
3. Is it correct?

Return ONLY a score from 1 to 5.
"""

        try:
            response = await self.llm.ainvoke(prompt)
            score_text = response.content.strip()
            score = float(score_text.split()[0])
            return max(1.0, min(5.0, score))

        except Exception as e:
            log.warning("llm_judge_failed", error=str(e))
            return 0.0

    # -------------------------------------------------
    # Citation Score
    # -------------------------------------------------

    def _citation_score(self, answer: str, chunks: List[RetrievedChunk]) -> float:
        """Check citation coverage."""

        if not chunks:
            return 0.0

        cited = answer.count("[")
        return min(1.0, cited / len(chunks))

    # -------------------------------------------------
    # RAGAS Integration
    # -------------------------------------------------

    def _run_ragas(self, query: str, answer: str, chunks: List[RetrievedChunk]) -> Dict[str, float]:
        """Run RAGAS metrics."""

        contexts = [c.content for c in chunks]

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })

        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return {
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "context_recall": float(scores["context_recall"]),
        }

    # -------------------------------------------------
    # DB Storage
    # -------------------------------------------------

    async def _store_result(self, result: Dict[str, Any]):
        """
        Persist evaluation results using structured schema.
        """

        try:
            async with db_utils.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_logs (
                        query,
                        answer,
                        status,
                        latency_ms,
                        grounding_score,
                        judge_score,
                        citation_score,
                        ragas,
                        data
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """,
                    result.get("query"),
                    result.get("answer"),
                    result.get("status"),
                    result.get("latency_ms"),
                    result.get("grounding_score"),
                    result.get("judge_score"),
                    result.get("citation_score"),
                    json.dumps(result.get("ragas")),
                    json.dumps(result),
                )

        except Exception as e:
            log.warning("eval_store_failed", error=str(e))

    # -------------------------------------------------
    # Aggregation
    # -------------------------------------------------

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate batch results."""

        success = [r for r in results if r["status"] == "success"]

        if not success:
            return {"total": len(results), "success": 0}

        return {
            "total": len(results),
            "success": len(success),
            "failed": len(results) - len(success),
            "avg_latency": int(sum(r["latency_ms"] for r in success) / len(success)),
            "avg_grounding": round(sum(r["grounding_score"] for r in success) / len(success), 3),
            "avg_judge_score": round(sum(r["judge_score"] for r in success) / len(success), 3),
            "avg_citation_score": round(sum(r["citation_score"] for r in success) / len(success), 3),
            "details": results,
        }