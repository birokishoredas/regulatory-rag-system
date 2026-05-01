from typing import List
from sentence_transformers import CrossEncoder

from src.shared.schemas import RetrievedChunk
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class CrossEncoderReranker:
    """
    Cross-encoder based reranker with structure-aware enhancements.

    Enhancements over base version:
        - Incorporates metadata (section, title) into scoring
        - Improves ranking for structured regulatory documents
        - Optional score threshold filtering
        - Robust fallback behavior

    Attributes:
        model_name (str): Cross-encoder model name.
        top_k (int): Number of top chunks to return.
        min_score (float): Minimum score threshold (optional filtering).
        model (CrossEncoder): Loaded model instance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 6,
        min_score: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize reranker.

        Args:
            model_name (str): HuggingFace model name.
            top_k (int): Number of results to return.
            min_score (float): Optional score threshold.
            device (str): "cpu" or "cuda".

        Raises:
            RegulatoryRAGException: If model fails to load.
        """

        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score

        try:
            self.model = CrossEncoder(
                model_name,
                device=device,
            )

            log.info(
                "reranker_initialized",
                model=model_name,
                top_k=top_k,
                device=device,
            )

        except Exception as e:
            log.error(
                "reranker_initialization_failed",
                error=str(e),
            )
            raise RegulatoryRAGException(e)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder scoring.

        Guarantees:
            - Always returns List[RetrievedChunk]
            - No dict leakage (defensive normalization applied)
            - Maintains metadata consistency

        Args:
            query (str): User query.
            chunks (List[RetrievedChunk]): Retrieved chunks.

        Returns:
            List[RetrievedChunk]: Top-k reranked chunks.
        """

        if not chunks:
            return []

        log.info(
            "reranking_started",
            input_chunks=len(chunks),
        )

        try:
            # -------------------------------------------------
            #  Defensive normalization (important)
            # -------------------------------------------------
            normalized_chunks: List[RetrievedChunk] = [
                c if isinstance(c, RetrievedChunk) else RetrievedChunk(**c)
                for c in chunks
            ]

            # -------------------------------------------------
            # Metadata-aware input construction
            # -------------------------------------------------
            pairs = []

            for c in normalized_chunks:
                section = c.metadata.get("section", "")
                section_title = c.metadata.get("section_title", "")

                enriched_text = f"""
                {c.content}

                Section: {section}
                Title: {section_title}
                """

                pairs.append((query, enriched_text))

            # -------------------------------------------------
            # Score computation
            # -------------------------------------------------
            scores = self.model.predict(pairs)

            reranked: List[RetrievedChunk] = []

            for chunk, score in zip(normalized_chunks, scores):

                score = float(score)

                # Optional filtering
                if score < self.min_score:
                    continue

                # Preserve metadata safely
                metadata = dict(chunk.metadata or {})
                metadata.update(
                    {
                        "rerank_score": score,
                        "reranked": True,
                    }
                )

                reranked.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=score,
                        source=chunk.source,
                        metadata=metadata,
                    )
                )

            # -------------------------------------------------
            # Sort + select top_k
            # -------------------------------------------------
            reranked.sort(key=lambda c: c.score, reverse=True)

            min_chunks = max(3, self.top_k)

            final = reranked[:min_chunks]

            # -------------------------------------------------
            # Fallback safety
            # -------------------------------------------------
            if not final:
                log.warning("reranker_empty_fallback")
                return normalized_chunks[: self.top_k]

            log.info(
                "reranking_completed",
                input_chunks=len(normalized_chunks),
                output_chunks=len(final),
                top_scores=[round(c.score, 3) for c in final[:3]],
            )

            return final

        except Exception as e:
            log.error(
                "reranking_failed",
                error=str(e),
            )

            # Hard fallback (still safe)
            return [
                c if isinstance(c, RetrievedChunk) else RetrievedChunk(**c)
                for c in chunks[: self.top_k]
            ]