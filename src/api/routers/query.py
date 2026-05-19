from fastapi import APIRouter, Request, HTTPException, status
from pydantic import ValidationError
import asyncio

from src.shared.schemas import QueryRequest
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

router = APIRouter(tags=["query"])


# -------------------------------------------------
# Background Evaluation Task
# -------------------------------------------------

async def _run_evaluation_background(
    evaluator,
    query: str,
    filters,
    result,
):
    """
    Run evaluation asynchronously without blocking API response.
    """
    try:
        eval_result = await evaluator.evaluate_sample(
            query=query,
            filters=filters,
            result=result,
        )

        log.info(
            "evaluation_completed",
            grounding_score=eval_result.get("grounding_score"),
            judge_score=eval_result.get("judge_score"),
            citation_score=eval_result.get("citation_score"),
        )

    except Exception as e:
        log.warning(
            "evaluation_failed",
            error=str(e),
        )


# -------------------------------------------------
# Query Endpoint
# -------------------------------------------------

@router.post("/query")
async def query(payload: dict, request: Request):
    """
    Main RAG query endpoint.

    Flow:
        1. Validate request
        2. Execute RAG pipeline
        3. Run async evaluation
        4. Return answer + citations + evaluation
    """

    # -------------------------------------------------
    # Validate payload
    # -------------------------------------------------

    try:
        qreq = QueryRequest.model_validate(payload)

    except ValidationError as e:

        log.warning(
            "query_validation_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # -------------------------------------------------
    # Require document selection
    # -------------------------------------------------

    if not qreq.filters or "title" not in qreq.filters:

        return {
            "answer": "Please select a document before asking a question.",
            "citations": [],
            "evaluation": None,
        }

    log.info(
        "query_received",
        has_filters=bool(qreq.filters),
    )

    # -------------------------------------------------
    # Run RAG pipeline
    # -------------------------------------------------

    try:
        result = await request.app.state.rag_pipeline.run(
            query=qreq.question,
            filters=qreq.filters,
        )

        answer = result.get("answer")

        raw_citations = result.get("citations", [])

        # -------------------------------------------------
        # Convert citations for API response
        # -------------------------------------------------

        citations = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in raw_citations
        ]

        # -------------------------------------------------
        # Run evaluation immediately
        # -------------------------------------------------

        evaluation = await request.app.state.evaluator.evaluate_sample(
            query=qreq.question,
            filters=qreq.filters,
            result=result,
        )

        # -------------------------------------------------
        # Optional background logging
        # -------------------------------------------------

        asyncio.create_task(
            _run_evaluation_background(
                evaluator=request.app.state.evaluator,
                query=qreq.question,
                filters=qreq.filters,
                result=result,
            )
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {
            "answer": answer,
            "citations": citations,
            "evaluation": {
                "grounding_score": evaluation.get("grounding_score"),
                "judge_score": evaluation.get("judge_score"),
                "citation_score": evaluation.get("citation_score"),
                "latency_ms": evaluation.get("latency_ms"),
                "num_citations": evaluation.get("num_citations"),
                "faithfulness": evaluation.get("faithfulness"),
                "answer_relevancy": evaluation.get("answer_relevancy"),
                "context_precision": evaluation.get("context_precision"),
                "context_recall": evaluation.get("context_recall"),
            },
        }

    except RegulatoryRAGException as e:

        log.error(
            "rag_query_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        )

    except Exception as e:

        log.error(
            "unexpected_query_error",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error",
        )