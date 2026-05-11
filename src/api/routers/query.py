from fastapi import APIRouter, HTTPException, Request, status
import json
import asyncio
from src.shared.schemas import QueryRequest
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

router = APIRouter(
    prefix="",
    tags=["RAG"],
)

# -------------------------------------------------
# Background Evaluation Task
# -------------------------------------------------

async def _run_evaluation_background(evaluator, query: str, filters: dict, result: dict):
    """
    Runs evaluation asynchronously without blocking API response.

    This executes:
        - grounding score
        - LLM judge
        - optional RAGAS
        - DB logging

    Failures are logged but do NOT affect API response.
    """

    try:
        await evaluator.evaluate_sample(
            query=query,
            filters=filters,
            result=result
        )

        log.info("background_evaluation_completed")

    except Exception as e:
        log.warning(
            "background_evaluation_failed",
            error=str(e),
        )


# -------------------------------------------------
# Query Endpoint
# -------------------------------------------------

@router.post("/query")
async def query_rag(request: Request):
    """
    Handles RAG-based question answering requests.

    Enhancements:
        - Returns answer immediately
        - Triggers async evaluation in background
        - Stores evaluation results in DB

    Returns:
        Dict:
            - answer
            - citations
    """

    payload = {}

    # ------------------------------
    # Flexible body parsing
    # ------------------------------
    try:
        payload = await request.json()
    except Exception:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    # ------------------------------
    # Normalize filters
    # ------------------------------
    if isinstance(payload.get("filters"), str):
        try:
            payload["filters"] = json.loads(payload["filters"])
        except Exception:
            pass

    # ------------------------------
    # Validate request
    # ------------------------------
    try:
        qreq = QueryRequest.model_validate(payload)
    except Exception as e:
        log.warning("query_validation_failed", error=str(e))

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    log.info(
        "query_received",
        has_filters=bool(qreq.filters),
    )

    # Require document selection
    if not qreq.filters or "title" not in qreq.filters:
        return {
            "answer": "Please select a document before asking a question.",
            "citations": []
        }

    # ------------------------------
    # Run RAG pipeline (FAST PATH)
    # ------------------------------
    try:
        result = await request.app.state.rag_pipeline.run(
            query=qreq.question,
            filters=qreq.filters,
        )

        answer = result.get("answer")
        raw_citations = result.get("citations", [])

        # -------------------------------------------------
        # Convert to dict for API response
        # -------------------------------------------------
        citations = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in raw_citations
        ]

        # ------------------------------
        # Trigger async evaluation
        # ------------------------------
        asyncio.create_task(
            _run_evaluation_background(
                evaluator=request.app.state.evaluator,
                query=qreq.question,
                filters=qreq.filters,
                result=result
            )
        )

        return {
            "answer": answer,
            "citations": citations,
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