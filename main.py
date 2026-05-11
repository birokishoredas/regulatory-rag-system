from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import os
import shutil
from src.core.pipeline.rag_pipeline import RAGPipeline
from src.core.evaluation.evaluator import RAGEvaluator

from src.infra.db.db_utils import init_db_pool, close_db_pool
from logger import GLOBAL_LOGGER as log

# Routers
from src.api.routers import (
    health,
    query,
    ingest,
    document,
    chunk
)

DATA_DIR = "data"


# -------------------------------------------------
# Lifespan (replaces on_event)
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # -------- Startup --------
    try:
        await init_db_pool()

        # Initialize ONCE
        pipeline = RAGPipeline()
        app.state.rag_pipeline = pipeline
        app.state.evaluator = RAGEvaluator(
            pipeline=pipeline,
            llm=pipeline.answer_generator.llm,
            embedder=pipeline.retriever.embedder,
        )

        log.info("api_startup_completed")

    except Exception as e:
        log.error("api_startup_failed", error=str(e))
        raise

    yield

    # -------- Shutdown --------
    try:
        await close_db_pool()
        log.info("db_pool_closed")
    except Exception as e:
        log.error("db_pool_close_failed", error=str(e))

    try:
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            log.info("data_dir_cleaned")
    except Exception as e:
        log.warning("data_dir_cleanup_failed", error=str(e))

# -------------------------------------------------
# App initialization
# -------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

# Static UI files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/templates")


# -------------------------------------------------
# Register Routers
# -------------------------------------------------

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(document.router)
app.include_router(chunk.router)


# -------------------------------------------------
# Root UI
# -------------------------------------------------

@app.get("/")
async def index():
    """Simple UI landing page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": {}},
    )