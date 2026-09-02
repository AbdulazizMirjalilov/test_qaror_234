"""
FastAPI Application Entry Point.

Configures the FastAPI application with:
- Lifespan events for startup/shutdown (retriever initialization)
- CORS middleware
- Exception handlers
- Logging middleware
- Versioned API routers
- Root and health endpoints
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api import router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import LoggingMiddleware, setup_logging
from app.services.llm import check_ollama


def _build_retriever():
    # Deferred import: pulls in sentence-transformers/torch/chromadb, which
    # are heavy and not needed just to import the app (e.g. in tests, which
    # substitute a fake retriever here).
    from app.services.retriever import Retriever

    return Retriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Startup: logging configuration, embedding model + Chroma index loading.
    """
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT.value}")

    logger.info("Loading embedding model and Chroma index...")
    app.state.retriever = _build_retriever()
    logger.info(f"Retriever ready: {app.state.retriever.count()} chunks indexed")

    logger.info("Application startup complete")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION or "FastAPI application",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Include API routers
    app.include_router(router)

    # Health check endpoints
    @app.get("/", include_in_schema=False)
    def root():
        """Root endpoint."""
        return {
            "message": f"{settings.APP_NAME} is running",
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    @app.get("/health", tags=["Health"])
    async def health_check(request: Request):
        """Readiness probe: verifies both the vector index and Ollama, so a
        load balancer (or a human) can see *which* dependency is broken."""
        retriever = getattr(request.app.state, "retriever", None)
        try:
            index_count = retriever.count() if retriever is not None else 0
        except Exception:
            index_count = 0
        ollama_ok = await check_ollama()

        healthy = ollama_ok and index_count > 0
        body = {
            "status": "ok" if healthy else "degraded",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "index": {"ok": index_count > 0, "chunks": index_count},
            "ollama": {"ok": ollama_ok},
        }
        return JSONResponse(body, status_code=200 if healthy else 503)

    # CORS configuration (origins from settings / QAROR_CORS_ORIGINS env var)
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Request/response logging middleware
    app.add_middleware(LoggingMiddleware)

    return app


app = create_app()
