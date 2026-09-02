"""
Centralized FastAPI dependencies for clean architecture.

Dependencies are organized by layer:

1. Infrastructure layer: the retriever (embedding model + Chroma), built
   once in the application lifespan and shared via app.state
2. Service layer: business logic services constructed per-request from
   the shared infrastructure
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from app.services.rag import RagService

# =============================================================================
# INFRASTRUCTURE LAYER DEPENDENCIES
# =============================================================================


def get_retriever(request: Request) -> Any:
    """The retriever is built once at startup (see app.main lifespan)."""
    return request.app.state.retriever


RetrieverDep = Annotated[Any, Depends(get_retriever)]


# =============================================================================
# SERVICE LAYER DEPENDENCIES
# =============================================================================


def get_rag_service(retriever: RetrieverDep) -> RagService:
    """Get RagService instance."""
    return RagService(retriever)


RagSvc = Annotated[RagService, Depends(get_rag_service)]
