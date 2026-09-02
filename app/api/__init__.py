"""
API Router Configuration.

Aggregates all versioned API routers under the /v1 prefix.
"""

from fastapi import APIRouter

from app.api.v1 import ask

router = APIRouter(prefix="/v1")

router.include_router(ask.router)
