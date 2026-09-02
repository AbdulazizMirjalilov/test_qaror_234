"""
Custom exceptions and exception handlers for the FastAPI application.

This module provides:
- Custom exception classes for common error scenarios
- Global exception handlers for a consistent error response envelope:
  {"success": false, "message": ..., "data": {...}}
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.constants import ErrorMessages


class BaseAPIException(Exception):
    """Base exception class for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ServiceUnavailableException(BaseAPIException):
    """Raised when a required upstream service (e.g. Ollama) is unreachable."""

    def __init__(self, message: str = ErrorMessages.LLM_UNAVAILABLE, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class BadGatewayException(BaseAPIException):
    """Raised when an upstream service answered but with an error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class GatewayTimeoutException(BaseAPIException):
    """Raised when an upstream service did not answer within the timeout."""

    def __init__(self, message: str = ErrorMessages.LLM_TIMEOUT, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details=details,
        )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "data": {"name": exc.__class__.__name__, "details": exc.details},
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": {"name": "HTTPException", "code": exc.status_code},
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    errors = [
        {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "success": False,
            "message": ErrorMessages.VALIDATION_ERROR,
            "data": {"name": "ValidationError", "details": {"errors": errors}},
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.exception("Unhandled exception details:")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": ErrorMessages.INTERNAL_ERROR,
            "data": {"name": "InternalError"},
        },
    )


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
