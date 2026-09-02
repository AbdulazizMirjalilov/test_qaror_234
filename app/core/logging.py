"""
Logging configuration using Loguru.

This module provides structured logging with:
- Console output with colors
- Rotating file logs (anchored to the project root, not the CWD)
- Request/response logging middleware
- Log levels based on environment
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import PROJECT_ROOT, settings

LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging() -> None:
    """
    Configure loguru logging based on environment.

    - Local: colorized console output with DEBUG level
    - Staging/production: console + rotating file logs with INFO level
    """
    LOG_DIR.mkdir(exist_ok=True)

    logger.remove()

    log_level = "DEBUG" if settings.ENVIRONMENT.value == "local" else "INFO"

    # Console handler with colors
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # File handler - rotating logs
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        level=log_level,
    )

    # Error file handler
    logger.add(
        LOG_DIR / "error_{time:YYYY-MM-DD}.log",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        level="ERROR",
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Logs:
    - Request method, path, and client IP
    - Response status code
    - Request duration
    - Any errors that occur
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process and log each request/response."""
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        start_time = time.time()
        logger.info(f"Request started: {method} {path} from {client_ip}")

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {method} {path} - "
                f"Status: {response.status_code} - "
                f"Duration: {duration:.3f}s"
            )
            return response
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {method} {path} - "
                f"Error: {exc.__class__.__name__} - "
                f"Duration: {duration:.3f}s"
            )
            logger.exception("Exception details:")
            raise
