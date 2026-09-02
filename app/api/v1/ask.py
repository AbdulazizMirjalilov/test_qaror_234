"""
Question answering API endpoints.

Endpoints stay thin: validation via Pydantic schemas, business logic in
RagService (injected from core/dependencies), errors via the exception
handlers registered in app.main.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.dependencies import RagSvc
from app.schemas.ask import AskRequest, AskResponse

router = APIRouter(prefix="/ask", tags=["Ask"])


@router.post(
    "",
    response_model=AskResponse,
    summary="Savolga javob olish",
    description="Hujjat boʻyicha savolga manba va score bilan toʻliq javob qaytaradi.",
)
async def ask(request: AskRequest, service: RagSvc) -> AskResponse:
    """
    Answer a question about the document.

    - **question**: the question, in Uzbek Latin or Cyrillic script
    """
    return await service.ask(request.question)


@router.post(
    "/stream",
    summary="Savolga javobni oqim koʻrinishida olish",
    description=(
        "NDJSON oqimi: avval `meta` (manba, score), soʻng `token` hodisalari, "
        "oxirida `done` (answer_grounded bilan)."
    ),
)
async def ask_stream(request: AskRequest, service: RagSvc) -> StreamingResponse:
    """Streaming variant of /ask."""
    return StreamingResponse(
        service.ask_stream(request.question),
        media_type="application/x-ndjson",
    )
