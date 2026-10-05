from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)
from app.core.logging import get_logger
from app.pipeline import ChatbotPipeline


router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Kiểm tra API đang hoạt động."""
    return HealthResponse()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    tags=["Chat"],
)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Nhận câu hỏi và trả về câu trả lời cùng các nguồn tham khảo."""
    pipeline: ChatbotPipeline | None = getattr(
        request.app.state,
        "pipeline",
        None,
    )
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot chưa sẵn sàng.",
        )

    try:
        return await pipeline.ask(payload.question)
    except Exception:
        logger.exception("Lỗi khi xử lý yêu cầu chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể xử lý câu hỏi vào lúc này.",
        ) from None
