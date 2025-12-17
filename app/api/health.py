from fastapi import APIRouter, Depends, Request

from app.predictor import Predictor
from app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor  # type: ignore[attr-defined]


@router.get("", response_model=HealthResponse)
async def health_check(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor.model_loaded)
