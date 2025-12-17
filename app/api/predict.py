from fastapi import APIRouter, Depends, Request

from app.predictor import Predictor
from app.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor  # type: ignore[attr-defined]


@router.post("", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest, predictor: Predictor = Depends(get_predictor)
) -> PredictionResponse:
    result = predictor.predict(payload.features)
    return PredictionResponse(**result)
