from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name to value mapping")
    correlation_id: Optional[str] = Field(None, description="Client-supplied trace identifier")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary fraud prediction")
    score: float = Field(..., description="Probability or decision score")
    model_version: Optional[str] = Field(None, description="Model identifier or checksum")
    cached: bool = Field(False, description="Indicates whether response was served from cache")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is in memory")
