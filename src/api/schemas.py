from pydantic import BaseModel, Field
from typing import Optional

class FeatureResponse(BaseModel):
    eyebrow_raise: float = Field(..., description="Normalized eyebrow raise intensity (0-1)")
    lip_tension: float = Field(..., description="Normalized lip tension intensity (0-1)")
    blink_intensity: float = Field(..., description="Normalized blink intensity (0-1)")
    head_nod: float = Field(..., description="Normalized head nod intensity (0-1)")
    symmetry_delta: float = Field(..., description="Normalized facial symmetry delta (0-1)")

class StressResponse(BaseModel):
    stress_score: float = Field(..., description="Computed stress score (0-1)")
    stress_level: str = Field(..., description="Stress category (Calm, Slight Stress, High Stress)")

class AnalysisResponse(BaseModel):
    face_detected: bool = Field(..., description="True if a face was detected in the frame")
    features: Optional[FeatureResponse] = Field(None, description="Extracted facial features")
    stress: StressResponse = Field(..., description="Stress analysis results")
