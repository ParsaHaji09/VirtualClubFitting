from pydantic import BaseModel, Field
from typing import Dict
from physics.models import PerformanceMetrics # Cross-reference if needed

class SwingParameters(BaseModel):
    clubhead_speed: float = Field(..., ge=50, le=150)
    attack_angle: float = Field(..., ge=-10, le=10)
    launch_angle: float = Field(..., ge=5, le=25)
    spin_rate: float = Field(..., ge=1000, le=6000)
    swing_path: float = Field(0, ge=-10, le=10)
    face_angle: float = Field(0, ge=-5, le=5)

class ClubConfiguration(BaseModel):
    loft: float; shaft_flex: str; shaft_weight: float
    head_weight: float; shaft_torque: float; shaft_length: float

class FittingRecommendation(BaseModel):
    recommended_config: ClubConfiguration
    confidence_score: float
    predicted_improvement: Dict[str, float]
    reasoning: Dict[str, str]