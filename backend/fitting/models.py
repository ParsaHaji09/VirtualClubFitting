from pydantic import BaseModel, Field
from typing import Dict
from physics.models import PerformanceMetrics # Cross-reference if needed

class SwingParameters(BaseModel):
    clubhead_speed: float = Field(..., ge=50, le=150, description="Clubhead speed in mph")
    attack_angle: float = Field(..., ge=-10, le=10, description="Attack angle in degrees")
    launch_angle: float = Field(..., ge=5, le=25, description="Launch angle in degrees")
    spin_rate: float = Field(..., ge=1000, le=6000, description="Spin rate in rpm")
    swing_path: float = Field(0, ge=-10, le=10, description="Swing path in degrees")
    face_angle: float = Field(0, ge=-5, le=5, description="Face angle at impact")


class ClubConfiguration(BaseModel):
    loft: float = Field(..., ge=7, le=15)
    shaft_flex: str = Field(..., description="L, A, R, S, X")
    shaft_weight: float = Field(..., ge=40, le=85)
    head_weight: float = Field(..., ge=190, le=210)
    shaft_torque: float = Field(3.0, ge=2, le=6)
    shaft_length: float = Field(45.5, ge=44, le=46.5)


class FittingRecommendation(BaseModel):
    recommended_config: ClubConfiguration
    confidence_score: float
    predicted_improvement: Dict[str, float]
    reasoning: Dict[str, str]