from pydantic import BaseModel, Field
from typing import List
from dataclasses import dataclass

@dataclass
class BallState:
    x: float = 0; y: float = 0; z: float = 0
    vx: float = 0; vy: float = 0; vz: float = 0
    wx: float = 0; wy: float = 0; wz: float = 0

class TrajectoryPoint(BaseModel):
    time: float
    x: float  # distance downrange (m)
    y: float  # height (m)
    z: float  # lateral deviation (m)
    vx: float
    vy: float
    vz: float


class PerformanceMetrics(BaseModel):
    carry_distance: float
    total_distance: float
    apex_height: float
    landing_angle: float
    flight_time: float
    max_height_distance: float
    lateral_deviation: float