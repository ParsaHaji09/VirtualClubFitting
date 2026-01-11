from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path

from physics.engine import PhysicsEngine
from fitting.engine import FittingEngine

from physics.models import TrajectoryPoint, PerformanceMetrics
from fitting.models import ClubConfiguration, SwingParameters, FittingRecommendation

app = FastAPI(title="Virtual Golf Fitting API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationResult(BaseModel):
    trajectory: List[TrajectoryPoint]
    metrics: PerformanceMetrics
    club_config: ClubConfiguration

physics_engine = PhysicsEngine()
fitting_engine = FittingEngine()


@app.get("/")
async def root():
    return {
        "message": "Virtual Golf Fitting API",
        "version": "1.0.0",
        "endpoints": ["/simulate", "/fit", "/health"]
    }


@app.post("/simulate", response_model=SimulationResult)
async def simulate_shot(
    swing_params: SwingParameters,
    club_config: Optional[ClubConfiguration] = None
):
    """Simulate ball flight for given swing and club parameters"""
    try:
        if club_config is None:
            club_config = ClubConfiguration(
                loft=10.5, shaft_flex="R", shaft_weight=60,
                head_weight=200, shaft_torque=3.5, shaft_length=45.5
            )
        
        trajectory, metrics = physics_engine.simulate_flight(swing_params, club_config)
        
        trajectory_points = [
            TrajectoryPoint(
                time=i * physics_engine.dt,
                x=state.x,
                y=state.y,
                z=state.z,
                vx=state.vx,
                vy=state.vy,
                vz=state.vz
            )
            for i, state in enumerate(trajectory[::10])  # Downsample
        ]
        
        return SimulationResult(
            trajectory=trajectory_points,
            metrics=metrics,
            club_config=club_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fit", response_model=FittingRecommendation)
async def fit_clubs(swing_params: SwingParameters):
    try:
        return fitting_engine.recommend_configuration(swing_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "physics_engine": "operational"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)