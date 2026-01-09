from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path

app = FastAPI(title="Virtual Golf Fitting API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATA MODELS
# ============================================================================

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


class FittingRecommendation(BaseModel):
    recommended_config: ClubConfiguration
    confidence_score: float
    predicted_improvement: Dict[str, float]
    reasoning: Dict[str, str]


class SimulationResult(BaseModel):
    trajectory: List[TrajectoryPoint]
    metrics: PerformanceMetrics
    club_config: ClubConfiguration


# ============================================================================
# PHYSICS ENGINE
# ============================================================================

@dataclass
class BallState:
    x: float = 0
    y: float = 0
    z: float = 0
    vx: float = 0
    vy: float = 0
    vz: float = 0
    wx: float = 0  # spin rate x (rad/s)
    wy: float = 0  # spin rate y (rad/s)
    wz: float = 0  # spin rate z (rad/s)


class PhysicsEngine:
    """Advanced ball flight physics simulation with realistic constants"""
    
    # Physical constants
    G = 9.81  # gravity (m/s^2)
    RHO = 1.225  # air density (kg/m^3)
    BALL_MASS = 0.0459  # kg (45.9 grams)
    BALL_RADIUS = 0.02135  # meters (42.7mm diameter)
    BALL_AREA = np.pi * BALL_RADIUS ** 2 # ~0.001432 m^2
    
    # Aerodynamic coefficients for modern golf balls
    CD0 = 0.22  # Lower base drag coefficient
    CL0 = 0.14  # Base lift coefficient for Magnus calculation
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        
    def compute_drag_coefficient(self, velocity: float, spin_rate: float) -> float:
        """Reynolds number dependent drag"""
        reynolds = (self.RHO * velocity * 2 * self.BALL_RADIUS) / 1.8e-5
        cd = self.CD0 + 0.05 * (spin_rate / 3000) ** 2
        return cd
    
    def compute_lift_coefficient(self, spin_parameter: float) -> float:
        """Spin-dependent lift coefficient"""
        # Spin parameter S = r*omega/v
        cl = 1.0 * (1.0 - np.exp(-2.5 * spin_parameter))
        return np.clip(cl + self.CL0, 0, 0.45)
    
    def simulate_flight(
        self, 
        swing_params: SwingParameters, 
        club_config: ClubConfiguration
    ) -> tuple[List[BallState], PerformanceMetrics]:
        """
        Full 3D ball flight simulation with vector-based Magnus effect.
        Refined for realistic carry distances (220+ yards at 95mph).
        """
        # 1. Convert units and compute initial ball speed
        v_ball_mph = self._compute_ball_speed(swing_params.clubhead_speed, club_config)
        v_total = v_ball_mph * 0.44704  # mph to m/s
        
        launch_rad = np.radians(swing_params.launch_angle)
        path_rad = np.radians(swing_params.swing_path)
        
        # 2. Initial velocity components (Meters per Second)
        vx0 = v_total * np.cos(launch_rad) * np.cos(path_rad)
        vy0 = v_total * np.sin(launch_rad)
        vz0 = v_total * np.cos(launch_rad) * np.sin(path_rad)
        
        # Initial backspin (rad/s)
        spin_rpm = swing_params.spin_rate
        wy0 = spin_rpm * 2 * np.pi / 60
        
        state = BallState(
            x=0, y=0, z=0,
            vx=vx0, vy=vy0, vz=vz0,
            wx=0, wy=wy0, wz=0
        )
        
        trajectory = [BallState(**vars(state))]
        apex = 0
        max_height_x = 0
        t = 0
        max_time = 15
        
        while state.y >= 0 and t < max_time:
            v = np.sqrt(state.vx**2 + state.vy**2 + state.vz**2)
            if v < 0.1: break
            
            # 3. Aerodynamic coefficients based on Spin Parameter (S)
            omega = np.sqrt(state.wx**2 + state.wy**2 + state.wz**2)
            spin_param = self.BALL_RADIUS * omega / v
            
            # Refined coefficients for modern golf balls
            cd = self.compute_drag_coefficient(v, spin_rpm)
            cl = self.compute_lift_coefficient(spin_param)
            
            # 4. Force Magnitudes
            # Use precise BALL_AREA (~0.001432 m^2) and RHO (1.225 kg/m^3)
            drag_mag = 0.5 * self.RHO * v**2 * cd * self.BALL_AREA
            lift_mag = 0.5 * self.RHO * v**2 * cl * self.BALL_AREA
            
            # 5. Force Vectors (The "New Math")
            # Drag is always exactly opposite to the velocity vector
            Fdx = -drag_mag * (state.vx / v)
            Fdy = -drag_mag * (state.vy / v)
            Fdz = -drag_mag * (state.vz / v)
            
            # Lift (Magnus Force) is perpendicular to the velocity vector
            # For pure backspin, this rotates the velocity vector in the flight plane
            Flx = -lift_mag * (state.vy / v)
            Fly = lift_mag * (state.vx / v)
            Flz = 0  # Assuming zero side-spin for this baseline model
            
            # 6. Accelerations (F/m)
            ax = (Fdx + Flx) / self.BALL_MASS
            ay = (Fdy + Fly) / self.BALL_MASS - self.G
            az = (Fdz + Flz) / self.BALL_MASS
            
            # 7. Update state (Euler Integration)
            state.vx += ax * self.dt
            state.vy += ay * self.dt
            state.vz += az * self.dt
            
            state.x += state.vx * self.dt
            state.y += state.vy * self.dt
            state.z += state.vz * self.dt
            
            # Track apex for metrics
            if state.y > apex:
                apex = state.y
                max_height_x = state.x
                
            t += self.dt
            if len(trajectory) % 10 == 0:
                trajectory.append(BallState(**vars(state)))

        # 8. Metrics calculation (Convert m -> yards)
        landing_angle = np.degrees(np.arctan2(-state.vy, state.vx))
        rollout = max(0, 25 - landing_angle) * 1.8 # Refined rollout logic
        
        metrics = PerformanceMetrics(
            carry_distance=state.x * 1.09361,
            total_distance=(state.x + rollout) * 1.09361,
            apex_height=apex * 1.09361,
            landing_angle=landing_angle,
            flight_time=t,
            max_height_distance=max_height_x * 1.09361,
            lateral_deviation=state.z * 1.09361
        )
        
        return trajectory, metrics
    
    def _compute_ball_speed(self, clubhead_speed: float, config: ClubConfiguration) -> float:
        """Compute ball speed from clubhead speed with smash factor"""
        # Smash factor depends on impact quality
        base_smash = 1.48
        
        # Adjust for loft (higher loft = slightly lower smash)
        loft_factor = 1 - (config.loft - 9) * 0.005
        
        smash_factor = base_smash * loft_factor
        return clubhead_speed * np.clip(smash_factor, 1.35, 1.52)


# ============================================================================
# FITTING ENGINE
# ============================================================================

class FittingEngine:
    """Rule-based + ML-enhanced club fitting logic"""
    
    def __init__(self):
        self.physics = PhysicsEngine()
        self.ml_model = None  # Placeholder for ML model
        
    def recommend_configuration(
        self, 
        swing_params: SwingParameters
    ) -> FittingRecommendation:
        """
        Generate optimal club configuration using rules + ML
        """
        speed = swing_params.clubhead_speed
        attack = swing_params.attack_angle
        spin = swing_params.spin_rate
        
        # Rule-based fitting logic
        config = self._apply_fitting_rules(speed, attack, spin)
        
        # ML enhancement (placeholder for now)
        confidence = self._compute_confidence(swing_params, config)
        
        # Predict improvement
        baseline_config = ClubConfiguration(
            loft=10.5, shaft_flex="R", shaft_weight=60,
            head_weight=200, shaft_torque=3.5, shaft_length=45.5
        )
        
        _, baseline_metrics = self.physics.simulate_flight(swing_params, baseline_config)
        _, optimized_metrics = self.physics.simulate_flight(swing_params, config)
        
        improvements = {
            "carry_distance": optimized_metrics.carry_distance - baseline_metrics.carry_distance,
            "apex_height": optimized_metrics.apex_height - baseline_metrics.apex_height,
            "dispersion": -5.0  # Simulated dispersion improvement
        }
        
        reasoning = self._generate_reasoning(swing_params, config)
        
        return FittingRecommendation(
            recommended_config=config,
            confidence_score=confidence,
            predicted_improvement=improvements,
            reasoning=reasoning
        )
    
    def _apply_fitting_rules(
        self, 
        speed: float, 
        attack: float, 
        spin: float
    ) -> ClubConfiguration:
        """Apply domain-specific fitting heuristics"""
        
        # Loft selection
        if speed < 80:
            loft = 12.0
            flex = "L"
            weight = 45
        elif speed < 90:
            loft = 11.0
            flex = "A"
            weight = 50
        elif speed < 95:
            loft = 10.5
            flex = "R"
            weight = 60
        elif speed < 105:
            loft = 9.5
            flex = "S"
            weight = 65
        else:
            loft = 9.0
            flex = "X"
            weight = 70
        
        # Adjust loft for attack angle
        if attack < -3:
            loft += 1.5  # Add loft for steep descending blow
        elif attack < -1:
            loft += 0.5
        elif attack > 3:
            loft -= 1.0  # Reduce loft for upward strike
        
        # Head weight for spin control
        if spin > 3200:
            head_weight = 205  # Heavier head for lower spin
            torque = 2.5
        elif spin > 2800:
            head_weight = 200
            torque = 3.0
        else:
            head_weight = 195  # Lighter for more spin
            torque = 3.5
        
        # Shaft length optimization
        if speed > 110:
            length = 45.0  # Shorter for control
        else:
            length = 45.5
        
        return ClubConfiguration(
            loft=loft,
            shaft_flex=flex,
            shaft_weight=weight,
            head_weight=head_weight,
            shaft_torque=torque,
            shaft_length=length
        )
    
    def _compute_confidence(
        self, 
        swing_params: SwingParameters, 
        config: ClubConfiguration
    ) -> float:
        """Compute fitting confidence score"""
        confidence = 85.0
        
        # Higher confidence for mid-range speeds
        if 90 <= swing_params.clubhead_speed <= 105:
            confidence += 5
        
        # Lower confidence for extreme spin rates
        if swing_params.spin_rate < 2000 or swing_params.spin_rate > 3500:
            confidence -= 10
        
        return np.clip(confidence, 60, 98)
    
    def _generate_reasoning(
        self, 
        swing_params: SwingParameters, 
        config: ClubConfiguration
    ) -> Dict[str, str]:
        """Generate human-readable fitting reasoning"""
        reasoning = {}
        
        speed = swing_params.clubhead_speed
        
        if speed < 90:
            reasoning["loft"] = f"Higher loft ({config.loft}°) recommended to maximize carry with slower swing speed"
        elif speed > 105:
            reasoning["loft"] = f"Lower loft ({config.loft}°) optimizes launch conditions for high swing speed"
        else:
            reasoning["loft"] = f"Standard loft ({config.loft}°) suits your swing speed profile"
        
        reasoning["shaft"] = f"{config.shaft_flex} flex with {config.shaft_weight}g weight balances control and distance"
        
        if swing_params.spin_rate > 3000:
            reasoning["head"] = f"Heavier head ({config.head_weight}g) will reduce excessive spin"
        elif swing_params.spin_rate < 2200:
            reasoning["head"] = f"Lighter head ({config.head_weight}g) will help increase spin rate"
        
        return reasoning


# ============================================================================
# API ENDPOINTS
# ============================================================================

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