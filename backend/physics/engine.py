import numpy as np
from .models import BallState, PerformanceMetrics

class PhysicsEngine:
    G = 9.81
    RHO = 1.225
    BALL_MASS = 0.0459
    BALL_RADIUS = 0.02135
    BALL_AREA = np.pi * BALL_RADIUS ** 2
    CD0 = 0.22
    CL0 = 0.14

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def compute_drag_coefficient(self, velocity, spin_rate):
        return self.CD0 + 0.05 * (spin_rate / 3000) ** 2

    def compute_lift_coefficient(self, spin_parameter):
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

    def _compute_ball_speed(self, clubhead_speed, config):
        base_smash = 1.48
        loft_factor = 1 - (config.loft - 9) * 0.005
        smash_factor = base_smash * loft_factor
        return clubhead_speed * np.clip(smash_factor, 1.35, 1.52)