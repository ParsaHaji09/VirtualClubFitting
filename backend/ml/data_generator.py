import pandas as pd
import numpy as np
from physics.engine import PhysicsEngine
from fitting.models import SwingParameters, ClubConfiguration

class SyntheticDataGenerator:
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics = physics_engine
    
    def generate_dataset(
            self,
            n_samples: int = 10000,
            add_noise: bool = True,
            noise_level: float = 0.02
    ) -> pd.DataFrame:
        print(f"Generating {n_samples} synthetic samples...")

        data = []

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Progress: {i}/{n_samples}")

            swing_params = self._sample_swing_parameters()
            club_config = self._sample_club_configuration()

            try:
                _, metrics = self.physics.simulate_flight(swing_params, club_config)

                if add_noise:
                    metrics.carry_distance *= np.random.normal(1.0, noise_level)
                    metrics.apex_height *= np.random.normal(1.0, noise_level)
                    metrics.lateral_deviation += np.random.normal(0, 2.0)

                data_point = {
                    'clubhead_speed': swing_params.clubhead_speed,
                    'attack_angle': swing_params.attack_angle,
                    'launch_angle': swing_params.launch_angle,
                    'spin_rate': swing_params.spin_rate,
                    'swing_path': swing_params.swing_path,
                    'face_angle': swing_params.face_angle,

                    'loft': club_config.loft,
                    'shaft_flex_numeric': self._flex_to_numeric(club_config.shaft_flex),
                    'shaft_weight': club_config.shaft_weight,
                    'head_weight': club_config.head_weight,
                    'shaft_torque': club_config.shaft_torque,
                    'shaft_length': club_config.shaft_length,

                    'carry_distance': metrics.carry_distance,
                    'total_distance': metrics.total_distance,
                    'apex_height': metrics.apex_height,
                    'landing_angle': metrics.landing_angle,
                    'flight_time': metrics.flight_time,
                    'lateral_deviation': abs(metrics.lateral_deviation)
                }

                data.append(data_point)
            
            except Exception as e:
                print(f"Simulation failed for sample {i}: {e}")
                continue
        
        df = pd.DataFrame(data)

        print(f"Generated {len(df)} valid samples")
        return df
    
    # creating sample realistic swing parameters fro empirical distributions
    def _sample_swing_parameters(self) -> SwingParameters:
        speed = np.random.normal(95,12)
        speed = np.clip(speed, 70, 130)

        attack = np.random.normal(0, 2.5)
        attack = np.clip(attack, -6, 5)

        launch = np.random.normal(12, 2.5)
        launch = np.clip(launch, 7, 20)

        spin_base = 3200 - (speed - 95) * 15
        spin = np.random.normal(spin_base, 300)
        spin = np.clip(spin, 1500, 4500)

        path = np.random.normal(0, 2)
        raw_face = np.random.normal(0, 1.5)
        face = np.clip(raw_face, -5.0, 5.0)

        return SwingParameters(
            clubhead_speed=float(speed),
            attack_angle=float(attack),
            launch_angle=float(launch),
            spin_rate=float(spin),
            swing_path=float(path),
            face_angle =float(face)
        )
    
    def _sample_club_configuration(self) -> ClubConfiguration:
        """Sample random but realistic club configurations"""
        loft = np.random.choice([8.5, 9.0, 9.5, 10.5, 11.0, 12.0])
        flex = np.random.choice(['L', 'A', 'R', 'S', 'X'], p=[0.05, 0.15, 0.40, 0.30, 0.10])
        
        # Shaft weight correlates with flex
        flex_weights = {'L': 45, 'A': 50, 'R': 60, 'S': 65, 'X': 70}
        weight = flex_weights[flex] + np.random.randint(-5, 5)
        
        head_weight = np.random.choice([195, 200, 205])
        torque = np.random.uniform(2.5, 4.5)
        length = np.random.choice([44.5, 45.0, 45.5, 46.0])
        
        return ClubConfiguration(
            loft=loft,
            shaft_flex=flex,
            shaft_weight=float(weight),
            head_weight=float(head_weight),
            shaft_torque=float(torque),
            shaft_length=float(length)
        )
    
    def _flex_to_numeric(self, flex: str) -> float:
        """Convert shaft flex to numeric value"""
        mapping = {'L': 1, 'A': 2, 'R': 3, 'S': 4, 'X': 5}
        return float(mapping.get(flex, 3))