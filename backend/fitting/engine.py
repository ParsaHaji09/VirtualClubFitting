import numpy as np
from physics.engine import PhysicsEngine
from typing import Dict
from .models import ClubConfiguration, FittingRecommendation, SwingParameters
from ml.training_model import FittingMLModel
from ml.data_generator import SyntheticDataGenerator
import os

class FittingEngine:
    """Rule-based + ML-enhanced club fitting logic"""
    
    def __init__(self):
        self.physics = PhysicsEngine()
        self.ml_model = FittingMLModel()
        # Try to load pre-trained model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fitting_model.pkl')
        if os.path.exists(model_path):
            try:
                self.ml_model.load(model_path)
                print("ML model loaded successfully")
            except Exception as e:
                print(f"Failed to load ML model: {e}")
                self.ml_model = None
        else:
            print("No pre-trained ML model found, using rule-based fitting only")
            self.ml_model = None
        
    def recommend_configuration(
        self, 
        swing_params: SwingParameters
    ) -> FittingRecommendation:
        """
        Generate optimal club configuration using rules + ML
        """
        if self.ml_model:
            # Use ML-based optimization
            config, score = self._ml_optimize_configuration(swing_params)
            confidence = 95.0  # Higher confidence with ML
        else:
            # Rule-based fitting logic
            config = self._apply_fitting_rules(
                swing_params.clubhead_speed, 
                swing_params.attack_angle, 
                swing_params.spin_rate
            )
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


    def _ml_optimize_configuration(self, swing_params: SwingParameters) -> tuple[ClubConfiguration, float]:
        """Use ML model to find optimal club configuration"""
        swing_dict = {
            'clubhead_speed': swing_params.clubhead_speed,
            'attack_angle': swing_params.attack_angle,
            'launch_angle': swing_params.launch_angle,
            'spin_rate': swing_params.spin_rate,
            'swing_path': swing_params.swing_path,
            'face_angle': swing_params.face_angle
        }
        
        config, score = self.ml_model.optimize_configuration(swing_dict)
        return config, score
    
    def train_ml_model(self, n_samples: int = 10000):
        """Train the ML model using synthetic data"""
        print("Training ML model...")
        data_gen = SyntheticDataGenerator(self.physics)
        df = data_gen.generate_dataset(n_samples)
        
        self.ml_model = FittingMLModel()
        results = self.ml_model.train(df)
        
        # Save the model
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fitting_model.pkl')
        self.ml_model.save(model_path)
        
        return results

