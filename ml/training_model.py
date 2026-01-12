class FittingMLModel:
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.target_columns = None
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        model_type: str = 'xgboost'
    ) -> Dict:
        print("Training models...")
        self.feature_columns = [
            'clubhead_speed', 'attack_angle', 'launch_angle', 'spin_rate',
            'swing_path', 'face_angle', 'loft', 'shaft_flex_numeric',
            'shaft_weight', 'head_weight', 'shaft_torque', 'shaft_length'
        ]
        self.target_columns = [
            'carry_distance', 'apex_height', 'lateral_deviation'
        ]

        X = df[self.feature_columns]

        results = {}

        for target in self.target_columns:
            print(f"Specified Target: {target}")
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            cv_mae = -cv_scores.mean()
            
            print(f"  Test MAE: {mae:.2f}")
            print(f"  Test R²: {r2:.3f}")
            print(f"  CV MAE: {cv_mae:.2f}")
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 5 features:")
            print(importance.head(5).to_string(index=False))
            
            self.models[target] = model
            
            results[target] = {
                'mae': mae,
                'r2': r2,
                'cv_mae': cv_mae,
                'feature_importance': importance.to_dict('records')
            }
        
        return results
    
    def predict(self, features: Dict) -> Dict:
        """Predict performance metrics for given swing + club combo"""
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Create feature vector
        X = pd.DataFrame([features])[self.feature_columns]
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = float(model.predict(X)[0])
        
        return predictions
    
    def optimize_configuration(
        self, 
        swing_params: Dict,
        objective: str = 'carry_distance'
    ) -> Tuple[ClubConfiguration, float]:
        """
        Find optimal club configuration for given swing parameters
        using grid search over configuration space
        """
        print("Optimizing club configuration...")
        
        best_config = None
        best_score = -np.inf
        
        # Define search space
        lofts = [8.5, 9.0, 9.5, 10.5, 11.0, 12.0]
        flexes = [1, 2, 3, 4, 5]  # L, A, R, S, X
        weights = [50, 55, 60, 65, 70]
        head_weights = [195, 200, 205]
        
        # Grid search
        for loft in lofts:
            for flex in flexes:
                for weight in weights:
                    for head_weight in head_weights:
                        config_features = {
                            **swing_params,
                            'loft': loft,
                            'shaft_flex_numeric': flex,
                            'shaft_weight': weight,
                            'head_weight': head_weight,
                            'shaft_torque': 3.0,
                            'shaft_length': 45.5
                        }
                        
                        predictions = self.predict(config_features)
                        
                        # Composite score (weighted objectives)
                        score = (
                            predictions['carry_distance'] * 1.0 +
                            predictions['apex_height'] * 0.1 -
                            predictions['lateral_deviation'] * 0.5
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_config = config_features
        
        flex_map = {1: 'L', 2: 'A', 3: 'R', 4: 'S', 5: 'X'}
        
        optimized_config = ClubConfiguration(
            loft=best_config['loft'],
            shaft_flex=flex_map[best_config['shaft_flex_numeric']],
            shaft_weight=best_config['shaft_weight'],
            head_weight=best_config['head_weight'],
            shaft_torque=best_config['shaft_torque'],
            shaft_length=best_config['shaft_length']
        )
        
        return optimized_config, best_score
    
    def save(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models from disk"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        print(f"Models loaded from {filepath}")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def run_training_pipeline(
    n_samples: int = 10000,
    model_type: str = 'xgboost',
    save_path: str = 'models/fitting_model.pkl'
):
    """Complete training pipeline"""
    
    print("=" * 80)
    print("GOLF CLUB FITTING - ML TRAINING PIPELINE")
    print("=" * 80)
    
    # Initialize
    physics = PhysicsEngine()
    data_gen = SyntheticDataGenerator(physics)
    
    # Generate synthetic data
    print("\n[1/4] Generating synthetic training data...")
    df = data_gen.generate_dataset(n_samples=n_samples)
    
    # Save raw data
    data_path = Path('data')
    data_path.mkdir(exist_ok=True)
    df.to_csv(data_path / 'synthetic_data.csv', index=False)
    print(f"Data saved to {data_path / 'synthetic_data.csv'}")
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df[['carry_distance', 'apex_height', 'lateral_deviation']].describe())
    
    # Train models
    print("\n[2/4] Training ML models...")
    ml_model = FittingMLModel()
    results = ml_model.train(df, model_type=model_type)
    
    # Save results
    results_path = data_path / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to {results_path}")
    
    # Save model
    print("\n[3/4] Saving trained models...")
    model_path = Path(save_path)
    model_path.parent.mkdir(exist_ok=True)
    ml_model.save(save_path)
    
    # Test optimization
    print("\n[4/4] Testing optimization...")
    test_swing = {
        'clubhead_speed': 95,
        'attack_angle': 0,
        'launch_angle': 12,
        'spin_rate': 2600,
        'swing_path': 0,
        'face_angle': 0
    }
    
    optimal_config, score = ml_model.optimize_configuration(test_swing)
    print("\nOptimized Configuration:")
    print(f"  Loft: {optimal_config.loft}°")
    print(f"  Shaft: {optimal_config.shaft_flex} flex, {optimal_config.shaft_weight}g")
    print(f"  Head: {optimal_config.head_weight}g")
    print(f"  Score: {score:.1f}")
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    
    return ml_model, df, results


if __name__ == "__main__":
    # Run full training pipeline
    ml_model, data, results = run_training_pipeline(
        n_samples=10000,
        model_type='xgboost'
    )
