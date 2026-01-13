import pickle  # or 'joblib'
from main import fitting_engine  # Import your existing engine instance

def run_training():
    print("Starting ML model training...")
    n_samples = 10000
    
    # 1. Train the model
    results = fitting_engine.train_ml_model(n_samples)
    print(f"Training complete. Results: {results}")

if __name__ == "__main__":
    run_training()