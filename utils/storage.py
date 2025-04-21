import pickle
import os

# Models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, model_id: str):
    """Save model to disk"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

def load_model(model_id: str):
    """Load model from disk"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_id} not found")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model