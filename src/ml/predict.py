import joblib
import pandas as pd
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_risk_band(prob):
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    else:
        return "High"

def predict_risk(data: pd.DataFrame):
    """
    Predicts default probability and risk band for given data.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        
    logger.info("Loading model...")
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline["model"]
    preprocessor = pipeline["preprocessor"]
    feature_names = pipeline["feature_names"]
    
    logger.info("Preprocessing input data...")
    # Add dummy target column if not present so preprocessor doesn't break if expecting it,
    # though our preprocessor drops target so it's fine.
    X_processed = preprocessor.transform(data)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    logger.info("Making predictions...")
    probs = model.predict_proba(X_processed_df)[:, 1]
    
    results = []
    for prob in probs:
        results.append({
            "Probability": round(prob, 4),
            "Risk Score": int(prob * 1000),
            "Risk Band": get_risk_band(prob)
        })
        
    return pd.DataFrame(results), X_processed_df
