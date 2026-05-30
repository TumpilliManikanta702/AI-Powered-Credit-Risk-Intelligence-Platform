import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_explainer(model, X_background=None):
    """Returns a SHAP TreeExplainer for the model."""
    return shap.TreeExplainer(model)

def explain_prediction(data_row: pd.DataFrame):
    """
    Generate SHAP explanations for a single prediction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
        
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline["model"]
    preprocessor = pipeline["preprocessor"]
    feature_names = pipeline["feature_names"]
    
    # Preprocess
    X_processed = preprocessor.transform(data_row)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X_processed_df)
    
    # SHAP returns a list of arrays for binary classification in some LightGBM versions, 
    # or a single array in newer SHAP/LightGBM versions.
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1][0]
    else:
        shap_values_class_1 = shap_values[0]
        
    # Get top contributing features
    feature_impacts = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values_class_1,
        "Value": X_processed_df.iloc[0].values
    })
    
    # Sort by absolute impact
    feature_impacts["Abs_Impact"] = feature_impacts["Impact"].abs()
    top_features = feature_impacts.sort_values(by="Abs_Impact", ascending=False).head(5)
    
    return top_features

def generate_shap_summary_plot(data: pd.DataFrame, max_display=10):
    """
    Generates a SHAP summary plot for global explainability.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
        
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline["model"]
    preprocessor = pipeline["preprocessor"]
    feature_names = pipeline["feature_names"]
    
    # Preprocess
    X_processed = preprocessor.transform(data)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X_processed_df)
    
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]
    else:
        shap_values_class_1 = shap_values
        
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_class_1, X_processed_df, max_display=max_display, show=False)
    plt.tight_layout()
    return fig
