import joblib
import lightgbm as lgb
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_data
from src.utils.config import MODEL_PATH, RANDOM_STATE
from src.utils.logger import get_logger
from src.ml.evaluate import evaluate_model

logger = get_logger(__name__)

def train_model():
    """Trains the LightGBM model and saves the pipeline."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
    logger.info("Initializing LightGBM model...")
    model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced"
    )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
        
    logger.info(f"Saving model and preprocessor to {MODEL_PATH}...")
    pipeline = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
