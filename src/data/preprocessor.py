import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from src.utils.config import TARGET_COL, TEST_SIZE, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess data:
    1. Separate features and target
    2. Train/Test split
    3. Missing value imputation
    4. Categorical encoding
    5. Handle class imbalance (SMOTE)
    """
    logger.info("Starting data preprocessing...")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
        
    X = df.drop(columns=[TARGET_COL, "SK_ID_CURR"], errors="ignore")
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Identify numeric and categorical columns dynamically
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    logger.info("Fitting preprocessor and transforming data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    import re
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_feature_names)
    
    # Clean feature names for LightGBM compatibility (no spaces or JSON special characters)
    feature_names = [re.sub(r'[^\w]', '_', name) for name in feature_names]
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
    
    return X_train_resampled, X_test_df, y_train_resampled, y_test, preprocessor, feature_names
