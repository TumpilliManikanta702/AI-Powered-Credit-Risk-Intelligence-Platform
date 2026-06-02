import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SQL_DIR = BASE_DIR / "sql"

# File Paths
RAW_DATA_PATH = DATA_DIR / "application_train.csv"
DB_PATH = SQL_DIR / "credit_risk.db"
MODEL_PATH = MODELS_DIR / "model.pkl"

# ML Configurations
TARGET_COL = "TARGET"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to use for the mock generator or modeling
NUMERIC_COLS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
CATEGORICAL_COLS = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"]

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SQL_DIR.mkdir(exist_ok=True)
