import pandas as pd
import numpy as np
from src.utils.config import RAW_DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_mock_data(n_samples=5000):
    """
    Generates a mock dataset resembling the Home Credit Default Risk dataset
    so the pipeline can run out-of-the-box if the real dataset is missing.
    """
    logger.info(f"Generating {n_samples} rows of mock data...")
    np.random.seed(42)
    
    data = {
        "SK_ID_CURR": range(100000, 100000 + n_samples),
        "TARGET": np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]), # 8% default rate
        "AMT_INCOME_TOTAL": np.random.lognormal(mean=11.5, sigma=0.5, size=n_samples).round(),
        "AMT_CREDIT": np.random.lognormal(mean=13, sigma=0.8, size=n_samples).round(),
        "AMT_ANNUITY": np.random.lognormal(mean=10, sigma=0.5, size=n_samples).round(),
        "DAYS_BIRTH": -np.random.randint(7000, 25000, size=n_samples), # Negative days
        "DAYS_EMPLOYED": -np.random.randint(100, 10000, size=n_samples), # Negative days
        "CODE_GENDER": np.random.choice(["M", "F"], size=n_samples, p=[0.35, 0.65]),
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], size=n_samples, p=[0.3, 0.7]),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], size=n_samples, p=[0.7, 0.3]),
        "NAME_INCOME_TYPE": np.random.choice(["Working", "Commercial associate", "Pensioner", "State servant"], size=n_samples),
        "NAME_EDUCATION_TYPE": np.random.choice(["Secondary / secondary special", "Higher education", "Incomplete higher"], size=n_samples),
        "OCCUPATION_TYPE": np.random.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", np.nan], size=n_samples),
        "CNT_FAM_MEMBERS": np.random.randint(1, 5, size=n_samples),
        "EXT_SOURCE_1": np.random.uniform(0, 1, size=n_samples),
        "EXT_SOURCE_2": np.random.uniform(0, 1, size=n_samples),
        "EXT_SOURCE_3": np.random.uniform(0, 1, size=n_samples),
    }
    
    # Introduce correlation for EDA and ML
    # High risk correlates with lower income and lower EXT_SOURCEs
    high_risk_idx = data["TARGET"] == 1
    data["AMT_INCOME_TOTAL"][high_risk_idx] *= 0.8
    data["EXT_SOURCE_1"][high_risk_idx] *= 0.7
    data["EXT_SOURCE_2"][high_risk_idx] *= 0.7
    
    # Introduce some NaN values to test preprocessing
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_ANNUITY"]:
        mask = np.random.rand(n_samples) < 0.1
        data[col][mask] = np.nan
        
    df = pd.DataFrame(data)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Mock data saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    if not RAW_DATA_PATH.exists():
        generate_mock_data()
    else:
        logger.info("Raw data already exists. Skipping mock generation.")
