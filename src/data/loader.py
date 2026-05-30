import pandas as pd
import sqlite3
from src.utils.config import RAW_DATA_PATH, DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Loads the raw CSV data."""
    if not RAW_DATA_PATH.exists():
        logger.error(f"Data file not found at {RAW_DATA_PATH}. Please run helpers.py to generate mock data.")
        raise FileNotFoundError(f"Missing {RAW_DATA_PATH}")
    
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)

def save_to_sqlite(df: pd.DataFrame, table_name="credit_risk_data"):
    """Saves DataFrame to SQLite database for NL-to-SQL functionality."""
    logger.info(f"Saving data to SQLite DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    # Convert objects to string to ensure safe insertion
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Successfully saved to table {table_name}")

if __name__ == "__main__":
    df = load_raw_data()
    save_to_sqlite(df)
