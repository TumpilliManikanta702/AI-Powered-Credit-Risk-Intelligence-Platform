import sqlite3
import pandas as pd
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "GRANT", "REVOKE"]

def run_query(query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the SQLite database after validating it.
    """
    query_upper = query.upper()
    
    # Strict validation
    if not query_upper.strip().startswith("SELECT"):
        logger.error(f"Invalid query attempted (Not a SELECT): {query}")
        raise ValueError("Only SELECT queries are allowed.")
        
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in query_upper:
            logger.error(f"Forbidden keyword '{keyword}' found in query: {query}")
            raise ValueError(f"Forbidden keyword '{keyword}' detected.")
            
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error executing query: {query}. Error: {e}")
        raise e
