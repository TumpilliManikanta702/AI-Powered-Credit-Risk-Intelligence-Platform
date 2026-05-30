import json
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

os.makedirs("sql", exist_ok=True)
with open("sql/schema.sql", "w") as f:
    f.write('''-- SQLite Schema for Credit Risk Platform
CREATE TABLE IF NOT EXISTS credit_risk_data (
    SK_ID_CURR INTEGER PRIMARY KEY,
    TARGET INTEGER,
    NAME_CONTRACT_TYPE TEXT,
    CODE_GENDER TEXT,
    FLAG_OWN_CAR TEXT,
    FLAG_OWN_REALTY TEXT,
    CNT_CHILDREN INTEGER,
    AMT_INCOME_TOTAL REAL,
    AMT_CREDIT REAL,
    AMT_ANNUITY REAL,
    AMT_GOODS_PRICE REAL,
    NAME_TYPE_SUITE TEXT,
    NAME_INCOME_TYPE TEXT,
    NAME_EDUCATION_TYPE TEXT,
    NAME_FAMILY_STATUS TEXT,
    NAME_HOUSING_TYPE TEXT,
    REGION_POPULATION_RELATIVE REAL,
    DAYS_BIRTH INTEGER,
    DAYS_EMPLOYED INTEGER,
    DAYS_REGISTRATION REAL,
    DAYS_ID_PUBLISH INTEGER,
    OWN_CAR_AGE REAL,
    FLAG_MOBIL INTEGER,
    FLAG_EMP_PHONE INTEGER,
    FLAG_WORK_PHONE INTEGER,
    FLAG_CONT_MOBILE INTEGER,
    FLAG_PHONE INTEGER,
    FLAG_EMAIL INTEGER,
    OCCUPATION_TYPE TEXT,
    CNT_FAM_MEMBERS REAL,
    REGION_RATING_CLIENT INTEGER,
    REGION_RATING_CLIENT_W_CITY INTEGER,
    WEEKDAY_APPR_PROCESS_START TEXT,
    HOUR_APPR_PROCESS_START INTEGER,
    REG_REGION_NOT_LIVE_REGION INTEGER,
    REG_REGION_NOT_WORK_REGION INTEGER,
    LIVE_REGION_NOT_WORK_REGION INTEGER,
    REG_CITY_NOT_LIVE_CITY INTEGER,
    REG_CITY_NOT_WORK_CITY INTEGER,
    LIVE_CITY_NOT_WORK_CITY INTEGER,
    ORGANIZATION_TYPE TEXT,
    EXT_SOURCE_1 REAL,
    EXT_SOURCE_2 REAL,
    EXT_SOURCE_3 REAL
);
''')

os.makedirs("notebooks", exist_ok=True)
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Exploratory Data Analysis (EDA)\n", "Home Credit Default Risk dataset EDA."]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["import pandas as pd\n", "df = pd.read_csv('../data/application_train.csv')\n", "df.head()"]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
with open("notebooks/eda.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)

os.makedirs("documents", exist_ok=True)
slides = [
    "1. Problem Statement: Predict credit default risk.",
    "2. Architecture: Streamlit UI, LightGBM ML, LangChain NL-to-SQL.",
    "3. Dataset Overview: 300k+ rows, 122 features (application_train).",
    "4. EDA Insights: Low income & missed payments correlate to risk.",
    "5. ML Pipeline: Preprocessing -> SMOTE -> LightGBM.",
    "6. Evaluation Metrics: ROC-AUC, F1, Precision, Recall.",
    "7. SHAP Explainability: Feature importance & local explanations.",
    "8. Business Rules: IF income < X AND EXT_SOURCE < Y THEN HIGH RISK.",
    "9. NL-to-SQL Chatbot: Groq API querying SQLite DB securely.",
    "10. Docker Deployment: docker-compose setup.",
    "11. Challenges & Improvements: Handled class imbalance via SMOTE.",
    "12. Conclusion: Production-ready credit risk platform deployed."
]

with PdfPages('documents/project_presentation.pdf') as pdf:
    for slide in slides:
        fig = plt.figure(figsize=(10, 5))
        fig.text(0.1, 0.5, slide, fontsize=16, ha='left')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()
