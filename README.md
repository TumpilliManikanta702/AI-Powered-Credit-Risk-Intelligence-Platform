# AI-Powered Credit Risk Intelligence Platform

## Project Overview

A lightweight, professional AI-powered credit risk platform using the Home Credit Default Risk dataset. This platform integrates Machine Learning, Explainable AI (XAI), and a Natural Language to SQL (NL-to-SQL) Chatbot to provide actionable insights for credit risk assessment.

## Architecture

1. **Frontend (Streamlit)**: Multi-page interactive UI.
2. **Machine Learning Pipeline**: Data preprocessing, LightGBM model training (with SMOTE for class imbalance), evaluation, and saving artifacts.
3. **Explainable AI (SHAP)**: Global and local interpretability of the model predictions.
4. **Business Rule Engine**: Translation of ML logic into interpretable business rules.
5. **NL-to-SQL Chatbot**: LangChain & Groq (llama3-70b-8192) to translate natural language questions into SQLite queries and generate business answers.

## Setup Instructions

### Local Environment Setup

1. **Clone the repository.**
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Rename `.env.example` to `.env` and add your Groq API Key:
   ```
   GROQ_API_KEY=your_key_here
   ```
5. **Data:**
   Download `application_train.csv` from the [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it in the `data/` directory. Alternatively, run the mock data generator (see below).

6. **Run End-to-End Pipeline:**
   ```bash
   python src/utils/helpers.py  # Generates mock data if needed
   python src/ml/train.py       # Trains the model
   python src/data/loader.py    # Loads data into SQLite for NL-to-SQL
   ```
7. **Run the App:**
   ```bash
   streamlit run src/ui/app.py
   ```

### Docker Instructions

To run the application using Docker Compose:

1. Ensure `.env` is created with your `GROQ_API_KEY`.
2. Run:
   ```bash
   docker-compose up --build
   ```
3. Access the application at `http://localhost:8501`.

## ML Strategy

- **Model Rationale**: LightGBM is chosen for its efficiency with tabular data and handling of missing values.
- **Imbalance Strategy**: Using SMOTE to synthetically oversample the minority class (defaulters) to ensure the model doesn't just predict the majority class.
- **Evaluation Metrics**: ROC-AUC, F1-Score, Precision, and Recall are prioritized over accuracy due to class imbalance.

## Screenshots

The platform includes the following views:
- **Dashboard**: Overview of credit risk metrics and model performance
- **Prediction Page**: Interactive form for individual risk assessment
- **SHAP Explainability**: Feature importance plots and local explanations
- **NL-to-SQL Chatbot**: Natural language interface to query the database
- **EDA Charts**: Exploratory data analysis visualizations

*Note: Screenshots should be added here from actual application runs.*

## Prompt Engineering & Hallucination Reduction

- **Schema-aware prompts**: The LLM is explicitly provided with the SQLite table schema.
- **Strict SQL Validation**: The `prompt_templates.py` strictly instructs the LLM to only use `SELECT` queries.
- **Execution Safeguards**: The `query_runner.py` validates the SQL string, blocking any `DROP`, `DELETE`, `UPDATE`, `INSERT`, or `ALTER` commands before execution.
- **Temperature=0**: Ensuring deterministic generation.

## Limitations & Future Improvements

- **Limitations**: The mock dataset generated (if real dataset is missing) is small and not perfectly representative of real-world complexity. The SQLite database is local and not suited for massive concurrent production usage.
- **Future Improvements**: Connect to a robust data warehouse (e.g., Snowflake, BigQuery), implement MLOps monitoring (e.g., MLflow), and deploy on a managed service (e.g., AWS ECS, Vercel).
