import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_raw_data

def get_eda_data():
    """Loads and returns data for EDA."""
    return load_raw_data()

def get_dataset_summary(df):
    """Returns dataset summary."""
    return df.describe()

def get_feature_categorization(df):
    """Categorizes features."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    return {"Numeric Features": num_cols, "Categorical Features": cat_cols}

def plot_missing_value_analysis(df):
    """Plots missing value percentages."""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False).reset_index()
    missing.columns = ["Feature", "Missing Percentage"]
    fig = px.bar(missing, x="Feature", y="Missing Percentage", title="Missing Value Analysis")
    return fig

def plot_target_distribution(df):
    """Plots the distribution of the TARGET variable."""
    target_counts = df["TARGET"].value_counts().reset_index()
    target_counts.columns = ["TARGET", "Count"]
    target_counts["TARGET"] = target_counts["TARGET"].map({0: "Repaid", 1: "Default"})
    
    fig = px.pie(target_counts, values="Count", names="TARGET", title="Target Distribution (Imbalance)", color="TARGET", color_discrete_sequence=["#1f77b4", "#d62728"])
    return fig

def plot_income_distribution_by_target(df):
    """Plots income distribution grouped by target."""
    fig = px.box(df, x="TARGET", y="AMT_INCOME_TOTAL", title="Income Distribution by Target", color="TARGET", color_discrete_sequence=["#1f77b4", "#d62728"])
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Repaid", "Default"])
    # Cap y-axis for better visibility due to outliers
    upper_bound = df["AMT_INCOME_TOTAL"].quantile(0.95)
    fig.update_yaxes(range=[0, upper_bound])
    return fig

def plot_outlier_analysis(df):
    """Plots outlier analysis using a boxplot for income."""
    fig = px.box(df, y="AMT_INCOME_TOTAL", title="Outlier Analysis: Total Income")
    return fig

def plot_correlation_matrix(df):
    """Plots correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    # Sample down if too many columns
    if len(numeric_df.columns) > 15:
        numeric_df = numeric_df.iloc[:, :15] 
    
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    return fig

def plot_age_distribution_by_target(df):
    """Plots age distribution by target."""
    df_copy = df.copy()
    df_copy["AGE_YEARS"] = abs(df_copy["DAYS_BIRTH"]) / 365
    fig = px.histogram(df_copy, x="AGE_YEARS", color="TARGET", barmode="overlay", title="Age Distribution by Target", nbins=50, color_discrete_sequence=["#1f77b4", "#d62728"])
    return fig

def generate_insights():
    """Returns static business insights generated from EDA."""
    return [
        "1. Extreme Class Imbalance: Defaulters account for ~8% of the dataset, requiring SMOTE for model training.",
        "2. Income Disparity: Lower-income applicants show a higher propensity to default.",
        "3. Age Factor: Younger applicants (under 30) have higher default rates compared to older demographics.",
        "4. Employment Stability: Shorter employment durations correlate with higher risk.",
        "5. External Scores: External credit sources (EXT_SOURCE_1/2/3) show the strongest negative correlation with defaults."
    ]
