import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv(override=True)

from notebooks.eda import (
    get_eda_data,
    get_dataset_summary,
    get_feature_categorization,
    plot_missing_value_analysis,
    plot_target_distribution,
    plot_income_distribution_by_target,
    plot_outlier_analysis,
    plot_correlation_matrix,
    plot_age_distribution_by_target,
    generate_insights
)
from src.ml.predict import predict_risk
from src.ml.explain import explain_prediction, generate_shap_summary_plot
from src.ml.rules import generate_business_rules
from src.talk_to_data.nl_to_sql import NLtoSQLChatbot
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide", page_icon="🏦")

# Sidebar navigation
st.sidebar.title("🏦 AI Credit Risk Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "EDA",
    "Risk Prediction",
    "Explainability",
    "Business Rules",
    "AI Chatbot"
])

# Global state for demo
if "demo_data" not in st.session_state:
    try:
        st.session_state.demo_data = get_eda_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.session_state.demo_data = pd.DataFrame()

# Global state for selected customer prediction
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

df = st.session_state.demo_data

# Page 1: Dashboard
if page == "Dashboard":
    st.title("Dashboard Overview")
    st.markdown("Welcome to the AI-powered platform for predicting and explaining credit default risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Applications Evaluated", value=f"{len(df):,}")
    with col2:
        if not df.empty and 'TARGET' in df.columns:
            default_rate = df['TARGET'].mean() * 100
            st.metric(label="Overall Default Rate", value=f"{default_rate:.2f}%")
        else:
            st.metric(label="Overall Default Rate", value="N/A")
    with col3:
        st.metric(label="Model Status", value="Active - LightGBM")
        
    st.markdown("### Platform Features")
    st.markdown("""
    - **EDA**: Explore patterns and risk factors in the dataset.
    - **Risk Prediction**: Score individual customers using a machine learning model.
    - **Explainability**: Understand *why* a customer received a specific score using SHAP.
    - **Business Rules**: Translate complex ML logic into human-readable banking rules.
    - **AI Chatbot**: Query the database using natural language.
    """)

# Page 2: EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    
    if df.empty:
        st.warning("No data available for EDA.")
    else:
        st.subheader("Key Business Insights")
        for insight in generate_insights():
            st.write(insight)
            
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Data Summary", "Distributions & Outliers", "Correlations & Missing Values"])
        
        with tab1:
            st.subheader("Dataset Summary")
            st.dataframe(get_dataset_summary(df))
            st.subheader("Feature Categorization")
            cats = get_feature_categorization(df)
            st.write(cats)
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_target_distribution(df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_age_distribution_by_target(df), use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(plot_income_distribution_by_target(df), use_container_width=True)
            with col4:
                st.plotly_chart(plot_outlier_analysis(df), use_container_width=True)
                
        with tab3:
            st.plotly_chart(plot_missing_value_analysis(df), use_container_width=True)
            st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

# Page 3: Risk Prediction
elif page == "Risk Prediction":
    st.title("Machine Learning Risk Prediction")
    
    if df.empty:
        st.warning("No data available.")
    else:
        sample_idx = st.selectbox("Select Customer (Index)", df.index[:100])
        customer_data = df.iloc[[sample_idx]].copy()
        
        st.session_state.selected_customer = customer_data
        
        st.write("### Customer Profile", customer_data.drop(columns=["TARGET", "SK_ID_CURR"], errors='ignore'))
        
        if st.button("Predict Risk"):
            with st.spinner("Running ML Pipeline..."):
                try:
                    results_df, _ = predict_risk(customer_data)
                    st.session_state.prediction_results = {
                        "results_df": results_df,
                        "risk_band": results_df.iloc[0]['Risk Band'],
                        "prob": results_df.iloc[0]['Probability'],
                        "score": results_df.iloc[0]['Risk Score']
                    }
                    st.success("Prediction complete! View 'Explainability' or 'Business Rules' for details.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        if st.session_state.prediction_results:
            res = st.session_state.prediction_results
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Default Probability", value=f"{res['prob']:.1%}")
            col2.metric(label="Risk Score", value=res['score'])
            col3.metric(label="Risk Band", value=res['risk_band'])

# Page 4: Explainability
elif page == "Explainability":
    st.title("Explainable AI (SHAP)")
    
    tab1, tab2 = st.tabs(["Local Explanation (Single Prediction)", "Global Explanation (Model Summary)"])
    
    with tab1:
        if st.session_state.prediction_results and st.session_state.selected_customer is not None:
            st.subheader("Top Risk Drivers for Selected Customer")
            with st.spinner("Generating SHAP local explanation..."):
                top_features = explain_prediction(st.session_state.selected_customer)
                st.dataframe(top_features[["Feature", "Impact", "Value"]])
        else:
            st.info("Please predict a customer risk in the 'Risk Prediction' tab first to view local explanations.")
            
    with tab2:
        st.subheader("Global Feature Importance")
        st.markdown("Shows overall feature impact across the dataset sample.")
        if st.button("Generate SHAP Summary Plot"):
            with st.spinner("Generating global SHAP summary..."):
                try:
                    sample_data = df.sample(min(100, len(df)))
                    fig = generate_shap_summary_plot(sample_data.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore"))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating plot: {e}")

# Page 5: Business Rules
elif page == "Business Rules":
    st.title("Business Rule Engine")
    
    if st.session_state.prediction_results and st.session_state.selected_customer is not None:
        st.subheader("Generated Business Rules for Selected Customer")
        res = st.session_state.prediction_results
        top_features = explain_prediction(st.session_state.selected_customer)
        rules = generate_business_rules(st.session_state.selected_customer.iloc[0].to_dict(), res['risk_band'], top_features)
        
        for rule in rules:
            st.markdown(f"> **{rule}**")
    else:
        st.info("Please predict a customer risk in the 'Risk Prediction' tab first to view business rules.")

# Page 6: AI Chatbot
elif page == "AI Chatbot":
    st.title("Talk to Data (NL-to-SQL)")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.warning("GROQ_API_KEY not found in environment variables. Please enter it below.")
        groq_api_key = st.text_input("Groq API Key", type="password")
        
    if groq_api_key:
        chatbot = NLtoSQLChatbot(api_key=groq_api_key)
        
        st.markdown("Ask business questions about the credit risk dataset in natural language.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("E.g., What is the average income of defaulters?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = chatbot.process_question(prompt)
                    
                    if response["status"] == "success":
                        st.markdown(f"**Answer:** {response['business_answer']}")
                        with st.expander("View Generated SQL & Data"):
                            st.code(response['sql_query'], language="sql")
                            st.dataframe(response['result_df'])
                        
                        st.session_state.messages.append({"role": "assistant", "content": response['business_answer']})
                    else:
                        st.error(response["business_answer"])

# App is ready.

