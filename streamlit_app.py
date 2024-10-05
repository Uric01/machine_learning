import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import io
import json
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_period_transactions
import altair as alt
import matplotlib.pyplot as plt

@st.cache_data
def load_and_preprocess_data(file):
    try:
        sales_data = pd.read_csv(file)
        required_columns = ['customer_id', 'date']
        if not all(col in sales_data.columns for col in required_columns):
            raise ValueError(f"The dataset must contain the following columns: {', '.join(required_columns)}")
        
        sales_data['date'] = pd.to_datetime(sales_data['date'], errors='coerce')
        if sales_data['date'].isnull().any():
            raise ValueError("Some date values couldn't be parsed. Please ensure all dates are in a valid format.")
        
        sales_data = sales_data.dropna(subset=['customer_id', 'date'])
        return sales_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(summary, penalizer_coef):
    with st.spinner("Training BG/NBD model..."):
        bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    return bgf

def make_predictions(bgf, summary, time_period):
    return bgf.conditional_expected_number_of_purchases_up_to_time(
        time_period,
        summary['frequency'],
        summary['recency'],
        summary['T']
    )

def plot_frequency_recency_heatmap(bgf, summary):
    max_frequency = min(int(summary['frequency'].max()), 50)
    max_recency = min(int(summary['T'].max()), 50)
    
    data = []
    for i in range(max_recency+1):
        for j in range(max_frequency+1):
            expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                30, j, i, max_recency
            )
            data.append({'Recency': i, 'Frequency': j, 'Expected Purchases': expected_purchases})
    
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('Frequency:O', title='Frequency'),
        y=alt.Y('Recency:O', title='Recency'),
        color=alt.Color('Expected Purchases:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(
        title='Frequency-Recency Matrix',
        width=600,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)
    st.write("Frequency-Recency Matrix: Darker colors indicate higher expected future purchases in the next 30 days.")


def plot_validation(bgf, summary):
    """Plots a validation curve comparing actual vs. predicted transactions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_period_transactions(bgf, ax=ax)
    st.pyplot(fig)
    st.write("This plot shows how well the model fits the data, comparing actual vs. predicted transactions.")

def save_model_params(bgf):
    return {
        'params': bgf.params_.to_dict(),
        'penalizer_coef': bgf.penalizer_coef
    }

def load_model_params(params_dict):
    bgf = BetaGeoFitter(penalizer_coef=params_dict['penalizer_coef'])
    bgf.params_ = pd.Series(params_dict['params'])
    return bgf

def main():
    st.title("Customer Lifetime Value Prediction with BG/NBD Model")
    
    uploaded_file = st.file_uploader("Upload your sales history dataset (CSV)- Please note required columns 'customer_id' and 'date' ", type=["csv"])
    
    if uploaded_file is not None:
        sales_data = load_and_preprocess_data(uploaded_file)
        
        if sales_data is not None:
            st.write("Sales Data Preview:")
            st.write(sales_data.head())
            
            with st.spinner("Transforming data..."):
                summary = summary_data_from_transaction_data(sales_data, 'customer_id', 'date', observation_period_end=sales_data['date'].max())
            
            st.write("Transformed Data Preview:")
            st.write(summary.head())
            
            penalizer_coef = st.slider("Penalizer Coefficient", 0.0, 1.0, 0.0, 0.01)
            bgf = train_model(summary, penalizer_coef)
            
            st.success("BG/NBD model trained successfully!")
            
            time_period = st.slider("Prediction Time Period (days)", 1, 365, 60)
            
            with st.spinner("Making predictions..."):
                summary['predicted_transactions'] = make_predictions(bgf, summary, time_period)
            
            st.write("Predictions:")
            st.write(summary[['frequency', 'recency', 'T', 'predicted_transactions']])
            
            st.subheader("Model Visualizations")
            plot_frequency_recency_heatmap(bgf, summary)
        

            st.subheader("Model Validation")
            plot_validation(bgf, summary)
            
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                csv_data = summary.to_csv(index=False)
                zip_file.writestr("predictions.csv", csv_data)
                
                model_params = save_model_params(bgf)
                zip_file.writestr("model_params.json", json.dumps(model_params))
            
            st.download_button(
                label="Download Predictions and Model Parameters as ZIP",
                data=buffer.getvalue(),
                file_name="predictions_and_model_params.zip",
                mime="application/zip"
            )
    else:
        st.info("Please upload a sales history dataset to proceed.")

if __name__ == "__main__":
    main()
