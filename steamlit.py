import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import joblib

# Load the trained model and scaler
try:
    clf = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Load the data with clusters for displaying the charts
try:
    data_train_with_labels = pd.read_csv('data_train_with_labels.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

# Define the Streamlit app
st.title('Customer Segmentation')

# Input fields
balance = st.number_input('BALANCE')
balance_frequency = st.number_input('BALANCE_FREQUENCY')
purchases = st.number_input('PURCHASES')
cash_advance = st.number_input('CASH_ADVANCE')
purchases_frequency = st.number_input('PURCHASES_FREQUENCY')
credit_limit = st.number_input('CREDIT_LIMIT')
payments = st.number_input('PAYMENTS')
minimum_payments = st.number_input('MINIMUM_PAYMENTS')
tenure = st.number_input('TENURE')

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([[balance, balance_frequency, purchases, cash_advance, purchases_frequency, credit_limit, payments, minimum_payments, tenure]])
    
    try:
        # Check the shape of input_data
        st.write(f"Input data shape: {input_data.shape}")

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict the cluster
        cluster = clf.predict(input_data_scaled)
        st.write(f'The customer belongs to cluster: {cluster[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Plot pie chart and bar chart
if st.checkbox('Show Customer Segment Distribution'):
    try:
        cluster_counts = data_train_with_labels['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']

        # Pie chart
        fig_pie = px.pie(cluster_counts, names='Cluster', values='Count', title='Customer Segment Distribution')
        st.plotly_chart(fig_pie)

        # Bar chart
        fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', title='Customer Segment Count', text='Count')
        st.plotly_chart(fig_bar)
    except Exception as e:
        st.error(f"Error creating charts: {e}")
