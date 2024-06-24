import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# Load the dataset
file_path = 'Customer Data.csv'
customer_data = pd.read_csv(file_path)

# Impute missing values for "CREDIT_LIMIT" and "MINIMUM_PAYMENTS"
imputer = SimpleImputer(strategy='mean')
customer_data[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']] = imputer.fit_transform(customer_data[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']])

# Drop the CUST_ID column as it's not useful for clustering or classification
customer_data = customer_data.drop(columns=['CUST_ID'])

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Split the data into two sets (70% training, 30% testing)
data_train, data_test = train_test_split(customer_data_scaled, test_size=0.3, random_state=42)



# Perform K-means clustering on the training set
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data_train)

# Add cluster labels to the training set
data_train_with_labels = pd.DataFrame(data_train, columns=customer_data.columns)
data_train_with_labels['Cluster'] = kmeans_labels



# Perform K-means clustering on the training set
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data_train)

# Add cluster labels to the training set
data_train_with_labels = pd.DataFrame(data_train, columns=customer_data.columns)
data_train_with_labels['Cluster'] = kmeans_labels



# Prepare the training and test sets for classification
X_train = data_train_with_labels.drop(columns=['Cluster'])
y_train = data_train_with_labels['Cluster']
X_test = pd.DataFrame(data_test, columns=customer_data.columns)

# Train a RandomForest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the clusters for the test set
y_test_pred = clf.predict(X_test)

# Evaluate the classifier
print('Accuracy:', accuracy_score(y_train, clf.predict(X_train)))

import joblib
# joblib.dump(clf, 'random_forest_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
clf = joblib.load('random_forest_model.pkl')  # Adjust the file path as needed
scaler = joblib.load('scaler.pkl')  # Adjust the file path as needed

# Load the data with clusters for displaying the charts
# This is a placeholder, replace with the actual data loading code
data_train_with_labels = pd.read_csv('Customer Data.csv')  # Adjust the file path as needed

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
    input_data_scaled = scaler.transform(input_data)

    # Predict the cluster
    cluster = clf.predict(input_data_scaled)
    st.write(f'The customer belongs to cluster: {cluster[0]}')

# Plot pie chart and bar chart
if st.checkbox('Show Customer Segment Distribution'):
    cluster_counts = data_train_with_labels['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # Pie chart
    fig_pie = px.pie(cluster_counts, names='Cluster', values='Count', title='Customer Segment Distribution')
    st.plotly_chart(fig_pie)

    # Bar chart
    fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', title='Customer Segment Count', text='Count')
    st.plotly_chart(fig_bar)