# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
features = data['data']
labels = data['target']
feature_names = data['feature_names']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

train, test, train_labels, test_labels = train_test_split(
    features_scaled, labels, test_size=0.33, random_state=42
)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
model = rfc.fit(train, train_labels)

predictions = model.predict(test)

accuracy = accuracy_score(test_labels, predictions)

st.title('Breast Cancer Prediction')

st.write("""
    Prediction of a tumor is Cancerous or Non_Cancerous based on 
    the breast cancer dataset.
""")

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

st.sidebar.header("Enter feature values")

input_values = {}
for feature in feature_names:
    input_values[feature] = st.sidebar.number_input(f"{feature}", value=10.0, min_value=0.0)

user_input = np.array([list(input_values.values())])

user_input_scaled = scaler.transform(user_input)

if st.sidebar.button("Predict"):
    user_prediction = model.predict(user_input_scaled)
    result = 'Cancerous' if user_prediction == 1 else 'Non_Cancerous'
    st.write(f"Prediction: The tumor is **{result}**.")

st.write("Dataset Overview:")
df = pd.DataFrame(features, columns=feature_names)
st.dataframe(df.head())
