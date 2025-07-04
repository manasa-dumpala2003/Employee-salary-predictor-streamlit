# Save this as app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Title
st.title("💼 Employee Salary Predictor")

# Load cleaned data
data = pd.read_csv("Cleaned_Salary_Data.csv")
data = data.dropna(subset=["Salary", "Degree"])

# Features and target
X = data[["Age", "Gender", "Degree", "Job_Title", "Experience_Years"]]
y = data["Salary"]

# Preprocessing
numeric_features = ["Age", "Experience_Years"]
categorical_features = ["Gender", "Degree", "Job_Title"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X, y)

# --- Streamlit User Input ---
st.header("📥 Enter Employee Details")

age = st.slider("Age", 18, 70, 30)
experience = st.slider("Years of Experience", 0, 40, 5)
gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique()))
degree = st.selectbox("Degree", sorted(data["Degree"].dropna().unique()))
job = st.selectbox("Job Title", sorted(data["Job_Title"].dropna().unique()))

# Predict Button
if st.button("Predict Salary 💰"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Degree": degree,
        "Job_Title": job,
        "Experience_Years": experience
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Salary: ₹ {int(prediction):,}")
