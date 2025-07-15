# Save this as app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Salary Predictor")

# Load data
data = pd.read_csv("Cleaned_Salary_Data.csv")
data = data.dropna(subset=["Salary", "Degree"])

# Define features
features = ["Age", "Gender", "Degree", "Job_Title", "Experience_Years"]
X = data[features]
y = data["Salary"]  # Monthly salary

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Model Metrics ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
st.metric("R² Score", f"{r2:.2f}")

# --- Input Section ---
st.header("🧾 Enter Employee Details")

degree = st.selectbox("🎓 Education Level", sorted(data["Degree"].dropna().unique()), index=0)
experience = st.number_input("💼 Experience (Years)", min_value=0, max_value=40, value=2)
job = st.selectbox("👔 Job Title", sorted(data["Job_Title"].dropna().unique()), index=0)
age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("🧍 Gender", sorted(data["Gender"].dropna().unique()), index=0)

# --- Predict Salary ---
if st.button("🔮 PREDICT MY SALARY"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Degree": degree,
        "Job_Title": job,
        "Experience_Years": experience
    }])
    
    prediction = model.predict(input_df)[0]
    monthly_salary = int(prediction)
    annual_salary = monthly_salary * 12

    st.success(f"🎯 Predicted Monthly Salary: ₹ {monthly_salary:,}")
    st.info(f"💰 Estimated Annual Salary: ₹ {annual_salary:,}")
    st.caption("📌 Note: Model is trained on monthly salary data.")

# --- Evaluation Plots ---
st.header("📈 Model Evaluation Plots")

# Plot: Actual vs Predicted (Sample 50)
fig1, ax1 = plt.subplots()
ax1.plot(y_test.values[:50], label='Actual', marker='o')
ax1.plot(y_pred[:50], label='Predicted', marker='x')
ax1.set_title("Actual vs Predicted Salary (Sample 50)")
ax1.legend()
st.pyplot(fig1)

# Plot: Residual Distribution
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax2)
ax2.set_title("Distribution of Residuals")
st.pyplot(fig2)

# Plot: Feature Importance
importances = model.named_steps["regressor"].feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)

# Plot: Scatter Plot - Actual vs Predicted
fig4, ax4 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax4.set_xlabel("Actual Salary")
ax4.set_ylabel("Predicted Salary")
ax4.set_title("Scatter Plot: Actual vs Predicted")
st.pyplot(fig4)

