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
st.set_page_config(page_title="Smart Salary Estimator Using Machine Learning", page_icon="💼", layout="centered")

# --- Custom CSS: Background + Font Styling ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #e6f2ff;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #d9eaf7;
    }

    html, body, [class*="css"]  {
        font-size: 18px;
        font-family: "Segoe UI", sans-serif;
    }

    .stMetric {
        background-color: #ffffffaa;
        padding: 1em;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
    }

    .stButton > button {
        background-color: #4da6ff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        font-size: 18px;
    }

    .stButton > button:hover {
        background-color: #1a8cff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("💼 Employee Salary Predictor")

# --- Load Data ---
data = pd.read_csv("Cleaned_Salary_Data.csv")
data = data.dropna(subset=["Salary", "Degree"])

# --- Define Features ---
features = ["Age", "Gender", "Degree", "Job_Title", "Experience_Years"]
X = data[features]
y = data["Salary"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing ---
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

# --- Model Pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:,.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# --- User Input Section ---
st.header("🧾 Enter Employee Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30)
    experience = st.number_input("💼 Experience (Years)", min_value=0, max_value=40, value=2)
    gender = st.selectbox("🧍 Gender", sorted(data["Gender"].dropna().unique()))

with col2:
    degree = st.selectbox("🎓 Education Level", sorted(data["Degree"].dropna().unique()))
    job = st.selectbox("👔 Job Title", sorted(data["Job_Title"].dropna().unique()))

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

# Plot 1: Actual vs Predicted (Sample 50)
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(y_test.values[:50], label='Actual', marker='o')
ax1.plot(y_pred[:50], label='Predicted', marker='x')
ax1.set_title("Actual vs Predicted Salary (Sample 50)", fontsize=16)
ax1.set_xlabel("Sample Index", fontsize=14)
ax1.set_ylabel("Salary ₹", fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
ax1.legend()
ax1.grid()
st.pyplot(fig1)

# Plot 2: Residual Distribution
residuals = y_test - y_pred
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True, ax=ax2)
ax2.set_title("Distribution of Residuals", fontsize=16)
ax2.set_xlabel("Residuals", fontsize=14)
ax2.set_ylabel("Count", fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
st.pyplot(fig2)

# Plot 3: Feature Importance
importances = model.named_steps["regressor"].feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), ax=ax3)
ax3.set_title("Feature Importance", fontsize=16)
ax3.set_xlabel("Importance", fontsize=14)
ax3.set_ylabel("Feature", fontsize=14)
ax3.tick_params(axis='both', labelsize=12)
st.pyplot(fig3)

# Plot 4: Scatter Plot - Actual vs Predicted
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax4.set_xlabel("Actual Salary", fontsize=14)
ax4.set_ylabel("Predicted Salary", fontsize=14)
ax4.set_title("Scatter Plot: Actual vs Predicted", fontsize=16)
ax4.tick_params(axis='both', labelsize=12)
st.pyplot(fig4)

