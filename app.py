
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #e6f2ff, #ffffff);
        padding: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #d9eaf7;
        padding-top: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
        font-size: 18px;
    }

    .stMetric {
        background-color: #ffffffcc;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        text-align: center;
    }

    .stButton > button {
        background-color: #4da6ff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1a8cff;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("💼 Employee Salary Predictor with Model Selection")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Salary_Data.csv")
    df.dropna(subset=["Salary", "Degree"], inplace=True)
    return df

data = load_data()

# --- Feature Definition ---
features = ["Age", "Gender", "Degree", "Job_Title", "Experience_Years"]
target = "Salary"
X = data[features]
y = data[target]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing Pipelines ---
numeric_features = ["Age", "Experience_Years"]
categorical_features = ["Gender", "Degree", "Job_Title"]

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

st.subheader("🔍 Comparing Models")

for name, reg in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": pipe, "MSE": mse, "R2": r2}
    st.write(f"📌 **{name}** — R²: `{r2:.4f}` | MSE: `{mse:,.2f}`")

# --- Select Best Model ---
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = results[best_model_name]["model"]
best_r2 = results[best_model_name]["R2"]
best_mse = results[best_model_name]["MSE"]

st.success(f"✅ Best Performing Model: **{best_model_name}** | R² Score: `{best_r2:.4f}`")

# --- Model Comparison Plots ---
st.header("📊 Model Comparison Plots")

# Prepare data for plotting
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]
mses = [results[m]["MSE"] for m in model_names]

# R² Score Plot
fig_r2, ax_r2 = plt.subplots()
sns.barplot(x=r2_scores, y=model_names, ax=ax_r2, palette="viridis")
ax_r2.set_title("R² Score by Model")
ax_r2.set_xlabel("R² Score")
ax_r2.set_xlim(0, 1)
for i, v in enumerate(r2_scores):
    ax_r2.text(v + 0.01, i, f"{v:.3f}", va='center')
st.pyplot(fig_r2)

# MSE Plot
fig_mse, ax_mse = plt.subplots()
sns.barplot(x=mses, y=model_names, ax=ax_mse, palette="rocket")
ax_mse.set_title("Mean Squared Error by Model")
ax_mse.set_xlabel("MSE")
for i, v in enumerate(mses):
    ax_mse.text(v + max(mses)*0.01, i, f"{int(v):,}", va='center')
st.pyplot(fig_mse)

# --- User Input for Prediction ---
st.header("🧾 Enter Employee Details to Predict Salary")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30)
    experience = st.number_input("💼 Experience (Years)", min_value=0, max_value=40, value=2)
    gender = st.selectbox("🧍 Gender", sorted(data["Gender"].dropna().unique()))

with col2:
    degree = st.selectbox("🎓 Education Level", sorted(data["Degree"].dropna().unique()))
    job = st.selectbox("👔 Job Title", sorted(data["Job_Title"].dropna().unique()))

# --- Predict Salary ---
if st.button("🔮 Predict My Salary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Degree": degree,
        "Job_Title": job,
        "Experience_Years": experience
    }])
    salary = best_model.predict(input_df)[0]
    st.success(f"🎯 Predicted Monthly Salary: ₹ {int(salary):,}")
    st.info(f"💰 Estimated Annual Salary: ₹ {int(salary * 12):,}")
    st.caption(f"📌 Prediction made using **{best_model_name}**")

# --- Evaluation Plots for Best Model ---
st.header("📈 Model Evaluation (Best Model)")

y_pred = best_model.predict(X_test)

# Plot 1: Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(y_test.values[:50], label='Actual', marker='o')
ax1.plot(y_pred[:50], label='Predicted', marker='x')
ax1.set_title(f"Actual vs Predicted Salary ({best_model_name})")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Salary ₹")
ax1.legend()
st.pyplot(fig1)

# Plot 2: Residuals
residuals = y_test - y_pred
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True, ax=ax2)
ax2.set_title("Residual Distribution")
ax2.set_xlabel("Residuals")
st.pyplot(fig2)

# Plot 3: Feature Importance (if applicable)
if hasattr(best_model.named_steps["regressor"], "feature_importances_"):
    importances = best_model.named_steps["regressor"].feature_importances_
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df.head(15), ax=ax3)
    ax3.set_title("Top Feature Importances")
    st.pyplot(fig3)

# Plot 4: Scatter Plot
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax4.set_xlabel("Actual Salary")
ax4.set_ylabel("Predicted Salary")
ax4.set_title("Scatter Plot: Actual vs Predicted")
st.pyplot(fig4)

