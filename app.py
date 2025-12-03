import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

def train_model_from_csv():
    df = pd.read_csv("Churn_Modelling.csv")
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    y = df["Exited"]
    X = df.drop("Exited", axis=1)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    return pipeline

@st.cache_resource
def load_model():
    try:
        if os.path.exists("churn_model.pkl"):
            return joblib.load("churn_model.pkl")
        else:
            st.warning("Model file not found. Training a new model from Churn_Modelling.csv.")
            model = train_model_from_csv()
            joblib.dump(model, "churn_model.pkl")
            return model
    except Exception:
        st.warning("Error loading churn_model.pkl. Training a new model from Churn_Modelling.csv.")
        try:
            model = train_model_from_csv()
            return model
        except Exception:
            st.error("Unable to train model. Make sure Churn_Modelling.csv is in the app folder and correctly formatted.")
            st.stop()

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        padding-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 16px;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    .success-box {
        background-color: #ecfdf3;
        border: 1px solid #4ade80;
    }
    .danger-box {
        background-color: #fef2f2;
        border: 1px solid #f87171;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("🔧 Input Customer Details")
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    has_card_val = 1 if has_card == "Yes" else 0
    is_active_val = 1 if is_active == "Yes" else 0
    predict_button = st.button("🔮 Predict Churn", use_container_width=True)

st.markdown('<div class="main-title">📉 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Use this tool to estimate whether a customer is likely to leave the bank based on their profile.</div>',
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1.2, 1])

if predict_button:
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card_val,
        "IsActiveMember": is_active_val,
        "EstimatedSalary": salary
    }])
    prediction = int(model.predict(input_data)[0])
    proba = float(model.predict_proba(input_data)[0][1])
    churn_probability = proba
    not_churn_probability = 1 - proba
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "ChurnPrediction": "Churn" if prediction == 1 else "Not Churn",
        "ChurnProbability": round(churn_probability, 4)
    }
    st.session_state.history.append(record)
    with col_left:
        box_class = "danger-box" if prediction == 1 else "success-box"
        st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("### ⚠️ Customer is likely to **CHURN**")
            st.markdown(f"Probability of churn: **{churn_probability:.2%}**")
        else:
            st.markdown("### ✅ Customer is likely to **STAY**")
            st.markdown(f"Probability of churn: **{churn_probability:.2%}**")
        st.markdown("</div>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown('<div class="metric-label">Churn Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{churn_probability:.2%}</div>', unsafe_allow_html=True)
        with mc2:
            st.markdown('<div class="metric-label">Not Churn Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{not_churn_probability:.2%}</div>', unsafe_allow_html=True)
    with col_right:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_probability * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"thickness": 0.3},
                    "steps": [
                        {"range": [0, 30], "color": "#bbf7d0"},
                        {"range": [30, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fecaca"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": churn_probability * 100,
                    },
                },
                title={"text": "Churn Risk Gauge"}
            )
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📊 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions made yet. Fill the details in the sidebar and click **Predict Churn** to see results here.")
else:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download History as CSV",
        data=csv_data,
        file_name="churn_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )
