import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import os
import time

import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

def train_model_from_csv():
    df = pd.read_csv("Churn_Modelling.csv")
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    y = df["Exited"]
    X = df.drop("Exited", axis=1)

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred)
    }

    return model, metrics


@st.cache_resource
def load_or_train_model():
    if os.path.exists("churn_model.pkl"):
        try:
            return joblib.load("churn_model.pkl")
        except:
            pass

    model, metrics = train_model_from_csv()
    joblib.dump(model, "churn_model.pkl")
    st.session_state.metrics = metrics
    return model


model = load_or_train_model()

if "history" not in st.session_state:
    st.session_state.history = []

if "metrics" not in st.session_state:
    _, st.session_state.metrics = train_model_from_csv()

metrics = st.session_state.metrics

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg,#2563eb,#1e40af);
    padding: 2rem;
    color: white;
    border-radius: 18px;
}
.card {
    background: white;
    padding: 1.4rem;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.metric {
    font-size: 30px;
    font-weight: 700;
}
.label {
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📊 Customer Churn Prediction System</h1>
<p>Machine Learning based customer retention analysis</p>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.header("🔍 Customer Profile")
    credit_score = st.slider("Credit Score", 300, 900, 600)
    geography = st.selectbox("Country", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Years with Bank", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 60000.0, step=1000.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.checkbox("Has Credit Card")
    is_active = st.checkbox("Active Member")
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0, step=1000.0)

    predict_btn = st.button("🚀 Predict Churn", use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary
    }])

    with st.spinner("🔄 Analyzing customer data..."):
        time.sleep(1)
        proba = model.predict_proba(input_df)[0][1]

    prediction = "Churn" if proba >= 0.5 else "Not Churn"

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": prediction,
        "Churn Probability": round(proba, 4)
    })

    st.subheader("🔎 Prediction Result")
    st.success(f"Prediction: **{prediction}**")
    st.info(f"Churn Probability: **{proba:.2%}**")

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": proba * 100
            }
        },
        title={"text": "Churn Risk Meter"}
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("---")
st.subheader("📈 Model Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
c2.metric("Precision", f"{metrics['precision']:.2%}")
c3.metric("Recall", f"{metrics['recall']:.2%}")
c4.metric("F1 Score", f"{metrics['f1']:.2%}")

st.markdown("### Confusion Matrix")
cm = metrics["confusion"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Churn", "Actual: Churn"],
    columns=["Predicted: Not Churn", "Predicted: Churn"]
)
st.dataframe(cm_df, use_container_width=True)

st.markdown("---")
st.subheader("📄 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions yet.")
else:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download Prediction History",
        history_df.to_csv(index=False),
        "churn_prediction_history.csv",
        "text/csv"
    )
