import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("🚨 Fraud Detection System")
st.markdown("Use this tool to predict fraudulent transactions using machine learning.")

@st.cache_data
def load_data():
    df = pd.read_csv("onlinefraud.csv")
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    return df, le

df, le = load_data()

@st.cache_resource
def train_model():
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balanceDiffOrig',
                'oldbalanceDest', 'newbalanceDest', 'balanceDiffDest', 'type']
    X = df[features]
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, features

model, feature_cols = train_model()

st.subheader("🔍 Fraud Prediction")
st.markdown("Enter transaction details below and click 'Predict' to check for fraud.")

with st.form(key="input_form"):
    amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=0.01)
    oldbalanceOrg = st.number_input("🏦 Old Balance Origin", min_value=0.0, step=0.01)
    newbalanceOrig = st.number_input("💳 New Balance Origin", min_value=0.0, step=0.01)
    oldbalanceDest = st.number_input("🏦 Old Balance Destination", min_value=0.0, step=0.01)
    newbalanceDest = st.number_input("💳 New Balance Destination", min_value=0.0, step=0.01)
    type_input = st.selectbox("🔄 Transaction Type", df['type'].unique())
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    predict_button = st.form_submit_button("🚀 Predict Fraud")

if predict_button:
    input_data = [amount, oldbalanceOrg, newbalanceOrig, balanceDiffOrig, oldbalanceDest, newbalanceDest, balanceDiffDest, type_input]
    prediction = model.predict(pd.DataFrame([input_data], columns=feature_cols))
    result = "🛑 Fraud Detected!" if prediction[0] == 1 else "✅ Transaction is Safe."
    st.markdown(f"### {result}")
