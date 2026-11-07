import json
import os

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Classification Model Comparison",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Classification Model Comparison Dashboard")

st.write(
    "Upload test data, select a trained classification model, "
    "and evaluate its performance."
)

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

with open("feature_columns.json", "r") as file:
    feature_columns = json.load(file)

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Classification Model",
    list(MODEL_PATHS.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("The uploaded CSV must contain a target column.")

    elif not set(feature_columns).issubset(data.columns):
        st.error("The uploaded CSV is missing required feature columns.")

    else:

        X_test = data[feature_columns]
        y_test = data["target"]

        model = joblib.load(MODEL_PATHS[selected_model])

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_probability = model.predict_proba(X_test)[:, 1]
        else:
            y_probability = model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probability)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
        )

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a CSV file containing features and target.")