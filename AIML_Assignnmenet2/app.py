
import json
from pathlib import Path

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

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

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

# --------------------------------------------------
# 2. Model file paths
# --------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": APP_DIR / "model" / "logistic_regression.joblib",
    "Decision Tree": APP_DIR / "model" / "decision_tree.joblib",
    "KNN": APP_DIR / "model" / "kn
