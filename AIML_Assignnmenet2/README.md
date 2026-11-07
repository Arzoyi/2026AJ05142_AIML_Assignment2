# Machine Learning Assignment 2
## Comparative Analysis of Classification Algorithms

### 1. Problem Statement

The objective of this assignment is to implement and compare five classification algorithms using a common dataset. The models are evaluated using multiple classification performance metrics to understand their strengths and weaknesses.

### 2. Dataset Description

The dataset contains 569 instances and 30 input features, with one target column named `target`.

- Total records: 569
- Input features: 30
- Target variable: `target`
- Target classes: 0 and 1
- Class 0 records: 357
- Class 1 records: 212
- Missing values: None

The dataset satisfies the assignment requirements of at least 500 instances and 12 input features.

### 3. Algorithms Implemented

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

The same training and testing split was used for all five models. The data was divided into 80% training and 20% testing data.

### 4. Model Comparison

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Fill result | Fill result | Fill result | Fill result | Fill result | Fill result |
| Decision Tree | Fill result | Fill result | Fill result | Fill result | Fill result | Fill result |
| KNN | Fill result | Fill result | Fill result | Fill result | Fill result | Fill result |
| Naive Bayes | Fill result | Fill result | Fill result | Fill result | Fill result | Fill result |
| SVM | Fill result | Fill result | Fill result | Fill result | Fill result | Fill result |

### 5. Observations

**Logistic Regression:**  
Write your observations based on its actual evaluation metrics.

**Decision Tree:**  
Write your observations based on its actual evaluation metrics.

**K-Nearest Neighbors:**  
Write your observations based on its actual evaluation metrics.

**Gaussian Naive Bayes:**  
Write your observations based on its actual evaluation metrics.

**Support Vector Machine:**  
Write your observations based on its actual evaluation metrics.

**Overall Best Model:**  
Identify the model with the highest F1-score and explain its performance using the other evaluation metrics.

### 6. GitHub Repository

Add your GitHub repository link here.

### 7. Streamlit Application

Add your deployed Streamlit application link here.

### 8. Application Features

The Streamlit application provides:

- CSV test data upload
- Classification model selection
- Accuracy, AUC, Precision, Recall, F1-score and MCC
- Confusion matrix
- Classification report

The application loads the previously trained models and evaluates the uploaded data without retraining.

### 9. Project Files

- `ML_Assignment_2.ipynb`: Notebook containing data exploration, training and evaluation.
- `dataset.csv`: Original dataset.
- `test_data.csv`: Test dataset used for evaluation.
- `model_comparison.csv`: Model performance comparison.
- `feature_columns.json`: List of required feature columns.
- `model/`: Saved trained classification models.
- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
