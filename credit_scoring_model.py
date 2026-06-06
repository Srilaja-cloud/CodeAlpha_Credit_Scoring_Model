# ==========================================
# Credit Scoring Model
# CodeAlpha Machine Learning Internship
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# Create Sample Credit Dataset
# ==========================================

data = {
    "Income": [50000, 60000, 25000, 80000, 45000, 70000, 30000, 90000, 40000, 55000,
               65000, 35000, 75000, 48000, 85000, 28000, 92000, 38000, 62000, 53000],

    "Debt": [10000, 12000, 15000, 5000, 11000, 7000, 18000, 4000, 13000, 9000,
             8000, 16000, 6000, 10000, 3000, 17000, 2000, 14000, 7500, 9500],

    "Credit_History": [8, 10, 2, 12, 7, 9, 3, 15, 5, 8,
                       11, 4, 13, 6, 14, 2, 16, 5, 10, 7],

    "Age": [35, 40, 24, 45, 32, 38, 22, 50, 29, 34,
            41, 26, 47, 31, 52, 23, 55, 28, 39, 33],

    "Creditworthy": [1,1,0,1,1,1,0,1,0,1,
                     1,0,1,1,1,0,1,0,1,1]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Logistic Regression Model
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")
print(f"ROC-AUC   : {roc_auc:.2f}")

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# Confusion Matrix Visualization
# ==========================================

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Bad Credit", "Good Credit"],
    yticklabels=["Bad Credit", "Good Credit"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# Example Prediction
# ==========================================

sample_customer = [[65000, 8000, 10, 40]]

prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("\nCustomer is Creditworthy")
else:
    print("\nCustomer is Not Creditworthy")
