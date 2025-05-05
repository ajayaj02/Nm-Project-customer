#customer_churn_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'customer_data.csv' with your dataset)
data = pd.read_csv('customer_data.csv')

# Basic EDA
print("First 5 rows of the dataset:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())

# Drop irrelevant columns (example: customerID)
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert categorical variables
categorical_features = data.select_dtypes(include=['object']).columns

for col in categorical_features:
    data[col] = data[col].astype('category').cat.codes

# Define features and label
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
