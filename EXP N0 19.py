
# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
data = pd.read_csv("/content/loan_approval_dataset.csv")

print("First 5 rows of dataset:")
print(data.head())

# Step 3: Handle missing values
data.ffill(inplace=True)

# Step 4: Encode categorical columns
le = LabelEncoder()

categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Step 5: Split features and target
# Change 'Loan_Status' if your dataset uses a different column name
X = data.drop('LoanApproved', axis=1)
y = data['LoanApproved']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
