import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
data = pd.read_csv('kerala_datasets.csv')
print(data.head())
print(data.info())
# Check for missing values
print(data.isnull().sum())

# Fill missing values with the mean (or other appropriate strategies)
# Fill missing values for numeric columns only
data.fillna(data.select_dtypes(include=np.number).mean(), inplace=True)

# Convert categorical columns using one-hot encoding (if any exist)
data = pd.get_dummies(data, drop_first=True)
# Perform one-hot encoding
encoded_data = pd.get_dummies(data, drop_first=True)


# Convert boolean values to integers (1 for True, 0 for False)
encoded_data['FLOODS_YES'] = encoded_data['FLOODS_YES'].astype(int)

# Display the updated DataFrame
print(encoded_data.head())

# Assuming the target column is named 'Flood' (1 for flood, 0 for no flood)
# Assuming the target column is 'FLOODS_YES' after one-hot encoding
X = data.drop('FLOODS_YES', axis=1)  # Features
y = data['FLOODS_YES']  # Target

# Display features (X) and target (y) --
print("Features (X):")
print(X.head())  # First 5 rows of features

print("\nTarget (y):")
print(y.head())  # First 5 rows of the target variable

# Plot the distribution of the target variable
sns.countplot(x='FLOODS_YES', data=data)
plt.title('Distribution of Flood Occurrence')
plt.show()

# Correlation heatmap to check relationships
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify shapes
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

from sklearn.preprocessing import StandardScaler

# Apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled features back to a DataFrame for better readability--
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the first few rows of the scaled features--
print("Scaled X_train:")
print(X_train_scaled_df.head())  # Show the first 5 rows

print("\nScaled X_test:")
print(X_test_scaled_df.head())  # Show the first 5 rows

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Accuracy, Precision, Recall, F1-Score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ROC-AUC Score
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
import numpy as np

# Function to take user input
def get_user_input():
    print("Enter the feature values for prediction:")
    user_input = {}
    for col in X.columns:  # Iterate through the feature columns
        value = float(input(f"Enter value for {col}: "))
        user_input[col] = value
    return user_input

# Get user input and convert to DataFrame
user_data = get_user_input()
user_df = pd.DataFrame([user_data])

# Scale the user input using the same scaler
user_scaled = scaler.transform(user_df)

# Make prediction
prediction = model.predict(user_scaled)
prediction_proba = model.predict_proba(user_scaled)[:, 1]

# Display the results
if prediction[0] == 1:
    print("\nPrediction: Flood is likely to occur.")
else:
    print("\nPrediction: Flood is not likely to occur.")

print(f"Prediction Probability (Flood Likelihood): {prediction_proba[0]:.2f}")