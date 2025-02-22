import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/goure/decision-tree-model/data/training dataset/synthetic_fraud_data.csv')

# Display basic information about the dataset
print("Basic Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(data.head())

# Prepare the data for training
# Assuming 'Fraud' is the target variable and the rest are features
X = data.drop('Fraud', axis=1)
y = data['Fraud']

# Convert categorical features to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nAccuracy Score:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Load new data for prediction
new_data = pd.read_csv('C:/Users/goure/decision-tree-model/data/prediction_dataset/new_data.csv')

# Preprocess the new data (ensure it matches the training data preprocessing)
new_data_preprocessed = pd.get_dummies(new_data, drop_first=True)

# Ensure the new data has the same columns as the training data
missing_cols = set(X_train.columns) - set(new_data_preprocessed.columns)
for col in missing_cols:
    new_data_preprocessed[col] = 0
new_data_preprocessed = new_data_preprocessed[X_train.columns]

# Make predictions on the new data
new_predictions = clf.predict(new_data_preprocessed)

# Add predictions to the new data
new_data['Predicted_Fraud'] = new_predictions

# Save the new data with predictions
new_data.to_csv('C:/Users/goure/decision-tree-model/data/prediction_dataset/new_data_with_predictions.csv', index=False)
print("Predictions saved to new_data_with_predictions.csv")