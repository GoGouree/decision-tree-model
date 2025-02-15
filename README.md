# Decision Tree Model Project

## Overview

This project involves building a decision tree model to detect fraudulent transactions. The steps include generating synthetic data, performing exploratory data analysis (EDA), training the model, running the model on new data, and interpreting the results.


## Steps

### 1. Generate Synthetic Data

We generated synthetic data to simulate fraudulent transactions using the `Synthetic_data.py` script. This script creates a synthetic dataset and saves it to `data/training dataset/synthetic_fraud_data.csv`.

### 2. Exploratory Data Analysis (EDA)

We performed EDA to understand the data better using the `data_exploration.py` script. This script loads the synthetic dataset, displays basic information, generates summary statistics, and creates various plots to visualize the data.

### 3. Train the Model

We trained a decision tree model using the `model_training.py` script. This script loads the synthetic dataset, preprocesses the data, splits it into training and testing sets, trains the decision tree model, evaluates its performance, and plots the confusion matrix.

### 4. Run the Model on New Data

We used the trained model to make predictions on new data using the `model_training.py` script. This script loads new data, preprocesses it to match the training data format, makes predictions using the trained model, and saves the predictions to `data/new_data_with_predictions.csv`.

### 5. Interpret the Results

The confusion matrix and classification report provide insights into the model's performance. The confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives. The classification report provides precision, recall, and F1-score for each class.

#### Confusion Matrix

- **TP (True Positive)**: Correctly predicted positive cases.
- **TN (True Negative)**: Correctly predicted negative cases.
- **FP (False Positive)**: Incorrectly predicted positive cases.
- **FN (False Negative)**: Incorrectly predicted negative cases.

#### Classification Report

The classification report includes:
- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model.
- **Recall (Sensitivity)**: The proportion of true positive predictions out of all actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall.

### Conclusion

This project demonstrates the process of generating synthetic data, performing exploratory data analysis, training a decision tree model, making predictions on new data, and interpreting the results. The model achieved an accuracy of 0.78, and further fine-tuning and feature engineering can be explored to improve performance.