import pandas as pd
import numpy as np
import os

# Number of samples
n_samples = 1000

# Generate random data
np.random.seed(42)
ids = range(1, n_samples + 1)
ages = np.random.randint(18, 70, size=n_samples)
prices = np.random.randint(10, 1000, size=n_samples)
categories = np.random.choice(['Groceries', 'Electronics', 'Entertainment', 'Travel'], size=n_samples)
fraud = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.1, 0.9])

# Create a DataFrame
data = pd.DataFrame({
    'ID': ids,
    'Age': ages,
    'Price': prices,
    'Category of Spend': categories,
    'Fraud': fraud
})

# Ensure the directory exists
output_dir = 'C:/Users/goure/decision-tree-model/data/training dataset'
os.makedirs(output_dir, exist_ok=True)

# Save the synthetic data to a CSV file in the specified folder
output_file = os.path.join(output_dir, 'synthetic_fraud_data.csv')
data.to_csv(output_file, index=False)
print(f"Synthetic data saved to {output_file}")