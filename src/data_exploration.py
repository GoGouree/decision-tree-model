import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset using an absolute path
data = pd.read_csv('C:/Users/goure/decision-tree-model/data/raw/Sample_Data.csv')

# Display basic information about the dataset
print("Basic Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(data.head())

# Generate summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Plot a histogram for each numerical feature
print("\nHistograms:")
data.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Plot a box plot for each numerical feature
print("\nBox Plots:")
data.plot(kind='box', subplots=True, layout=(5, 5), figsize=(15, 10))
plt.suptitle('Box Plots of Numerical Features')
plt.show()

# Plot a scatter plot for pairs of features
print("\nScatter Plot Matrix:")
sns.pairplot(data)
plt.suptitle('Scatter Plot Matrix')
plt.show()

# Plot a correlation heatmap
print("\nCorrelation Heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Plot a bar plot for categorical features
print("\nBar Plots for Categorical Features:")
categorical_features = data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data[feature])
    plt.title(f'Bar Plot of {feature}')
    plt.show()