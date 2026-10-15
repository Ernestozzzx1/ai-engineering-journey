
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a pandas DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the flower species labels
df["species"] = iris.target

# Display dataset information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nFeature names:")
print(iris.feature_names)

print("\nFlower species:")
print(iris.target_names)

print("\nDataset statistics:")
print(df.describe())
