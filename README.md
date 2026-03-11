# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, sklearn, matplotlib, and seaborn and Load the dataset HeightsWeights.csv.
2. Select the features Height (Inches) and Weight (Pounds).
3. Visualize the original data distribution using a scatter plot.
4. Apply StandardScaler to standardize the feature values and Apply PCA to reduce the dimensionality of the data.
5. Calculate the explained variance ratio of the principal components and Transform the dataset into principal components.
6. Visualize the PCA-transformed data using a scatter plot.
## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Lenasri R
RegisterNumber: 212225040199 
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())
X=data[['Height(Inches)', 'Weight(Pounds)']]
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="664" height="161" alt="image" src="https://github.com/user-attachments/assets/74b3bbb8-5b46-4674-a1e8-5edb5915cda3" />

<img width="779" height="609" alt="image" src="https://github.com/user-attachments/assets/477d8ddb-a375-431b-9a17-7a6bc6fa9732" />

<img width="760" height="628" alt="image" src="https://github.com/user-attachments/assets/bc539ce7-6a9c-4034-af4b-ebf771b79c64" />

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
