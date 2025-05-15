import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# Add a column to distinguish wine types
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# Combine the two datasets
wine = pd.concat([red_wine, white_wine], axis=0)

# Check the data
print(wine.head())
print(wine.info())
print(wine.describe())

# Check for missing values
print(wine.isnull().sum())

# Convert quality into categories for classification
# For example, we can divide into 3 classes: bad (0-4), medium (5-6), good (7-10)
def quality_to_class(quality):
    if quality <= 4:
        return 'bad'
    elif quality <= 6:
        return 'medium'
    else:
        return 'good'

wine['quality_class'] = wine['quality'].apply(quality_to_class)

# Number of samples per class
print(wine['quality_class'].value_counts())

# Visualizations
# 1. 2D scatter plot to see class separability
plt.figure(figsize=(10, 6))
sns.scatterplot(x='alcohol', y='density', hue='quality_class', data=wine)
plt.title('Alcohol vs Density by Wine Quality')
plt.savefig('scatter_alcohol_density.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='fixed acidity', y='pH', hue='quality_class', data=wine)
plt.title('Fixed Acidity vs pH by Wine Quality')
plt.savefig('scatter_acidity_ph.png')

# 2. Histograms by class
plt.figure(figsize=(12, 8))
for i, quality_class in enumerate(['bad', 'medium', 'good']):
    plt.subplot(3, 1, i+1)
    subset = wine[wine['quality_class'] == quality_class]
    sns.histplot(subset['alcohol'], kde=True)
    plt.title(f'Alcohol Distribution for {quality_class.capitalize()} Wine')
plt.tight_layout()
plt.savefig('hist_alcohol_by_class.png')

# 3. Distribution of features of interest
plt.figure(figsize=(12, 8))
sns.histplot(wine['quality'], kde=False, discrete=True)
plt.title('Distribution of Wine Quality Scores')
plt.savefig('hist_quality.png')

plt.figure(figsize=(12, 8))
sns.histplot(wine['alcohol'], kde=True)
plt.title('Distribution of Alcohol Content')
plt.savefig('hist_alcohol.png')

# 4. Statistical indicators for each class
stats_by_class = wine.groupby('quality_class').describe()
print(stats_by_class)

# Correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = wine.drop(columns=['quality_class', 'wine_type']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')