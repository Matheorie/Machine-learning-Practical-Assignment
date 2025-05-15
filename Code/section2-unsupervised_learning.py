import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram

# Load data
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# Add a column to distinguish wine types
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# Combine the two datasets
wine = pd.concat([red_wine, white_wine], axis=0)

# Convert quality into categories for classification
def quality_to_class(quality):
    if quality <= 4:
        return 'bad'
    elif quality <= 6:
        return 'medium'
    else:
        return 'good'

wine['quality_class'] = wine['quality'].apply(quality_to_class)

# Data preparation
# Select numerical features
features = wine.drop(columns=['quality', 'quality_class', 'wine_type'])

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

print("Data loaded and preprocessed.")
print(f"Total number of samples: {len(wine)}")
print(f"Feature dimensions: {features.shape}")

# 1. K-means algorithm
# Calculate silhouette coefficient for different k values
print("\nCalculating silhouette scores for K-means...")
silhouette_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, clusters))
    print(f"k = {k}: silhouette score = {silhouette_scores[-1]:.4f}")

# Visualize silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Values of k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig('kmeans_silhouette.png')
print("Silhouette score graph saved: kmeans_silhouette.png")

# Apply K-means with the best k
best_k = np.argmax(silhouette_scores) + 2
print(f"\nThe best number of clusters (k) is {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to the dataset
wine['kmeans_cluster'] = clusters

# Visualize clusters (for the first 2 features)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title(f'K-means Clustering (k={best_k})')
plt.xlabel('Fixed Acidity (Scaled)')
plt.ylabel('Volatile Acidity (Scaled)')
plt.savefig('kmeans_clusters.png')
print(f"K-means clusters visualization saved: kmeans_clusters.png")

# 2. Hierarchical clustering
# Function to plot dendrogram
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

print("\nApplying hierarchical clustering...")
# Test different cut-off lines with 3, 4, and 5 clusters
for n_clusters in [3, 4, 5]:
    print(f"Hierarchical clustering with {n_clusters} clusters...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    hierarchical_clusters = hierarchical.fit_predict(X_scaled)
    
    wine[f'hierarchical_cluster_{n_clusters}'] = hierarchical_clusters
    
    # Visualize dendrogram (for a subset of data if necessary)
    plt.figure(figsize=(12, 8))
    plt.title(f'Hierarchical Clustering Dendrogram (n_clusters={n_clusters})')
    
    # If dataset is too large, use a subset
    if len(features) > 100:
        print("  Using a subset for the dendrogram...")
        sample_indices = np.random.choice(len(features), 100, replace=False)
        X_sample = X_scaled[sample_indices]
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')
        model = model.fit(X_sample)
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plot_dendrogram(model, truncate_mode='level', p=3)
    else:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')
        model = model.fit(X_scaled)
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plot_dendrogram(model, truncate_mode='level', p=3)
    
    plt.axhline(y=0.5 * max(model.distances_), color='r', linestyle='--')
    plt.savefig(f'hierarchical_dendrogram_{n_clusters}.png')
    print(f"  Dendrogram saved: hierarchical_dendrogram_{n_clusters}.png")
    
    # Visualize clusters (for the first 2 features)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Hierarchical Clustering (n_clusters={n_clusters})')
    plt.xlabel('Fixed Acidity (Scaled)')
    plt.ylabel('Volatile Acidity (Scaled)')
    plt.savefig(f'hierarchical_clusters_{n_clusters}.png')
    print(f"  Clusters visualization saved: hierarchical_clusters_{n_clusters}.png")

# Compare clusters with actual classes
print("\nComparing clusters with actual quality classes...")
for n_clusters in [best_k, 3, 4, 5]:
    if n_clusters == best_k:
        cluster_col = 'kmeans_cluster'
        algo_name = 'K-means'
    else:
        cluster_col = f'hierarchical_cluster_{n_clusters}'
        algo_name = 'Hierarchical'
    
    plt.figure(figsize=(10, 6))
    crosstab = pd.crosstab(wine[cluster_col], wine['quality_class'])
    crosstab.plot(kind='bar', stacked=True)
    plt.title(f'{algo_name} Clusters vs. Quality Classes (n_clusters={n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Quality Class')
    plt.tight_layout()
    plt.savefig(f'{algo_name.lower()}_vs_quality_{n_clusters}.png')
    print(f"  {algo_name} comparison saved: {algo_name.lower()}_vs_quality_{n_clusters}.png")

print("\nUnsupervised learning analysis completed!")