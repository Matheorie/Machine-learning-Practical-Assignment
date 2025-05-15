import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

print("Data loaded and preprocessed.")
print(f"Total number of samples: {len(wine)}")
print("Distribution of quality classes:")
print(wine['quality_class'].value_counts())

# Data preparation
# Select features and target
X = wine.drop(columns=['quality', 'quality_class', 'wine_type'])
y = wine['quality_class']

# Convert target to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and test sets (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Information about training and test sets
print(f"\nTotal number of objects in training set: {len(X_train)} ({len(X_train) / len(X) * 100:.2f}%)")
print(f"Total number of objects in test set: {len(X_test)} ({len(X_test) / len(X) * 100:.2f}%)")

# Number of objects per class in training and test sets
train_class_counts = np.bincount(y_train)
test_class_counts = np.bincount(y_test)

for i, class_name in enumerate(le.classes_):
    print(f"\nClass '{class_name}':")
    print(f"  - Training: {train_class_counts[i]} objects ({train_class_counts[i] / len(y_train) * 100:.2f}%)")
    print(f"  - Test: {test_class_counts[i]} objects ({test_class_counts[i] / len(y_test) * 100:.2f}%)")

# 1. Artificial Neural Networks
print("\n--- Experiments with Artificial Neural Networks ---")
# Experiments with different hyperparameters
nn_experiments = [
    {'hidden_layer_sizes': (100,), 'activation': 'logistic', 'learning_rate_init': 0.1, 'max_iter': 500},
    {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'learning_rate_init': 0.01, 'max_iter': 1000},
    {'hidden_layer_sizes': (100, 50, 25), 'activation': 'tanh', 'learning_rate_init': 0.001, 'max_iter': 1500}
]

nn_results = []

for i, params in enumerate(nn_experiments):
    print(f"\nExperiment {i+1} with Neural Network:")
    print(params)
    
    mlp = MLPClassifier(random_state=42, **params)
    mlp.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Predictions on training set
    y_train_pred = mlp.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy on training set: {train_accuracy:.4f}")
    
    # Store results
    nn_results.append({
        'model': mlp,
        'params': params,
        'cv_mean': cv_scores.mean(),
        'train_accuracy': train_accuracy
    })

# Select the best neural network model
best_nn = max(nn_results, key=lambda x: x['cv_mean'])
print(f"\nBest Neural Network model:")
print(best_nn['params'])
print(f"Mean cross-validation score: {best_nn['cv_mean']:.4f}")

# 2. Random Forest
print("\n--- Experiments with Random Forest ---")
# Experiments with different hyperparameters
rf_experiments = [
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10}
]

rf_results = []

for i, params in enumerate(rf_experiments):
    print(f"\nExperiment {i+1} with Random Forest:")
    print(params)
    
    rf = RandomForestClassifier(random_state=42, **params)
    rf.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Predictions on training set
    y_train_pred = rf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy on training set: {train_accuracy:.4f}")
    
    # Store results
    rf_results.append({
        'model': rf,
        'params': params,
        'cv_mean': cv_scores.mean(),
        'train_accuracy': train_accuracy
    })

# Select the best Random Forest model
best_rf = max(rf_results, key=lambda x: x['cv_mean'])
print(f"\nBest Random Forest model:")
print(best_rf['params'])
print(f"Mean cross-validation score: {best_rf['cv_mean']:.4f}")

# 3. SVM
print("\n--- Experiments with SVM ---")
# Experiments with different hyperparameters
svm_experiments = [
    {'C': 1.0, 'kernel': 'linear'},
    {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 100.0, 'kernel': 'poly', 'degree': 3}
]

svm_results = []

for i, params in enumerate(svm_experiments):
    print(f"\nExperiment {i+1} with SVM:")
    print(params)
    
    svm = SVC(random_state=42, **params)
    svm.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Predictions on training set
    y_train_pred = svm.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy on training set: {train_accuracy:.4f}")
    
    # Store results
    svm_results.append({
        'model': svm,
        'params': params,
        'cv_mean': cv_scores.mean(),
        'train_accuracy': train_accuracy
    })

# Select the best SVM model
best_svm = max(svm_results, key=lambda x: x['cv_mean'])
print(f"\nBest SVM model:")
print(best_svm['params'])
print(f"Mean cross-validation score: {best_svm['cv_mean']:.4f}")

# Test the best models on the test set
print("\n--- Evaluation of best models on test set ---")
best_models = [
    ('Neural Network', best_nn['model']),
    ('Random Forest', best_rf['model']),
    ('SVM', best_svm['model'])
]

for name, model in best_models:
    # Predictions on test set
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTest results for {name}:")
    print(f"Accuracy on test set: {test_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    print(f"Confusion matrix saved: confusion_matrix_{name.replace(' ', '_').lower()}.png")

# Feature importance (for Random Forest)
if hasattr(best_rf['model'], 'feature_importances_'):
    importances = best_rf['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance saved: feature_importance.png")

print("\nSupervised learning analysis completed!")