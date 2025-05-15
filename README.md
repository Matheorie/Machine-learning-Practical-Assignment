# Wine Quality Classification

This repository contains the code and documentation for our Wine Quality prediction project, developed as part of the "Fundamentals of artificial intelligence" course at Riga Technical University.

## Project Overview

In this project, we analyze the Wine Quality dataset from the UCI Machine Learning Repository, which contains physicochemical properties of red and white Portuguese "Vinho Verde" wines. We apply both unsupervised and supervised machine learning techniques to explore the data structure and predict wine quality classes.

## Dataset

The dataset includes:
- 6,497 wine samples (1,599 red wine samples and 4,898 white wine samples)
- 11 input features (physicochemical properties)
- Wine quality ratings (scores from 3 to 9)
- We've converted the original scores into three quality classes:
  - Bad (scores 3-4)
  - Medium (scores 5-6)
  - Good (scores 7-9)

## Project Structure

The repository is organized into three main parts:

1. **Data Exploration** (`section1-data_exploration.py`)
   - Loading and preprocessing the data
   - Statistical analysis of features
   - Visualization of feature relationships
   - Correlation analysis

2. **Unsupervised Learning** (`section2-unsupervised_learning.py`)
   - Hierarchical clustering with single linkage method
   - K-means clustering with silhouette analysis
   - Comparison of clustering results with actual quality classes

3. **Supervised Learning** (`section3-supervised_learning.py`)
   - Neural Networks (MLP classifier)
   - Random Forest classifier
   - Support Vector Machine (SVM) classifier
   - Performance evaluation and comparison

## Key Findings

- **Feature Importance**: Alcohol content, volatile acidity, and sulphates were identified as the most important features for predicting wine quality.
- **Model Performance**: Random Forest achieved the highest accuracy (84.87%), followed by Neural Network (81.18%) and SVM (80.56%).
- **Class Imbalance Effects**: All models struggled with the minority "bad" class, with Random Forest showing the highest precision but low recall.

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Installation

```bash
git clone https://github.com/Matheorie/Machine-learning-Practical-Assignment.git
cd Machine-learning-Practical-Assignment
pip install -r requirements.txt
```

### Running the Code

```bash
# Data exploration
python section1-data_exploration.py

# Unsupervised learning
python section2-unsupervised_learning.py

# Supervised learning
python section3-supervised_learning.py
```

## Team Members

Team-10:
- Bahadir Mammadli (231ADB243)
- Matheo Gilbert Judenne (250AEB016)
- Ruslan Nasirov (231ADB129)
- Preowei Samuel Fakrogha (231ADB012)
- Emil Babayev (231ADB291)

## Acknowledgments

- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis for the original Wine Quality dataset
- Teaching staff at RTU DITEF: Alla Anohina-Naumeca
