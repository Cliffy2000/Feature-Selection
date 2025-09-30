import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.data_loader import load_data
from core.genetic_algorithms import *

# Load data
X, y = load_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Dataset shape: {X.shape}")

# Single 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("\nKNN Performance (Full Dataset):")
print("-" * 30)


chromosome = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
 0.9, 0.9, 0.9, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.33537587])


ga_temp = ThresholdDecodingGA(X, y, population_size=2, generations=1,
                              elitism_ratio=0.5, crossover_rate=0.5, mutation_rate=0.1)
decoded = ga_temp.decode(chromosome)
weights = decoded[:-1]

n_selected = np.sum(weights > 0)
print(f"Features selected: {int(n_selected)} out of {len(weights)}")

for k in [5, 10, 25, 50, 100, 200, 500]:
    scores = []
    for trial in range(5):
        idx = np.random.choice(len(X), 50000, replace=True)
        X_sample = X[idx] * weights
        y_sample = y[idx]

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sample, y_sample, test_size=0.3, random_state=None
        )

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_s, y_train_s)
        scores.append(knn.score(X_test_s, y_test_s))

    print(f"k={k:3d}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")