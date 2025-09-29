"""

KNN Performance (Full Dataset):
------------------------------
    k=  5: 0.7028
    k= 10: 0.7140
    k= 25: 0.7319
    k= 50: 0.7357
  * k=100: 0.7388
    k=200: 0.7394
    k=500: 0.7387

KNN Performance (n=5000 samples, 5 trials):
------------------------------
    k=  5: 0.7045 (±0.0115)
    k= 10: 0.6996 (±0.0226)
  * k= 25: 0.7333 (±0.0116)
    k= 50: 0.7156 (±0.0061)
    k=100: 0.7232 (±0.0100)
    k=200: 0.7307 (±0.0132)
    k=500: 0.7444 (±0.0104)


"""



import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.data_loader import load_data

# Load data
X, y = load_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Dataset shape: {X.shape}")

# Single 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nKNN Performance (Full Dataset):")
print("-" * 30)

for k in [5, 10, 25, 50, 100, 200, 500]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"k={k:3d}: {accuracy:.4f}")

# Test with n=5000 sampling
print("\nKNN Performance (n=5000 samples, 5 trials):")
print("-" * 30)

k_values = [5, 10, 25, 50, 100, 200, 500]
for k in k_values:
    scores = []
    for trial in range(5):
        idx = np.random.choice(len(X), 5000, replace=True)
        X_sample = X[idx]
        y_sample = y[idx]

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sample, y_sample, test_size=0.3, random_state=None
        )

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_s, y_train_s)
        scores.append(knn.score(X_test_s, y_test_s))

    print(f"k={k:3d}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")