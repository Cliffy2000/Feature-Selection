import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.data_loader import *


def test_epoch_sensitivity(n_trials=2500):
    X, y = load_clean_iris()
    noise = np.random.randn(X.shape[0], 5)
    X = np.hstack([X, noise])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    epoch_ranges = range(1, 26, 1)
    clean_means = []
    clean_stds = []
    noisy_means = []
    noisy_stds = []

    for n_epochs in tqdm(epoch_ranges, desc="Testing epochs"):
        results = []

        for trial in range(n_trials):
            n_features = np.random.randint(2, 10)
            feature_mask = np.random.choice(9, n_features, replace=False)

            X_subset = X_tensor[:, feature_mask]

            model = nn.Sequential(
                nn.Linear(n_features, 6),
                nn.Tanh(),
                nn.Linear(6, 3)
            )

            for m in model.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.xavier_normal_(m.weight, gain=2.0)
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.zeros_(m.bias)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.05)

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                outputs = model(X_subset)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_subset)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()

            has_noise = any(f >= 4 for f in feature_mask)

            results.append({
                'accuracy': accuracy,
                'has_noise': has_noise
            })

        clean_accs = [r['accuracy'] for r in results if not r['has_noise']]
        noisy_accs = [r['accuracy'] for r in results if r['has_noise']]

        clean_means.append(np.mean(clean_accs) if clean_accs else np.nan)
        clean_stds.append(np.std(clean_accs) if clean_accs else 0)
        noisy_means.append(np.mean(noisy_accs) if noisy_accs else np.nan)
        noisy_stds.append(np.std(noisy_accs) if noisy_accs else 0)

    epoch_list = list(epoch_ranges)
    clean_means = np.array(clean_means)
    clean_stds = np.array(clean_stds)
    noisy_means = np.array(noisy_means)
    noisy_stds = np.array(noisy_stds)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, clean_means, 'b-', label='Clean features', linewidth=2)
    plt.fill_between(epoch_list, clean_means - clean_stds, clean_means + clean_stds, alpha=0.3, color='b')
    plt.plot(epoch_list, noisy_means, 'r-', label='With noise features', linewidth=2)
    plt.fill_between(epoch_list, noisy_means - noisy_stds, noisy_means + noisy_stds, alpha=0.3, color='r')

    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Feature Quality Discrimination vs Training Epochs', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    separation = clean_means - noisy_means
    valid_separation = separation[~np.isnan(separation)]
    valid_epochs = [epoch_list[i] for i, s in enumerate(separation) if not np.isnan(s)]

    if valid_separation.size > 0:
        optimal_epoch = valid_epochs[np.argmax(valid_separation)]
        print(f"\nOptimal epochs for discrimination: {optimal_epoch}")
        print(f"Max separation: {np.max(valid_separation):.3f}")
    else:
        print("\nInsufficient data for analysis")


test_epoch_sensitivity()