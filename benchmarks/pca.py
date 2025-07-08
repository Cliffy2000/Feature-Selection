import pandas as pd
import numpy as np
import os
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from utils.util import load_and_preprocess_data, print_section
from core.models import benchmark_models


def analyze_pca_components(data_path="data/kaggle_diabetes_data.csv", target_column="Diabetes_binary", overwrite=False):
    print_section("PCA Component Analysis")

    # Load data
    X, y = load_and_preprocess_data(data_path, target_column)

    # Run PCA with all components
    pca = PCA(n_components=None)
    pca_result = pca.fit_transform(X)

    # Extract component information
    feature_names = X.columns.tolist()
    n_components = pca.n_components_

    print(f"Number of features: {len(feature_names)}")
    print(f"Number of components: {n_components}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Log all components to file
    os.makedirs("results/pca", exist_ok=True)

    # Extract dataset name from path for filename
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = f"results/pca/{dataset_name}_components.txt"

    # Check if file exists and handle overwrite
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to replace.")
        return pca, pca_result, X, y

    with open(output_file, "w") as f:
        f.write("PCA Component Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Target Column: {target_column}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Number of Components: {n_components}\n")
        f.write(f"Total Samples: {len(X)}\n\n")

        for i in range(n_components):
            f.write(f"PC{i + 1} (Explained Variance: {pca.explained_variance_ratio_[i]:.4f})\n")
            f.write("-" * 40 + "\n")

            # Get component loadings
            loadings = pca.components_[i]

            # Create feature-loading pairs and sort by absolute value
            feature_loadings = [(feature_names[j], loadings[j]) for j in range(len(feature_names))]
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)

            for feature, loading in feature_loadings:
                f.write(f"  {feature:<25}: {loading:>8.4f}\n")

            f.write("\n")

    print(f"Component analysis logged to {output_file}")

    return pca, pca_result, X, y


def run_pca_model_benchmarks(data_path="data/kaggle_diabetes_data.csv", target_column="Diabetes_binary", overwrite=False):
    print_section("PCA Model Benchmarks")

    # Load data
    X, y = load_and_preprocess_data(data_path, target_column)
    X_full = X.values
    y_full = y.values

    # Setup output file
    os.makedirs("results/pca", exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = f"results/pca/{dataset_name}_model_benchmarks.txt"

    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to replace.")
        return

    # Test different numbers of PCA components
    results = []
    component_counts = [1, 2, 3, 5, 10, 15, 16, 18, 20, 21]

    for k in component_counts:
        print(f"Testing {k} components...")

        if k < 21:
            X_k = PCA(n_components=k).fit_transform(X_full)
        else:
            X_k = X_full

        X_train, X_test, y_train, y_test = train_test_split(X_k, y_full, test_size=0.3)

        for name, clf in benchmark_models.items():
            t0 = time.time()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            prob = clf.predict_proba(X_test)[:, 1]
            ll = log_loss(y_test, prob)

            results.append({
                'Method': name,
                'Components': k,
                'Accuracy': accuracy_score(y_test, preds),
                'AUC': roc_auc_score(y_test, prob),
                'F1': f1_score(y_test, preds),
                'LogLossScore': 1 / (1 + ll),
                'Runtime_s': round(time.time() - t0, 4)
            })

    # Create results DataFrame
    pca_df = pd.DataFrame(results)
    methods = list(benchmark_models.keys())
    metrics = ['Accuracy', 'AUC', 'F1', 'LogLossScore']

    # Log results to file
    with open(output_file, "w") as f:
        f.write("PCA Model Benchmarks\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Target Column: {target_column}\n")
        f.write(f"Total Samples: {len(X_full)}\n")
        f.write(f"Test Split: 30%\n\n")

        for k in component_counts:
            f.write(f"PCA Benchmarks: {k} Components\n")
            f.write("-" * 40 + "\n")

            subset = pca_df[pca_df['Components'] == k].set_index('Method')
            subset = subset.reindex(methods)

            f.write(f"{'Method':<20}")
            for metric in metrics:
                f.write(f"{metric:>12}")
            f.write(f"{'Runtime_s':>12}\n")
            f.write("-" * (20 + 12 * 5) + "\n")

            for method in methods:
                f.write(f"{method:<20}")
                for metric in metrics:
                    f.write(f"{subset.loc[method, metric]:>12.4f}")
                f.write(f"{subset.loc[method, 'Runtime_s']:>12.4f}\n")
            f.write("\n")

    # Print to console
    for k in component_counts:
        subset = pca_df[pca_df['Components'] == k].set_index('Method')
        subset = subset.reindex(methods)
        print_section(f"PCA Benchmarks: {k} Components")
        print(subset[metrics])

    print(f"\nModel benchmarks logged to {output_file}")